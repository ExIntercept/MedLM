"""FastAPI backend with JWT auth, hybrid RAG retrieval, and real-time MedGemma streaming from Colab."""
import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from src.agents.emergency_triage import EMERGENCY_WARNING, check_emergency_triage
from src.agents.guardrails import check_hard_rules, format_guardrail_intercept, parse_embedded_options
from src.agents.intake_extraction import extract_intake_fields
from src.agents.prompting import build_prompt, format_patient_profile, sanitize_patient_output
from src.agents.verifier import verify_evidence
from src.auth.dependencies import get_current_user
from src.auth.security import create_access_token, hash_password, verify_password
from src.config import DOCS_DIR, MODEL_ID, PROMPT_STYLE, ROOT
from src.db.models import ApiUser, Conversation, ConversationMessage
from src.db.orm import ENGINE, get_db_session, init_db
from src.llm.client import health_sync, stream_generate
from src.retrieval.retriever import collection, embedder, reranker, retrieve_context
from src.security import run_input_defenses, sanitize_user_inputs, warmup
from src.security.logging import log_decision

app = FastAPI(title="Medical RAG & MedGemma Clinical API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8600",
        "http://127.0.0.1:8600",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup():
    init_db()
    # Load defense detectors in the background so startup isn't blocked.
    # The first chat request will wait on each detector's load lock if warmup
    # hasn't finished yet.
    threading.Thread(target=warmup, daemon=True, name="defense-warmup").start()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None


class UserPublic(BaseModel):
    id: int
    username: str
    email: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PatientProfile(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    duration: Optional[str] = None
    conditions: Optional[str] = None
    medications: Optional[str] = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatStreamRequest(BaseModel):
    conversation_id: Optional[Union[str, int]] = None
    message: str
    patient_profile: Optional[PatientProfile] = None
    # DEPRECATED: server now uses DB history exclusively to prevent turn-forgery attacks.
    # Field is kept so old clients pass schema validation, but its value is ignored.
    history: list[ChatTurn] = Field(default_factory=list)
    mode: Literal["patient", "clinician"] = "patient"


# Patient-facing refusal message for security blocks.
# Deliberately vague: reveals nothing about which detector fired.
BLOCK_MESSAGE = (
    "I can only help with genuine health questions, and I can't respond to role-play or "
    "instruction-override requests. If you have a medical question, please ask it directly "
    "and I'll do my best to help. If this is an emergency, contact your local emergency number."
)



class NewConversationRequest(BaseModel):
    title: Optional[str] = None
    patient_profile: Optional[PatientProfile] = None


class MedicationCheckRequest(BaseModel):
    medications: Optional[str] = None
    conditions: Optional[str] = None


def _format_patient_profile(profile: Optional[PatientProfile]) -> str:
    if not profile:
        return ""
    return format_patient_profile(
        age=profile.age,
        sex=profile.sex,
        duration=profile.duration,
        conditions=profile.conditions,
        medications=profile.medications,
    )


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def _save_message(db: Session, conversation_id: int, role: str, content: str, meta: Optional[dict] = None) -> None:
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        meta=json.dumps(meta) if meta is not None else None,
    )
    db.add(msg)
    db.commit()


def _conversation_history(db: Session, conversation_id: int, last_n: int = 6) -> list:
    rows = db.exec(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.id.desc())
        .limit(last_n)
    ).all()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _conversation_messages_full(db: Session, conversation_id: int) -> list:
    rows = db.exec(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.id.asc())
    ).all()
    return [{"role": r.role, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]


def _conversation_payload(db: Session, conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "date": conv.created_at.isoformat(),
        "subject": conv.title or f"Consultation #{conv.id}",
        "patient_profile": json.loads(conv.patient_profile) if conv.patient_profile else None,
        "messages": _conversation_messages_full(db, conv.id),
    }


def _get_owned_conversation(db: Session, conversation_id: int, user_id: int) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# --------------------------------------------------------------------------
# Health Endpoint
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    colab_health = health_sync()

    try:
        chromadb_count = collection.count()
        chromadb_ok = chromadb_count > 0
    except Exception:
        chromadb_count = 0
        chromadb_ok = False

    reranker_ok = reranker is not None and embedder is not None

    return {
        "colab_medgemma": colab_health.get("ok", False),
        "colab_details": colab_health,
        "chromadb": chromadb_ok,
        "chromadb_chunks": chromadb_count,
        "reranker": reranker_ok,
        "model": MODEL_ID,
        "prompt_style": PROMPT_STYLE,
        "status": "ok" if (colab_health.get("ok", False) and chromadb_ok and reranker_ok) else "degraded",
    }


# --------------------------------------------------------------------------
# Evaluation & Benchmark Endpoints
# --------------------------------------------------------------------------
_BENCHMARK_RESULTS_PATH = DOCS_DIR / "benchmark_results.json"
_BENCHMARK_REPORT_PATH = DOCS_DIR / "BENCHMARK_REPORT.md"


@app.get("/api/evaluation/summary")
def evaluation_summary():
    if not _BENCHMARK_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found in docs/benchmark_results.json",
        )
    try:
        payload = json.loads(_BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"benchmark_results.json is not valid JSON: {exc}")
    return {
        "generated_at": payload.get("generated_at"),
        "model": payload.get("model", MODEL_ID),
        "dataset_version": payload.get("dataset_version"),
        "num_cases": payload.get("num_cases"),
        "metrics": payload.get("metrics"),
        "cases": payload.get("cases", []),
    }


@app.get("/api/evaluation/report")
def evaluation_report():
    if not _BENCHMARK_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report found in docs/BENCHMARK_REPORT.md",
        )
    return {"markdown": _BENCHMARK_REPORT_PATH.read_text(encoding="utf-8")}


# --------------------------------------------------------------------------
# Auth Endpoints
# --------------------------------------------------------------------------
@app.post("/api/auth/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db_session)):
    username = payload.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db.exec(select(ApiUser).where(ApiUser.username == username)).first()
    if existing:
        raise HTTPException(status_code=409, detail="That username is already taken.")

    user = ApiUser(username=username, email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserPublic(id=user.id, username=user.username, email=user.email)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_session)):
    user = db.exec(select(ApiUser).where(ApiUser.username == form_data.username.strip())).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=user.username)
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserPublic)
def read_me(current_user: ApiUser = Depends(get_current_user)):
    return UserPublic(id=current_user.id, username=current_user.username, email=current_user.email)


# --------------------------------------------------------------------------
# Conversation Endpoints
# --------------------------------------------------------------------------
@app.get("/api/conversations")
def list_conversations(
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    rows = db.exec(
        select(Conversation).where(Conversation.user_id == current_user.id).order_by(Conversation.id.desc())
    ).all()
    return [_conversation_payload(db, c) for c in rows]


@app.get("/api/conversations/{conversation_id}")
def get_conversation_detail(
    conversation_id: int,
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    conv = _get_owned_conversation(db, conversation_id, current_user.id)
    return _conversation_payload(db, conv)


@app.post("/api/conversations/new")
def new_conversation(
    payload: NewConversationRequest = NewConversationRequest(),
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    title = (payload.title or "New Consultation").strip()[:60] or "New Consultation"
    profile_json = json.dumps(payload.patient_profile.model_dump()) if payload.patient_profile else None
    conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"conversation_id": conv.id, "title": title}


# --------------------------------------------------------------------------
# Clinical Utility Endpoints
# --------------------------------------------------------------------------
@app.post("/api/medications/check")
def check_medications(payload: MedicationCheckRequest, current_user: ApiUser = Depends(get_current_user)):
    medications_text = (payload.medications or "").strip()
    if not medications_text:
        return {"medications": [], "alert": None}

    medications = [m.strip() for m in medications_text.split(",") if m.strip()]
    combined_context = f"{medications_text}. Reported conditions: {payload.conditions or 'none reported'}."
    hit = check_hard_rules(combined_context)
    alert = hit[2] if hit else None
    return {"medications": medications, "alert": alert}


_PATIENT_FRIENDLY_FIELDS = {"clinical_guideline", "indications"}


@app.get("/api/evidence/search")
def search_evidence(q: str = "", mode: str = "clinician", current_user: ApiUser = Depends(get_current_user)):
    query = q.strip()
    if not query:
        return {"results": []}
    results = retrieve_context(query, top_k=8)
    payload = [
        {
            "chunk_id": str(chunk_id),
            "title": meta.get("title", "Guideline"),
            "source": meta.get("source", ""),
            "field": meta.get("field", ""),
            "score": round(float(meta.get("cross_score", 0.0)), 4),
            "excerpt": text[:400],
        }
        for chunk_id, text, meta in results
    ]
    if mode == "patient":
        payload.sort(key=lambda r: 0 if r["field"] in _PATIENT_FRIENDLY_FIELDS else 1)
    return {"results": payload}


# --------------------------------------------------------------------------
# Pipeline Configuration & Dynamic Layer Ablation Controls
# --------------------------------------------------------------------------
PIPELINE_CONFIG: dict[str, bool] = {
    "defense_enabled": True,
    "layer_sanitize": True,
    "layer_triage": True,
    "layer_guardrails": True,
    "layer_prompt_guard": True,
    "layer_template_matcher": True,
    "layer_prompt_hardening": True,
}


# --------------------------------------------------------------------------
# Real-Time Chat Stream (SSE)
# --------------------------------------------------------------------------
def _run_pipeline(req: ChatStreamRequest, conversation_id: int, sink: "queue.Queue"):
    with Session(ENGINE) as db:
        try:
            # ── Layer S: Sanitise all user-controlled inputs ──────────────────
            if PIPELINE_CONFIG.get("layer_sanitize", True):
                san = sanitize_user_inputs(req.message, req.patient_profile)
                raw_input = san.message.model_text  # sanitised, used hereafter
            else:
                from src.security.sanitize import SanitizedField, SanitizedInputs
                san = SanitizedInputs(
                    message=SanitizedField(model_text=req.message, detector_text=req.message, modified=False, notes=[]),
                    profile={},
                )
                raw_input = req.message

            # ── Layer 0: Emergency Red-Flag Triage ──
            if PIPELINE_CONFIG.get("layer_triage", True):
                triage_category = check_emergency_triage(raw_input)
                if triage_category:
                    _save_message(db, conversation_id, "user", raw_input)
                    _save_message(
                        db,
                        conversation_id,
                        "assistant",
                        EMERGENCY_WARNING,
                        meta={"emergency_triage": triage_category},
                    )
                    sink.put({"event": "sources", "data": []})
                    sink.put({"event": "triage_status", "data": {"level": "EMERGENCY", "category": triage_category}})
                    sink.put({"event": "token", "data": EMERGENCY_WARNING})
                    sink.put({"event": "done", "data": {"status": "EMERGENCY_ESCALATION", "score": 1.0}})
                    return

            # ── Layer 1: Adversarial Guardrail ─────
            if PIPELINE_CONFIG.get("layer_guardrails", True):
                hit = check_hard_rules(raw_input)
                if hit:
                    entity, _, warning = hit
                    response_text = format_guardrail_intercept(entity, warning, mode=req.mode)
                    _save_message(db, conversation_id, "user", raw_input)
                    _save_message(db, conversation_id, "assistant", response_text, meta={"guardrail_intercept": True})
                    sink.put({"event": "sources", "data": []})
                    sink.put({"event": "token", "data": response_text})
                    sink.put({"event": "done", "data": {"status": "INTERCEPTED", "score": 1.0}})
                    return

            # ── Layer D: Input Defense (Prompt Guard + Template Matcher) ──────
            # Always use server-side DB history; ignore req.history to prevent
            # turn-forgery attacks via client-supplied "assistant" turns.
            db_history = _conversation_history(db, conversation_id, last_n=6)
            prior_user_turns = [t["content"] for t in db_history if t["role"] == "user"][-2:]

            decision = None
            flagged = False
            if PIPELINE_CONFIG.get("defense_enabled", True):
                disable_a = not PIPELINE_CONFIG.get("layer_prompt_guard", True)
                disable_b = not PIPELINE_CONFIG.get("layer_template_matcher", True)
                decision = run_input_defenses(
                    san, prior_user_turns, disable_layer_a=disable_a, disable_layer_b=disable_b
                )
                log_decision(decision, conversation_id, None, san.message.detector_text)

                if decision.action == "BLOCK":
                    _save_message(db, conversation_id, "user", raw_input)
                    _save_message(
                        db,
                        conversation_id,
                        "assistant",
                        BLOCK_MESSAGE,
                        meta={
                            "security_block": decision.category,
                            "pg": decision.pg_score,
                            "tm": decision.tm_max_sim,
                            "n_components": decision.tm_n_components,
                        },
                    )
                    sink.put({"event": "sources", "data": []})
                    sink.put({"event": "security_status", "data": {"action": "BLOCK", "category": decision.category}})
                    sink.put({"event": "token", "data": BLOCK_MESSAGE})
                    sink.put({"event": "done", "data": {"status": "BLOCKED", "score": None}})
                    return

                # FLAG: continue, but pass flagged=True to build_prompt and emit status
                if decision.action == "FLAG":
                    sink.put({"event": "security_status", "data": {"action": "FLAG"}})
                    flagged = True

            # ── Layer 2: Extract real-time profile update ─────────────────────
            extracted_profile = extract_intake_fields(raw_input)
            if extracted_profile:
                sink.put({"event": "profile_update", "data": extracted_profile})

            # ── Layer 3: Hybrid RAG Retrieval (BM25 + ChromaDB + RRF + MedCPT) ─
            vignette_text, detected_options = parse_embedded_options(raw_input)
            evidence = retrieve_context(vignette_text, top_k=3)

            sources = [
                {
                    "chunk_id": str(chunk_id),
                    "title": meta.get("title", "Guideline"),
                    "source": meta.get("source", ""),
                    "score": round(float(meta.get("cross_score", 0.0)), 4),
                    "excerpt": text[:280],
                }
                for chunk_id, text, meta in evidence
            ]
            sink.put({"event": "sources", "data": sources})

            evidence_md = ""
            for idx, (_, text, meta) in enumerate(evidence, 1):
                evidence_md += f"### [Evidence {idx}] {meta.get('title', 'Guideline')} (`{meta.get('source', '')}`)\n"
                evidence_md += f"*MedCPT Rerank Score:* `{meta.get('cross_score', 0):.4f}`\n\n> {text}\n\n---\n"

            # ── Layer 4: Tone-aware, hardened MedGemma Prompt Construction ────
            # Build profile_note from sanitised profile model_texts.
            profile_note = format_patient_profile(
                age=san.profile.get("age") and san.profile["age"].model_text or None,
                sex=san.profile.get("sex") and san.profile["sex"].model_text or None,
                duration=san.profile.get("duration") and san.profile["duration"].model_text or None,
                conditions=san.profile.get("conditions") and san.profile["conditions"].model_text or None,
                medications=san.profile.get("medications") and san.profile["medications"].model_text or None,
            )
            forced_tone = "patient" if req.mode == "patient" else "clinical_qa"
            hardened = PIPELINE_CONFIG.get("layer_prompt_hardening", True)
            prompt = build_prompt(
                vignette_text,
                detected_options,
                evidence_md,
                db_history,
                forced_tone=forced_tone,
                prompt_style=PROMPT_STYLE,
                patient_profile_text=profile_note,
                hardened=hardened,
                flagged=flagged,
            )

            # ── Layer 5: Real-time token streaming from Colab MedGemma ────────
            accumulated_response = ""
            for delta in stream_generate(prompt):
                accumulated_response += delta
                sink.put({"event": "token", "data": delta})

            # Sanitize output
            accumulated_response = sanitize_patient_output(accumulated_response)

            # ── Layer 6: Post-generation evidence audit ────────────────────────
            audit = verify_evidence(accumulated_response, evidence, vignette_text)

            _save_message(
                db,
                conversation_id,
                "user",
                raw_input,
            )
            _save_message(
                db,
                conversation_id,
                "assistant",
                accumulated_response,
                meta={
                    "faithfulness_score": audit["score"],
                    "status": audit["status"],
                    "security": {
                        "action": decision.action if decision else "BYPASSED",
                        "pg": decision.pg_score if decision else None,
                        "tm_max": decision.tm_max_sim if decision else 0.0,
                        "n_components": decision.tm_n_components if decision else 0,
                    },
                },
            )

            sink.put(
                {
                    "event": "done",
                    "data": {
                        "status": audit["status"],
                        "score": audit["score"],
                        "verified_claims": audit["verified_claims"],
                        "flagged_claims": audit["flagged_claims"],
                    },
                }
            )
        except Exception as exc:
            sink.put({"event": "error", "data": str(exc)})
        finally:
            sink.put(None)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest, current_user: ApiUser = Depends(get_current_user)):
    with Session(ENGINE) as db:
        if req.conversation_id:
            try:
                conversation_id = int(req.conversation_id)
            except ValueError:
                conversation_id = None

            if conversation_id:
                _get_owned_conversation(db, conversation_id, current_user.id)
            else:
                title = req.message.strip().splitlines()[0][:60] if req.message.strip() else "New Consultation"
                profile_json = json.dumps(req.patient_profile.model_dump()) if req.patient_profile else None
                conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
                db.add(conv)
                db.commit()
                db.refresh(conv)
                conversation_id = conv.id
        else:
            title = req.message.strip().splitlines()[0][:60] if req.message.strip() else "New Consultation"
            profile_json = json.dumps(req.patient_profile.model_dump()) if req.patient_profile else None
            conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            conversation_id = conv.id

    sink: "queue.Queue" = queue.Queue()
    threading.Thread(target=_run_pipeline, args=(req, conversation_id, sink), daemon=True).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        yield {"event": "conversation", "data": str(conversation_id)}
        while True:
            item = await loop.run_in_executor(None, sink.get)
            if item is None:
                break
            data = item["data"]
            if not isinstance(data, str):
                data = json.dumps(data)
            yield {"event": item["event"], "data": data}

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------------------
# Admin & Security Audit Endpoints (Spec §13)
# --------------------------------------------------------------------------
class DefenseConfigUpdate(BaseModel):
    toggles: Optional[dict[str, bool]] = None
    thresholds: Optional[dict[str, float]] = None


@app.get("/api/admin/defense-config")
def get_defense_config():
    from src.config import DEFENSE_ENABLED, DEFENSE_THRESHOLDS_PATH
    from src.security.decision import load_thresholds
    from src.security.prompt_guard import _pg_instance, _pg_unavailable
    return {
        "toggles": PIPELINE_CONFIG,
        "thresholds": load_thresholds(DEFENSE_THRESHOLDS_PATH),
        "prompt_guard_status": "READY" if _pg_instance is not None else ("UNAVAILABLE" if _pg_unavailable else "PENDING"),
        "defense_enabled_env": DEFENSE_ENABLED,
    }


@app.post("/api/admin/defense-config")
def update_defense_config(update: DefenseConfigUpdate):
    if update.toggles:
        PIPELINE_CONFIG.update(update.toggles)
    if update.thresholds:
        from src.config import DEFENSE_THRESHOLDS_PATH
        from src.security.decision import load_thresholds
        current = load_thresholds(DEFENSE_THRESHOLDS_PATH)
        current.update(update.thresholds)
        DEFENSE_THRESHOLDS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return {"status": "ok", "toggles": PIPELINE_CONFIG}


@app.get("/api/admin/defense-log")
def get_defense_log(limit: int = 100):
    from src.config import DEFENSE_LOG_PATH
    if not DEFENSE_LOG_PATH.exists():
        return {"logs": []}
    lines = DEFENSE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
        if len(records) >= limit:
            break
    return {"logs": records}


from fastapi.responses import FileResponse

# Mount static directory
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

dist_dir = ROOT / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/app", StaticFiles(directory=dist_dir, html=True), name="frontend_app")


@app.get("/")
def index():
    if (ROOT / "static" / "index.html").exists():
        return FileResponse(ROOT / "static" / "index.html")
    if dist_dir.exists():
        return FileResponse(dist_dir / "index.html")
    return {"message": "Medical RAG API is running."}
