"""FastAPI app: serves the UI, holds accounts and memory, proxies to Colab."""
import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, llm, memory, storage
from .config import MAX_MODEL_LEN, MODEL_ID, PROMPT_STYLE, RESERVE_OUTPUT_TOKENS, ROOT

app = FastAPI(title="Meditron Local")

SESSION_COOKIE = "meditron_session"


# --- schemas -----------------------------------------------------------------

class Credentials(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    display_name: str = ""
    age: str = ""
    sex_at_birth: str = ""
    height_cm: str = ""
    weight_kg: str = ""
    conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    notes: str = ""


class FactIn(BaseModel):
    text: str


class Ask(BaseModel):
    message: str


# --- helpers -----------------------------------------------------------------

def current_user(token: Optional[str]) -> str:
    username = auth.resolve_session(token)
    if not username:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return username


async def token_counter():
    """Prefer the server's real tokenizer; fall back to an estimate once per request."""
    probe = await llm.count_tokens_remote("token budget probe")
    if probe is None:
        return llm.count_tokens_estimate, False

    cache: Dict[str, int] = {}

    def counter(text: str) -> int:
        if text not in cache:
            cache[text] = llm.count_tokens_estimate(text)
        return cache[text]

    # /tokenize is one round trip per call, which is too slow per history turn.
    # Use it for the full prompt only; estimate the parts.
    return counter, True


async def build_prompt(username: str, question: str):
    profile = storage.read_profile(username)
    facts = storage.read_memory(username)["facts"]
    history = storage.read_chat(username)
    system_block = memory.build_system_block(profile, facts)
    counter, exact_available = await token_counter()
    prompt, breakdown, dropped = memory.assemble(
        system_block=system_block,
        history=history,
        question=question,
        prompt_style=PROMPT_STYLE,
        input_budget_tokens=llm.input_budget(),
        count_tokens=counter,
    )
    exact = await llm.count_tokens_remote(prompt) if exact_available else None
    if exact is not None and breakdown["total_input"] > 0:
        # The per-section numbers are estimates; rescale them so the ledger's
        # segments add up to the tokenizer's real count instead of over-drawing.
        scale = exact / breakdown["total_input"]
        for key in ("system_and_priming", "history", "question"):
            breakdown[key] = int(round(breakdown[key] * scale))
        breakdown["total_input"] = exact
    breakdown["exact"] = exact is not None
    breakdown["budget"] = llm.input_budget()
    breakdown["reserved_for_answer"] = RESERVE_OUTPUT_TOKENS
    breakdown["dropped_turns"] = dropped
    return prompt, breakdown


# --- auth routes -------------------------------------------------------------

@app.post("/api/register")
def register(body: Credentials, response: Response):
    try:
        token = auth.register(body.username.strip(), body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return {"username": body.username.strip()}


@app.post("/api/login")
def login(body: Credentials, response: Response):
    try:
        token = auth.login(body.username.strip(), body.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
    return {"username": body.username.strip()}


@app.post("/api/logout")
def logout(response: Response, meditron_session: Optional[str] = Cookie(None)):
    auth.end_session(meditron_session)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(meditron_session: Optional[str] = Cookie(None)):
    username = auth.resolve_session(meditron_session)
    if not username:
        return {"username": None, "known_profiles": auth.list_users()}
    return {
        "username": username,
        "profile": storage.read_profile(username),
        "memory": storage.read_memory(username),
        "messages": storage.read_chat(username),
        "model": MODEL_ID,
        "prompt_style": PROMPT_STYLE,
        "max_model_len": MAX_MODEL_LEN,
    }


# --- profile and memory ------------------------------------------------------

@app.put("/api/profile")
def save_profile(body: ProfileIn, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    return storage.write_profile(username, body.model_dump())


@app.post("/api/memory")
def create_fact(body: FactIn, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="A fact needs some text.")
    return storage.add_fact(username, body.text)


@app.delete("/api/memory/{fact_id}")
def remove_fact(fact_id: str, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    storage.delete_fact(username, fact_id)
    return {"ok": True}


@app.post("/api/memory/{fact_id}/confirm")
def confirm_fact(fact_id: str, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    item = storage.confirm_suggestion(username, fact_id)
    if not item:
        raise HTTPException(status_code=404, detail="That suggestion is gone.")
    return item


@app.get("/api/memory")
def get_memory(meditron_session: Optional[str] = Cookie(None)):
    return storage.read_memory(current_user(meditron_session))


# --- chat --------------------------------------------------------------------

@app.delete("/api/chat")
def clear_chat(meditron_session: Optional[str] = Cookie(None)):
    storage.clear_chat(current_user(meditron_session))
    return {"ok": True}


@app.post("/api/context/preview")
async def preview_context(body: Ask, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    prompt, breakdown = await build_prompt(username, body.message or "(your question)")
    return {"prompt": prompt, "tokens": breakdown}


@app.post("/api/chat")
async def chat(body: Ask, meditron_session: Optional[str] = Cookie(None)):
    username = current_user(meditron_session)
    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Type a question first.")

    prompt, breakdown = await build_prompt(username, question)

    async def event_stream():
        yield f"event: context\ndata: {json.dumps(breakdown)}\n\n"
        collected: List[str] = []
        try:
            async for piece in llm.stream_completion(prompt, memory.stop_sequences(PROMPT_STYLE)):
                collected.append(piece)
                yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"
        except llm.ColabUnreachable as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        except asyncio.CancelledError:
            return

        answer = "".join(collected)
        # vLLM applies the stop strings, but a swapped-in model may emit them anyway.
        for marker in memory.stop_sequences(PROMPT_STYLE):
            answer = answer.split(marker)[0]
        answer = answer.strip()
        if not answer:
            yield f"event: error\ndata: {json.dumps({'detail': 'The model returned nothing. Lower RESERVE_OUTPUT_TOKENS or check the Colab logs.'})}\n\n"
            return

        storage.append_chat(username, "user", question)
        body_text, followups = memory.split_followups(answer)
        storage.append_chat(username, "assistant", body_text)
        suggestions = storage.add_suggestions(username, memory.suggest_facts(question))
        yield f"event: done\ndata: {json.dumps({'suggested': suggestions, 'followups': followups})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- status ------------------------------------------------------------------

@app.get("/api/health")
async def health():
    result = await llm.health()
    result["prompt_style"] = PROMPT_STYLE
    return result


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
