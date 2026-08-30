"""Turns a profile + saved facts + history into a prompt that fits in 2048 tokens."""
import re
from typing import Any, Dict, List, Tuple

from .config import MAX_HISTORY_TURNS

SYSTEM_PROMPT = (
    "You are a careful medical information assistant. Use the patient record below "
    "when relevant. Be concise: a patient is reading this, not a colleague.\n\n"
    "Reply using exactly these five Markdown headers, in this order, and write only "
    "the content under each — never repeat these instructions in your answer:\n\n"
    "## Understanding your symptoms\n"
    "Two or three sentences restating the pattern and what it suggests.\n\n"
    "## Possible explanations\n"
    "A numbered list of at most FOUR conditions, most to least likely. Bold the "
    "condition name, then one sentence on why it fits. Do not pad the list to four "
    "if fewer are plausible.\n\n"
    "## What to ask your doctor\n"
    "Three to five bullets: specific tests or questions.\n\n"
    "## When to seek urgent care\n"
    "Two to four bullets: red-flag symptoms needing immediate attention.\n\n"
    "## Follow-up questions\n"
    "Exactly three short questions the patient could ask you next, each on its own "
    "line starting with a hyphen. This section is required — always include it last.\n\n"
    "Keep the whole reply under about 300 words. Never state a diagnosis as certain; "
    "you describe possibilities, you do not diagnose. If the question is not about "
    "health, say you can only help with medical questions and decline."
)

# epfl-llm/meditron-7b was released without instruction tuning, so it does not
# follow a bare system prompt. This pseudo-demonstration primes the answer shape
# without supplying any real clinical content.
ONE_SHOT_QUESTION = (
    "For a week I've had more thirst than usual, I'm up at night to urinate, and "
    "I'm tired during the day. What could explain this and what should I ask about?"
)
ONE_SHOT_ANSWER = (
    "## Understanding your symptoms\n"
    "Increased thirst, needing to urinate overnight, and daytime tiredness tend to "
    "cluster together, and the combination is worth taking seriously rather than "
    "treating each one on its own.\n\n"
    "## Possible explanations\n"
    "1. **New or poorly controlled diabetes** — high blood sugar pulls fluid into "
    "the urine, which drives both the thirst and the night-time urination, and the "
    "swings leave you tired. This is the first thing to rule out.\n"
    "2. **A urinary tract or prostate issue** — can explain frequent urination on "
    "its own, though it fits the thirst less well.\n"
    "3. **Too much caffeine or fluid before bed** — a benign explanation, but one "
    "to confirm rather than assume.\n\n"
    "## What to ask your doctor\n"
    "- A fasting blood glucose or HbA1c test\n"
    "- A urine dipstick for glucose and infection\n"
    "- Whether any current medication increases urination\n\n"
    "## When to seek urgent care\n"
    "- Vomiting, deep or rapid breathing, or drowsiness and confusion\n"
    "- Breath that smells fruity\n"
    "These can signal dangerously high blood sugar and need same-day attention.\n\n"
    "This is general information about what your symptoms could mean, not a diagnosis."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
STOP_SEQUENCES = [IM_END, IM_START, "\n<|im"]

# Gemma 3 (and so MedGemma) uses its own turn markers and has no system role.
GEMMA_START = "<start_of_turn>"
GEMMA_END = "<end_of_turn>"
GEMMA_STOP = [GEMMA_END, GEMMA_START]


def stop_sequences(prompt_style: str):
    return GEMMA_STOP if prompt_style == "gemma" else STOP_SEQUENCES


def render_profile(profile: Dict[str, Any]) -> str:
    """Compact patient record. Empty fields are dropped, not padded with 'unknown'."""
    rows: List[str] = []
    name = profile.get("display_name") or ""
    if name:
        rows.append(f"Name: {name}")
    if profile.get("age"):
        rows.append(f"Age: {profile['age']}")
    if profile.get("sex_at_birth"):
        rows.append(f"Sex at birth: {profile['sex_at_birth']}")
    if profile.get("height_cm"):
        rows.append(f"Height: {profile['height_cm']} cm")
    if profile.get("weight_kg"):
        rows.append(f"Weight: {profile['weight_kg']} kg")
    for label, key in (
        ("Existing conditions", "conditions"),
        ("Current medications", "medications"),
        ("Allergies", "allergies"),
    ):
        values = profile.get(key) or []
        if values:
            rows.append(f"{label}: {', '.join(values)}")
    if profile.get("notes"):
        rows.append(f"Other notes: {profile['notes']}")
    if not rows:
        return ""
    return "PATIENT RECORD\n" + "\n".join(rows)


def render_facts(facts: List[Dict[str, Any]], limit: int = 40) -> str:
    if not facts:
        return ""
    lines = [f"- {f['text']}" for f in facts[-limit:]]
    return "REMEMBERED FROM EARLIER CONVERSATIONS\n" + "\n".join(lines)


def build_system_block(profile: Dict[str, Any], facts: List[Dict[str, Any]]) -> str:
    parts = [SYSTEM_PROMPT]
    record = render_profile(profile)
    if record:
        parts.append(record)
    remembered = render_facts(facts)
    if remembered:
        parts.append(remembered)
    return "\n\n".join(parts)


def _turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def _gemma_turn(role: str, content: str) -> str:
    """role is 'user' or 'model' — Gemma has no system or assistant role."""
    return f"{GEMMA_START}{role}\n{content}{GEMMA_END}\n"


def assemble(
    system_block: str,
    history: List[Dict[str, Any]],
    question: str,
    prompt_style: str,
    input_budget_tokens: int,
    count_tokens,
) -> Tuple[str, Dict[str, int], int]:
    """Build the prompt, dropping the oldest turns until it fits the budget.

    Returns (prompt, per-section token counts, number of turns dropped).
    """
    if prompt_style == "gemma":
        # No system role exists, so the record and memory ride on the first user
        # turn. No one-shot priming either — MedGemma is instruction-tuned.
        head = ""
        tail = (
            _gemma_turn("user", f"{system_block}\n\n{question}")
            + f"{GEMMA_START}model\n"
        )

        def render(message):
            role = "user" if message["role"] == "user" else "model"
            return _gemma_turn(role, message["content"])
    else:
        head = _turn("system", system_block)
        if prompt_style == "base":
            head += _turn("question", ONE_SHOT_QUESTION) + _turn("answer", ONE_SHOT_ANSWER)
        tail = _turn("question", question) + f"{IM_START}answer\n"

        def render(message):
            role = "question" if message["role"] == "user" else "answer"
            return _turn(role, message["content"])

    recent = [m for m in history if m["role"] in ("user", "assistant")][-MAX_HISTORY_TURNS * 2:]

    head_tokens = count_tokens(head)
    tail_tokens = count_tokens(tail)
    available = input_budget_tokens - head_tokens - tail_tokens

    kept: List[str] = []
    used = 0
    dropped = 0
    for message in reversed(recent):
        chunk = render(message)
        cost = count_tokens(chunk)
        if used + cost > max(available, 0):
            dropped += 1
            continue
        kept.insert(0, chunk)
        used += cost

    prompt = head + "".join(kept) + tail
    breakdown = {
        "system_and_priming": head_tokens,
        "history": used,
        "question": tail_tokens,
        "total_input": head_tokens + used + tail_tokens,
    }
    return prompt, breakdown, dropped


# --- lightweight fact extraction ---------------------------------------------
# A 7B base model cannot reliably emit structured JSON, so nothing is written to
# memory automatically. These patterns only *suggest* facts; the user confirms
# them in the UI.

_PATTERNS = [
    (re.compile(r"\bI(?:'m| am)\s+(\d{1,3})\s*(?:years old|yo\b|y/o\b)", re.I), "Age: {0}"),
    (re.compile(r"\bI(?:'m| am)\s+allergic to\s+([^.,;]{2,60})", re.I), "Allergic to {0}"),
    (re.compile(r"\bI\s+(?:take|am on|'m on)\s+([^.,;]{2,60})", re.I), "Takes {0}"),
    (re.compile(r"\bI\s+(?:have|'ve got|was diagnosed with)\s+([^.,;]{2,60})", re.I), "Has {0}"),
    (re.compile(r"\bI\s+(?:smoke|vape)\b[^.,;]{0,40}", re.I), "Reported smoking/vaping"),
    (re.compile(r"\bmy\s+(?:doctor|gp|physician)\s+(?:said|told me)\s+([^.,;]{2,80})", re.I),
     "Clinician said: {0}"),
]


def suggest_facts(text: str) -> List[str]:
    out: List[str] = []
    for pattern, template in _PATTERNS:
        for match in pattern.finditer(text):
            groups = [g.strip() for g in match.groups()] if match.groups() else []
            try:
                out.append(template.format(*groups) if groups else template)
            except IndexError:
                continue
    seen = set()
    unique = []
    for item in out:
        key = item.lower()
        if key not in seen and len(item) < 120:
            seen.add(key)
            unique.append(item)
    return unique[:5]

FOLLOWUP_HEADER_RE = re.compile(
    r"(?:#{1,3}\s*|\*\*\s*)?Follow[\s-]?up\s+questions\s*:?\s*\**",
    re.I,
)


FALLBACK_FOLLOWUPS = [
    "Which of these is most likely in my case?",
    "What tests would tell these apart?",
    "What should I watch for while I wait to see a doctor?",
]


def split_followups(answer: str):
    """Return (body_without_followups, [questions]).

    If the model omitted the follow-up section, fall back to three generic
    questions so the UI always shows chips.
    """
    match = FOLLOWUP_HEADER_RE.search(answer)
    if not match:
        return answer.strip(), list(FALLBACK_FOLLOWUPS)
    body = answer[: match.start()].strip()
    tail = answer[match.end():]
    questions = []
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "\u2022")):
            q = line.lstrip("-*\u2022 ").strip()
        elif line[:2].strip().rstrip(".").isdigit():
            q = line.split(".", 1)[1].strip() if "." in line else ""
        else:
            continue
        if q:
            questions.append(q)
    if not questions:
        return body, list(FALLBACK_FOLLOWUPS)
    return body, questions[:3]
