"""Ask a few questions before answering a new symptom complaint.

A doctor hearing "I've had a headache" asks how long, how bad and whether any
warning signs are present before giving advice. This module decides, with
plain rules rather than the model, whether a patient's message is such a
complaint and which of those details are still missing.

Used only on the patient chat path, after the security stages have passed and
emergency triage has had its chance to escalate. The gateway calls it only
when the server asks; benchmark runs never pass it, so their prompts and
outputs are unchanged.

Not asked when:
* the message asks a direct factual or medication question ("can I take...")
  - answering is more useful than questioning;
* it is about someone else or reads like an exam case;
* the patient asks for a straight answer ("just tell me");
* enough is already known (fewer than two key details missing).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from src.records.health_record import about_someone_else, extract_health_facts

MAX_QUESTIONS = 3

# ── Is this a personal symptom complaint? ───────────────────────────────
_SELF_CUE = re.compile(
    r"\b(?:I(?:'ve|'m|\s+have|\s+am)?\s+(?:have|had|got|keep\s+getting|having|"
    r"feel|feeling|been\s+having|been\s+feeling|suffering|woke\s+up\s+with|noticed|get)\b|"
    r"I(?:'m|\s+am)\s+(?:dizzy|nauseous|sick|tired|exhausted|coughing|vomiting|itchy|bloated|"
    r"constipated|wheezing|short\s+of\s+breath)|"
    r"my\s+\w+(?:\s+\w+)?\s+(?:hurts?|aches?|is\s+(?:sore|swollen|itchy|painful|red|burning)|"
    r"are\s+(?:sore|swollen|itchy|painful|red))|"
    r"there(?:'s|\s+is)\s+(?:a\s+)?(?:pain|lump|rash|swelling))",
    re.IGNORECASE,
)

# Symptom families: (name, trigger regex, warning signs to ask about).
_FAMILIES: list[tuple[str, re.Pattern, list[str]]] = [
    ("headache", re.compile(r"\b(?:headaches?|migraines?|head\s+(?:hurts|aches|pain))\b", re.I),
     ["sudden, very severe onset", "stiff neck or fever", "confusion, weakness or vision changes"]),
    ("chest", re.compile(r"\bchest\b", re.I),
     ["pain spreading to the arm, jaw or back", "shortness of breath", "sweating or feeling faint"]),
    ("breathing", re.compile(r"\b(?:short(?:ness)?\s+of\s+breath|breathless|wheez\w*|can'?t\s+breathe)\b", re.I),
     ["struggling to speak in full sentences", "blue or grey lips", "chest pain"]),
    ("cough", re.compile(r"\b(?:cough\w*|phlegm|sputum)\b", re.I),
     ["coughing up blood", "shortness of breath", "chest pain"]),
    ("fever", re.compile(r"\b(?:fever|feverish|temperature|chills)\b", re.I),
     ["a rash that doesn't fade when pressed", "stiff neck", "confusion or very drowsy"]),
    ("stomach", re.compile(r"\b(?:stomach|abdomen|abdominal|tummy|belly)\b|\b(?:vomit\w*|nause\w*|diarrh\w*|constipat\w*)\b", re.I),
     ["blood in vomit or black stools", "severe constant pain", "unable to keep fluids down"]),
    ("urinary", re.compile(r"\b(?:urin\w*|pee\w*|bladder|burning\s+when)\b", re.I),
     ["fever or back pain", "blood in the urine", "vomiting"]),
    ("skin", re.compile(r"\b(?:rash|hives|itch\w*|spots?|blisters?|lump)\b", re.I),
     ["swelling of the lips, face or tongue", "difficulty breathing", "rapidly spreading redness or fever"]),
    ("dizziness", re.compile(r"\b(?:dizz\w*|faint\w*|light-?headed|vertigo)\b", re.I),
     ["fainting or blacking out", "chest pain or palpitations", "slurred speech or weakness"]),
    ("pain", re.compile(r"\b(?:pain\w*|aches?|aching|hurts?|sore(?:ness)?|cramps?|swollen|swelling|stiff)\b", re.I),
     ["numbness or weakness", "fever", "unable to move or bear weight"]),
    ("general", re.compile(r"\b(?:tired|fatigue\w*|exhausted|weak(?:ness)?|insomnia|can'?t\s+sleep|"
                           r"anxious|anxiety|low\s+mood|palpitations?|bleeding|numb\w*)\b", re.I),
     ["chest pain or fainting", "unexplained weight loss", "thoughts of harming yourself"]),
]

# ── Reasons not to ask ─────────────────────────────────────────────────
_DIRECT_QUESTION = re.compile(
    r"\b(?:can|could|should|may)\s+I\s+(?:take|use|have|drink|eat|mix|combine|give)\b|"
    r"\bis\s+it\s+(?:safe|ok|okay|alright|fine)\b|"
    r"\b(?:dose|dosage|how\s+much|how\s+many|interact\w*|side[- ]effects?|contraindicat\w*)\b|"
    r"\b(?:what\s+is|what\s+are|what\s+does|define|explain)\b(?!.*\b(?:caus\w*|wrong|happening)\b)",
    re.IGNORECASE,
)
_WANTS_ANSWER = re.compile(
    r"\b(?:just\s+(?:tell|answer|give)|no\s+questions|don'?t\s+ask|skip\s+the\s+questions|quick\s+answer)\b",
    re.IGNORECASE,
)

# ── Is a detail already given? ─────────────────────────────────────────
_SEVERITY = re.compile(
    r"\b(?:mild|slight|moderate|severe|terrible|unbearable|excruciating|worst|bad|awful|"
    r"intense|really\s+(?:bad|painful)|\d{1,2}\s*(?:/|out\s+of)\s*10)\b",
    re.IGNORECASE,
)
_ONSET = re.compile(
    r"\b(?:since|started|began|for\s+(?:a|an|the\s+past|\d+|a\s+few|several|two|three|four|five|six|seven)|"
    r"(?:yesterday|today|tonight|this\s+(?:morning|afternoon|evening|week))|"
    r"\d+\s*(?:hours?|days?|weeks?|months?|years?)|(?:hours?|days?|weeks?|months?)\s+ago|"
    r"on\s+and\s+off|comes\s+and\s+goes)\b",
    re.IGNORECASE,
)
_NO_BACKGROUND = re.compile(
    r"\b(?:on\s+no|not\s+on\s+any|no\s+regular|don'?t\s+take\s+any|no)\s+"
    r"(?:regular\s+)?(?:medicines?|medications?|meds|tablets|pills|conditions?|health\s+problems)\b",
    re.IGNORECASE,
)
_NO_WARNING = re.compile(r"\bno\s+(?:other\s+symptoms|fever|warning\s+signs|red\s+flags)\b", re.I)


def _known(profile: Any, key: str) -> bool:
    if not profile:
        return False
    value = profile.get(key) if isinstance(profile, dict) else getattr(profile, key, None)
    return bool(str(value or "").strip())


def _family(text: str) -> Optional[tuple[str, list[str]]]:
    for name, rx, signs in _FAMILIES:
        if rx.search(text):
            return name, signs
    return None


def is_symptom_complaint(message: str) -> bool:
    """True when the speaker describes their own symptom and is not asking a
    direct factual or medication question: such messages get a consultation-
    style answer (what it could be, what to do, when to get help)."""
    text = (message or "").replace("’", "'").strip()
    if not text or about_someone_else(text) or _DIRECT_QUESTION.search(text):
        return False
    return bool(_SELF_CUE.search(text) and _family(text))


def plan_clarification(message: str, profile: Any = None) -> Optional[dict]:
    """Questions to ask before answering ``message``, or None to answer now.

    ``profile`` is what is already known about the patient (their saved
    record and this consultation's details), as a dict or an object with the
    same attributes. Returns ``{"family", "questions", "text"}`` where each
    question is ``{"id", "text", "options", "kind"}``: ``kind`` "choice" offers
    the options as quick replies, "text" asks for a short typed answer.
    """
    text = (message or "").replace("’", "'").strip()
    if not text or len(text) > 600:
        return None
    if _WANTS_ANSWER.search(text) or about_someone_else(text):
        return None
    if _DIRECT_QUESTION.search(text):
        return None
    if not _SELF_CUE.search(text):
        return None
    family = _family(text)
    if family is None:
        return None
    name, signs = family

    facts = extract_health_facts(text)
    questions: list[dict] = []
    if not (facts.get("duration") or _ONSET.search(text)):
        questions.append({
            "id": "onset", "kind": "choice", "text": "When did it start?",
            "options": ["Today", "1–3 days ago", "4–7 days ago", "Over a week ago"],
        })
    if not _SEVERITY.search(text):
        questions.append({
            "id": "severity", "kind": "choice", "text": "How bad is it?",
            "options": ["Mild", "Moderate", "Severe"],
        })
    if not _NO_WARNING.search(text):
        questions.append({
            "id": "warning", "kind": "choice",
            "text": "Do you have any of these: " + ", ".join(signs) + "?",
            "options": ["None of these", "Yes, at least one"],
        })
    has_meds = facts.get("medications") or _known(profile, "medications")
    has_conditions = facts.get("conditions") or _known(profile, "conditions")
    if not (has_meds or has_conditions or _NO_BACKGROUND.search(text)):
        questions.append({
            "id": "background", "kind": "text",
            "text": "Do you have any long-term conditions or take any regular medicines?",
            "options": [],
        })
    if not (facts.get("age") or _known(profile, "age")):
        questions.append({"id": "age", "kind": "text", "text": "How old are you?", "options": []})

    if len(questions) < 2:
        return None  # enough to go on: answer straight away
    questions = questions[:MAX_QUESTIONS]
    return {"family": name, "questions": questions, "text": clarification_text(questions)}


def clarification_text(questions: list[dict]) -> str:
    """The plain-text message shown (and stored) for a clarification turn."""
    lines = ["To give you advice that fits your situation, a few quick questions first:", ""]
    lines += [f"{i}. {q['text']}" for i, q in enumerate(questions, 1)]
    lines += [
        "",
        "You can tap an answer, type your own, or skip and I'll answer straight away. "
        "If you feel very unwell or it's getting worse quickly, call your local emergency number now.",
    ]
    return "\n".join(lines)


def combine_with_answers(original: str, answers: str) -> str:
    """The message the pipeline answers after a clarification turn."""
    answers = (answers or "").strip()
    if not answers or re.fullmatch(r"(?:please\s+)?(?:just\s+answer|skip)\.?", answers, re.IGNORECASE):
        return original
    return f"{original.strip()}\n\nMore details: {answers}"
