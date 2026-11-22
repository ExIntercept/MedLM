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

# Up to three questions chosen by priority, plus the warning-signs question,
# which is always kept when it is still unanswered.
MAX_QUESTIONS = 4


def _q(qid: str, label: str, text: str, options: list[str], known: str, kind: str = "choice") -> dict:
    """A symptom-specific question; ``known`` matches messages that already answer it."""
    return {"id": qid, "label": label, "text": text, "options": options, "kind": kind,
            "known": re.compile(known, re.IGNORECASE)}


# What a doctor would ask next for each symptom group, most useful first.
_SPECIFIC: dict[str, list[dict]] = {
    "headache": [
        _q("location", "Where", "Where is the pain?",
           ["One side", "Both sides or all over", "Forehead or behind the eyes", "Back of the head"],
           r"\b(?:one\s+side|both\s+sides|forehead|temples?|behind\s+(?:my|the)\s+eyes?|back\s+of\s+(?:my|the)\s+head|all\s+over)\b"),
        _q("with", "Along with it", "Is anything else happening with it?",
           ["Nausea or light hurts my eyes", "Blocked nose or a cold", "Stress or poor sleep", "Nothing else"],
           r"\b(?:nause\w*|light|sensitiv\w*|blocked\s+nose|congest\w*|a\s+cold|stress\w*|sleep\w*)\b"),
    ],
    "chest": [
        _q("character", "Feels like", "What does it feel like?",
           ["Tight or pressing", "Sharp, worse when I breathe in", "Burning, after eating", "Sore when I press on it"],
           r"\b(?:tight\w*|pressure|pressing|squeez\w*|sharp|stabbing|burning|heartburn|tender)\b"),
        _q("when", "Happens", "When does it happen?",
           ["During activity", "At rest", "After meals", "When I move or breathe in"],
           r"\b(?:exercis\w*|running|walking|stairs|at\s+rest|after\s+(?:eating|meals?)|breath(?:e|ing)\s+in)\b"),
    ],
    "breathing": [
        _q("when", "Breathless", "When are you short of breath?",
           ["Only when active", "At rest too", "When lying flat", "At night"],
           r"\b(?:exert\w*|exercis\w*|stairs|walking|at\s+rest|lying|at\s+night)\b"),
        _q("lungs", "Lung history", "Do you have asthma or another lung condition?",
           ["Asthma", "COPD", "Another lung condition", "No"],
           r"\b(?:asthma|copd|emphysema|bronchitis|lung\s+condition)\b"),
    ],
    "cough": [
        _q("type", "Cough type", "Is it dry, or are you bringing up phlegm?",
           ["Dry", "Clear or white phlegm", "Yellow or green phlegm"],
           r"\b(?:dry|phlegm|mucus|sputum|productive|chesty)\b"),
        _q("with", "Along with it", "Do you have any of these with it?",
           ["Fever", "Runny or blocked nose", "Sore throat", "None of these"],
           r"\b(?:fever|temperature|runny|blocked\s+nose|sore\s+throat|a\s+cold)\b"),
    ],
    "fever": [
        _q("height", "Temperature", "How high has it been?",
           ["Under 38 °C (100.4 °F)", "38–39 °C", "Over 39 °C (102.2 °F)", "Not measured"],
           r"\b\d{2,3}(?:\.\d)?\s*°?\s*[cf]\b|\b(?:degrees?|high\s+fever|low[- ]grade)\b"),
        _q("with", "Along with it", "Any other symptoms with the fever?",
           ["Cough or sore throat", "Burning when I pee", "Diarrhoea or vomiting", "None of these"],
           r"\b(?:cough\w*|sore\s+throat|urin\w*|pee|diarrh\w*|vomit\w*|rash)\b"),
    ],
    "stomach": [
        _q("location", "Where", "Where in your tummy is it?",
           ["Upper middle", "Lower right", "Lower left", "All over"],
           r"\b(?:upper|lower|right|left|middle|belly\s+button|navel|all\s+over)\b"),
        _q("with", "Along with it", "Is anything else happening?",
           ["Vomiting", "Diarrhoea", "Constipation", "Nothing else"],
           r"\b(?:vomit\w*|diarrh\w*|constipat\w*|loose\s+stools?)\b"),
    ],
    "urinary": [
        _q("what", "Noticing", "What are you noticing?",
           ["Burning or pain when I pee", "Going more often", "Both", "Something else"],
           r"\b(?:burn\w*|sting\w*|pain\w*|frequen\w*|more\s+often|urgen\w*)\b"),
    ],
    "skin": [
        _q("feel", "Feels", "Is it itchy or painful?", ["Itchy", "Painful", "Both", "Neither"],
           r"\b(?:itch\w*|pain\w*|sore|burn\w*|sting\w*)\b"),
        _q("where", "Where", "Where is it?", ["Face", "Arms or legs", "Body", "Several places"],
           r"\b(?:face|arms?|legs?|hands?|feet|chest|back|body|neck|scalp|all\s+over)\b"),
    ],
    "dizziness": [
        _q("type", "Feels like", "What does it feel like?",
           ["The room is spinning", "Light-headed or about to faint", "Unsteady on my feet"],
           r"\b(?:spin\w*|light-?headed|faint\w*|unsteady|off\s+balance|vertigo)\b"),
        _q("when", "Happens", "When does it happen?",
           ["When I stand up", "When I move my head", "All the time"],
           r"\b(?:stand\w*|get(?:ting)?\s+up|(?:turn|mov)\w*\s+(?:my\s+)?head|all\s+the\s+time|constant\w*)\b"),
    ],
    "pain": [
        _q("where", "Where", "Where exactly is the pain?", [],
           r"\b(?:knee|back|neck|shoulder|hip|ankle|wrist|elbow|foot|feet|hand|leg|arm|joint|muscle|jaw|tooth|teeth|ear|throat|eye)s?\b",
           kind="text"),
        _q("injury", "Injury", "Did it start after an injury or strain?",
           ["Yes, after an injury or strain", "No, it came on by itself"],
           r"\b(?:fell|fall(?:en)?|injur\w*|sprain\w*|twist\w*|strain\w*|lift\w*|hit|knock\w*)\b"),
    ],
    "general": [
        _q("pattern", "Pattern", "Has this happened before?",
           ["This is the first time", "On and off", "Getting worse over time"],
           r"\b(?:first\s+time|before|again|on\s+and\s+off|recurr\w*|getting\s+worse|keeps?\s+happening)\b"),
    ],
}

# Display order of every question kind.
_ORDER = ["onset", "severity", "location", "where", "character", "type", "feel", "what", "height",
          "when", "with", "injury", "pattern", "lungs", "warning", "background", "age"]

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
    onset = severity = warning = background = age = None
    if not (facts.get("duration") or _ONSET.search(text)):
        onset = {
            "id": "onset", "label": "Started", "kind": "choice", "text": "When did it start?",
            "options": ["Today", "1–3 days ago", "4–7 days ago", "Over a week ago"],
        }
    if not _SEVERITY.search(text):
        severity = {
            "id": "severity", "label": "Severity", "kind": "choice", "text": "How bad is it?",
            "options": ["Mild", "Moderate", "Severe"],
        }
    if not _NO_WARNING.search(text):
        warning = {
            "id": "warning", "label": "Warning signs", "kind": "choice",
            "text": "Do you have any of these: " + ", ".join(signs) + "?",
            "options": ["None of these", "Yes, at least one"],
        }
    has_meds = facts.get("medications") or _known(profile, "medications")
    has_conditions = facts.get("conditions") or _known(profile, "conditions")
    if not (has_meds or has_conditions or _NO_BACKGROUND.search(text)):
        background = {
            "id": "background", "label": "Conditions or medicines", "kind": "text",
            "text": "Do you have any long-term conditions or take any regular medicines?",
            "options": [],
        }
    if not (facts.get("age") or _known(profile, "age")):
        age = {"id": "age", "label": "Age", "kind": "text", "text": "How old are you?", "options": []}

    specific = [
        {k: v for k, v in q.items() if k != "known"}
        for q in _SPECIFIC.get(name, []) if not q["known"].search(text)
    ]
    first, second = (specific + [None, None])[:2]

    # Pick by usefulness, then show in a natural consultation order. The
    # warning-signs question is always kept while it is unanswered.
    by_priority = [q for q in (onset, first, severity, second, background, age) if q]
    chosen = by_priority[: MAX_QUESTIONS - (1 if warning else 0)]
    if warning:
        chosen.append(warning)
    if len(chosen) < 2:
        return None  # enough to go on: answer straight away
    rank = {qid: i for i, qid in enumerate(_ORDER)}
    questions = sorted(chosen, key=lambda q: rank.get(q["id"], len(rank)))
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
