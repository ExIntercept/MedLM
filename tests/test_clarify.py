"""Rule-based clarification before answering a new symptom complaint."""
import pytest

from src.agents.clarify import (
    MAX_QUESTIONS,
    clarification_text,
    combine_with_answers,
    plan_clarification,
)


def _ids(plan):
    return [q["id"] for q in plan["questions"]] if plan else None


@pytest.mark.parametrize("message", [
    "I have a headache",
    "My stomach hurts",
    "I've got a bad cough",
    "I’m feeling dizzy",
])
def test_vague_symptom_complaints_get_questions(message):
    plan = plan_clarification(message)
    assert plan is not None
    assert 2 <= len(plan["questions"]) <= MAX_QUESTIONS
    assert plan["text"].startswith("To give you advice")


@pytest.mark.parametrize("message", [
    "Can I take ibuprofen with my blood pressure tablets?",
    "I'm 58 and take warfarin. Is it safe to take ibuprofen for knee pain?",
    "What are the early signs of dehydration?",
    "What is the usual dose of paracetamol?",
    "My son has a fever",                                 # someone else
    "A 45-year-old man presents with chest pain",         # exam vignette
    "I feel dizzy when I stand up, just tell me what to do",
    "I had a great day",                                  # no symptom
    "",
])
def test_direct_questions_and_other_cases_are_answered_straight_away(message):
    assert plan_clarification(message) is None


def test_details_already_given_are_not_asked_again():
    plan = plan_clarification("I've had a headache for three days. What could be causing it?")
    assert "onset" not in _ids(plan)
    assert "severity" in _ids(plan)


def test_enough_details_means_no_questions():
    msg = "I've had a mild rash on my arm since yesterday, no fever, I'm 30 and on no medicines"
    assert plan_clarification(msg) is None


def test_saved_record_counts_as_known():
    plan = plan_clarification("I've got a bad cough", {"medications": "metformin", "age": "52"})
    assert "background" not in _ids(plan)
    assert "age" not in _ids(plan)


def test_warning_signs_match_the_symptom():
    plan = plan_clarification("I have a headache")
    warning = next(q for q in plan["questions"] if q["id"] == "warning")
    assert "stiff neck" in warning["text"]
    assert warning["options"][0] == "None of these"


@pytest.mark.parametrize("message,expected", [
    ("I have a headache", "location"),
    ("My stomach hurts", "location"),
    ("I've got a bad cough", "type"),
    ("I have a fever", "height"),
    ("I'm feeling dizzy", "type"),
    ("I have pain when I pee", None),         # urinary: pain already said
])
def test_symptom_specific_question_is_asked(message, expected):
    plan = plan_clarification(message)
    assert plan is not None
    ids = _ids(plan)
    if expected:
        assert expected in ids
    assert ids[-1] != "onset"                  # display order, not priority order
    assert all(q.get("label") for q in plan["questions"])
    assert all("known" not in q for q in plan["questions"])


def test_specific_question_skipped_when_already_answered():
    plan = plan_clarification("I have a headache on one side of my head")
    assert "location" not in _ids(plan)
    plan = plan_clarification("I've got a dry cough")
    assert "type" not in _ids(plan)


def test_warning_question_always_kept():
    plan = plan_clarification("My stomach hurts")
    assert "warning" in _ids(plan)
    assert len(plan["questions"]) == MAX_QUESTIONS
    order = _ids(plan)
    assert order.index("onset") < order.index("warning")


def test_text_mentions_emergency_number():
    text = clarification_text([{"id": "x", "text": "How bad is it?", "options": [], "kind": "text"}])
    assert "emergency number" in text
    assert "1. How bad is it?" in text


def test_combine_with_answers():
    assert combine_with_answers("I have a headache", "Today; Mild") == "I have a headache\n\nMore details: Today; Mild"
    assert combine_with_answers("I have a headache", "Just answer") == "I have a headache"
    assert combine_with_answers("I have a headache", "") == "I have a headache"


@pytest.mark.parametrize("message,expected", [
    ("I have a headache", True),
    ("I've had a mild rash on my arm since yesterday, no fever, I'm 30 and on no medicines", True),
    ("Can I take ibuprofen with warfarin?", False),
    ("My son has a fever", False),
    ("What is metformin?", False),
])
def test_is_symptom_complaint(message, expected):
    from src.agents.clarify import is_symptom_complaint

    assert is_symptom_complaint(message) is expected


@pytest.mark.parametrize("compact", [False, True])
def test_consult_format_only_when_asked(compact):
    from src.agents.prompting import CONSULT_FORMAT, build_messages

    args = ("I have a headache", None, "Headache guideline text.", [])
    plain = build_messages(*args, forced_tone="patient", nonce="abcd1234", compact=compact)
    again = build_messages(*args, forced_tone="patient", nonce="abcd1234", compact=compact, consult=False)
    consult = build_messages(*args, forced_tone="patient", nonce="abcd1234", compact=compact, consult=True)
    assert plain == again                       # default prompt unchanged
    assert "What it could be" not in plain[0]["content"]
    assert CONSULT_FORMAT in consult[0]["content"]
    clinical = build_messages(*args, forced_tone="clinical_qa", nonce="abcd1234", compact=compact, consult=True)
    assert "What it could be" not in clinical[0]["content"]   # clinicians never get it
