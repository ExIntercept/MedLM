"""Self-reported facts are captured, merged across turns, and never taken from
messages about someone else."""
import pytest

from src.records.health_record import NONE_KNOWN, extract_health_facts, merge_record


@pytest.mark.parametrize("text, expected", [
    ("I am a 45 year old woman with diabetes, taking metformin 500mg, chest pain for 2 days",
     {"age": "45", "sex": "Female", "duration": "2 days", "conditions": ["diabetes"], "medications": ["metformin 500mg"]}),
    ("I'm 62, male. I have hypertension and asthma. I take lisinopril and an inhaler. Allergic to penicillin.",
     {"age": "62", "sex": "Male", "conditions": ["hypertension", "asthma"],
      "medications": ["lisinopril", "inhaler"], "allergies": ["penicillin"]}),
    ("I used to take atorvastatin but stopped last year. Now I am on rosuvastatin 10mg daily.",
     {"medications": ["rosuvastatin 10mg daily"], "past_medications": ["atorvastatin"]}),
    ("I have no known allergies. I was diagnosed with type 2 diabetes in 2019.",
     {"conditions": ["type 2 diabetes"], "allergies": [NONE_KNOWN]}),
    ("headache for the past 3 days, 29F, on the pill",
     {"age": "29", "sex": "Female", "duration": "past 3 days", "medications": ["contraceptive pill"]}),
    ("I'm 58, male, I take warfarin and I'm allergic to penicillin.",
     {"age": "58", "sex": "Male", "medications": ["warfarin"], "allergies": ["penicillin"]}),
    ("I've been taking sertraline 50 mg for anxiety",
     {"conditions": ["anxiety"], "medications": ["sertraline 50 mg"]}),
])
def test_extracts_self_reported_facts(text, expected):
    assert extract_health_facts(text) == expected


@pytest.mark.parametrize("text", [
    "My son is 8 and has had a fever since yesterday",
    "My mother takes warfarin and is allergic to aspirin",
    "A 45-year-old man presents with chest pain. Which of the following is the best next step?",
    "Can diabetes cause blurry vision?",
    "What are the side effects of ibuprofen?",
    "Is it safe to take ibuprofen with warfarin?",
    "I am on my way to the clinic",
])
def test_ignores_other_people_cases_and_questions(text):
    assert extract_health_facts(text) == {}


def test_negated_condition_is_not_recorded():
    facts = extract_health_facts("I don't have high blood pressure but my cholesterol is high")
    assert facts == {"conditions": ["high cholesterol"]}


def test_merge_accumulates_moves_stopped_meds_and_replaces_none_known():
    record = {}
    for turn in [
        "I am a 45 year old woman with diabetes, taking metformin 500mg",
        "I also take atorvastatin. I have no allergies.",
        "I stopped taking atorvastatin last month",
        "Turns out I am allergic to penicillin",
        "I am 46 now",
    ]:
        record = merge_record(record, extract_health_facts(turn))
    assert record["age"] == "46"
    assert record["sex"] == "Female"
    assert record["conditions"] == "diabetes"
    assert record["medications"] == "metformin 500mg"
    assert record["past_medications"] == "atorvastatin"
    assert record["allergies"] == "penicillin"


def test_merge_keeps_user_edits_and_updates_dose():
    record = {"medications": "Metformin, Lisinopril", "allergies": "Penicillin"}
    merged = merge_record(record, extract_health_facts("I take metformin 1000 mg now"))
    assert merged["medications"] == "Lisinopril, metformin 1000 mg"
    assert merged["allergies"] == "Penicillin"


def test_profile_prompt_unchanged_without_new_fields():
    """Benchmarks send no profile; the prompt text for old fields must not change."""
    from src.agents.prompting import format_patient_profile

    assert format_patient_profile(age="40", medications="metformin") == (
        "Patient Profile: The patient is 40 years old; current medications: metformin."
    )
    assert "allergies: penicillin" in format_patient_profile(allergies="penicillin")
