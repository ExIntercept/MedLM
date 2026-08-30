"""Lightweight regex-based entity extraction for patient intake forms."""
import re

_AGE_EXTRACT = re.compile(
    r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b|\bage[d]?\s*[:\-]?\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_SEX_EXTRACT = re.compile(r"\b(male|female|man|woman)\b", re.IGNORECASE)
_DURATION_EXTRACT = re.compile(
    r"\b((?:past\s+)?(?:\d+|a|an|few|couple(?:\s+of)?|several)\s*"
    r"(?:hours?|days?|weeks?|months?|years?))\b(?!\s*old\b)",
    re.IGNORECASE,
)
_CONDITIONS_EXTRACT = re.compile(
    r"\b(?:history of|diagnosed with|known case of)\s+([a-zA-Z0-9 ,/-]{3,60}?)(?=[.,;]|$)",
    re.IGNORECASE,
)
_MEDICATIONS_EXTRACT = re.compile(
    r"\b(?:taking|currently on|prescribed)\s+([a-zA-Z0-9 ,/-]{3,60}?)(?=[.,;]|$)",
    re.IGNORECASE,
)

_SEX_NORMALIZE = {"male": "Male", "man": "Male", "female": "Female", "woman": "Female"}


def extract_intake_fields(text: str) -> dict:
    """Extracts age, sex, duration, conditions, medications from text."""
    if not text:
        return {}

    fields = {}

    age_match = _AGE_EXTRACT.search(text)
    if age_match:
        fields["age"] = age_match.group(1) or age_match.group(2)

    sex_match = _SEX_EXTRACT.search(text)
    if sex_match:
        fields["sex"] = _SEX_NORMALIZE[sex_match.group(1).lower()]

    duration_match = _DURATION_EXTRACT.search(text)
    if duration_match:
        fields["duration"] = duration_match.group(1).strip()

    conditions_match = _CONDITIONS_EXTRACT.search(text)
    if conditions_match:
        fields["conditions"] = conditions_match.group(1).strip()

    medications_match = _MEDICATIONS_EXTRACT.search(text)
    if medications_match:
        fields["medications"] = medications_match.group(1).strip()

    return fields
