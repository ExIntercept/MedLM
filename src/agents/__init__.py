"""Clinical agents and safety guardrails package."""
from .emergency_triage import EMERGENCY_WARNING, check_emergency_triage
from .guardrails import check_hard_rules, format_guardrail_intercept, parse_embedded_options
from .intake_extraction import extract_intake_fields
from .prompting import build_prompt, detect_prompt_tone, format_patient_profile, sanitize_patient_output
from .verifier import verify_evidence

__all__ = [
    "EMERGENCY_WARNING",
    "check_emergency_triage",
    "check_hard_rules",
    "format_guardrail_intercept",
    "parse_embedded_options",
    "extract_intake_fields",
    "build_prompt",
    "detect_prompt_tone",
    "format_patient_profile",
    "sanitize_patient_output",
    "verify_evidence",
]
