"""Adversarial safety guardrails and USMLE option parsing."""
import re

HARD_RULES = [
    ("warfarin", ["pregnant", "pregnancy", "gestation"], "Warfarin is teratogenic (FDA Boxed Warning Category X) and contraindicated in pregnancy. LMWH (Enoxaparin) is indicated."),
    ("metformin", ["egfr < 30", "egfr 18", "severe renal", "contrast"], "Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min) and must be withheld prior to iodinated contrast due to lactic acidosis risk."),
    ("ceftriaxone", ["neonate", "calcium"], "Ceftriaxone is contraindicated in neonates receiving IV calcium due to fatal particulate precipitation."),
    ("potassium", ["push", "bolus", "2 minutes", "90 seconds"], "Direct IV push of concentrated Potassium Chloride is lethal (causes cardiac arrest). Infuse slowly (max 10-20 mEq/hr)."),
    ("acetaminophen", ["8000", "8,000", "8g"], "Maximum daily dose of acetaminophen is 4,000 mg/day. Escalation to 8,000 mg/day causes fatal acute hepatic necrosis.")
]


def check_hard_rules(raw_input):
    q_lower = raw_input.lower()
    for drug_or_entity, triggers, warning in HARD_RULES:
        if drug_or_entity in q_lower and any(t in q_lower for t in triggers):
            return drug_or_entity, triggers, warning
    return None


PATIENT_ACTION_GUIDANCE = {
    "warfarin": {
        "contraindication": "Warfarin must NOT be taken during pregnancy because it is teratogenic (causes severe birth defects and bleeding risks).",
        "action": "Stop taking Warfarin immediately and contact your OB-GYN or primary doctor right away to discuss safe pregnancy alternatives (such as LMWH / Enoxaparin or approved joint pain management).",
    },
    "metformin": {
        "contraindication": "Metformin must be stopped in severe kidney impairment or before certain contrast-dye imaging scans, because it can build up in the body and cause a dangerous condition called lactic acidosis.",
        "action": "Stop taking Metformin and contact your primary doctor or kidney specialist right away to discuss safe alternatives for managing your blood sugar.",
    },
    "ceftriaxone": {
        "contraindication": "Ceftriaxone must NOT be given together with IV calcium in a newborn, because the combination can form dangerous particles in the bloodstream.",
        "action": "Do not give this combination. Contact the prescribing doctor or pharmacist immediately to arrange a safe alternative antibiotic or calcium regimen.",
    },
    "potassium": {
        "contraindication": "Potassium chloride must NOT be given as a fast IV push, because it can stop the heart.",
        "action": "Stop the injection immediately and contact the prescribing doctor or nursing supervisor right away — potassium must only be given as a slow infusion.",
    },
    "acetaminophen": {
        "contraindication": "Taking more than 4,000 mg of acetaminophen (Tylenol) in a single day can cause severe, potentially fatal liver damage.",
        "action": "Stop taking any more acetaminophen today and contact your doctor, pharmacist, or Poison Control right away.",
    },
}


def format_guardrail_intercept(entity, warning, mode="patient"):
    if mode == "clinician":
        return (
            "### ⛔ Contraindication Alert\n\n"
            f"**Agent:** {entity.title()}\n\n"
            f"**Clinical Detail:** {warning}\n\n"
            "*This guardrail intercepted the request before retrieval/generation ran, "
            "per a hard-coded contraindication rule.*"
        )

    guidance = PATIENT_ACTION_GUIDANCE.get(entity)
    contraindication = guidance["contraindication"] if guidance else warning
    action = (
        guidance["action"]
        if guidance
        else "Stop this medication or action immediately and contact your prescribing doctor or pharmacist right away."
    )
    return (
        "⚠️ **Critical Medication Safety Warning**\n"
        f"- **Contraindication:** {contraindication}\n"
        f"- **Action Required:** {action}"
    )


_OPTION_MARKER = r'[\(\[]?[A-E][\)\]\.\:\-]\s+'
_OPTION_START = r'(?:(?<=\n)|(?<=\s)|(?<=^))'


def parse_embedded_options(full_text):
    pattern = _OPTION_START + r'[\(\[]?([A-E])[\)\]\.\:\-]\s+(.+?)(?=' + _OPTION_MARKER + r'|\Z)'
    matches = re.findall(pattern, full_text, re.DOTALL | re.IGNORECASE)
    if len(matches) >= 2:
        options = {m[0].upper(): m[1].strip() for m in matches}
        split_pos = re.search(_OPTION_START + r'[\(\[]?[A-E][\)\]\.\:\-]\s+', full_text)
        vignette_only = full_text[:split_pos.start()].strip() if split_pos else full_text
        return vignette_only, options
    return full_text.strip(), None
