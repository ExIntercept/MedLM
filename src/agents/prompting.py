"""Shared clinical prompt construction supporting MedGemma formatting and tone detection."""
import re

from src.agents.intake_extraction import extract_intake_fields
from src.config import PROMPT_STYLE

_USMLE_OPTION_PATTERN = re.compile(r"\([A-D]\)")
_FIRST_PERSON_PATTERN = re.compile(
    r"\b(i am|i have|my|i'm|i feel|feelings|help me|hi doc|hello doc|doctor)\b", re.IGNORECASE
)
_SYMPTOM_KEYWORD_PATTERN = re.compile(
    r"\b(pain|ache|aches|aching|hurt|hurts|fever|cough|nausea|nauseous|vomit|vomiting|"
    r"dizzy|dizziness|tired|fatigue|bleeding|swelling|swollen|rash|headache|sore|"
    r"burning|numb|numbness|breathless|shortness of breath|chills|cramp|cramps|"
    r"diarrhea|constipation|symptom|symptoms|sick|ill|weak|weakness)\b",
    re.IGNORECASE,
)

_SUMMARY_REQUEST_PATTERN = re.compile(
    r"\b(summar\w+|written report|recap|overview of (my|this|the) (visit|consultation|conversation))\b",
    re.IGNORECASE,
)


def wants_structured_summary(text: str) -> bool:
    return bool(_SUMMARY_REQUEST_PATTERN.search(text or ""))


CRITICAL_SAFETY_RULE = (
    "CRITICAL SAFETY RULE: Under no circumstances should you engage in exploratory intake "
    "questioning when the patient describes acute life-threatening symptoms (e.g., ischemic "
    "chest pain, stroke signs, anaphylaxis, severe respiratory distress). Escalate to emergency "
    "services immediately."
)


def format_history(conversation_history):
    if not conversation_history:
        return "None (this is the first message in the consultation)."
    lines = []
    for turn in conversation_history:
        speaker = "Patient/Clinician" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


def detect_prompt_tone(user_input: str) -> str:
    if not user_input:
        return "clinical_qa"
    if len(_USMLE_OPTION_PATTERN.findall(user_input)) >= 2:
        return "usmle"
    if _FIRST_PERSON_PATTERN.search(user_input) and _SYMPTOM_KEYWORD_PATTERN.search(user_input):
        return "patient"
    return "clinical_qa"


def format_patient_profile(age=None, sex=None, duration=None, conditions=None, medications=None):
    bits = []
    if age:
        bits.append(f"The patient is {age} years old")
    if sex:
        bits.append(f"sex: {sex}")
    if duration:
        bits.append(f"symptoms have been present for {duration}")
    if conditions:
        bits.append(f"existing conditions/history: {conditions}")
    if medications:
        bits.append(f"current medications: {medications}")
    if not bits:
        return ""
    return "Patient Profile: " + "; ".join(bits) + "."


def build_prompt(
    vignette_text,
    detected_options=None,
    evidence_md="",
    conversation_history=None,
    forced_tone=None,
    prompt_style=None,
):
    """Builds the clinical prompt, optionally wrapping with MedGemma turn markers."""
    if prompt_style is None:
        prompt_style = PROMPT_STYLE

    history_block = format_history(conversation_history)
    if detected_options:
        tone = "usmle"
    elif forced_tone in ("patient", "clinical_qa"):
        tone = forced_tone
    else:
        prior_user_text = "\n".join(
            turn.get("content", "") for turn in (conversation_history or []) if turn.get("role") == "user"
        )
        tone_text = f"{prior_user_text}\n{vignette_text}" if prior_user_text else vignette_text
        tone = detect_prompt_tone(tone_text)

    if tone == "usmle":
        opts_str = "\n".join([f"({k}) {v}" for k, v in sorted(detected_options.items())])
        raw_prompt = f"""You are an expert clinical physician taking the USMLE board examination.
Analyze the clinical vignette using the retrieved guidelines and prior conversation context. Explain your diagnostic deduction step-by-step and conclude with the correct option.

### Previous Case History
{history_block}

### Retrieved Guidelines
{evidence_md if evidence_md else "Standard clinical guidelines apply."}

### Current Question
Clinical Vignette:
{vignette_text}

Options:
{opts_str}

Base your reasoning strictly on the pathophysiological principles in the Retrieved Guidelines and the vignette's findings. Evaluate the most likely standard etiology first before considering the options.

Keep your deduction concise (under 250 words). Your response must strictly end with "Answer: [Option Letter]" on a new line, with no text after it."""

    elif tone == "patient":
        known_fields = extract_intake_fields(f"{history_block}\n{vignette_text}")
        known_info_line = (
            "Already known from this conversation — do NOT ask the patient for these again: "
            + "; ".join(f"{k}: {v}" for k, v in known_fields.items()) + "."
            if known_fields
            else "No structured intake details have been confirmed yet in this conversation."
        )

        negative_constraints = """STRICT NEGATIVE CONSTRAINTS:
- NEVER start with "Dear Patient", "Hello Patient", or any letter-style greeting.
- NEVER end with "Best regards", "[Your Name]", "Clinical Physician", or any signature.
- NEVER recommend prescription drugs (e.g. Sertraline, Methimazole) or offer to schedule clinic appointments.
- NEVER give unsafe dietary advice.
- CRITICAL ANTI-HALLUCINATION RULE: NEVER attribute symptoms to the patient (such as frothy sputum, coughing blood, nausea, vomiting, dizziness, radiating pain, diaphoresis) unless the patient EXPLICITLY reported them in their message. Do NOT copy symptoms from the retrieved textbook guidelines into what the patient feels.
- Clearly distinguish between what the patient actually reported vs what the textbook guidelines describe as hypothetical warning signs to watch out for."""

        if wants_structured_summary(vignette_text):
            format_instruction = """The patient has explicitly asked for a written summary. Produce ONE structured Markdown report using EXACTLY these headings, in this order, each grounded in the Retrieved Guidelines:

### 📋 Virtual Patient Intake Summary
### 💡 Safe At-Home Self-Care Measures
### 🚫 What to Avoid
### 😌 What Not to Panic About
### 🩺 Questions & Tests to Discuss With Your Doctor
### 🚨 Emergency Red Flags"""
        else:
            format_instruction = "Respond in direct prose (with bullet points where they aid clarity) — do not produce a fixed multi-section report with formal headings unless the patient explicitly asks for a written summary."

        persona = f"""You are an evidence-based clinical triage and medical education assistant. You are NOT a doctor, physician, or clinic scheduler.

{CRITICAL_SAFETY_RULE}

{negative_constraints}

{known_info_line}

Maintain full awareness of previous conversation turns, grounded in the Retrieved Guidelines and the Previous Case History. Do NOT add generic empathetic preamble, conversational filler, greetings, or assumptions not supported by the context — every sentence should carry a concrete fact, criterion, dosage, or recommendation. State facts directly.

CORE INSTRUCTIONS:
1. STRICT PATIENT SYMPTOM FIDELITY: Only address the exact symptoms the user explicitly reported. If explaining a condition from the guidelines, introduce other textbook symptoms as "Potential warning signs to monitor" or "Clinical signs described in guidelines", NEVER as symptoms the patient already has.
2. DIRECT ANSWER FIRST: If the user asks specific questions (e.g., medication safety, food interactions, standard adult dosages, specialist recommendations, or home care), answer those questions IMMEDIATELY in your response using retrieved clinical evidence. List contraindications and safety advice. Never recommend a prescription-only drug.
3. CONTEXTUAL AWARENESS: Do NOT ask for information the user already provided. You may ask 1-2 relevant follow-up questions ONLY if critical details are missing to assess urgency.
4. SAFETY & RED FLAGS: Briefly highlight relevant warning signs at the end.

{format_instruction}"""

        raw_prompt = f"""{persona}

### Previous Case History
{history_block}

### Retrieved Guidelines
{evidence_md if evidence_md else "Standard clinical guidelines apply."}

### Current Question
{vignette_text}

Provide your response:"""

    else:
        persona = f"""You are an expert clinical physician and medical educator.

{CRITICAL_SAFETY_RULE}

Answer the clinical question strictly and concisely using the provided context chunks. Do not add generic empathetic preamble, conversational filler, greetings, or assumptions not supported by the context.
- STRICT GROUNDING: Never falsely claim the patient has symptoms from the textbook guidelines unless explicitly presented in the case.
- State facts, criteria, dosages, and contraindications directly, grounded in the Retrieved Guidelines.
- Use clear clinical headings (Mechanism / Pathophysiology, Diagnostic Evaluation, Management) where they help organize the answer.
- Every factual assertion should be directly traceable to the Retrieved Guidelines."""

        raw_prompt = f"""{persona}

### Previous Case History
{history_block}

### Retrieved Guidelines
{evidence_md if evidence_md else "Standard clinical guidelines apply."}

### Current Question
{vignette_text}

Provide your response:"""

    # Wrap according to model prompt style
    if prompt_style == "gemma":
        return f"<start_of_turn>user\n{raw_prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif prompt_style == "chat":
        return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return raw_prompt


def sanitize_patient_output(text: str) -> str:
    """Strips letter-style greetings or signatures."""
    if not text:
        return text
    text = re.sub(r'^(Dear\s+Patient,?\s*|Hello\s+Patient,?\s*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Best\s+regards,?\s*(\[.*?\]|\bClinical\s+Physician\b.*)?)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\[Your\s+Name\]|\bClinical\s+Physician\s*(&\s*Medical\s*Educator)?\b)', '', text, flags=re.IGNORECASE)
    return text.strip()
