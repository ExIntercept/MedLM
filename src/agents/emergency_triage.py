"""Emergency red-flag triage deterministic classifier."""
import re

EMERGENCY_WARNING = (
    "🚨 **EMERGENCY MEDICAL WARNING**\n\n"
    "**Please call emergency services (such as 911 / 112) or proceed to the nearest emergency "
    "department immediately.**\n\n"
    "The symptoms you described indicate a potential medical emergency that requires urgent, "
    "in-person clinical care:\n"
    "- **Do not drive yourself** to the hospital; wait for emergency responders or have someone "
    "transport you.\n"
    "- **Stop all physical exertion** and remain seated or lying down in a comfortable position.\n"
    "- Do not delay seeking care to complete further intake questions."
)


def _any(text, *patterns):
    return any(p.search(text) for p in patterns)


def _flags(*words):
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# Cardiovascular
_CARDIAC_PRESSURE = re.compile(r"\b(chest|heart)\b.{0,25}\b(crushing|squeezing|pressure|tightness|pain)\b|\b(crushing|squeezing)\b.{0,25}\b(chest|heart)\b", re.I)
_CARDIAC_RADIATION = re.compile(r"\b(radiat\w*|spread\w*|shooting)\b.{0,30}\b(jaw|neck|left arm|arm|shoulder)\b", re.I)
_CARDIAC_CLAMMY = re.compile(r"\bcold and clammy\b|\bclammy skin\b|\bcold sweat\w*\b", re.I)
_CARDIAC_TEARING = re.compile(r"\b(tearing|ripping)\b.{0,20}\b(back|chest)\s*pain\b", re.I)
_CARDIAC_SYNCOPE = re.compile(r"\bfainted\b|\bfainting\b|\bsyncope\b|\bpassed out\b|\blost consciousness\b", re.I)
_CARDIAC_EXERTION = re.compile(r"\b(heart|chest)\s*pain\b.{0,30}\b(gym|exercise|workout|running|exertion)\b|\b(gym|exercise|workout|running|exertion)\b.{0,30}\b(heart|chest)\s*pain\b", re.I)
_CARDIAC_VALVE = re.compile(r"\bbicuspid\b.{0,30}\b(heart|chest|pain)\b|\b(heart|chest|pain)\b.{0,30}\bbicuspid\b", re.I)

# Neurological (FAST)
_NEURO_FACIAL_DROOP = re.compile(r"\bfacial droop\w*\b|\bface\b.{0,15}\bdroop\w*\b|\bdrooping face\b", re.I)
_NEURO_WEAKNESS = re.compile(
    r"\b(sudden|one[- ]sided)\b.{0,30}\b(weakness|numbness|paraly\w*)\b|"
    r"\b(weakness|numbness|paraly\w*)\b.{0,30}\bone side\b",
    re.I,
)
_NEURO_SPEECH = re.compile(r"\bslurred speech\b|\bcan'?t speak\b|\baphasia\b|\bcan'?t find (my|the) words\b", re.I)
_NEURO_THUNDERCLAP = re.compile(r"\bworst headache\b|\bthunderclap headache\b", re.I)
_NEURO_SEIZURE_WORD = re.compile(r"\bseiz\w*\b", re.I)
_NEURO_SEIZURE_PROLONGED = re.compile(r"\bprolonged\b|\bwon'?t stop\b|\bstill (seizing|going)\b|\bcontinuous\b", re.I)
_NEURO_SEIZURE_MINUTES = re.compile(r"(\d+)\s*min", re.I)
_NEURO_NECK_STIFF = re.compile(r"\bneck stiff\w*\b|\bstiff neck\b", re.I)
_NEURO_FEVER = _flags("fever", "high temperature")
_NEURO_PHOTOPHOBIA = re.compile(r"\bphotophobia\b|\blight sensitiv\w*\b|\bhurts? to look at light\b", re.I)


def _seizure_over_5_min(text):
    if not _NEURO_SEIZURE_WORD.search(text):
        return False
    if _NEURO_SEIZURE_PROLONGED.search(text):
        return True
    match = _NEURO_SEIZURE_MINUTES.search(text)
    return bool(match and int(match.group(1)) > 5)


# Respiratory
_RESP_STRIDOR_CHOKING = re.compile(r"\bstridor\b|\bchoking\b", re.I)
_RESP_CANT_BREATHE = re.compile(r"\b(can'?t|cannot|unable to)\b.{0,15}\b(breathe|breathing|speak|talk)\b", re.I)
_RESP_CYANOSIS = re.compile(r"\bblue lips\b|\bcyanosis\b|\bturning blue\b|\bblue (skin|face|fingers)\b", re.I)
_RESP_HEMOPTYSIS = re.compile(r"\bcoughing up blood\b|\bhemoptysis\b|\bblood.{0,10}\bcough\w*\b", re.I)

# Anaphylaxis
_ANAPHYLAXIS_WORD = re.compile(r"\banaphylax\w*\b", re.I)
_ANAPHYLAXIS_THROAT = re.compile(r"\bthroat\b.{0,20}\b(tight\w*|closing|swelling|swoll?en)\b", re.I)
_ANAPHYLAXIS_SWELLING = re.compile(r"\bswelling\b.{0,20}\b(lips|tongue|face)\b|\b(lips|tongue|face)\b.{0,20}\bswoll?en\b", re.I)
_ANAPHYLAXIS_WHEEZE = _flags("wheez\\w*")
_ANAPHYLAXIS_HIVES = _flags("hives")
_ANAPHYLAXIS_ALLERGEN = re.compile(r"\ballerg\w*\b|\bafter eating\b|\bbee sting\b|\binsect sting\b", re.I)

# Gastrointestinal / Surgical Abdomen
_GI_HEMATEMESIS = re.compile(r"\bvomit\w*\b.{0,15}\bblood\b|\bhematemesis\b", re.I)
_GI_MELENA = re.compile(r"\bblack\b.{0,10}(tarry)?\s*stool\w*\b|\bmelena\b|\btarry stool\w*\b", re.I)
_GI_RIGID_ABDOMEN = re.compile(r"\brigid\b.{0,15}\babdomen\b|\bboard[- ]like abdomen\b|\brebound tenderness\b", re.I)

# Trauma & Sepsis
_TRAUMA_ARTERIAL_BLEED = re.compile(r"\b(uncontrolled|severe|heavy|spurting|won'?t stop)\b.{0,15}\bbleed\w*\b|\barterial bleed\w*\b", re.I)
_TRAUMA_HEAD_LOC = re.compile(
    r"\bhead\b.{0,10}\b(trauma|injury|hit)\b.{0,30}\b(loss of consciousness|unconscious|passed out|blacked out)\b|"
    r"\b(unconscious|loss of consciousness|passed out|blacked out)\b.{0,30}\bhead\b.{0,10}\b(trauma|injury|hit)\b",
    re.I,
)
_SEPSIS_FEVER = _flags("fever", "high temperature")
_SEPSIS_SIGNS = re.compile(r"\bmottled skin\b|\baltered mental status\b|\bextreme(ly)? lethargic\b|\bextreme lethargy\b|\bconfus\w*\b|\bunresponsive\b", re.I)

# Psychiatric
_PSYCH_SUICIDE = re.compile(r"\bsuicid\w*\b|\bkill myself\b|\bend my life\b|\bwant to die\b|\bdon'?t want to live\b", re.I)
_PSYCH_SELF_HARM = re.compile(r"\bself[- ]harm\w*\b|\bcutting myself\b|\bhurt(ing)? myself\b", re.I)
_PSYCH_DELIRIUM = re.compile(r"\bdelirium\b|\bacutely confused\b|\bsevere(ly)? confus\w*\b", re.I)


def _check_cardiovascular(text):
    return _any(
        text,
        _CARDIAC_PRESSURE,
        _CARDIAC_RADIATION,
        _CARDIAC_CLAMMY,
        _CARDIAC_TEARING,
        _CARDIAC_SYNCOPE,
        _CARDIAC_EXERTION,
        _CARDIAC_VALVE,
    )


def _check_neurological(text):
    if _any(text, _NEURO_FACIAL_DROOP, _NEURO_WEAKNESS, _NEURO_SPEECH, _NEURO_THUNDERCLAP):
        return True
    if _seizure_over_5_min(text):
        return True
    if _NEURO_NECK_STIFF.search(text) and _NEURO_FEVER.search(text) and _NEURO_PHOTOPHOBIA.search(text):
        return True
    return False


def _check_respiratory(text):
    return _any(text, _RESP_STRIDOR_CHOKING, _RESP_CANT_BREATHE, _RESP_CYANOSIS, _RESP_HEMOPTYSIS)


def _check_anaphylaxis(text):
    if _ANAPHYLAXIS_WORD.search(text):
        return True
    if _any(text, _ANAPHYLAXIS_THROAT, _ANAPHYLAXIS_SWELLING):
        return True
    if _ANAPHYLAXIS_WHEEZE.search(text) and (_ANAPHYLAXIS_HIVES.search(text) or _ANAPHYLAXIS_ALLERGEN.search(text)):
        return True
    return False


def _check_gi(text):
    return _any(text, _GI_HEMATEMESIS, _GI_MELENA, _GI_RIGID_ABDOMEN)


def _check_trauma_sepsis(text):
    if _TRAUMA_ARTERIAL_BLEED.search(text) or _TRAUMA_HEAD_LOC.search(text):
        return True
    if _SEPSIS_FEVER.search(text) and _SEPSIS_SIGNS.search(text):
        return True
    return False


def _check_psychiatric(text):
    return _any(text, _PSYCH_SUICIDE, _PSYCH_SELF_HARM, _PSYCH_DELIRIUM)


_RED_FLAG_CATEGORIES = [
    ("Cardiovascular", _check_cardiovascular),
    ("Neurological (FAST)", _check_neurological),
    ("Respiratory", _check_respiratory),
    ("Anaphylaxis", _check_anaphylaxis),
    ("Gastrointestinal / Surgical Abdomen", _check_gi),
    ("Trauma & Sepsis", _check_trauma_sepsis),
    ("Psychiatric", _check_psychiatric),
]


def check_emergency_triage(text: str):
    """Returns matched category label if text describes an acute medical emergency."""
    if not text:
        return None
    for label, check in _RED_FLAG_CATEGORIES:
        if check(text):
            return label
    return None
