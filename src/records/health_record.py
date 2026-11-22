"""Running health record: facts a patient states about themselves while chatting.

``extract_health_facts`` reads one (already sanitized) message and returns the
self-reported facts it finds: age, sex, symptom duration, conditions, current
and previous medications, allergies. ``merge_record`` folds those into the
stored record: single values are replaced, list fields gain new items, and a
medication the patient says they stopped moves from current to previous.

Deliberately conservative: a fact is only taken from a clause that is clearly
about the speaker ("I have…", "I take…", "allergic to…"), and messages about
someone else ("my son…") or exam-style case vignettes are skipped entirely, so
they never end up in the user's own record.

This is separate from ``src.agents.intake_extraction`` on purpose: that module
feeds the model prompt, and the paper's experiments depend on it unchanged.
"""
from __future__ import annotations

import re
from typing import Optional

SCALAR_FIELDS = ("age", "sex", "duration")
LIST_FIELDS = ("conditions", "medications", "past_medications", "allergies")
FIELDS = SCALAR_FIELDS + LIST_FIELDS
MAX_FIELD_CHARS = 300
NONE_KNOWN = "None known"

# ── Who is this about? ────────────────────────────────────────────────────
_OTHER_PERSON = re.compile(
    r"\bmy\s+(?:\w+\s+)?(?:son|daughter|child|children|kid|kids|baby|toddler|infant|mother|mom|mum|father|dad|"
    r"parent|parents|wife|husband|partner|boyfriend|girlfriend|brother|sister|sibling|grandmother|grandma|"
    r"grandfather|grandpa|grandson|granddaughter|aunt|uncle|cousin|niece|nephew|friend|neighbou?r|patient|"
    r"client|colleague|roommate)\b",
    re.IGNORECASE,
)
_VIGNETTE = re.compile(
    r"\b(?:presents?|presented|presenting)\s+(?:to|with)\b|\bwhich of the following\b|"
    r"\ba\s+\d{1,3}[- ]?(?:year|yr)[- ]?old\s+(?:man|woman|male|female|boy|girl|patient|child)\b|"
    r"^\s*[A-E][.)]\s",
    re.IGNORECASE | re.MULTILINE,
)

# ── Age and sex ───────────────────────────────────────────────────────────
_AGE = [
    re.compile(r"\bI(?:'m| am)\s+(?:a\s+)?(\d{1,3})(?:\s*(?:years?|yrs?)(?:[- ]old)?|\s*y/?o)?(?=$|[\s,.;!?])", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:-\s*)?(?:years?|yrs?)[- ]old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*y/?o\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+)?age(?:d)?\s*(?:is|:|-)?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*([MF])\b"),
]
_SEX = re.compile(
    r"\bI(?:'m| am)\s+(?:a\s+)?(?:\d{1,3}[- ]?(?:years?|yrs?)?[- ]?(?:old)?\s*,?\s*(?:and\s+)?)?"
    r"(male|female|man|woman|guy|lady|boy|girl)\b"
    r"|\b(?:\d{1,3}[- ]?(?:years?|yrs?)?[- ]?old|\d{1,3})\s*,?\s*(male|female|man|woman)\b"
    r"|\b(?:sex|gender)\s*(?:is|:)\s*(male|female|man|woman)\b"
    r"|^\s*(male|female)\b",
    re.IGNORECASE,
)
_SEX_NORMALIZE = {"male": "Male", "man": "Male", "guy": "Male", "boy": "Male", "m": "Male",
                  "female": "Female", "woman": "Female", "lady": "Female", "girl": "Female", "f": "Female"}

# ── Duration ──────────────────────────────────────────────────────────────
_DURATION = re.compile(
    r"\b(?:for|since|over|in)\s+(?:the\s+)?((?:past|last)\s+(?:\d+|a|an|few|couple(?:\s+of)?|several)?\s*"
    r"(?:hours?|days?|weeks?|months?|years?)|(?:\d+|a|an|one|two|three|four|five|six|seven|few|couple(?:\s+of)?|several)"
    r"\s*(?:hours?|days?|weeks?|months?|years?)|yesterday|last\s+(?:night|week|month)|this\s+morning|this\s+week)\b"
    r"|\b((?:\d+|a|an|few|couple(?:\s+of)?|several)\s*(?:hours?|days?|weeks?|months?))\s+(?:now|ago)\b",
    re.IGNORECASE,
)

# ── Conditions ────────────────────────────────────────────────────────────
# Chronic conditions and long-term states worth keeping on a record. Symptoms
# ("chest pain", "fever") are deliberately not here: they belong to the visit.
_CONDITION_TERMS = [
    ("type 1 diabetes", r"type\s*1\s+diabetes|t1dm"),
    ("type 2 diabetes", r"type\s*2\s+diabetes|t2dm"),
    ("diabetes", r"diabet(?:es|ic)"),
    ("prediabetes", r"pre-?diabet(?:es|ic)"),
    ("hypertension", r"hypertension|high\s+blood\s+pressure|high\s+bp|blood\s+pressure\s+(?:is|was|has\s+been)\s+(?:high|raised|elevated)"),
    ("high cholesterol", r"high\s+cholesterol|cholesterol\s+(?:is|was|has\s+been)\s+(?:high|raised|elevated)|hypercholesterol[a-z]*|hyperlipid[a-z]*"),
    ("asthma", r"asthma(?:tic)?"),
    ("COPD", r"copd|emphysema|chronic\s+bronchitis"),
    ("heart failure", r"heart\s+failure|chf"),
    ("coronary artery disease", r"coronary\s+artery\s+disease|\bcad\b|angina"),
    ("previous heart attack", r"heart\s+attack|myocardial\s+infarction"),
    ("atrial fibrillation", r"atrial\s+fibrillation|\bafib\b|\ba-fib\b"),
    ("previous stroke", r"(?:had\s+a\s+)?stroke"),
    ("chronic kidney disease", r"chronic\s+kidney\s+disease|\bckd\b|kidney\s+disease|renal\s+failure"),
    ("liver disease", r"liver\s+disease|cirrhosis|fatty\s+liver"),
    ("hepatitis", r"hepatitis\s*[a-c]?"),
    ("HIV", r"\bhiv\b"),
    ("hypothyroidism", r"hypothyroid(?:ism)?|underactive\s+thyroid"),
    ("hyperthyroidism", r"hyperthyroid(?:ism)?|overactive\s+thyroid|graves"),
    ("epilepsy", r"epilep(?:sy|tic)|seizure\s+disorder"),
    ("migraine", r"migraines?"),
    ("depression", r"depression|depressed"),
    ("anxiety", r"anxiety(?:\s+disorder)?"),
    ("bipolar disorder", r"bipolar"),
    ("arthritis", r"(?:rheumatoid\s+|osteo)?arthritis"),
    ("osteoporosis", r"osteoporosis"),
    ("gout", r"\bgout\b"),
    ("GERD", r"\bgerd\b|acid\s+reflux"),
    ("IBS", r"\bibs\b|irritable\s+bowel"),
    ("Crohn's disease", r"crohn'?s"),
    ("ulcerative colitis", r"ulcerative\s+colitis"),
    ("celiac disease", r"c(?:o)?eliac"),
    ("cancer", r"cancer|lymphoma|leukemia|leukaemia"),
    ("anemia", r"an(?:a)?emi[ac]"),
    ("sleep apnea", r"sleep\s+apn(?:o)?ea"),
    ("PCOS", r"\bpcos\b|polycystic\s+ovar"),
    ("pregnant", r"pregnant|pregnancy"),
    ("obesity", r"obes(?:e|ity)"),
    ("psoriasis", r"psoriasis"),
    ("eczema", r"eczema"),
]
_CONDITION_RES = [(name, re.compile(rf"\b(?:{pat})\b", re.IGNORECASE)) for name, pat in _CONDITION_TERMS]
# A clause only counts when it is about the speaker having the condition.
_POSSESSION = re.compile(
    r"\bI\s+(?:have|'ve|have\s+got|'ve\s+got|got|suffer|am|'m|was\s+diagnosed|was\s+told)\b|\bI(?:'m| am)\b|"
    r"\bdiagnosed\s+with\b|\bhistory\s+of\b|\bknown\s+case\s+of\b|\bsuffer(?:ing)?\s+from\b|\bliving\s+with\b|"
    r"\b(?:woman|man|male|female|guy|lady|patient)\s+with\b|\bmy\s+(?:diabetes|asthma|blood\s+pressure|bp|thyroid|"
    r"epilepsy|migraines?|arthritis|depression|anxiety|copd|heart|kidneys?|liver|cholesterol)\b|\bas\s+a\s+diabetic\b|"
    r"\b(?:take|taking|on|prescribed|use)\b.+\bfor\s+(?:my\s+)?",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:no|not|don'?t|do\s+not|never|without|neither|nor|isn'?t|aren'?t|haven'?t|denies?)\b", re.IGNORECASE)
_QUESTION_ABOUT = re.compile(r"^\s*(?:what|how|why|can|could|does|do|is|are|should|will|would)\b", re.IGNORECASE)

# ── Medications ───────────────────────────────────────────────────────────
_DRUG_NAMES = (
    "metformin|insulin|glipizide|gliclazide|glimepiride|sitagliptin|empagliflozin|dapagliflozin|semaglutide|"
    "ozempic|liraglutide|lisinopril|enalapril|ramipril|losartan|valsartan|amlodipine|nifedipine|"
    "hydrochlorothiazide|hctz|furosemide|lasix|spironolactone|bisoprolol|metoprolol|atenolol|propranolol|"
    "carvedilol|atorvastatin|simvastatin|rosuvastatin|pravastatin|aspirin|clopidogrel|warfarin|apixaban|"
    "rivaroxaban|dabigatran|heparin|digoxin|nitroglycerin|levothyroxine|thyroxine|eltroxin|methimazole|"
    "carbimazole|salbutamol|albuterol|ventolin|budesonide|fluticasone|symbicort|seretide|montelukast|"
    "tiotropium|prednisone|prednisolone|hydrocortisone|dexamethasone|ibuprofen|naproxen|diclofenac|"
    "paracetamol|acetaminophen|tylenol|advil|codeine|tramadol|morphine|oxycodone|gabapentin|pregabalin|"
    "sertraline|fluoxetine|citalopram|escitalopram|paroxetine|venlafaxine|duloxetine|mirtazapine|"
    "amitriptyline|bupropion|lithium|quetiapine|olanzapine|risperidone|aripiprazole|diazepam|lorazepam|"
    "alprazolam|clonazepam|zolpidem|melatonin|levetiracetam|lamotrigine|valproate|carbamazepine|phenytoin|"
    "omeprazole|pantoprazole|esomeprazole|lansoprazole|ranitidine|famotidine|amoxicillin|augmentin|"
    "azithromycin|doxycycline|ciprofloxacin|cephalexin|clindamycin|metronidazole|nitrofurantoin|"
    "allopurinol|colchicine|methotrexate|hydroxychloroquine|adalimumab|tamoxifen|letrozole|"
    "contraceptive|birth\\s+control|the\\s+pill|iron|folic\\s+acid|vitamin\\s+[a-z0-9]+|antihistamines?|"
    "cetirizine|loratadine|inhaler|statins?|blood\\s+thinners?|antibiotics?|antidepressants?|steroids?"
)
_DRUG_SUFFIX = r"[a-z]+(?:pril|sartan|olol|statin|formin|azole|prazole|cillin|mycin|cycline|floxacin|dipine|tidine|oxetine|triptyline|pam|lam|gliptin|gliflozin|glutide|mab|nib)"
_DRUG = re.compile(rf"\b(?:{_DRUG_NAMES}|{_DRUG_SUFFIX})\b", re.IGNORECASE)
_DOSE = r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|units?|iu)\b)?(?:\s+(?:once|twice|daily|a\s+day|per\s+day|at\s+night|every\s+\w+))*"
_DRUG_WITH_DOSE = re.compile(rf"\b(?:{_DRUG_NAMES}|{_DRUG_SUFFIX})\b{_DOSE}", re.IGNORECASE)

_CURRENT_MED_CUE = re.compile(
    r"\bI\s+(?:take|use|am\s+taking|'m\s+taking|am\s+on|'m\s+on|have\s+been\s+taking|'ve\s+been\s+taking|"
    r"have\s+been\s+on|'ve\s+been\s+on|was\s+prescribed|got\s+prescribed|still\s+take)\b|"
    r"\b(?:currently\s+(?:taking|on)|taking|prescribed|my\s+(?:medications?|meds|tablets|pills)\s+(?:are|is|include))\b|"
    r"\b(?:on|with)\s+(?=(?:" + _DRUG_NAMES + r"|" + _DRUG_SUFFIX + r")\b)",
    re.IGNORECASE,
)
_PAST_MED_CUE = re.compile(
    r"\b(?:I\s+)?(?:used\s+to\s+(?:take|be\s+on)|stopped(?:\s+taking)?|quit(?:\s+taking)?|was\s+(?:taking|on)|"
    r"(?:have|'ve)\s+(?:stopped|come\s+off)|came\s+off|no\s+longer\s+(?:take|on)|don'?t\s+take|previously\s+(?:on|took|taking)|"
    r"took\s+\w+\s+(?:before|in\s+the\s+past)|was\s+switched\s+(?:from|off))\b",
    re.IGNORECASE,
)

# ── Allergies ─────────────────────────────────────────────────────────────
_NO_ALLERGIES = re.compile(
    r"\b(?:no\s+(?:known\s+)?(?:drug\s+)?allergies|nkda|not\s+allergic\s+to\s+anything|don'?t\s+have\s+any\s+allergies|"
    r"I\s+have\s+no\s+allergies)\b",
    re.IGNORECASE,
)
_ALLERGY_LIST = re.compile(
    r"\b(?:I(?:'m| am)\s+)?(?:allergic\s+to|(?:have\s+an?\s+)?allerg(?:y|ies)\s+(?:to|:|-|are|is|include))\s+"
    r"([^.;!?\n]{2,120})",
    re.IGNORECASE,
)
_ALLERGY_BEFORE = re.compile(
    r"\b([A-Za-z][A-Za-z -]{1,30}?)\s+allerg(?:y|ic)\b(?!\s+to)",
    re.IGNORECASE,
)
_ALLERGY_STOP = re.compile(r"\b(?:but|and\s+(?:I|my|it|they|have|take|am)|because|which|so|when|since)\b.*$", re.IGNORECASE)

_CLAUSE_SPLIT = re.compile(r"[.;!?\n]+|\s+but\s+|\s+however\s+", re.IGNORECASE)
_ITEM_SPLIT = re.compile(r"\s*(?:,|\band\b|\bplus\b|&|/|\bas\s+well\s+as\b)\s*", re.IGNORECASE)
_FILLER = re.compile(r"^(?:also|some|an?|the|my|daily|regular(?:ly)?)\s+", re.IGNORECASE)


def _tidy(item: str) -> str:
    item = re.sub(r"\s+", " ", item).strip(" ,.-:")
    while True:
        new = _FILLER.sub("", item)
        if new == item:
            break
        item = new
    return item[:80]


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def about_someone_else(text: str) -> bool:
    """True when the message describes another person or an exam-style case."""
    if _OTHER_PERSON.search(text):
        return True
    # "I am a 45 year old woman" is the speaker, not a case vignette.
    own = re.sub(r"\bI(?:'m| am)\s+an?\s+\d{1,3}[- ]?(?:years?|yrs?)[- ]?old\s+\w+", " ", text, flags=re.IGNORECASE)
    return bool(_VIGNETTE.search(own))


def _age(text: str) -> Optional[str]:
    for rx in _AGE:
        m = rx.search(text)
        if m:
            n = int(m.group(1))
            if 0 < n < 120:
                return str(n)
    return None


def _sex(text: str) -> Optional[str]:
    m = _SEX.search(text)
    if m:
        word = next(g for g in m.groups() if g)
        return _SEX_NORMALIZE[word.lower()]
    m = _AGE[-1].search(text)  # "62M"
    if m:
        return _SEX_NORMALIZE[m.group(2).lower()]
    return None


def _duration(text: str) -> Optional[str]:
    m = _DURATION.search(text)
    if not m:
        return None
    return re.sub(r"\s+", " ", (m.group(1) or m.group(2))).strip()


def _conditions(clauses: list[str]) -> list[str]:
    found: list[str] = []
    for clause in clauses:
        if not _POSSESSION.search(clause) or _QUESTION_ABOUT.match(clause):
            continue
        for name, rx in _CONDITION_RES:
            m = rx.search(clause)
            if not m:
                continue
            # "I don't have diabetes" / "no history of asthma"
            if _NEGATION.search(clause[: m.start()][-40:]):
                continue
            # "type 2 diabetes" already covers "diabetes"
            if name in ("diabetes", "prediabetes") and any("diabetes" in f for f in found):
                continue
            if name == "diabetes" and re.search(r"pre-?diabet", clause, re.IGNORECASE):
                continue
            found.append(name)
    return found


_DRUG_LABEL = {"pill": "contraceptive pill", "birth control": "contraceptive pill"}


def _drugs_in(fragment: str) -> list[str]:
    out = []
    for m in _DRUG_WITH_DOSE.finditer(fragment):
        item = _tidy(m.group(0))
        out.append(_DRUG_LABEL.get(item.lower(), item))
    return out


_ALLERGY_WORDS = re.compile(r"\b(?:allerg(?:ic|y|ies)|reactions?\s+to|intoleran(?:t|ce)|can'?t\s+(?:take|tolerate))\b", re.IGNORECASE)


def _medications(clauses: list[str]) -> tuple[list[str], list[str]]:
    current: list[str] = []
    past: list[str] = []
    for clause in clauses:
        # "I take warfarin and I'm allergic to penicillin": penicillin is not a medication.
        cut = _ALLERGY_WORDS.search(clause)
        if cut:
            clause = clause[: cut.start()]
        if _QUESTION_ABOUT.match(clause) and not re.search(r"\bI\b", clause):
            continue
        past_cue = _PAST_MED_CUE.search(clause)
        cur_cue = _CURRENT_MED_CUE.search(clause)
        if past_cue:
            past.extend(_drugs_in(clause[past_cue.end():]))
            before = clause[: past_cue.start()]
            if _CURRENT_MED_CUE.search(before):
                current.extend(_drugs_in(before))
            continue
        if cur_cue:
            tail = clause[cur_cue.start():]
            drugs = _drugs_in(tail)
            if drugs:
                current.extend(drugs)
            elif re.match(r"(?:taking|I\s+take|I(?:'m| am)\s+taking|prescribed)", tail, re.IGNORECASE):
                # Unknown drug name right after a strong cue: keep the first item, dose included.
                head = _ITEM_SPLIT.split(tail[cur_cue.end() - cur_cue.start():].strip())[0] if tail else ""
                head = _tidy(head)
                if head and len(head.split()) <= 4 and re.search(r"\d\s*(?:mg|mcg|ml|units?)\b", head, re.IGNORECASE):
                    current.append(head)
    return current, past


def _allergies(text: str) -> list[str]:
    found: list[str] = []
    for m in _ALLERGY_LIST.finditer(text):
        span = _ALLERGY_STOP.sub("", m.group(1))
        for item in _ITEM_SPLIT.split(span):
            item = _tidy(item)
            if item and not _NEGATION.fullmatch(item) and len(item) > 1:
                found.append(item)
    for m in _ALLERGY_BEFORE.finditer(text):
        item = _tidy(m.group(1).split()[-1] if len(m.group(1).split()) > 2 else m.group(1))
        if item.lower() not in {"an", "a", "any", "no", "my", "food", "seasonal", "known", "drug", "i", "have"}:
            found.append(item)
    return found


def extract_health_facts(text: str) -> dict:
    """Self-reported facts in one message. Empty when it is about someone else."""
    if not text or not text.strip() or about_someone_else(text):
        return {}
    clauses = _clauses(text)
    facts: dict = {}
    for key, value in (("age", _age(text)), ("sex", _sex(text)), ("duration", _duration(text))):
        if value:
            facts[key] = value
    conditions = _conditions(clauses)
    current, past = _medications(clauses)
    if _NO_ALLERGIES.search(text):
        allergies = [NONE_KNOWN]
    else:
        allergies = _allergies(text)
    allergic = {_key(a) for a in allergies}
    current = [m for m in current if _key(m) not in allergic]
    for key, items in (("conditions", conditions), ("medications", current),
                       ("past_medications", past), ("allergies", allergies)):
        items = _dedupe(items)
        if items:
            facts[key] = items
    return facts


# ── Record merge ──────────────────────────────────────────────────────────

def _key(item: str) -> str:
    """Identity of a list item: the first word, so 'metformin 500mg' == 'Metformin'."""
    words = re.findall(r"[a-z0-9']+", item.lower())
    return words[0] if words else item.lower()


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        k = _key(item)
        if k and k not in seen:
            seen.add(k)
            out.append(item)
    return out


def split_items(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s for s in (_tidy(p) for p in re.split(r"\s*[,;\n]\s*", str(value))) if s]


def merge_record(record: Optional[dict], facts: dict) -> dict:
    """Fold newly stated facts into a record (both as {field: str}); returns a new record."""
    out = {k: (record or {}).get(k) or "" for k in FIELDS}
    for k in SCALAR_FIELDS:
        if facts.get(k):
            out[k] = str(facts[k])[:MAX_FIELD_CHARS]
    lists = {k: split_items(out[k]) for k in LIST_FIELDS}
    new_past = split_items(facts.get("past_medications"))
    new_current = split_items(facts.get("medications"))
    stopped = {_key(i) for i in new_past}
    restarted = {_key(i) for i in new_current}
    lists["medications"] = [i for i in lists["medications"] if _key(i) not in stopped]
    lists["past_medications"] = [i for i in lists["past_medications"] if _key(i) not in restarted]
    for k in LIST_FIELDS:
        incoming = split_items(facts.get(k))
        if k == "allergies":
            if incoming == [NONE_KNOWN]:
                lists[k] = lists[k] or [NONE_KNOWN]
                continue
            if incoming:
                lists[k] = [i for i in lists[k] if i != NONE_KNOWN]
        # A newer mention (e.g. with a dose) replaces the older wording.
        incoming_keys = {_key(i) for i in incoming}
        lists[k] = _dedupe([i for i in lists[k] if _key(i) not in incoming_keys] + incoming)
    for k in LIST_FIELDS:
        out[k] = ", ".join(lists[k])[:MAX_FIELD_CHARS]
    return out


def clean_record(raw: Optional[dict]) -> dict:
    """Keep only known fields as trimmed strings (for records the user edits)."""
    raw = raw or {}
    out = {}
    for k in FIELDS:
        v = raw.get(k)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        out[k] = ("" if v is None else str(v)).strip()[:MAX_FIELD_CHARS]
    return out
