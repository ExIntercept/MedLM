"""Evidence auditing and faithfulness verification."""
import re
import nltk
from pathlib import Path
_local_nltk = Path(__file__).resolve().parent.parent.parent / ".nltk_data"
if _local_nltk.exists() and str(_local_nltk) not in nltk.data.path:
    nltk.data.path.insert(0, str(_local_nltk))

from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = set()

OVERLAP_THRESHOLD = 0.18
PASS_THRESHOLD = 0.50
NEGATION_TERMS = {
    "no", "not", "never", "without", "absent", "contraindicated",
    "avoid", "unlikely", "negative",
}


def _salient_terms(text):
    return {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text) if w.lower() not in STOPWORDS}


def _token_shape(text: str) -> set[str]:
    words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text) if w.lower() not in STOPWORDS]
    bigrams = {f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)}
    return set(words) | bigrams


def _has_conflicting_negation(claim: str, evidence: str) -> bool:
    claim_words = set(re.findall(r"\b[A-Za-z]+\b", claim.lower()))
    evidence_words = set(re.findall(r"\b[A-Za-z]+\b", evidence.lower()))
    evidence_text = evidence.lower()
    claim_text = claim.lower()
    evidence_no_contraindication = bool(re.search(r"\b(?:no|not|without|never)\s+contraindication", evidence_text))
    claim_contraindicated = bool(re.search(r"\bcontraindicat(?:ed|ion)\b", claim_text))
    if evidence_no_contraindication and claim_contraindicated:
        return True
    claim_neg = bool(claim_words & NEGATION_TERMS)
    evidence_neg = bool(evidence_words & NEGATION_TERMS)
    return claim_neg != evidence_neg


def verify_evidence(response_text, retrieved_evidence, vignette="", options=None, pass_threshold=None):
    """Audits generated response sentences against retrieved evidence chunks.

    Args:
        vignette:       The user's question; its terms count as grounded.
        options:        Multiple-choice options ``{"A": "...", ...}`` stripped from
                        the vignette; naming one is restating the question.
        pass_threshold: Share of sentences that must be grounded (default PASS_THRESHOLD).
    """
    if not retrieved_evidence:
        return {
            "status": "UNGROUNDED",
            "score": 0.0,
            "verified_claims": [],
            "flagged_claims": [{"claim": "No evidence retrieved.", "overlap": 0.0}],
        }

    evidence_text = " ".join(text for _, text, _ in retrieved_evidence)
    reference_text = evidence_text + " " + vignette + " " + " ".join((options or {}).values())
    reference_terms = _salient_terms(reference_text)
    evidence_shape = _token_shape(reference_text)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if len(s.strip()) > 15]

    verified_claims, flagged_claims = [], []
    for sent in sentences:
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", sent) if w.lower() not in STOPWORDS]
        if not words:
            continue
        matched = [w for w in words if w in reference_terms]
        overlap = round(len(matched) / len(words), 2)
        shape = _token_shape(sent)
        shape_overlap = len(shape & evidence_shape) / max(1, len(shape))
        negation_conflict = _has_conflicting_negation(sent, evidence_text)
        score = min(1.0, overlap * 0.75 + shape_overlap * 0.25)
        if negation_conflict:
            score = 0.0
        record = {"claim": sent, "overlap": round(score, 2)}
        if score >= OVERLAP_THRESHOLD and not negation_conflict:
            verified_claims.append(record)
        else:
            flagged_claims.append(record)

    total = len(verified_claims) + len(flagged_claims)
    score = round(len(verified_claims) / total, 2) if total > 0 else 1.0
    threshold = PASS_THRESHOLD if pass_threshold is None else pass_threshold
    status = "PASS" if score >= threshold else "FLAGGED"

    return {
        "status": status,
        "score": score,
        "verified_claims": verified_claims,
        "flagged_claims": flagged_claims,
    }
