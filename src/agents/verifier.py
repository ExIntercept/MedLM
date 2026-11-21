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


def _salient_terms(text):
    return {w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", text) if w.lower() not in STOPWORDS}


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

    reference_terms = _salient_terms(
        " ".join(text for _, text, _ in retrieved_evidence) + " " + vignette
        + " " + " ".join((options or {}).values())
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if len(s.strip()) > 15]

    verified_claims, flagged_claims = [], []
    for sent in sentences:
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", sent) if w.lower() not in STOPWORDS]
        if not words:
            continue
        matched = [w for w in words if w in reference_terms]
        overlap = round(len(matched) / len(words), 2)
        record = {"claim": sent, "overlap": overlap}
        if overlap >= OVERLAP_THRESHOLD:
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
