"""Evidence-grounding gate (gateway stage 8)."""
from src.agents.verifier import verify_evidence

EVIDENCE = [("c1", "Opioid withdrawal presents with mydriasis, piloerection, yawning and diarrhea. "
                   "Withdrawal is managed with supportive care.", {})]
VIGNETTE = "A 30-year-old man has yawning, piloerection and diarrhea after stopping heroin."
OPTIONS = {"A": "Haloperidol", "B": "Methadone maintenance", "C": "Ondansetron"}
ANSWER = ("The findings of yawning, piloerection and diarrhea indicate opioid withdrawal. "
          "The best choice here is methadone maintenance. "
          "Haloperidol and ondansetron would not address the dependence. Answer: B")


def test_naming_the_chosen_option_counts_as_grounded():
    without = verify_evidence(ANSWER, EVIDENCE, VIGNETTE)
    with_options = verify_evidence(ANSWER, EVIDENCE, VIGNETTE, options=OPTIONS)
    assert without["status"] == "FLAGGED"
    assert with_options["status"] == "PASS"


def test_options_do_not_ground_unrelated_claims():
    invented = ("Quantum resonance therapy realigns neural frequencies permanently. "
                "Crystals amplify the healing signal considerably.")
    assert verify_evidence(invented, EVIDENCE, VIGNETTE, options=OPTIONS)["status"] == "FLAGGED"


def test_pass_threshold_override():
    audit = verify_evidence(ANSWER, EVIDENCE, VIGNETTE)
    assert verify_evidence(ANSWER, EVIDENCE, VIGNETTE, pass_threshold=audit["score"])["status"] == "PASS"


def test_no_evidence_is_ungrounded():
    assert verify_evidence(ANSWER, [], VIGNETTE, options=OPTIONS)["status"] == "UNGROUNDED"


def test_conflicting_negation_is_not_grounded():
    evidence = [("c1", "The patient has no contraindication to methadone.", {})]
    audit = verify_evidence(
        "Methadone is contraindicated for this patient.",
        evidence,
        "The patient is asking about methadone.",
    )
    assert audit["status"] == "FLAGGED"
    assert audit["score"] == 0.0
