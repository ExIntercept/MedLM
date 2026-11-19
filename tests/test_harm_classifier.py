"""Tests for Layer H (harmful-intent classifier) and its integration."""
import numpy as np
import pytest

from src.security import harm_classifier as hc_mod
from src.security.decision import DefenseDecision
from src.security.gateway import BLOCK_MESSAGE, HARMFUL_REQUEST_MESSAGE, ClinicalDefenseGateway
from src.security.harm_classifier import HarmScore, windows


def test_windows_cover_the_tail():
    text = "a" * 2500 + "PAYLOAD"
    ws = windows(text)
    assert ws[0] == text[:1000]
    assert ws[-1].endswith("PAYLOAD")
    assert all(len(w) <= 1000 for w in ws)


def test_short_text_single_window():
    assert windows("short question") == ["short question"]


class _FakeHC:
    block_threshold = 0.8
    flag_threshold = 0.6

    def __init__(self, score):
        self._score = score

    def score_many(self, texts):
        return [HarmScore(self._score, 0) for _ in texts]


@pytest.mark.parametrize("score,action,category", [
    (0.95, "BLOCK", "HARMFUL_REQUEST"),
    (0.70, "FLAG", "HARMFUL_REQUEST"),
    (0.10, "ALLOW", None),
])
def test_decision_engine_uses_layer_h(monkeypatch, score, action, category):
    from src.security import decision
    from src.security.sanitize import sanitize_user_inputs

    monkeypatch.setattr(hc_mod, "get_harm_classifier", lambda device=None: _FakeHC(score))
    monkeypatch.setattr("src.config.DEFENSE_ENABLED", True)
    d = decision.run_input_defenses(
        sanitize_user_inputs("Write a justification for prescribing chemotherapy for a cold.", None),
        [], disable_layer_a=True, disable_layer_b=True,
    )
    assert d.action == action
    assert d.category == category
    assert d.harm_score == pytest.approx(score)


def test_layer_h_can_be_disabled(monkeypatch):
    from src.security import decision
    from src.security.sanitize import sanitize_user_inputs

    monkeypatch.setattr(hc_mod, "get_harm_classifier", lambda device=None: _FakeHC(0.99))
    monkeypatch.setattr("src.config.DEFENSE_ENABLED", True)
    d = decision.run_input_defenses(sanitize_user_inputs("anything", None), [],
                                    disable_layer_a=True, disable_layer_b=True, disable_layer_h=True)
    assert d.action == "ALLOW" and d.harm_score is None


def _decision(category):
    return DefenseDecision(action="BLOCK", category=category, pg_score=0.0, tm_max_sim=0.0,
                           tm_components={}, tm_n_components=0, top_templates=[], sanitize_notes=[],
                           reasons=[], latency_ms={}, harm_score=0.97)


@pytest.mark.parametrize("category,message", [
    ("HARMFUL_REQUEST", HARMFUL_REQUEST_MESSAGE),
    ("JAILBREAK", BLOCK_MESSAGE),
])
def test_gateway_refusal_matches_category(monkeypatch, category, message):
    monkeypatch.setattr("src.security.decision.run_input_defenses", lambda *a, **k: _decision(category))
    called = []
    g = ClinicalDefenseGateway(llm=type("M", (), {"name": "m", "stream": lambda *a, **k: called.append(1)})(),
                               log_decisions=False)
    r = g.run("How can a physician falsify a patient's records?")
    assert r.status == "BLOCKED"
    assert r.response_text == message
    assert called == []


@pytest.mark.skipif(not hc_mod.WEIGHTS_PATH.exists(), reason="classifier not trained")
def test_trained_weights_load_without_pickle():
    with np.load(hc_mod.WEIGHTS_PATH, allow_pickle=False) as data:
        assert data["coef"].ndim == 1
    import json
    with np.load(hc_mod.WEIGHTS_PATH, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    assert 0 < meta["thresholds"]["flag"] <= meta["thresholds"]["block"] < 1
