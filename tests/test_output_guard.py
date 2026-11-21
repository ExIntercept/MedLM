"""Stage 9: Llama Guard on the model's answer, and seeded generation."""
import pytest

from src.llm import adapters
from src.llm.adapters import CallableAdapter, LLMUnavailable, for_medical_model
from src.llm.model_profiles import resolve_profile
from src.security.decision import DefenseDecision
from src.security.gateway import OUTPUT_BLOCKED_MESSAGE, ClinicalDefenseGateway
from src.security.llama_guard import InputGuard, parse_guard_output

ANSWER = "Here is the answer you asked for."


def _allow(*_a, **_k):
    return DefenseDecision(action="ALLOW", category=None, pg_score=0.0, tm_max_sim=0.0, tm_components={},
                           tm_n_components=0, top_templates=[], sanitize_notes=[], reasons=[], latency_ms={})


class FakeGuard:
    def __init__(self, verdict="safe", fail=False):
        self.verdict, self.fail, self.calls = verdict, fail, []

    def classify_response(self, user_text, answer):
        self.calls.append((user_text, answer))
        if self.fail:
            raise LLMUnavailable("guard down")
        return parse_guard_output(self.verdict)


def _gateway(monkeypatch, guard, **toggles):
    monkeypatch.setattr("src.security.decision.run_input_defenses", _allow)
    monkeypatch.setattr("src.security.llama_guard.get_input_guard", lambda: guard)
    model = CallableAdapter(lambda m, temperature, max_tokens: ANSWER, name="stub",
                            profile=resolve_profile("medgemma"))
    return ClinicalDefenseGateway(model, retriever=None, log_decisions=False,
                                  toggles={"layer_triage": False, "layer_guardrails": False, **toggles})


def test_unsafe_answer_is_replaced(monkeypatch):
    guard = FakeGuard("unsafe\nS10")
    res = _gateway(monkeypatch, guard).run("why should group X be excluded?")
    assert res.output_blocked and res.response_text == OUTPUT_BLOCKED_MESSAGE
    assert res.draft_text == ANSWER and "output_guard" in res.trace
    assert res.meta["output_guard"]["categories"] == ["S10"]
    assert guard.calls == [("why should group X be excluded?", ANSWER)]


@pytest.mark.parametrize("verdict", ["safe", "unsafe\nS6", "unsafe\nS11", "garbled"])
def test_safe_or_non_blocking_answer_passes(monkeypatch, verdict):
    res = _gateway(monkeypatch, FakeGuard(verdict)).run("How is asthma managed?")
    assert not res.output_blocked and res.response_text == ANSWER


def test_guard_failure_is_bypassed(monkeypatch):
    res = _gateway(monkeypatch, FakeGuard(fail=True)).run("How is asthma managed?")
    assert not res.output_blocked and res.response_text == ANSWER
    assert res.meta["output_guard"] is None


def test_toggle_off_skips_the_guard(monkeypatch):
    guard = FakeGuard("unsafe\nS1")
    res = _gateway(monkeypatch, guard, layer_output_guard=False).run("How is asthma managed?")
    assert not res.output_blocked and guard.calls == []


def test_no_guard_configured(monkeypatch):
    res = _gateway(monkeypatch, None).run("How is asthma managed?")
    assert not res.output_blocked and res.response_text == ANSWER


def test_classify_response_sends_both_turns_and_keeps_the_request_end():
    seen = {}

    def guard(messages, temperature, max_tokens):
        seen["messages"] = messages
        return "safe"

    long_prompt = "x" * 20000 + " the actual request"
    InputGuard(CallableAdapter(guard)).classify_response(long_prompt, "answer")
    user, assistant = seen["messages"]
    assert (user["role"], assistant["role"]) == ("user", "assistant")
    assert user["content"].endswith("the actual request") and len(user["content"]) <= 12000
    assert assistant["content"] == "answer"


@pytest.mark.parametrize("provider", ["openai_chat", "completions"])
def test_seed_is_sent_only_when_set(monkeypatch, provider):
    payloads = []

    def fake_post(url, key, payload, name):
        payloads.append(payload)
        raise LLMUnavailable("stop here")

    monkeypatch.setattr(adapters, "_post_stream", fake_post)
    for seed in (13, None):
        a = for_medical_model("google/medgemma-4b-it", "http://x", provider, seed=seed)
        with pytest.raises(LLMUnavailable):
            list(a.stream([{"role": "user", "content": "hi"}], temperature=0.0))
    assert payloads[0]["seed"] == 13 and "seed" not in payloads[1]
