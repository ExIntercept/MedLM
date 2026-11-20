"""Layer G: Llama Guard input moderation (guard model stubbed)."""
import pytest

from src.llm.adapters import CallableAdapter, LLMUnavailable
from src.security import llama_guard
from src.security.decision import run_input_defenses
from src.security.llama_guard import InputGuard, parse_guard_output
from src.security.sanitize import sanitize_user_inputs


@pytest.mark.parametrize("text,unsafe,cats,blocks", [
    ("safe", False, [], False),
    ("unsafe\nS1", True, ["S1"], True),
    ("unsafe\nS1,S9", True, ["S1", "S9"], True),
    ("\n unsafe \n S6 \n", True, ["S6"], False),      # specialized advice: this product's purpose
    ("unsafe\nS11", True, ["S11"], False),            # self-harm goes to crisis triage
    ("unsafe\nS6,S1", True, ["S6", "S1"], True),
    ("unsafe", True, [], True),                       # no category given: still blocks
    ("I cannot classify that", None, [], False),
    ("", None, [], False),
])
def test_parse_guard_output(text, unsafe, cats, blocks):
    r = parse_guard_output(text)
    assert r.unsafe is unsafe
    assert r.categories == cats
    assert r.blocks is blocks


def test_input_guard_sends_only_the_user_turn():
    seen = {}

    def guard(messages, temperature, max_tokens):
        seen.update(messages=messages, temperature=temperature)
        return "unsafe\nS1"

    r = InputGuard(CallableAdapter(guard)).classify("some request")
    assert seen["messages"] == [{"role": "user", "content": "some request"}]
    assert seen["temperature"] == 0.0
    assert r.blocks and r.latency_ms >= 0


def _decide(monkeypatch, guard_reply, text="a request", disable_g=False):
    calls = []

    def guard(messages, temperature, max_tokens):
        calls.append(messages)
        if isinstance(guard_reply, Exception):
            raise guard_reply
        return guard_reply

    monkeypatch.setattr(llama_guard, "_instance", InputGuard(CallableAdapter(guard)))
    d = run_input_defenses(sanitize_user_inputs(text, {}), [], disable_layer_a=True,
                           disable_layer_b=True, disable_layer_h=True, disable_layer_g=disable_g)
    return d, calls


def test_guard_unsafe_blocks_as_harmful_request(monkeypatch):
    d, calls = _decide(monkeypatch, "unsafe\nS1")
    assert d.action == "BLOCK"
    assert d.category == "HARMFUL_REQUEST"
    assert "guard_block" in d.reasons
    assert d.guard_unsafe is True and d.guard_categories == ["S1"]
    assert len(calls) == 1


def test_guard_safe_or_medical_advice_allows(monkeypatch):
    for reply in ("safe", "unsafe\nS6"):
        d, _ = _decide(monkeypatch, reply)
        assert d.action == "ALLOW", reply


def test_guard_toggle_off_skips_the_call(monkeypatch):
    d, calls = _decide(monkeypatch, "unsafe\nS1", disable_g=True)
    assert d.action == "ALLOW" and calls == [] and d.guard_unsafe is None


def test_guard_endpoint_down_is_bypassed(monkeypatch):
    d, _ = _decide(monkeypatch, LLMUnavailable("connection refused"))
    assert d.action == "ALLOW" and d.guard_unsafe is None


def test_guard_not_configured_is_off(monkeypatch):
    monkeypatch.setattr(llama_guard, "_instance", None)
    monkeypatch.setattr("src.config.GUARD_BASE_URL", "")
    assert llama_guard.get_input_guard() is None
