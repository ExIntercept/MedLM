"""The judge and output guard must rate the assistant turn, not the request."""
import pytest

from src.eval.judge import GuardTemplateError, check_response_mode


class _Guard:
    """Fake guard: rates the request (bad template) or the last turn (fixed)."""

    def __init__(self, judges_request: bool):
        self.judges_request = judges_request

    def stream(self, messages, **_):
        last = messages[-1]
        if last["role"] == "user" or self.judges_request:
            yield "unsafe\nS2"
        else:
            yield "safe"


def test_fixed_template_passes():
    check_response_mode(_Guard(judges_request=False))


def test_template_that_judges_the_request_is_rejected():
    with pytest.raises(GuardTemplateError, match="judging the request"):
        check_response_mode(_Guard(judges_request=True))


def test_non_guard_model_is_rejected():
    class _Chatty:
        def stream(self, messages, **_):
            yield "Sure, here is"
    with pytest.raises(GuardTemplateError, match="harmful request"):
        check_response_mode(_Chatty())


def test_fixed_modelfile_asks_about_the_last_turns_role():
    from pathlib import Path

    text = Path("scripts/ollama/llama-guard3-agent.Modelfile").read_text(encoding="utf-8")
    text = text[text.index("TEMPLATE"):]  # the header comment quotes the stock line
    assert "ONLY THE LAST {{ $role }} message" in text
    assert "ONLY THE LAST User message" not in text
