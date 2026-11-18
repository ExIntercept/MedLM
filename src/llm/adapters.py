"""Model-agnostic LLM adapters for the clinical defense gateway.

The gateway never talks to a model directly. It hands a list of chat messages
(``[{"role": "system"|"user"|"assistant", "content": str}, ...]``) to an
:class:`LLMAdapter`, which streams back text deltas. Swapping the protected
model is therefore a configuration change, not a code change.

Adapters provided:

* :class:`OpenAIChatAdapter` — ``/v1/chat/completions``. The serving stack
  applies the model's own chat template. Works with vLLM, TGI, Ollama,
  LM Studio, llama.cpp server, OpenAI and other compatible APIs.
* :class:`CompletionsAdapter` — ``/v1/completions`` with a template rendered
  here (gemma, chatml, llama3, mistral, plain). Use for base models or servers
  without a chat endpoint.
* :class:`CallableAdapter` — wraps any Python function, e.g. a local
  Hugging Face pipeline, a vendor SDK, or a stub in tests.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Iterator, Optional, Protocol, Union, runtime_checkable

import requests

from src.llm.model_profiles import ModelProfile, profile_from_config, resolve_profile
from src.security.sanitize import neutralise_control_tokens

Message = dict  # {"role": str, "content": str}


class LLMUnavailable(RuntimeError):
    """Raised when the configured model backend cannot be reached or errors."""


@runtime_checkable
class LLMAdapter(Protocol):
    """Anything that can turn chat messages into streamed text."""

    name: str

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 640,
    ) -> Iterator[str]:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_system_into_user(messages: list[Message]) -> list[Message]:
    """Fold system messages into the first user message.

    For models whose chat template has no system role (e.g. Gemma 2), or for
    raw-completion templates that only have user/model turns.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system_parts:
        return rest
    prefix = "\n\n".join(system_parts)
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{prefix}\n\n{m['content']}"
            return rest
    return [{"role": "user", "content": prefix}] + rest


def _normalise_base_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    return base_url


def _headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _iter_sse_json(response: requests.Response) -> Iterator[dict]:
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data:"):
            continue
        data = line_str[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _post_stream(url: str, api_key: str, payload: dict, backend: str) -> requests.Response:
    try:
        response = requests.post(
            url, headers=_headers(api_key), json=payload, stream=True, timeout=(15.0, 180.0)
        )
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailable(f"Could not reach {backend} at {url} ({type(exc).__name__}: {exc}).") from exc
    if response.status_code != 200:
        raise LLMUnavailable(f"{backend} returned HTTP {response.status_code}: {response.text[:300]}")
    return response


# ---------------------------------------------------------------------------
# Chat-completions adapter
# ---------------------------------------------------------------------------

class OpenAIChatAdapter:
    """OpenAI-compatible ``/v1/chat/completions`` backend."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        system_role: bool = True,
        profile: Optional["ModelProfile"] = None,
    ):
        if not base_url:
            raise LLMUnavailable("LLM_BASE_URL (or COLAB_API_BASE) is not set in .env.")
        self.base_url = _normalise_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.system_role = system_role
        self.name = f"openai_chat:{model}"
        self.profile = profile or resolve_profile(model)

    def stream(self, messages, *, temperature=0.2, max_tokens=640):
        msgs = messages if self.system_role else merge_system_into_user(messages)
        payload = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": True,
        }
        response = _post_stream(f"{self.base_url}/v1/chat/completions", self.api_key, payload, self.name)
        try:
            for chunk in _iter_sse_json(response):
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        yield piece
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailable(f"{self.name} stream failed ({type(exc).__name__}: {exc}).") from exc


# ---------------------------------------------------------------------------
# Raw-completions adapter with locally rendered templates
# ---------------------------------------------------------------------------

def _render_gemma(msgs: list[Message]) -> str:
    out = []
    for m in merge_system_into_user(msgs):
        role = "model" if m["role"] == "assistant" else "user"
        out.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
    return "".join(out) + "<start_of_turn>model\n"


def _render_chatml(msgs: list[Message]) -> str:
    out = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
    return "".join(out) + "<|im_start|>assistant\n"


def _render_llama3(msgs: list[Message]) -> str:
    out = ["<|begin_of_text|>"]
    for m in msgs:
        out.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
    return "".join(out) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def _render_mistral(msgs: list[Message]) -> str:
    out = ["<s>"]
    for m in merge_system_into_user(msgs):
        if m["role"] == "user":
            out.append(f"[INST] {m['content']} [/INST]")
        else:
            out.append(f" {m['content']}</s>")
    return "".join(out)


def _render_plain(msgs: list[Message]) -> str:
    return "\n\n".join(m["content"] for m in msgs)


TEMPLATES: dict[str, tuple[Callable[[list[Message]], str], list[str]]] = {
    "gemma": (_render_gemma, ["<end_of_turn>", "<start_of_turn>"]),
    "chatml": (_render_chatml, ["<|im_end|>", "<|im_start|>"]),
    "llama3": (_render_llama3, ["<|eot_id|>", "<|start_header_id|>"]),
    "mistral": (_render_mistral, ["</s>", "[INST]"]),
    # Base models have no turn markers of their own; stop on any common one so
    # they do not hallucinate a follow-up turn (matches the legacy client).
    "plain": (_render_plain, ["<end_of_turn>", "<start_of_turn>", "<|im_end|>", "<|im_start|>"]),
}
_TEMPLATE_ALIASES = {"chat": "chatml", "base": "plain", "llama": "llama3", "none": "plain"}


def render_template(template: str, messages: list[Message]) -> str:
    """Render messages with a named template after defanging control tokens.

    Every message's content has all model families' control tokens neutralised
    before rendering, so no content - user text, retrieved evidence, or history -
    can open or close a turn in the rendered prompt.
    """
    template = _TEMPLATE_ALIASES.get(template, template)
    if template not in TEMPLATES:
        raise ValueError(f"Unknown LLM template {template!r}; choose from {sorted(TEMPLATES)}")
    safe = [{"role": m["role"], "content": neutralise_control_tokens(m["content"])[0]} for m in messages]
    return TEMPLATES[template][0](safe)


class CompletionsAdapter:
    """OpenAI-compatible ``/v1/completions`` backend with a local chat template."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        template: str = "gemma",
        profile: Optional["ModelProfile"] = None,
    ):
        if not base_url:
            raise LLMUnavailable("LLM_BASE_URL (or COLAB_API_BASE) is not set in .env.")
        self.base_url = _normalise_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.template = _TEMPLATE_ALIASES.get(template, template)
        if self.template not in TEMPLATES:
            raise ValueError(f"Unknown LLM template {template!r}; choose from {sorted(TEMPLATES)}")
        self.name = f"completions:{self.template}:{model}"
        self.profile = profile or resolve_profile(model, fallback_template=self.template)

    def stream(self, messages, *, temperature=0.2, max_tokens=640):
        prompt = render_template(self.template, messages)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": True,
        }
        stop = TEMPLATES[self.template][1]
        if stop:
            payload["stop"] = stop
        response = _post_stream(f"{self.base_url}/v1/completions", self.api_key, payload, self.name)
        try:
            for chunk in _iter_sse_json(response):
                for choice in chunk.get("choices", []):
                    piece = choice.get("text", "")
                    if piece:
                        yield piece
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailable(f"{self.name} stream failed ({type(exc).__name__}: {exc}).") from exc


# ---------------------------------------------------------------------------
# Callable adapter
# ---------------------------------------------------------------------------

class CallableAdapter:
    """Wrap any ``fn(messages, temperature=..., max_tokens=...)``.

    ``fn`` may return a string or an iterable of string deltas.
    """

    def __init__(
        self,
        fn: Callable[..., Union[str, Iterable[str]]],
        name: str = "callable",
        profile: Optional["ModelProfile"] = None,
    ):
        self.fn = fn
        self.name = name
        self.profile = profile or resolve_profile(name)

    def stream(self, messages, *, temperature=0.2, max_tokens=640):
        result = self.fn(messages, temperature=temperature, max_tokens=max_tokens)
        if isinstance(result, str):
            yield result
        else:
            yield from result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def for_medical_model(
    model: str,
    base_url: str,
    provider: str = "completions",
    api_key: str = "",
    profile: Optional[ModelProfile] = None,
) -> LLMAdapter:
    """Build an adapter for a medical model, configured from its profile.

    The profile (see ``model_profiles.py``) supplies the prompt template,
    system-role support and context window, so callers only name the model::

        for_medical_model("aaditya/Llama3-OpenBioLLM-8B", "http://localhost:8000")
        for_medical_model("BioMistral/BioMistral-7B", "http://localhost:8001", provider="openai_chat")
    """
    profile = profile or resolve_profile(model)
    if provider in ("openai_chat", "chat", "openai"):
        return OpenAIChatAdapter(base_url, model, api_key, system_role=profile.system_role, profile=profile)
    if provider == "completions":
        return CompletionsAdapter(base_url, model, api_key, template=profile.template, profile=profile)
    raise ValueError(f"Unknown provider {provider!r}; use 'completions' or 'openai_chat'.")


def adapter_from_config() -> LLMAdapter:
    """Build the adapter described by ``LLM_*`` settings in ``.env``."""
    from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER

    return for_medical_model(
        LLM_MODEL, LLM_BASE_URL, provider=LLM_PROVIDER, api_key=LLM_API_KEY, profile=profile_from_config()
    )


_cached: Optional[LLMAdapter] = None


def get_default_adapter() -> LLMAdapter:
    """Lazily build and cache the configured adapter."""
    global _cached
    if _cached is None:
        _cached = adapter_from_config()
    return _cached
