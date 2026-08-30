"""Client for the MedGemma model server running on Google Colab behind ngrok."""
import json
from typing import AsyncIterator, Dict, Iterator, List, Optional
import httpx
import requests

from src.config import (
    COLAB_API_BASE,
    COLAB_API_KEY,
    MAX_MODEL_LEN,
    MODEL_ID,
    PROMPT_STYLE,
    RESERVE_OUTPUT_TOKENS,
)

HEADERS = {
    "Authorization": f"Bearer {COLAB_API_KEY}" if COLAB_API_KEY else "",
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}

# Stop tokens for MedGemma / Gemma
GEMMA_STOP = ["<end_of_turn>", "<start_of_turn>", "<|im_end|>", "<|im_start|>"]


class ColabUnreachable(RuntimeError):
    pass


def get_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if COLAB_API_KEY:
        headers["Authorization"] = f"Bearer {COLAB_API_KEY}"
    return headers


def input_budget() -> int:
    return max(1024, MAX_MODEL_LEN - RESERVE_OUTPUT_TOKENS)


def health_sync() -> Dict:
    if not COLAB_API_BASE:
        return {"ok": False, "detail": "COLAB_API_BASE is not set in .env"}
    try:
        resp = requests.get(f"{COLAB_API_BASE}/v1/models", headers=get_headers(), timeout=6.0)
        if resp.status_code == 401:
            return {"ok": False, "detail": "Server rejected the API key (401)."}
        resp.raise_for_status()
        served = [m.get("id") for m in resp.json().get("data", [])]
        return {
            "ok": True,
            "served_models": served,
            "configured_model": MODEL_ID,
            "match": MODEL_ID in served or len(served) > 0,
            "max_model_len": MAX_MODEL_LEN,
        }
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


async def health_async() -> Dict:
    if not COLAB_API_BASE:
        return {"ok": False, "detail": "COLAB_API_BASE is not set in .env"}
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(f"{COLAB_API_BASE}/v1/models", headers=get_headers())
        if resp.status_code == 401:
            return {"ok": False, "detail": "Server rejected the API key (401)."}
        resp.raise_for_status()
        served = [m.get("id") for m in resp.json().get("data", [])]
        return {
            "ok": True,
            "served_models": served,
            "configured_model": MODEL_ID,
            "match": MODEL_ID in served or len(served) > 0,
            "max_model_len": MAX_MODEL_LEN,
        }
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def count_tokens_estimate(text: str) -> int:
    """Approximate token count for budgeting (Gemma averages ~3.5 chars/token on clinical English)."""
    return max(1, int(len(text) / 3.4) + 4)


def stream_generate(prompt: str, temperature: float = 0.2, stop: Optional[List[str]] = None) -> Iterator[str]:
    """Synchronous generator yielding token deltas from Colab MedGemma endpoint."""
    if not COLAB_API_BASE:
        raise ColabUnreachable("COLAB_API_BASE is not set in .env. Please configure your Colab ngrok URL.")

    stop_seqs = stop or GEMMA_STOP

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": RESERVE_OUTPUT_TOKENS,
        "temperature": temperature,
        "top_p": 0.9,
        "stop": stop_seqs,
        "stream": True,
    }

    try:
        response = requests.post(
            f"{COLAB_API_BASE}/v1/completions",
            headers=get_headers(),
            json=payload,
            stream=True,
            timeout=(15.0, 180.0),
        )
        if response.status_code != 200:
            raise ColabUnreachable(f"Colab model server returned HTTP {response.status_code}: {response.text[:300]}")

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data:"):
                continue
            data = line_str[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    yield piece
    except requests.exceptions.RequestException as exc:
        raise ColabUnreachable(
            f"Could not reach Colab MedGemma ngrok tunnel ({type(exc).__name__}: {exc}). "
            "Please check that your Colab notebook is active and COLAB_API_BASE in .env is current."
        ) from exc


async def stream_completion_async(prompt: str, temperature: float = 0.2, stop: Optional[List[str]] = None) -> AsyncIterator[str]:
    """Async generator yielding token deltas from Colab MedGemma endpoint."""
    if not COLAB_API_BASE:
        raise ColabUnreachable("COLAB_API_BASE is not set in .env. Please configure your Colab ngrok URL.")

    stop_seqs = stop or GEMMA_STOP

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": RESERVE_OUTPUT_TOKENS,
        "temperature": temperature,
        "top_p": 0.9,
        "stop": stop_seqs,
        "stream": True,
    }

    _timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)
    try:
        async with httpx.AsyncClient(timeout=_timeout) as client:
            async with client.stream(
                "POST", f"{COLAB_API_BASE}/v1/completions", headers=get_headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="ignore")[:300]
                    raise ColabUnreachable(f"Colab model server returned HTTP {response.status_code}: {body}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        piece = choice.get("text", "")
                        if piece:
                            yield piece
    except httpx.HTTPError as exc:
        raise ColabUnreachable(
            f"Could not reach Colab MedGemma ngrok tunnel ({type(exc).__name__}: {exc}). "
            "Please check that your Colab notebook is active and COLAB_API_BASE in .env is current."
        ) from exc
