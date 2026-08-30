"""Client for the vLLM OpenAI-compatible server running in Colab behind ngrok."""
import json
from typing import AsyncIterator, Dict, List, Optional

import httpx

from .config import (
    COLAB_API_BASE,
    COLAB_API_KEY,
    MAX_MODEL_LEN,
    MODEL_ID,
    RESERVE_OUTPUT_TOKENS,
)

# ngrok's free tier serves an interstitial HTML page to browsers; this header
# skips it. Harmless on a paid tunnel or a direct URL.
HEADERS = {
    "Authorization": f"Bearer {COLAB_API_KEY}",
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}

_timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)


def input_budget() -> int:
    return MAX_MODEL_LEN - RESERVE_OUTPUT_TOKENS


class ColabUnreachable(RuntimeError):
    pass


async def health() -> Dict:
    if not COLAB_API_BASE:
        return {"ok": False, "detail": "COLAB_API_BASE is not set in .env"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COLAB_API_BASE}/v1/models", headers=HEADERS)
        if response.status_code == 401:
            return {"ok": False, "detail": "Server rejected the API key (401)."}
        response.raise_for_status()
        served = [m["id"] for m in response.json().get("data", [])]
        return {
            "ok": True,
            "served_models": served,
            "configured_model": MODEL_ID,
            "match": MODEL_ID in served,
            "max_model_len": MAX_MODEL_LEN,
        }
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


async def count_tokens_remote(text: str) -> Optional[int]:
    """vLLM exposes POST /tokenize. Returns None if unavailable."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{COLAB_API_BASE}/tokenize",
                headers=HEADERS,
                json={"model": MODEL_ID, "prompt": text},
            )
        if response.status_code != 200:
            return None
        return response.json().get("count")
    except httpx.HTTPError:
        return None


def count_tokens_estimate(text: str) -> int:
    """Fallback when /tokenize is unreachable. Llama tokenizers land near 3.6
    characters per token on clinical English; rounded down to over-count slightly."""
    return max(1, int(len(text) / 3.4) + 4)


async def stream_completion(prompt: str, stop: List[str], temperature: float = 0.3) -> AsyncIterator[str]:
    if not COLAB_API_BASE:
        raise ColabUnreachable("COLAB_API_BASE is not set in .env")

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": RESERVE_OUTPUT_TOKENS,
        "temperature": temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "stop": stop,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=_timeout) as client:
            async with client.stream(
                "POST", f"{COLAB_API_BASE}/v1/completions", headers=HEADERS, json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode()[:400]
                    raise ColabUnreachable(f"Model server returned {response.status_code}: {body}")
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
            f"Could not reach the Colab tunnel ({type(exc).__name__}). "
            "Check the notebook is still running and the ngrok URL in .env is current."
        ) from exc
