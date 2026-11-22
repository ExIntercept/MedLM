"""Admin overview statistics and model catalogue."""
from datetime import datetime, timedelta, timezone

from src.llm.catalog import MEDICAL_MODELS, find
from src.security.pipeline_stats import per_user, stopped_at, summarize

NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)


def _rec(user, status, steps, hours_ago=1, flagged=False, model=None, ms=100.0):
    return {
        "timestamp": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "username": user, "status": status, "flagged": flagged, "model": model,
        "steps": steps, "total_ms": ms,
    }


BLOCKED = _rec("eve", "BLOCKED", [
    {"stage": "sanitize", "outcome": "modified", "ms": 1},
    {"stage": "triage", "outcome": "clear", "ms": 1},
    {"stage": "guardrails", "outcome": "clear", "ms": 1},
    {"stage": "detect", "outcome": "block", "ms": 500,
     "detail": {"category": "JAILBREAK", "reasons": ["pg_score_block", "tm_sim_flag"]}},
], ms=503)
ANSWERED = _rec("ann", "PASS", [
    {"stage": "sanitize", "outcome": "clean", "ms": 1},
    {"stage": "detect", "outcome": "flag", "ms": 400, "detail": {"reasons": ["tm_sim_flag"]}},
    {"stage": "generate", "outcome": "answered", "ms": 3000},
    {"stage": "verify", "outcome": "pass", "ms": 10},
    {"stage": "output_guard", "outcome": "safe", "ms": 200},
], flagged=True, model="openai_chat:medgemma", ms=3611)
OUTPUT_BLOCKED = _rec("eve", "OUTPUT_BLOCKED", [
    {"stage": "detect", "outcome": "allow", "ms": 400},
    {"stage": "generate", "outcome": "answered", "ms": 3000},
    {"stage": "output_guard", "outcome": "blocked", "ms": 200},
], model="openai_chat:medgemma")
OLD = _rec("old", "PASS", [], hours_ago=24 * 30)


def test_stopped_at():
    assert stopped_at(BLOCKED) == "detect"
    assert stopped_at(OUTPUT_BLOCKED) == "output_guard"
    assert stopped_at(ANSWERED) is None


def test_summary_counts_window_and_reasons():
    s = summarize([BLOCKED, ANSWERED, OUTPUT_BLOCKED, OLD], days=7, now=NOW)
    assert s["total"] == 3  # OLD is outside the window
    assert (s["answered"], s["stopped"], s["flagged"], s["sanitized"]) == (1, 2, 1, 1)
    assert s["stopped_by_stage"] == {"detect": 1, "output_guard": 1}
    assert s["top_reasons"]["tm_sim_flag"] == 2 and s["top_reasons"]["pg_score_block"] == 1
    assert s["categories"] == {"JAILBREAK": 1}
    assert s["models"] == {"openai_chat:medgemma": 2}
    assert s["active_users"] == 2
    assert len(s["per_day"]) == 7 and s["per_day"][-1]["day"] == "2026-11-22"


def test_per_user():
    u = per_user([BLOCKED, ANSWERED, OUTPUT_BLOCKED])
    assert u["eve"]["requests"] == 2 and u["eve"]["stopped"] == 2 and u["eve"]["sanitized"] == 1
    assert u["ann"]["flagged"] == 1


def test_catalogue_matches_benchmark_models():
    import json
    from pathlib import Path

    bench = json.loads(Path("scripts/cross_model_local.json").read_text())
    assert {m["model"] for m in MEDICAL_MODELS} == {b["model"] for b in bench}
    for b in bench:
        assert find(b["model"])["provider"] == b["provider"]
    assert find("not-a-model") is None
