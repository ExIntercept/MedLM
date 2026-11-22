"""Usage and safety statistics over the per-request pipeline log, for the admin overview."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

STOPPING = {"block", "blocked", "intercepted", "emergency", "withheld"}
WARNING = {"flag", "modified", "case_discussion", "unsafe_allowed", "ungrounded"}
ANSWERED = {"PASS", "FLAGGED", "UNGROUNDED"}


def _ts(rec) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(rec["timestamp"])
    except (KeyError, ValueError, TypeError):
        return None


def stopped_at(rec) -> Optional[str]:
    """The stage that ended the request early or replaced its answer, if any."""
    for s in rec.get("steps") or []:
        if s.get("outcome") in STOPPING or (s.get("stage") == "output_guard" and s.get("outcome") == "blocked"):
            return s["stage"]
    return None


def summarize(records: Iterable[dict], days: int = 7, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    recs = [r for r in records if (_ts(r) or now) >= since]

    per_day = defaultdict(Counter)
    reasons = Counter()
    categories = Counter()
    stop_stage = Counter()
    models = Counter()
    users = set()
    latencies = []
    for r in recs:
        day = (_ts(r) or now).date().isoformat()
        c = per_day[day]
        c["requests"] += 1
        status = r.get("status")
        if status in ANSWERED:
            c["answered"] += 1
        else:
            c["stopped"] += 1
        if r.get("flagged"):
            c["flagged"] += 1
        steps = {s["stage"]: s for s in r.get("steps") or []}
        if steps.get("sanitize", {}).get("outcome") == "modified":
            c["sanitized"] += 1
        stage = stopped_at(r)
        if stage:
            stop_stage[stage] += 1
        det = steps.get("detect", {}).get("detail", {})
        if steps.get("detect", {}).get("outcome") in ("block", "flag"):
            reasons.update(det.get("reasons") or [])
            if det.get("category"):
                categories[det["category"]] += 1
        if r.get("model") and steps.get("generate"):
            models[r["model"]] += 1
        if r.get("username"):
            users.add(r["username"])
        if r.get("total_ms") is not None:
            latencies.append(float(r["total_ms"]))

    days_list = [(since + timedelta(days=i + 1)).date().isoformat() for i in range(days)]
    latencies.sort()
    total = len(recs)
    stopped = sum(per_day[d]["stopped"] for d in per_day)
    return {
        "days": days,
        "total": total,
        "answered": total - stopped,
        "stopped": stopped,
        "flagged": sum(per_day[d]["flagged"] for d in per_day),
        "sanitized": sum(per_day[d]["sanitized"] for d in per_day),
        "active_users": len(users),
        "latency_ms": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        },
        "per_day": [{"day": d, **{k: per_day[d][k] for k in ("requests", "answered", "stopped", "flagged", "sanitized")}}
                    for d in days_list],
        "stopped_by_stage": dict(stop_stage.most_common()),
        "top_reasons": dict(reasons.most_common(8)),
        "categories": dict(categories.most_common()),
        "models": dict(models.most_common()),
    }


def per_user(records: Iterable[dict]) -> dict:
    """{username: {requests, stopped, flagged, sanitized, last}} over the whole log."""
    out: dict = defaultdict(lambda: Counter())
    last: dict = {}
    for r in records:
        u = r.get("username")
        if not u:
            continue
        c = out[u]
        c["requests"] += 1
        if r.get("status") not in ANSWERED:
            c["stopped"] += 1
        if r.get("flagged"):
            c["flagged"] += 1
        if any(s.get("stage") == "sanitize" and s.get("outcome") == "modified" for s in r.get("steps") or []):
            c["sanitized"] += 1
        ts = r.get("timestamp")
        if ts and (u not in last or ts > last[u]):
            last[u] = ts
    return {u: {**dict(c), "last": last.get(u)} for u, c in out.items()}
