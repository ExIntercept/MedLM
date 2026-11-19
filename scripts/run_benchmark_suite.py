"""Evaluate the clinical defense gateway on public adversarial and benign benchmarks.

Stage ``screen`` (no GPU, minutes)
    Runs every prompt through the gateway's pre-model stages only
    (sanitize -> triage -> guardrails -> detect). Measures how many attacks are
    stopped or flagged before reaching any model, and how many benign clinical
    questions are wrongly blocked. ``--ablate`` repeats it with each detector
    layer switched off.

Stage ``e2e`` (needs model endpoints)
    For each medical model in ``--config``: sends a sample of every suite to the
    raw model (undefended) and through the full gateway (defended), then judges
    the responses. Reports attack success rate, benign refusal rate and MedQA
    accuracy per model, undefended vs defended.

Examples::

    python scripts/fetch_benchmarks.py
    python scripts/run_benchmark_suite.py screen --limit 200 --ablate
    python scripts/run_benchmark_suite.py e2e --config scripts/cross_model_config.example.json \
        --limit 25 --judge-config judge.json

``judge.json``: {"model": "meta-llama/Llama-Guard-3-8B", "base_url": "http://...", "provider": "openai_chat"}.
Without it a refusal-phrase heuristic is used and harmfulness is not measured.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.eval.benchmarks import SUITES, BenchItem, load_suites  # noqa: E402
from src.eval.judge import HeuristicJudge, LlamaGuardJudge, extract_choice  # noqa: E402
from src.llm.adapters import CallableAdapter, LLMUnavailable, OpenAIChatAdapter, for_medical_model  # noqa: E402
from src.security.gateway import DEFAULT_TOGGLES, ClinicalDefenseGateway  # noqa: E402

STOPPED = {"BLOCKED", "EMERGENCY_ESCALATION", "INTERCEPTED"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


def _mode(item: BenchItem) -> str:
    return "clinician" if item.suite == "medqa" else "patient"


# ---------------------------------------------------------------------------
# Stage 1: pre-model screening
# ---------------------------------------------------------------------------

ABLATIONS = {
    "full": {},
    "without_harm_classifier": {"layer_harm_classifier": False},
    "prompt_guard_only": {"layer_template_matcher": False, "layer_harm_classifier": False},
    "template_matcher_only": {"layer_prompt_guard": False, "layer_harm_classifier": False},
    "harm_classifier_only": {"layer_prompt_guard": False, "layer_template_matcher": False},
}


CHECKPOINT_DIR = ROOT / "data" / "benchmarks" / "runs"


def _load_checkpoint(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue  # a partially written last line after a crash
    return done


def run_screen(items: list[BenchItem], ablate: bool, seed: int = 13, limit=None) -> dict:
    """Screen every item; rows are appended to a checkpoint so an interrupted run resumes."""
    never_called = CallableAdapter(lambda *a, **k: "", name="unused")
    configs = ABLATIONS if ablate else {"full": {}}
    report = {"stage": "screen", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {}}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_name, overrides in configs.items():
        toggles = {**DEFAULT_TOGGLES, **overrides}
        gateway = ClinicalDefenseGateway(never_called, retriever=None, toggles=toggles, log_decisions=False)
        ckpt = CHECKPOINT_DIR / f"screen_{cfg_name}_seed{seed}_limit{limit}.jsonl"
        done = _load_checkpoint(ckpt)
        if done:
            print(f"  [{cfg_name}] resuming: {len(done)} prompts already screened ({ckpt.name})", flush=True)
        rows = []
        t0 = time.time()
        fh = open(ckpt, "a", encoding="utf-8")
        for n, item in enumerate(items, 1):
            if item.id in done:
                rows.append(done[item.id])
                continue
            start = time.perf_counter()
            r = gateway.screen(item.prompt, mode=_mode(item))
            ms = (time.perf_counter() - start) * 1000
            d = r.decision
            row = {
                "id": item.id, "suite": item.suite, "kind": item.kind,
                "status": r.status, "flagged": r.flagged,
                "action": d.action if d else None, "category": d.category if d else None,
                "pg": d.pg_score if d else None, "tm": round(d.tm_max_sim, 4) if d else None,
                "harm": d.harm_score if d else None,
                "reasons": d.reasons if d else [], "ms": round(ms, 1),
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 100 == 0:
                print(f"  [{cfg_name}] {n}/{len(items)}  ({time.time() - t0:.0f}s)", flush=True)
        fh.close()
        report["configs"][cfg_name] = {"summary": summarise_screen(rows), "rows": rows}
    return report


def summarise_screen(rows: list[dict]) -> dict:
    by_suite = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)
    out = {}
    for suite, rs in by_suite.items():
        n = len(rs)
        blocked = sum(r["status"] == "BLOCKED" for r in rs)
        stopped = sum(r["status"] in STOPPED for r in rs)
        flagged = sum(r["flagged"] for r in rs)
        escalated = sum(r["status"] == "EMERGENCY_ESCALATION" for r in rs)
        intercepted = sum(r["status"] == "INTERCEPTED" for r in rs)
        lat = [r["ms"] for r in rs]
        out[suite] = {
            "n": n, "kind": SUITES[suite]["kind"], "seen_by_layer_a": SUITES[suite].get("seen", False),
            "blocked": blocked, "flagged": flagged, "stopped": stopped,
            "caught": stopped + flagged, "escalated": escalated, "intercepted": intercepted,
            "p50_ms": round(statistics.median(lat), 1),
            "p95_ms": round(sorted(lat)[int(0.95 * (n - 1))], 1),
        }
    return out


def screen_markdown(report: dict, limit) -> str:
    lines = [
        "# Benchmark Suite - Pre-Model Screening",
        "",
        f"Generated {report['generated_at']}. Each prompt goes through the gateway's pre-model stages only "
        "(sanitize, emergency triage, contraindication guardrails, Prompt Guard + Template Matcher). "
        f"No model is called. Up to {limit or 'all'} prompts per suite, fixed seed. "
        "Rates show the 95% Wilson interval in brackets.",
        "",
        "- **Stopped** = BLOCKED, emergency-escalated or contraindication-intercepted before the model.",
        "- **Caught** = stopped, or FLAGGED (forwarded with the mitigation rule).",
        "- Suites marked * overlap Prompt Guard's published training data; treat them as in-distribution, not held-out.",
        "",
    ]
    for cfg, body in report["configs"].items():
        s = body["summary"]
        lines += [f"## Configuration: `{cfg}`", "", "### Attacks (higher is better)", "",
                  "| Suite | n | Stopped | Caught (stopped + flagged) | p50 ms |", "|---|---|---|---|---|"]
        for suite, m in s.items():
            if m["kind"] == "attack":
                star = "*" if m["seen_by_layer_a"] else ""
                lines.append(f"| {suite}{star} | {m['n']} | {pct(m['stopped'], m['n'])} "
                             f"| {pct(m['caught'], m['n'])} | {m['p50_ms']} |")
        lines += ["", "### Benign (lower is better)", "",
                  "| Suite | n | Wrongly stopped | Flagged | of which emergency-escalated | p50 ms |",
                  "|---|---|---|---|---|---|"]
        for suite, m in s.items():
            if m["kind"] == "benign":
                star = "*" if m["seen_by_layer_a"] else ""
                lines.append(f"| {suite}{star} | {m['n']} | {pct(m['stopped'], m['n'])} "
                             f"| {pct(m['flagged'], m['n'])} | {m['escalated']} | {m['p50_ms']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stage 2: end-to-end across medical models
# ---------------------------------------------------------------------------

def _undefended_prompt(item: BenchItem) -> list[dict]:
    text = item.prompt
    if item.suite == "medqa":
        text += "\n\nRespond with your reasoning and end with 'Answer: <letter>'."
    return [{"role": "user", "content": text}]


def run_e2e(items: list[BenchItem], models: list[dict], judge, use_rag: bool) -> dict:
    retriever = None
    if use_rag:
        from src.retrieval.retriever import retrieve_context

        retriever = lambda q: retrieve_context(q, top_k=3)  # noqa: E731

    report = {"stage": "e2e", "judge": judge.name, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "models": []}
    for spec in models:
        key = os.environ.get(spec.get("api_key_env", ""), "") if spec.get("api_key_env") else ""
        adapter = for_medical_model(spec["model"], spec["base_url"], spec.get("provider", "completions"), key)
        gateway = ClinicalDefenseGateway(adapter, retriever=retriever, log_decisions=False)
        print(f"\n=== {spec['model']} [{adapter.profile.family}] ===", flush=True)
        rows = []
        for n, item in enumerate(items, 1):
            for condition in ("undefended", "defended"):
                row = {"id": item.id, "suite": item.suite, "kind": item.kind, "condition": condition}
                try:
                    if condition == "undefended":
                        resp = "".join(adapter.stream(_undefended_prompt(item), max_tokens=384)).strip()
                        status = "RAW"
                    else:
                        res = gateway.run(item.prompt, mode=_mode(item))
                        resp, status = res.response_text, res.status + ("/WITHHELD" if res.withheld else "")
                except LLMUnavailable as exc:
                    row.update(status="ERROR", error=str(exc)[:200])
                    rows.append(row)
                    continue
                stopped = status.split("/")[0] in STOPPED or status.endswith("WITHHELD")
                verdict = judge.judge(item.prompt, resp) if not stopped else None
                row.update(
                    status=status, stopped=stopped,
                    refused=True if stopped else verdict.refused,
                    harmful=False if stopped else verdict.harmful,
                    response_excerpt=resp[:300],
                )
                if item.suite == "medqa":
                    row["correct"] = extract_choice(resp) == item.reference.get("answer")
                rows.append(row)
            if n % 10 == 0:
                print(f"  {n}/{len(items)}", flush=True)
        report["models"].append({"model": spec["model"], "family": adapter.profile.family,
                                 "summary": summarise_e2e(rows), "rows": rows})
    return report


def summarise_e2e(rows: list[dict]) -> dict:
    out: dict = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["condition"])].append(r)
    for (suite, cond), rs in groups.items():
        ok = [r for r in rs if r.get("status") != "ERROR"]
        n = len(ok)
        m = {"n": n, "errors": len(rs) - n,
             "refused": sum(r["refused"] for r in ok),
             "complied": sum(not r["refused"] for r in ok),
             "harmful": sum(r["harmful"] is True for r in ok),
             "judged": sum(r["harmful"] is not None for r in ok)}
        if suite == "medqa":
            m["correct"] = sum(r.get("correct", False) for r in ok)
        out[suite][cond] = m
    return dict(out)


def e2e_markdown(report: dict) -> str:
    harm = not report["judge"].startswith("heuristic")
    lines = [
        "# Benchmark Suite - End-to-End Across Medical Models", "",
        f"Generated {report['generated_at']}. Judge: `{report['judge']}`. "
        + ("" if harm else "The heuristic judge measures non-refusal only, not harmfulness. "
           "Use a Llama Guard judge for attack success rates suitable for reporting."),
        "",
    ]
    for m in report["models"]:
        lines += [f"## {m['model']} ({m['family']})", "",
                  "| Suite | Metric | Undefended | Defended |", "|---|---|---|---|"]
        for suite, conds in m["summary"].items():
            u, d = conds.get("undefended", {}), conds.get("defended", {})
            kind = SUITES[suite]["kind"]
            if suite == "medqa":
                lines.append(f"| {suite} | accuracy | {pct(u.get('correct', 0), u.get('n', 0))} "
                             f"| {pct(d.get('correct', 0), d.get('n', 0))} |")
            if kind == "attack":
                metric, key = ("attack success (unsafe)", "harmful") if harm else ("complied (non-refusal)", "complied")
            else:
                metric, key = "refused / blocked", "refused"
            lines.append(f"| {suite} | {metric} | {pct(u.get(key, 0), u.get('n', 0))} "
                         f"| {pct(d.get(key, 0), d.get('n', 0))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark the clinical defense gateway.")
    ap.add_argument("stage", choices=["screen", "e2e"])
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--limit", type=int, default=None, help="max prompts per suite (seeded sample)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ablate", action="store_true", help="screen: also run each detector layer alone")
    ap.add_argument("--config", type=Path, help="e2e: JSON list of medical models")
    ap.add_argument("--judge-config", type=Path, help="e2e: JSON for a Llama Guard endpoint")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()

    limit = args.limit if args.limit is not None else (200 if args.stage == "screen" else 25)
    items = load_suites(args.suites, limit=limit, seed=args.seed)
    print(f"Loaded {len(items)} prompts from {len(args.suites)} suites (limit {limit}/suite).", flush=True)
    args.out.mkdir(exist_ok=True)

    if args.stage == "screen":
        report = run_screen(items, args.ablate, seed=args.seed, limit=limit)
        md = screen_markdown(report, limit)
        stem = "benchmark_suite_screen"
    else:
        if not args.config:
            ap.error("e2e needs --config")
        models = json.loads(args.config.read_text(encoding="utf-8"))
        if args.judge_config:
            jc = json.loads(args.judge_config.read_text(encoding="utf-8"))
            judge = LlamaGuardJudge(OpenAIChatAdapter(jc["base_url"], jc["model"],
                                                      os.environ.get(jc.get("api_key_env", ""), "")))
        else:
            judge = HeuristicJudge()
        report = run_e2e(items, models, judge, use_rag=not args.no_rag)
        md = e2e_markdown(report)
        stem = "benchmark_suite_e2e"

    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.out / f"{stem.upper()}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out / (stem.upper() + '.md')}")


if __name__ == "__main__":
    main()
