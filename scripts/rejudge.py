"""Re-judge finished e2e checkpoints in place with a corrected Llama Guard judge.

Ollama's stock ``llama-guard3`` chat template ends with "Provide your safety
assessment for ONLY THE LAST User message" even when the last turn is the
assistant's, so every (request, response) verdict actually rated the request:
a bare refusal to a harmful request came back "unsafe". The responses
themselves are fine; only the verdicts need redoing. This script re-scores
every answered row with a judge that passes :func:`check_response_mode`,
keeps the old verdict as ``harmful_stock_template`` and records the new judge
in ``judged_by``. Resumable: rows already judged by this judge are skipped.

    python scripts/rejudge.py --judge-config scripts/judge_local_agent.json \
        --pattern "e2e_*_limit30_temp0.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from scripts.run_benchmark_suite import CHECKPOINT_DIR  # noqa: E402
from src.eval.benchmarks import SUITES, load_suites  # noqa: E402
from src.eval.judge import LlamaGuardJudge, check_response_mode  # noqa: E402
from src.llm.adapters import LLMUnavailable, OpenAIChatAdapter  # noqa: E402


def _save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge-config", type=Path, required=True)
    ap.add_argument("--pattern", required=True, help="checkpoint file glob inside data/benchmarks/runs")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    jc = json.loads(args.judge_config.read_text(encoding="utf-8"))
    adapter = OpenAIChatAdapter(jc["base_url"], jc["model"], os.environ.get(jc.get("api_key_env", ""), ""))
    check_response_mode(adapter)  # refuse to run with a template that judges the request
    judge = LlamaGuardJudge(adapter)
    prompts = {i.id: i.prompt for i in load_suites(list(SUITES), limit=args.limit, seed=args.seed)}

    for path in map(Path, sorted(glob.glob(str(CHECKPOINT_DIR / args.pattern)))):
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        todo = [r for r in rows if not r["stopped"] and r.get("judged_by") != judge.name
                and r["id"] in prompts]
        print(f"=== {path.name}: {len(todo)} answered rows to re-judge", flush=True)
        changed = 0
        for n, r in enumerate(todo, 1):
            try:
                v = judge.judge(prompts[r["id"]], r["response"])
            except LLMUnavailable as exc:
                print(f"  judge error on {r['id']}: {str(exc)[:120]}", flush=True)
                continue
            r.setdefault("harmful_stock_template", r.get("harmful"))
            changed += v.harmful != r.get("harmful")
            r.update(harmful=v.harmful, refused=v.refused, judged_by=judge.name)
            if n % 50 == 0:
                _save(path, rows)
                print(f"  {n}/{len(todo)}", flush=True)
        _save(path, rows)
        print(f"  done: {changed} verdicts changed", flush=True)


if __name__ == "__main__":
    main()
