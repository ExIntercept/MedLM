"""Score finished e2e checkpoints with a second, independent judge.

The main runs are judged by Llama Guard 3 8B, which is also one of the gateway's
defense layers (input and output), so the defended attack success rate could be
flattered. This script re-judges every answered row with a rubric judge from a
different model family and saves its verdicts in a separate file, so the main
checkpoints are never modified. Resumable.

    python scripts/second_judge.py --judge-config scripts/judge2_local.json \
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
from src.eval.judge import RubricJudge  # noqa: E402
from src.llm.adapters import LLMUnavailable, OpenAIChatAdapter  # noqa: E402


def verdict_file(judge_name: str) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in judge_name.lower()).strip("-")
    return CHECKPOINT_DIR / f"judge2_{slug}.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge-config", type=Path, required=True)
    ap.add_argument("--pattern", required=True, help="checkpoint file glob inside data/benchmarks/runs")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--kind", choices=["attack", "benign"],
                    help="only judge rows of this kind (the paper's judge-2 tables use attack rows only)")
    args = ap.parse_args()

    jc = json.loads(args.judge_config.read_text(encoding="utf-8"))
    judge = RubricJudge(OpenAIChatAdapter(jc["base_url"], jc["model"], jc.get("api_key", ""),
                                               extra=jc.get("extra")))
    prompts = {i.id: i.prompt for i in load_suites(list(SUITES), limit=args.limit, seed=args.seed)}
    out_path = verdict_file(judge.name)
    done = set()
    if out_path.exists():
        done = {(r["ckpt"], r["id"], r["condition"])
                for r in map(json.loads, out_path.read_text(encoding="utf-8").splitlines())}

    with open(out_path, "a", encoding="utf-8") as out:
        for path in sorted(glob.glob(str(CHECKPOINT_DIR / args.pattern))):
            ckpt = Path(path).name
            rows = [json.loads(line) for line in open(path, encoding="utf-8")]
            todo = [r for r in rows if not r["stopped"] and (ckpt, r["id"], r["condition"]) not in done
                    and r["id"] in prompts and (args.kind is None or r["kind"] == args.kind)]
            # Defended rows first: they are few and test the headline claim.
            todo.sort(key=lambda r: r["condition"] != "defended")
            print(f"=== {ckpt}: {len(todo)} answered rows to judge", flush=True)
            for n, r in enumerate(todo, 1):
                try:
                    v = judge.judge(prompts[r["id"]], r["response"])
                except LLMUnavailable as exc:
                    print(f"  judge error on {r['id']}: {str(exc)[:120]}", flush=True)
                    continue
                out.write(json.dumps({"ckpt": ckpt, "id": r["id"], "condition": r["condition"],
                                      "suite": r["suite"], "kind": r["kind"], "harmful": v.harmful,
                                      "refused": v.refused, "raw": v.raw[:200],
                                      "judge": judge.name}) + "\n")
                out.flush()
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
    print(f"verdicts in {out_path}")


if __name__ == "__main__":
    main()
