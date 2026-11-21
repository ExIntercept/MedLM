"""Build the paper's result tables from e2e checkpoints.

Reads the main run (all models, undefended and defended), the ablation runs
(defended only, tagged) and the second judge's verdicts, and writes
docs/paper/RESULTS.md and results.json:

  1. Main results per model: attack success (both judges), medical and general
     refusal, MedQA accuracy, with 95% Wilson intervals and an exact McNemar
     test pairing each attack prompt undefended vs defended.
  2. Attack success per attack suite, undefended vs defended, pooled over models.
  3. Which pipeline stage stopped each defended attack.
  4. Agreement between the two judges (Cohen's kappa).
  5. Ablations: the gateway with layers removed, and a Llama-Guard-only baseline.

    python scripts/paper_tables.py --limit 30 --ablation-model medgemma --subset 15
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmark_suite import CHECKPOINT_DIR, MEDICAL_BENIGN, wilson  # noqa: E402

FAMILY = {
    "medgemma": "MedGemma 4B", "openbiollm": "OpenBioLLM 8B",
    "biomistral": "BioMistral 7B", "jsl-medllama": "JSL-MedLlama 3 8B",
}
ATTACK_SUITES = ["medsafetybench", "medical_jailbreak", "in_the_wild", "jbb_harmful",
                 "deepset_injection", "xstest_unsafe"]


def model_name(ckpt: str) -> str:
    return next((v for k, v in FAMILY.items() if k in ckpt), ckpt)


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_judge2() -> dict:
    out = {}
    for path in glob.glob(str(CHECKPOINT_DIR / "judge2_*.jsonl")):
        for r in load_rows(path):
            out[(r["ckpt"], r["id"], r["condition"])] = r
    return out


def harmful2(row, ckpt, j2):
    """Second-judge harm verdict; stopped rows are never harmful, unjudged rows are None."""
    if row["stopped"]:
        return False
    v = j2.get((ckpt, row["id"], row["condition"]))
    return None if v is None else v["harmful"]


def refused2(row, ckpt, j2):
    if row["stopped"]:
        return True
    v = j2.get((ckpt, row["id"], row["condition"]))
    return None if v is None else v["refused"]


def rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "pct": None, "lo": None, "hi": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": 100 * k / n, "lo": 100 * lo, "hi": 100 * hi}


def fmt(r: dict) -> str:
    return "n/a" if r["n"] == 0 else f"{r['pct']:.1f} [{r['lo']:.0f}-{r['hi']:.0f}]"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def stage_of(row: dict) -> str:
    s = row["status"]
    if s.endswith("OUTPUT_BLOCKED"):
        return "output guard"
    if s.endswith("WITHHELD"):
        return "grounding check"
    if s.startswith("BLOCKED"):
        return "input detectors"
    if s.startswith("EMERGENCY"):
        return "emergency triage"
    if s.startswith("INTERCEPTED"):
        return "guardrails"
    return "model refused" if row["refused"] else "answered"


def summarise(rows, ckpt, j2, cond):
    rs = [r for r in rows if r["condition"] == cond]
    atk = [r for r in rs if r["kind"] == "attack"]
    med = [r for r in rs if r["suite"] in MEDICAL_BENIGN]
    gen = [r for r in rs if r["kind"] == "benign" and r["suite"] not in MEDICAL_BENIGN]
    mq = [r for r in rs if r["suite"] == "medqa"]
    h2 = [harmful2(r, ckpt, j2) for r in atk]
    return {
        "asr_j1": rate(sum(r["harmful"] is True for r in atk), len(atk)),
        # Only reported once the second judge has seen every answered attack row.
        "asr_j2": rate(sum(h is True for h in h2), len(h2) if None not in h2 else 0),
        "med_refused": rate(sum(r["refused"] for r in med), len(med)),
        "gen_refused": rate(sum(r["refused"] for r in gen), len(gen)),
        "medqa_acc": rate(sum(r.get("correct", False) for r in mq), len(mq)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "paper")
    args = ap.parse_args()
    j2 = load_judge2()
    base = f"_rag*_seed{args.seed}_limit{args.limit}_temp0"
    main_files = sorted(p for p in glob.glob(str(CHECKPOINT_DIR / f"e2e_*{base}.jsonl")))
    lines, data = ["# Results", ""], {"main": {}, "suites": {}, "stages": {}, "kappa": {}, "ablations": {}}

    # 1. Main results -------------------------------------------------------
    lines += ["## Table 1. Main results (percent, 95% Wilson interval)", "",
              "| Model | Attack success, judge 1 (undef / def) | McNemar p | Attack success, judge 2 (undef / def) "
              "| Medical refused (undef / def) | MedQA accuracy (undef / def) | General refused (undef / def) |",
              "|---|---|---|---|---|---|---|"]
    pooled = defaultdict(list)
    for path in main_files:
        ckpt, rows = Path(path).name, load_rows(path)
        name = model_name(ckpt)
        u, d = summarise(rows, ckpt, j2, "undefended"), summarise(rows, ckpt, j2, "defended")
        by_id = defaultdict(dict)
        for r in rows:
            if r["kind"] == "attack":
                by_id[r["id"]][r["condition"]] = r["harmful"] is True
        b = sum(v.get("undefended") and not v.get("defended") for v in by_id.values() if len(v) == 2)
        c = sum(v.get("defended") and not v.get("undefended") for v in by_id.values() if len(v) == 2)
        p = mcnemar_exact(b, c)
        data["main"][name] = {"undefended": u, "defended": d, "mcnemar_p": p, "discordant": [b, c],
                              "n_rows": len(rows)}
        lines.append(f"| {name} | {fmt(u['asr_j1'])} / {fmt(d['asr_j1'])} | {p:.1e} | "
                     f"{fmt(u['asr_j2'])} / {fmt(d['asr_j2'])} | {fmt(u['med_refused'])} / {fmt(d['med_refused'])} | "
                     f"{fmt(u['medqa_acc'])} / {fmt(d['medqa_acc'])} | {fmt(u['gen_refused'])} / {fmt(d['gen_refused'])} |")
        for r in rows:
            pooled[(r["suite"], r["condition"])].append((r, ckpt))
    lines += ["", "Judge 1: Llama Guard 3 8B. Judge 2: qwen3 8B with a harm/refusal rubric "
              "(a different model family from the guard used inside the gateway). McNemar: exact "
              "two-sided test pairing each attack prompt undefended vs defended (judge 1).", ""]

    # 2. Per attack suite, pooled over models --------------------------------
    lines += ["## Table 2. Attack success by attack suite, pooled over models (judge 1 / judge 2)", "",
              "| Suite | Undefended | Defended |", "|---|---|---|"]
    for suite in ATTACK_SUITES:
        cells = []
        for cond in ("undefended", "defended"):
            rs = pooled.get((suite, cond), [])
            r1 = rate(sum(r["harmful"] is True for r, _ in rs), len(rs))
            h2 = [harmful2(r, ck, j2) for r, ck in rs]
            r2 = rate(sum(h is True for h in h2), len(h2) if None not in h2 else 0)
            cells.append(f"{fmt(r1)} / {fmt(r2)}")
            data["suites"].setdefault(suite, {})[cond] = {"j1": r1, "j2": r2}
        lines.append(f"| {suite} | {cells[0]} | {cells[1]} |")
    lines.append("")

    # 3. Which stage stopped defended attacks --------------------------------
    stages = ["emergency triage", "guardrails", "input detectors", "grounding check", "output guard",
              "model refused", "answered"]
    lines += ["## Table 3. Outcome of defended attack prompts by pipeline stage (counts)", "",
              "| Model | " + " | ".join(stages) + " | Answered and harmful (judge 1) |",
              "|---|" + "---|" * (len(stages) + 1)]
    for path in main_files:
        ckpt, rows = Path(path).name, load_rows(path)
        atk = [r for r in rows if r["condition"] == "defended" and r["kind"] == "attack"]
        cnt = defaultdict(int)
        for r in atk:
            cnt[stage_of(r)] += 1
        data["stages"][model_name(ckpt)] = dict(cnt)
        lines.append(f"| {model_name(ckpt)} | " + " | ".join(str(cnt.get(s, 0)) for s in stages)
                     + f" | {sum(r['harmful'] is True for r in atk)} |")
    lines.append("")

    # 4. Judge agreement -----------------------------------------------------
    lines += ["## Table 4. Agreement between judges on answered attack responses", "",
              "| Model | Responses compared | Both harmful | Judge 1 only | Judge 2 only | Cohen's kappa |",
              "|---|---|---|---|---|---|"]
    for path in main_files:
        ckpt, rows = Path(path).name, load_rows(path)
        pairs = [(r["harmful"], harmful2(r, ckpt, j2)) for r in rows
                 if r["kind"] == "attack" and not r["stopped"]]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        k = cohen_kappa(pairs)
        data["kappa"][model_name(ckpt)] = k
        lines.append(f"| {model_name(ckpt)} | {len(pairs)} | {sum(a and b for a, b in pairs)} | "
                     f"{sum(a and not b for a, b in pairs)} | {sum(b and not a for a, b in pairs)} | "
                     + ("n/a" if k is None else f"{k:.2f}") + " |")
    lines.append("")

    # 5. Ablations -------------------------------------------------------------
    abl_files = sorted(glob.glob(str(CHECKPOINT_DIR / f"e2e_*{base}_*.jsonl")))
    if abl_files:
        lines += ["## Table 5. Ablations (defended condition only)", "",
                  "| Model | Configuration | Prompts | Attack success, judge 1 | Attack success, judge 2 "
                  "| Medical refused | MedQA accuracy | General refused |",
                  "|---|---|---|---|---|---|---|---|"]
        for path in abl_files:
            ckpt, rows = Path(path).name, load_rows(path)
            tag = ckpt.rsplit("_temp0_", 1)[1].removesuffix(".jsonl")
            ids = {r["id"] for r in rows}
            full_path = next((p for p in main_files if model_name(Path(p).name) == model_name(ckpt)), None)
            configs = [(tag, ckpt, rows)]
            if full_path:  # the full gateway on the same prompts, for comparison
                full = [r for r in load_rows(full_path) if r["id"] in ids and r["condition"] == "defended"]
                configs.insert(0, ("full gateway", Path(full_path).name, full))
            for name, ck, rs in configs:
                s = summarise(rs, ck, j2, "defended")
                data["ablations"].setdefault(model_name(ckpt), {})[name] = s
                lines.append(f"| {model_name(ckpt)} | {name} | {len(rs)} | {fmt(s['asr_j1'])} | {fmt(s['asr_j2'])} | "
                             f"{fmt(s['med_refused'])} | {fmt(s['medqa_acc'])} | {fmt(s['gen_refused'])} |")
        lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
