"""Download the public benchmark datasets used by run_benchmark_suite.py.

Only plain data files (CSV / JSONL / Parquet) are fetched - no dataset
loading scripts or other code - into ``data/benchmarks/raw/`` (gitignored).
Re-running skips files that already exist.

    python scripts/fetch_benchmarks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "benchmarks" / "raw"

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
GH = "https://raw.githubusercontent.com/{repo}/main/{path}"

MANIFEST: list[tuple[str, str, str]] = [
    # (local subdir/filename, url, description)
    ("jbb/harmful-behaviors.csv",
     HF.format(repo="JailbreakBench/JBB-Behaviors", path="data/harmful-behaviors.csv"),
     "JailbreakBench harmful behaviors (100)"),
    ("jbb/benign-behaviors.csv",
     HF.format(repo="JailbreakBench/JBB-Behaviors", path="data/benign-behaviors.csv"),
     "JailbreakBench matched benign behaviors (100)"),
    ("deepset/test.parquet",
     HF.format(repo="deepset/prompt-injections", path="data/test-00000-of-00001-701d16158af87368.parquet"),
     "deepset prompt-injections, test split"),
    ("in_the_wild/jailbreak_2023_12_25.parquet",
     HF.format(repo="TrustAIRLab/in-the-wild-jailbreak-prompts", path="jailbreak_2023_12_25/train-00000-of-00001.parquet"),
     "In-the-wild jailbreak prompts (Shen et al., CCS 2024)"),
    ("medqa/test.jsonl",
     HF.format(repo="GBaker/MedQA-USMLE-4-options", path="phrases_no_exclude_test.jsonl"),
     "MedQA USMLE 4-option, test split"),
    ("medquad/medDataset_processed.csv",
     HF.format(repo="keivalya/MedQuad-MedicalQnADataset", path="medDataset_processed.csv"),
     "MedQuAD consumer health questions"),
    ("xstest/xstest_prompts.csv",
     GH.format(repo="paul-rottger/xstest", path="xstest_prompts.csv"),
     "XSTest exaggerated-safety prompts"),
    # ── Training-only files for the harmful-intent classifier (never evaluated on) ──
    ("train/medsafetybench_train_med_n900.json",
     GH.format(repo="AI4LIFE-GROUP/med-safety-bench",
               path="datasets/train-splits-used-for-ft/med_safety/ft_safety_med_n900.json"),
     "MedSafetyBench training split (harmful medical requests)"),
    ("train/medqa_train.jsonl",
     HF.format(repo="GBaker/MedQA-USMLE-4-options", path="phrases_no_exclude_train.jsonl"),
     "MedQA USMLE training split (benign)"),
    ("train/in_the_wild_regular_2023_12_25.parquet",
     HF.format(repo="TrustAIRLab/in-the-wild-jailbreak-prompts", path="regular_2023_12_25/train-00000-of-00001.parquet"),
     "In-the-wild regular (non-jailbreak) user prompts"),
    ("train/or-bench-80k.csv",
     HF.format(repo="bench-llm/or-bench", path="or-bench-80k.csv"),
     "OR-Bench seemingly-toxic but safe prompts (hard negatives; hard-1k excluded)"),
    ("train/or-bench-toxic.csv",
     HF.format(repo="bench-llm/or-bench", path="or-bench-toxic.csv"),
     "OR-Bench genuinely toxic prompts (general harmful positives)"),
    # Evaluation: OR-Bench hard-1k over-refusal probes (excluded from training).
    ("orbench/or-bench-hard-1k.csv",
     HF.format(repo="bench-llm/or-bench", path="or-bench-hard-1k.csv"),
     "OR-Bench hard-1k over-refusal prompts"),
] + [
    (f"medsafetybench/{gen}/category_{c}.csv",
     GH.format(repo="AI4LIFE-GROUP/med-safety-bench",
               path=f"datasets/test/{gen}/med_safety_demonstrations_category_{c}.csv"),
     f"MedSafetyBench test, {gen}, category {c}")
    for gen in ("gpt4", "llama2") for c in range(1, 10)
]

ALLOWED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}


def main() -> int:
    failures = 0
    for rel, url, desc in MANIFEST:
        dest = RAW / rel
        assert dest.suffix in ALLOWED_SUFFIXES, dest
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  have  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(url, timeout=120, headers={"User-Agent": "medlm-benchmark-fetch"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  FAIL  {rel}: {exc}")
            failures += 1
            continue
        dest.write_bytes(resp.content)
        print(f"  got   {rel}  ({len(resp.content) / 1024:.0f} KB)  {desc}")
    print(f"\n{len(MANIFEST) - failures}/{len(MANIFEST)} files in {RAW}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
