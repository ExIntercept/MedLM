"""Train Layer H, the harmful medical intent classifier.

Training data (none of it is used for evaluation):
  harmful : MedSafetyBench *training* split (900 requests) and OR-Bench toxic
            prompts (655, general harm categories).
  benign  : MedQuAD questions from the training partition (see
            src/eval/benchmarks.partition), MedQA *training* vignettes,
            in-the-wild *regular* user prompts (imperative "write ..." requests
            are not harmful by themselves), and OR-Bench-80k seemingly-toxic but
            safe prompts as hard negatives (OR-Bench hard-1k excluded: it is an
            evaluation suite).

Every training prompt with cosine similarity > 0.90 to any evaluation prompt is
dropped, so no benchmark item (or near-copy of one) leaks into training.

Thresholds are calibrated on out-of-fold scores (5-fold cross-validation, so
every score comes from a model that did not see that prompt) against the
*worst* benign source, not the pooled benign set. A pooled 1% budget is
dominated by easy negatives (MedQA vignettes score near zero) and hides a much
higher rate on harmless prompts that merely sound alarming. ``block`` keeps
every benign source at or below 1% false positives, ``flag`` at or below 3%.

    python scripts/fetch_benchmarks.py
    python scripts/train_harm_classifier.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from src.eval.benchmarks import RAW, _medsafety_requests, medquad_questions, partition  # noqa: E402
from src.security.harm_classifier import WEIGHTS_PATH  # noqa: E402


def load_training_data(rng: random.Random, n_medquad: int, n_medqa: int, n_regular: int, n_orbench: int):
    import pandas as pd

    harmful = [x["instruction"].strip() for x in
               json.loads((RAW / "train" / "medsafetybench_train_med_n900.json").read_text(encoding="utf-8"))]
    toxic = pd.read_csv(RAW / "train" / "or-bench-toxic.csv")["prompt"].dropna().astype(str).tolist()

    hard_eval = set(pd.read_csv(RAW / "orbench" / "or-bench-hard-1k.csv")["prompt"].dropna().astype(str))
    orbench = [p for p in pd.read_csv(RAW / "train" / "or-bench-80k.csv")["prompt"].dropna().astype(str)
               if p not in hard_eval]

    medquad = [q for _, q, _, _ in medquad_questions() if partition(q) == "train"]
    medqa = []
    with open(RAW / "train" / "medqa_train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            medqa.append(row["question"])
    regular = pd.read_parquet(RAW / "train" / "in_the_wild_regular_2023_12_25.parquet")["prompt"].dropna()
    regular = [p for p in regular.astype(str) if 15 < len(p) < 1500]

    benign, sources = [], []
    for name, pool, n in (("medquad", medquad, n_medquad), ("medqa", medqa, n_medqa),
                          ("regular", regular, n_regular), ("orbench_80k", orbench, n_orbench)):
        picked = rng.sample(pool, min(n, len(pool)))
        benign += picked
        sources += [name] * len(picked)
    harmful_sources = ["medsafety"] * len(harmful) + ["orbench_toxic"] * len(toxic)
    return harmful + toxic, harmful_sources, benign, sources


def threshold_for_fpr(neg_scores: np.ndarray, max_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on negatives is <= max_fpr."""
    s = np.sort(neg_scores)
    k = int(np.floor(len(s) * (1 - max_fpr)))
    k = min(max(k, 0), len(s) - 1)
    return float(s[k]) + 1e-6


def worst_group_threshold(scores: np.ndarray, y: np.ndarray, sources: np.ndarray, max_fpr: float) -> float:
    """Smallest threshold that keeps *each* benign source at or below max_fpr."""
    neg = (y == 0)
    return max(threshold_for_fpr(scores[neg & (sources == s)], max_fpr) for s in np.unique(sources[neg]))


def oof_scores(X: np.ndarray, y: np.ndarray, C: float, seed: int, folds: int = 5) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    out = np.zeros(len(y), dtype=np.float64)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n-medquad", type=int, default=1500)
    ap.add_argument("--n-medqa", type=int, default=3000)
    ap.add_argument("--n-regular", type=int, default=1500)
    ap.add_argument("--n-orbench", type=int, default=8000)
    ap.add_argument("--block-fpr", type=float, default=0.01, help="max FPR per benign source at block")
    ap.add_argument("--flag-fpr", type=float, default=0.03, help="max FPR per benign source at flag")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(args.seed)
    harmful, harmful_sources, benign, benign_sources = load_training_data(
        rng, args.n_medquad, args.n_medqa, args.n_regular, args.n_orbench)
    embedder = SentenceTransformer(args.embed_model, device="cpu")
    t0 = time.time()

    def enc(texts):
        return np.asarray(embedder.encode(texts, normalize_embeddings=True, batch_size=64,
                                          show_progress_bar=False), dtype=np.float32)

    # Hold every benchmark out: drop training prompts close to any evaluation prompt.
    from src.eval.benchmarks import SUITES, load_suite

    eval_texts = []
    for suite in SUITES:
        if suite in ("in_the_wild", "medical_jailbreak"):
            continue  # long templates; their medical payloads are MedSafetyBench test requests
        eval_texts += [i.prompt[:2000] for i in load_suite(suite)]
    eval_emb = enc(eval_texts)

    def drop_near_eval(texts, sources, label):
        emb = enc(texts)
        nearest = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(eval_emb), 4096):
            nearest = np.maximum(nearest, (emb @ eval_emb[i:i + 4096].T).max(axis=1))
        keep = nearest <= 0.90
        print(f"{label}: kept {int(keep.sum())}/{len(texts)} "
              f"(dropped {int((~keep).sum())} near-duplicates of evaluation prompts)")
        return emb[keep], [x for x, k in zip(sources, keep) if k]

    h_emb, harmful_sources = drop_near_eval(harmful, harmful_sources, "Harmful")
    b_emb, benign_sources = drop_near_eval(benign, benign_sources, "Benign")
    print(f"Embedding took {time.time() - t0:.0f}s")

    X = np.vstack([h_emb, b_emb])
    y = np.array([1] * len(h_emb) + [0] * len(b_emb))
    src = np.array(harmful_sources + benign_sources)

    best = None
    for C in (0.5, 1.0, 2.0, 4.0, 8.0):
        p = oof_scores(X, y, C, args.seed)
        t_block = worst_group_threshold(p, y, src, args.block_fpr)
        tpr = float((p[y == 1] >= t_block).mean())
        print(f"  C={C:<4}  out-of-fold TPR at worst-source {args.block_fpr:.0%} FPR = {tpr:.3f}")
        if best is None or tpr > best[0]:
            best = (tpr, C, p)
    _, C, p = best

    t_block = worst_group_threshold(p, y, src, args.block_fpr)
    t_flag = min(t_block, worst_group_threshold(p, y, src, args.flag_fpr))
    val = {
        "method": f"5-fold out-of-fold scores, worst benign source <= {args.block_fpr:.0%} (block) / "
                  f"{args.flag_fpr:.0%} (flag)",
        "tpr_at_block": float((p[y == 1] >= t_block).mean()),
        "tpr_at_flag": float((p[y == 1] >= t_flag).mean()),
        "fpr_at_block": float((p[y == 0] >= t_block).mean()),
        "fpr_at_flag": float((p[y == 0] >= t_flag).mean()),
        "fpr_by_source_at_block": {s: float((p[(y == 0) & (src == s)] >= t_block).mean())
                                   for s in sorted(set(benign_sources))},
        "fpr_by_source_at_flag": {s: float((p[(y == 0) & (src == s)] >= t_flag).mean())
                                  for s in sorted(set(benign_sources))},
        "tpr_by_source_at_block": {s: float((p[(y == 1) & (src == s)] >= t_block).mean())
                                   for s in sorted(set(harmful_sources))},
    }

    # Final model on all training data. Its scores track the out-of-fold models'
    # closely (same C, 25% more data), so the thresholds carry over.
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
    meta = {
        "embed_model_id": args.embed_model,
        "C": C,
        "thresholds": {"block": t_block, "flag": t_flag},
        "n_harmful": int(y.sum()), "n_benign": int((y == 0).sum()),
        "validation": val,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "harmful": "MedSafetyBench train split, OR-Bench toxic",
            "benign": "MedQuAD train partition, MedQA train split, in-the-wild regular prompts, "
                      "OR-Bench-80k minus hard-1k",
            "dedup": "training prompts with cosine > 0.90 to any evaluation prompt removed",
        },
    }
    np.savez(args.out, coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             meta=np.array(json.dumps(meta)))
    print(json.dumps(meta["validation"], indent=1))
    print(f"thresholds: block={t_block:.4f} flag={t_flag:.4f}  ->  {args.out}")


if __name__ == "__main__":
    main()
