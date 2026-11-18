"""Load public adversarial and benign benchmarks into one normalised format.

Raw files come from ``scripts/fetch_benchmarks.py`` (``data/benchmarks/raw``).
Every loader returns :class:`BenchItem` records so the suite runner treats all
sources the same way.

Suites
------
Attacks
  medsafetybench       harmful medical requests (MedSafetyBench test split)
  medical_jailbreak    in-the-wild jailbreak template wrapped around a
                       MedSafetyBench request (the realistic threat)
  in_the_wild          in-the-wild jailbreak templates on their own
  jbb_harmful          JailbreakBench harmful behaviours (general domain)
  deepset_injection    deepset prompt-injection test split, label 1
  xstest_unsafe        XSTest unsafe contrast prompts

Benign
  medqa                MedQA USMLE questions with options (clinician style)
  medquad              MedQuAD consumer health questions (patient style)
  jbb_benign           JailbreakBench benign behaviours matched to the harmful ones
  deepset_benign       deepset prompt-injection test split, label 0
  xstest_safe          XSTest safe prompts that *look* unsafe (over-refusal)

``seen_by_layer_a`` marks suites that overlap the training data listed on the
ProtectAI prompt-injection model card (XSTest, jackhhao/jailbreak-classification,
which draws on the same jailbreakchat.com prompts as the in-the-wild set).
Results on those suites are reported separately, not as held-out evidence.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "benchmarks" / "raw"


@dataclass
class BenchItem:
    id: str
    suite: str
    kind: str                 # "attack" | "benign"
    attack_type: Optional[str]  # jailbreak | injection | harmful_direct | medical_jailbreak | None
    prompt: str
    category: str = ""
    medical: bool = False
    seen_by_layer_a: bool = False
    reference: dict = field(default_factory=dict)  # e.g. MedQA answer letter


SUITES: dict[str, dict] = {
    "medsafetybench":    {"kind": "attack", "attack_type": "harmful_direct", "medical": True},
    "medical_jailbreak": {"kind": "attack", "attack_type": "medical_jailbreak", "medical": True, "seen": True},
    "in_the_wild":       {"kind": "attack", "attack_type": "jailbreak", "seen": True},
    "jbb_harmful":       {"kind": "attack", "attack_type": "harmful_direct"},
    "deepset_injection": {"kind": "attack", "attack_type": "injection"},
    "xstest_unsafe":     {"kind": "attack", "attack_type": "harmful_direct", "seen": True},
    "medqa":             {"kind": "benign", "medical": True},
    "medquad":           {"kind": "benign", "medical": True},
    "jbb_benign":        {"kind": "benign"},
    "deepset_benign":    {"kind": "benign"},
    "xstest_safe":       {"kind": "benign", "seen": True},
}

_PLACEHOLDER_RE = re.compile(r"\[(?:INSERT PROMPT HERE|PROMPT|INSERT QUESTION HERE|YOUR PROMPT HERE)\]", re.I)


def _pd():
    import pandas as pd

    return pd


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/fetch_benchmarks.py")
    return path


def _medsafety_requests() -> list[tuple[str, str]]:
    pd = _pd()
    out = []
    for gen in ("gpt4", "llama2"):
        for c in range(1, 10):
            df = pd.read_csv(_require(RAW / "medsafetybench" / gen / f"category_{c}.csv"))
            for i, req in enumerate(df["harmful_medical_request"].dropna()):
                out.append((f"{gen}-c{c}-{i}", str(req).strip()))
    return out


def _wild_templates() -> list[str]:
    df = _pd().read_parquet(_require(RAW / "in_the_wild" / "jailbreak_2023_12_25.parquet"))
    return [str(p) for p in df["prompt"].dropna()]


def _load_raw(suite: str, rng: random.Random) -> list[tuple[str, str, str, dict]]:
    """Return (id, prompt, category, reference) tuples for one suite."""
    pd = _pd()
    if suite == "medsafetybench":
        return [(i, r, i.split("-")[1], {}) for i, r in _medsafety_requests()]

    if suite == "medical_jailbreak":
        requests_ = _medsafety_requests()
        out = []
        for t_idx, template in enumerate(_wild_templates()):
            rid, req = rng.choice(requests_)
            if _PLACEHOLDER_RE.search(template):
                prompt = _PLACEHOLDER_RE.sub(lambda _m: req, template, count=1)
            else:
                prompt = f"{template.rstrip()}\n\n{req}"
            out.append((f"t{t_idx}+{rid}", prompt, rid.split("-")[1], {"request": req}))
        return out

    if suite == "in_the_wild":
        return [(f"t{i}", p, "", {}) for i, p in enumerate(_wild_templates())]

    if suite in ("jbb_harmful", "jbb_benign"):
        name = "harmful-behaviors.csv" if suite == "jbb_harmful" else "benign-behaviors.csv"
        df = pd.read_csv(_require(RAW / "jbb" / name))
        return [(str(r.Index), str(r.Goal), str(r.Category), {}) for r in df.itertuples()]

    if suite in ("deepset_injection", "deepset_benign"):
        df = pd.read_parquet(_require(RAW / "deepset" / "test.parquet"))
        want = 1 if suite == "deepset_injection" else 0
        df = df[df["label"] == want]
        return [(str(i), str(t), "", {}) for i, t in zip(df.index, df["text"])]

    if suite in ("xstest_safe", "xstest_unsafe"):
        df = pd.read_csv(_require(RAW / "xstest" / "xstest_prompts.csv"))
        want = "safe" if suite == "xstest_safe" else "unsafe"
        df = df[df["label"] == want]
        return [(str(r.id), str(r.prompt), str(r.type), {}) for r in df.itertuples()]

    if suite == "medqa":
        out = []
        with open(_require(RAW / "medqa" / "test.jsonl"), encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                opts = "\n".join(f"({k}) {v}" for k, v in sorted(row["options"].items()))
                out.append((str(i), f"{row['question']}\n{opts}", row.get("meta_info", ""),
                            {"answer": row["answer_idx"]}))
        return out

    if suite == "medquad":
        df = pd.read_csv(_require(RAW / "medquad" / "medDataset_processed.csv"))
        df = df.dropna(subset=["Question"]).drop_duplicates(subset=["Question"])
        return [(str(i), str(q).replace("? ?", "?").strip(), str(t), {}) for i, q, t in
                zip(df.index, df["Question"], df["qtype"])]

    raise KeyError(f"Unknown suite {suite!r}; choose from {sorted(SUITES)}")


def load_suite(suite: str, limit: Optional[int] = None, seed: int = 13) -> list[BenchItem]:
    """Load one suite, optionally down-sampled with a fixed seed."""
    spec = SUITES[suite]
    rng = random.Random(f"{seed}:{suite}")
    rows = _load_raw(suite, rng)
    if limit is not None and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return [
        BenchItem(
            id=f"{suite}:{rid}",
            suite=suite,
            kind=spec["kind"],
            attack_type=spec.get("attack_type"),
            prompt=prompt,
            category=category,
            medical=spec.get("medical", False),
            seen_by_layer_a=spec.get("seen", False),
            reference=ref,
        )
        for rid, prompt, category, ref in rows
    ]


def load_suites(names: list[str], limit: Optional[int] = None, seed: int = 13) -> list[BenchItem]:
    items: list[BenchItem] = []
    for name in names:
        items.extend(load_suite(name, limit=limit, seed=seed))
    return items
