"""Tune the evidence-grounding gate (gateway stage 8) on real model drafts.

``capture``  Re-runs, through the full gateway, every defended benchmark prompt
             that reached a model in an e2e checkpoint, and saves the model's
             draft answer (before the gate), the retrieved evidence, the question
             and its answer options. Layer G is switched off for the capture:
             these prompts already passed it.

``judge``    Scores each captured attack draft with a Llama Guard judge, so a
             looser gate can be checked for releasing harmful answers.

``analyze``  Replays the drafts through gate variants offline and reports, per
             variant, how many medical answers are withheld and how many unsafe
             attack drafts would be released.

    python scripts/tune_grounding.py capture --config scripts/cross_model_local.json --limit 15
    python scripts/tune_grounding.py judge --judge-config scripts/judge_local.json
    python scripts/tune_grounding.py analyze
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

from scripts.run_benchmark_suite import (  # noqa: E402
    CHECKPOINT_DIR, MEDICAL_BENIGN, _e2e_text, _mode, _slug,
)
from src.eval.benchmarks import SUITES, load_suites  # noqa: E402

CAPTURE = CHECKPOINT_DIR / "grounding_drafts.jsonl"


def capture(args):
    from src.agents.guardrails import parse_embedded_options
    from src.llm.adapters import LLMUnavailable, for_medical_model
    from src.retrieval.retriever import retrieve_context
    from src.security.gateway import ClinicalDefenseGateway

    items = {i.id: i for i in load_suites(list(SUITES), limit=args.limit, seed=args.seed)}
    done = set()
    if CAPTURE.exists():
        done = {(r["model"], r["id"]) for r in map(json.loads, CAPTURE.read_text(encoding="utf-8").splitlines())}
    models = json.loads(args.config.read_text(encoding="utf-8"))
    out = open(CAPTURE, "a", encoding="utf-8")
    for spec in models:
        pattern = str(CHECKPOINT_DIR / f"e2e_{_slug(spec['model'])}_guard-*_rag1_seed{args.seed}_limit{args.limit}.jsonl")
        files = glob.glob(pattern)
        if not files:
            print(f"no checkpoint for {spec['model']}")
            continue
        rows = [json.loads(line) for line in open(files[0], encoding="utf-8")]
        todo = [r for r in rows if r["condition"] == "defended" and r["status"].startswith("ANSWERED")
                and (spec["model"], r["id"]) not in done]
        print(f"=== {spec['model']}: {len(todo)} prompts to capture", flush=True)
        adapter = for_medical_model(spec["model"], spec["base_url"], spec.get("provider", "completions"), "")
        seen = {}

        def retriever(q):
            ev = retrieve_context(q, top_k=3)
            seen["evidence"] = [(str(c), t, {}) for c, t, _m in ev]
            return ev

        gw = ClinicalDefenseGateway(adapter, retriever=retriever, log_decisions=False,
                                    max_tokens=int(spec.get("max_tokens", 512)),
                                    toggles={"layer_llama_guard": False})
        for n, row in enumerate(todo, 1):
            item = items[row["id"]]
            text = _e2e_text(item)
            seen.clear()
            try:
                res = gw.run(text, mode=_mode(item))
            except LLMUnavailable as exc:
                print(f"  skipped {item.id}: {str(exc)[:120]}", flush=True)
                continue  # not saved, retried on the next run
            if res.status != "ANSWERED":
                continue
            vignette, options = parse_embedded_options(res.user_text)
            out.write(json.dumps({
                "model": spec["model"], "id": item.id, "suite": item.suite, "kind": item.kind,
                "prompt": item.prompt, "user_text": res.user_text, "vignette": vignette,
                "options": options, "evidence": seen.get("evidence", []), "draft": res.draft_text,
                "withheld": res.withheld, "score": res.audit["score"] if res.audit else None,
                "answer": item.reference.get("answer"),
            }) + "\n")
            out.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    out.close()


def judge(args):
    from src.eval.judge import LlamaGuardJudge
    from src.llm.adapters import OpenAIChatAdapter

    jc = json.loads(args.judge_config.read_text(encoding="utf-8"))
    j = LlamaGuardJudge(OpenAIChatAdapter(jc["base_url"], jc["model"], ""))
    rows = [json.loads(line) for line in CAPTURE.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["kind"] == "attack" and "draft_harmful" not in r:
            r["draft_harmful"] = j.judge(r["prompt"], r["draft"]).harmful
    CAPTURE.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"judged {sum(r['kind'] == 'attack' for r in rows)} attack drafts")


def analyze(_args):
    from src.agents import verifier

    rows = [json.loads(line) for line in CAPTURE.read_text(encoding="utf-8").splitlines()]
    variants = {
        "current": dict(include_options=False),
        "with options": dict(include_options=True),
    }
    for th in (0.40, 0.34, 0.25):
        variants[f"with options, pass {th}"] = dict(include_options=True, pass_threshold=th)
    print(f"{len(rows)} drafts; medical = {sorted(MEDICAL_BENIGN)}\n")
    print("| Gate | Medical withheld | General benign withheld | Attack drafts withheld | Unsafe attack drafts released |")
    print("|---|---|---|---|---|")
    for name, kw in variants.items():
        stats = {"med": [0, 0], "gen": [0, 0], "atk": [0, 0], "unsafe_out": [0, 0]}
        for r in rows:
            ev = [tuple(e) for e in r["evidence"]]
            audit = verifier.verify_evidence(r["draft"], ev, r["vignette"],
                                             options=r["options"] if kw.get("include_options") else None,
                                             pass_threshold=kw.get("pass_threshold"))
            held = audit["status"] == "FLAGGED"
            group = "med" if r["suite"] in MEDICAL_BENIGN else ("atk" if r["kind"] == "attack" else "gen")
            stats[group][0] += held
            stats[group][1] += 1
            if r["kind"] == "attack" and r.get("draft_harmful"):
                stats["unsafe_out"][0] += not held
                stats["unsafe_out"][1] += 1
        cell = lambda k: f"{stats[k][0]}/{stats[k][1]}"  # noqa: E731
        print(f"| {name} | {cell('med')} | {cell('gen')} | {cell('atk')} | {cell('unsafe_out')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["capture", "judge", "analyze"])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--judge-config", type=Path)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    {"capture": capture, "judge": judge, "analyze": analyze}[args.step](args)


if __name__ == "__main__":
    main()
