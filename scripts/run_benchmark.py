#!/usr/bin/env python
# ==============================================================================
# EVALUATION & METRICS PROOF SUITE
#
# Runs every case in tests/eval_dataset.json through the real production
# safety/retrieval/prompting code paths (not reimplementations of them):
#   - src/agents/emergency_triage.check_emergency_triage
#   - src/agents/guardrails.check_hard_rules
#   - src/retrieval/retriever.retrieve_context (BM25 + ChromaDB + MedCPT rerank)
#   - src/agents/prompting.build_prompt
#   - src/agents/verifier.verify_evidence
#
# For category-A cases (open-ended, no guardrail/triage trigger) it generates
# both a pure-baseline response (no retrieved context) and a hybrid-RAG
# response, and audits BOTH against the same retrieved evidence set so
# faithfulness is an apples-to-apples comparison. Category-B (contraindication)
# and category-C (emergency) cases are scored on whether the guardrail/triage
# layer fires correctly — the production pipeline short-circuits before
# retrieval/generation for those, so this script does too.
#
# Usage:
#   python scripts/run_benchmark.py [--max-tokens 250] [--dataset PATH] [--out-dir docs]
# ==============================================================================

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.emergency_triage import EMERGENCY_WARNING, check_emergency_triage  # noqa: E402
from src.agents.guardrails import check_hard_rules, parse_embedded_options  # noqa: E402
from src.agents.prompting import build_prompt, format_patient_profile  # noqa: E402
from src.agents.verifier import verify_evidence  # noqa: E402
from src.config import MODEL_ID  # noqa: E402
from src.llm.client import stream_generate  # noqa: E402
from src.retrieval.retriever import retrieve_context  # noqa: E402


def generate(prompt, max_tokens):
    """Non-streaming wrapper over stream_generate()."""
    return "".join(stream_generate(prompt))


def _text_contains_all(text, phrases):
    text_lower = (text or "").lower()
    hits = [p for p in phrases if p.lower() in text_lower]
    return hits, len(hits) / len(phrases) if phrases else 1.0


def run_case(case, max_tokens):
    query = case["query"]
    profile = case.get("patient_profile") or {}
    result = {"case_id": case["case_id"], "category": case["category"], "query": query}

    try:
        # --- Layer 0/1 safety checks, exactly as the production pipeline runs them ---
        triage_category = check_emergency_triage(query)
        hard_rule_hit = check_hard_rules(query)

        result["emergency_detected"] = triage_category is not None
        result["emergency_category"] = triage_category
        result["contraindication_detected"] = hard_rule_hit is not None

        if triage_category or hard_rule_hit:
            # Production short-circuits before retrieval/generation for these —
            # the benchmark follows the same code path rather than forcing a
            # generation that would never happen in real use.
            final_text = EMERGENCY_WARNING if triage_category else hard_rule_hit[2]
            result["retrieval"] = None
            result["retrieval_hit_at_3"] = None
            result["retrieval_hit_at_5"] = None
            result["baseline_response"] = None
            result["rag_response"] = final_text
            result["groundedness_baseline"] = None
            result["groundedness_rag"] = None
            hits, coverage = _text_contains_all(final_text, case.get("required_key_phrases", []))
            result["key_phrase_hits"] = hits
            result["key_phrase_coverage"] = coverage
            return result

        # --- Category A: full hybrid pipeline ---
        profile_note = format_patient_profile(**profile) if profile else ""
        vignette_text, detected_options = parse_embedded_options(query)
        vignette_with_profile = f"{profile_note}\n{vignette_text}" if profile_note else vignette_text

        t0 = time.time()
        evidence = retrieve_context(vignette_text, top_k=5)
        retrieval_latency = round(time.time() - t0, 3)

        expected_source = case.get("expected_evidence_source")
        retrieval_summary = [
            {"chunk_id": str(chunk_id), "title": meta.get("title", ""), "source": meta.get("source", "")}
            for chunk_id, _, meta in evidence
        ]
        result["retrieval"] = retrieval_summary
        result["retrieval_latency_s"] = retrieval_latency
        if expected_source:
            haystack_3 = " ".join(f"{r['title']} {r['source']}" for r in retrieval_summary[:3]).lower()
            haystack_5 = " ".join(f"{r['title']} {r['source']}" for r in retrieval_summary[:5]).lower()
            result["retrieval_hit_at_3"] = expected_source.lower() in haystack_3
            result["retrieval_hit_at_5"] = expected_source.lower() in haystack_5
        else:
            result["retrieval_hit_at_3"] = None
            result["retrieval_hit_at_5"] = None

        evidence_md = ""
        for idx, (_, text, meta) in enumerate(evidence, 1):
            evidence_md += f"### [Evidence {idx}] {meta.get('title', 'Guideline')} (`{meta.get('source', '')}`)\n\n> {text}\n\n---\n"

        # (a) Pure baseline LLM: no retrieved context, no safety/persona framing.
        baseline_prompt = (
            "Answer the following clinical question concisely and accurately.\n\n"
            f"Question: {vignette_with_profile}"
        )
        result["baseline_response"] = generate(baseline_prompt, max_tokens)

        # (b) Hybrid RAG pipeline: the real prompt template with real retrieved evidence.
        forced_tone = "patient" if case.get("mode") == "patient" else "clinical_qa"
        rag_prompt = build_prompt(
            vignette_with_profile, detected_options, evidence_md, conversation_history=None, forced_tone=forced_tone
        )
        result["rag_response"] = generate(rag_prompt, max_tokens)

        # Groundedness: audit BOTH responses against the SAME retrieved evidence set,
        # so the comparison isolates the effect of RAG grounding, not retrieval variance.
        baseline_audit = verify_evidence(result["baseline_response"], evidence, vignette_text)
        rag_audit = verify_evidence(result["rag_response"], evidence, vignette_text)
        result["groundedness_baseline"] = baseline_audit["score"]
        result["groundedness_rag"] = rag_audit["score"]

        hits, coverage = _text_contains_all(result["rag_response"], case.get("required_key_phrases", []))
        result["key_phrase_hits"] = hits
        result["key_phrase_coverage"] = coverage

    except Exception as exc:  # keep the whole run resilient to a single bad case
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    return result


def _avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def compute_metrics(cases_by_id, case_results):
    a_results = [r for r in case_results if cases_by_id[r["case_id"]]["category"] == "A" and "error" not in r]
    b_results = [r for r in case_results if cases_by_id[r["case_id"]]["category"] == "B" and "error" not in r]
    all_results = [r for r in case_results if "error" not in r]

    retrieval_flags_3 = [r["retrieval_hit_at_3"] for r in a_results if r.get("retrieval_hit_at_3") is not None]
    retrieval_flags_5 = [r["retrieval_hit_at_5"] for r in a_results if r.get("retrieval_hit_at_5") is not None]

    contraindication_catch = [
        r["contraindication_detected"]
        for r in b_results
        if cases_by_id[r["case_id"]]["expected_contraindications"]
    ]

    emergency_correct = [
        r["emergency_detected"] == cases_by_id[r["case_id"]]["emergency_flag"] for r in all_results
    ]

    key_phrase_scores = [r["key_phrase_coverage"] for r in all_results if r.get("key_phrase_coverage") is not None]

    return {
        "retrieval_hit_rate_at_3": _avg([1.0 if f else 0.0 for f in retrieval_flags_3]),
        "retrieval_hit_rate_at_5": _avg([1.0 if f else 0.0 for f in retrieval_flags_5]),
        "faithfulness_rag_avg": _avg([r.get("groundedness_rag") for r in a_results]),
        "faithfulness_baseline_avg": _avg([r.get("groundedness_baseline") for r in a_results]),
        "contraindication_catch_rate": _avg([1.0 if f else 0.0 for f in contraindication_catch]),
        "emergency_escalation_accuracy": _avg([1.0 if f else 0.0 for f in emergency_correct]),
        "key_phrase_coverage_avg": _avg(key_phrase_scores),
        "cases_evaluated": len(case_results),
        "cases_errored": len(case_results) - len(all_results),
    }


def render_markdown_report(payload):
    m = payload["metrics"]

    def pct(v):
        return "N/A" if v is None else f"{v * 100:.1f}%"

    lines = [
        "# Benchmark Report",
        "",
        f"**Generated:** {payload['generated_at']}  ",
        f"**Model:** `{payload['model']}`  ",
        f"**Dataset:** `tests/eval_dataset.json` (v{payload['dataset_version']}, {payload['num_cases']} cases)  ",
        f"**Max tokens per generation:** {payload['max_tokens']}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value | Scope |",
        "|---|---|---|",
        f"| Retrieval Hit Rate @3 | {pct(m['retrieval_hit_rate_at_3'])} | Category A (open-ended, {sum(1 for c in payload['cases'] if c['category']=='A')} cases) |",
        f"| Retrieval Hit Rate @5 | {pct(m['retrieval_hit_rate_at_5'])} | Category A |",
        f"| Faithfulness — Hybrid RAG | {pct(m['faithfulness_rag_avg'])} | Category A |",
        f"| Faithfulness — Pure Baseline LLM | {pct(m['faithfulness_baseline_avg'])} | Category A |",
        f"| Safety & Contraindication Catch Rate | {pct(m['contraindication_catch_rate'])} | Category B (guardrail, {sum(1 for c in payload['cases'] if c['category']=='B')} cases) |",
        f"| Emergency Escalation Accuracy | {pct(m['emergency_escalation_accuracy'])} | All {payload['num_cases']} cases (TP + TN) |",
        f"| Required Key-Phrase Coverage | {pct(m['key_phrase_coverage_avg'])} | All cases |",
        "",
        f"Cases evaluated: {m['cases_evaluated']} · Cases errored: {m['cases_errored']}",
        "",
        "## Per-Case Results",
        "",
        "| Case ID | Cat | Emergency? | Contraindication? | Hit@3 | Hit@5 | Groundedness (RAG / Baseline) | Key-Phrase Coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt_bool(v):
        return "—" if v is None else ("✅" if v else "❌")

    def fmt_score(v):
        return "—" if v is None else f"{v:.2f}"

    for c in payload["cases"]:
        lines.append(
            f"| `{c['case_id']}` | {c['category']} | {fmt_bool(c.get('emergency_detected'))} | "
            f"{fmt_bool(c.get('contraindication_detected'))} | {fmt_bool(c.get('retrieval_hit_at_3'))} | "
            f"{fmt_bool(c.get('retrieval_hit_at_5'))} | {fmt_score(c.get('groundedness_rag'))} / "
            f"{fmt_score(c.get('groundedness_baseline'))} | {pct(c.get('key_phrase_coverage'))} |"
        )

    lines += [
        "",
        "## Methodology",
        "",
        "- **Category A** (open-ended queries): run through the full hybrid retrieval pipeline "
        "(BM25 + ChromaDB dense search + RRF fusion + MedCPT cross-encoder rerank, `top_k=5`). "
        "Both a pure-baseline LLM response (no retrieved context) and a hybrid-RAG response "
        "(real prompt template from `src/agents/prompting.py`) are generated, then **both** are "
        "audited against the *same* retrieved evidence set via `src/agents/verifier.verify_evidence` "
        "so the faithfulness comparison isolates the effect of grounding, not retrieval variance.",
        "- **Category B** (contraindications): scored on whether `src/agents/guardrails.check_hard_rules` "
        "fires — this check runs before retrieval/generation in production, so no LLM call is made "
        "for these cases in the benchmark either.",
        "- **Category C** (emergencies): scored on whether `src/agents/emergency_triage.check_emergency_triage` "
        "fires. Emergency Escalation Accuracy is computed over **all** cases (true positives on C, "
        "true negatives on A/B), not just recall on C, so a classifier that over-fires on benign "
        "queries is penalized.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation & metrics benchmark suite.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "tests" / "eval_dataset.json"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "docs"))
    parser.add_argument("--max-tokens", type=int, default=250, help="num_predict per LLM generation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    cases_by_id = {c["case_id"]: c for c in cases}

    print(f"Loaded {len(cases)} cases from {dataset_path}")
    case_results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['case_id']} ({case['category']}) ... ", end="", flush=True)
        t0 = time.time()
        result = run_case(case, args.max_tokens)
        elapsed = time.time() - t0
        status = "ERROR" if "error" in result else "ok"
        print(f"{status} ({elapsed:.1f}s)")
        case_results.append(result)

    metrics = compute_metrics(cases_by_id, case_results)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "dataset_version": dataset.get("version", 1),
        "num_cases": len(cases),
        "max_tokens": args.max_tokens,
        "metrics": metrics,
        "cases": case_results,
    }

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "BENCHMARK_REPORT.md"
    report_path.write_text(render_markdown_report(payload), encoding="utf-8")

    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print()
    print("Summary:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
