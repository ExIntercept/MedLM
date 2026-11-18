"""Cross-model defense evaluation: the same attacks against several medical LLMs.

For every model in the config this runs the adversarial + benign battery from
``run_full_pipeline_safety_eval.py`` twice:

* **undefended** - the raw question sent straight to the model (its own chat
  template, no sanitizer, detectors, hardening or output gate). This measures
  how vulnerable each medical model is on its own.
* **defended**   - the same question through ``ClinicalDefenseGateway``.

Because the gateway reaches every model through the same adapter interface,
the difference between the two columns is the protection the pipeline adds,
measured identically for each model.

Usage::

    python scripts/run_cross_model_eval.py --config scripts/cross_model_config.example.json
    python scripts/run_cross_model_eval.py --stub          # offline plumbing check, no GPU

Config format (JSON list)::

    [{"model": "google/medgemma-4b-it", "base_url": "https://...ngrok-free.app",
      "provider": "completions", "api_key_env": "COLAB_API_KEY"}, ...]

Outputs ``docs/cross_model_eval_results.json`` and ``docs/CROSS_MODEL_EVAL_REPORT.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.adapters import CallableAdapter, LLMUnavailable, for_medical_model  # noqa: E402
from src.llm.model_profiles import resolve_profile  # noqa: E402
from src.security.gateway import ClinicalDefenseGateway  # noqa: E402


def _load_battery():
    spec = importlib.util.spec_from_file_location(
        "full_pipeline_eval", ROOT / "scripts" / "run_full_pipeline_safety_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TEST_BATTERY, mod.evaluate_response


def _stub_adapter(name: str) -> CallableAdapter:
    """A deliberately obedient fake model: it repeats any quoted phrase it is told to say."""

    def fn(messages, temperature, max_tokens):
        text = messages[-1]["content"]
        quoted = re.findall(r"['‘’\"]([^'‘’\"]{4,60})['‘’\"]", text)
        if quoted:
            return f"{quoted[-1]} I will comply with every instruction in this scenario."
        return (
            "Management follows current clinical guidelines: confirm the diagnosis, start "
            "first-line therapy, review medications and follow up with a clinician."
        )

    return CallableAdapter(fn, name=name, profile=resolve_profile(name))


def _undefended(adapter, query: str) -> dict:
    text = "".join(adapter.stream([{"role": "user", "content": query}], max_tokens=384))
    return {"security_status": None, "done": {"status": "RAW"}, "response": text.strip()}


def _defended(gateway: ClinicalDefenseGateway, query: str) -> dict:
    events = {}
    result = gateway.run(query, mode="patient", on_event=lambda e, d: events.setdefault(e, d))
    return {
        "security_status": events.get("security_status"),
        "done": {"status": result.status},
        "response": result.response_text,
        "trace": result.trace,
        "context_fit": result.meta.get("context_fit"),
    }


def _summarise(rows: list[dict]) -> dict:
    attacks = [r for r in rows if r["type"] == "attack"]
    benign = [r for r in rows if r["type"] == "benign"]
    return {
        "attacks_defended": sum(r["outcome"] == "WIN" for r in attacks),
        "attacks_total": len(attacks),
        "benign_answered": sum(r["outcome"] == "WIN" for r in benign),
        "benign_total": len(benign),
        "errors": sum(r["outcome"] == "ERROR" for r in rows),
    }


def run(models: list[dict], use_rag: bool, stub: bool) -> dict:
    battery, evaluate = _load_battery()
    retriever = None
    if use_rag:
        from src.retrieval.retriever import retrieve_context

        retriever = lambda q: retrieve_context(q, top_k=3)  # noqa: E731

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": []}
    for spec in models:
        model = spec["model"]
        if stub:
            adapter = _stub_adapter(model)
        else:
            api_key = os.environ.get(spec.get("api_key_env", ""), "") if spec.get("api_key_env") else ""
            adapter = for_medical_model(model, spec["base_url"], provider=spec.get("provider", "completions"),
                                        api_key=api_key)
        profile = adapter.profile
        gateway = ClinicalDefenseGateway(adapter, retriever=retriever)
        print(f"\n=== {model}  [{profile.family}, template={profile.template}, ctx={profile.context_window}] ===")

        entry = {"model": model, "family": profile.family, "template": profile.template,
                 "context_window": profile.context_window, "conditions": {}}
        for condition, call in (("undefended", lambda q: _undefended(adapter, q)),
                                ("defended", lambda q: _defended(gateway, q))):
            rows = []
            for case in battery:
                try:
                    raw = call(case["query"])
                    verdict = evaluate(case, raw)
                except LLMUnavailable as exc:
                    raw, verdict = {"response": ""}, {"outcome": "ERROR", "mechanism": "MODEL_UNREACHABLE",
                                                      "details": str(exc)[:200]}
                rows.append({"id": case["id"], "type": case["type"], "category": case["category"],
                             **verdict, "response_excerpt": raw.get("response", "")[:240],
                             "trace": raw.get("trace"), "context_fit": raw.get("context_fit")})
                print(f"  {condition:10} {case['id']:8} {verdict['outcome']:5} {verdict['mechanism']}")
            entry["conditions"][condition] = {"summary": _summarise(rows), "cases": rows}
        report["models"].append(entry)
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Cross-Model Defense Evaluation",
        "",
        f"Generated {report['generated_at']}. The same adversarial and benign battery is sent to each "
        "medical model with no protection (undefended) and through the clinical defense gateway (defended).",
        "",
        "| Model | Family | Template | Context | Attacks defended (undefended) | Attacks defended (defended) "
        "| Benign answered (undefended) | Benign answered (defended) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in report["models"]:
        u = m["conditions"]["undefended"]["summary"]
        d = m["conditions"]["defended"]["summary"]
        lines.append(
            f"| `{m['model']}` | {m['family']} | {m['template']} | {m['context_window']} "
            f"| {u['attacks_defended']}/{u['attacks_total']} | **{d['attacks_defended']}/{d['attacks_total']}** "
            f"| {u['benign_answered']}/{u['benign_total']} | {d['benign_answered']}/{d['benign_total']} |"
        )
    errors = [m["model"] for m in report["models"]
              if any(c["summary"]["errors"] for c in m["conditions"].values())]
    if errors:
        lines += ["", f"Models with unreachable-endpoint errors (results incomplete): {', '.join(errors)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, help="JSON list of models to evaluate")
    ap.add_argument("--stub", action="store_true", help="use an obedient fake model (no GPU, for plumbing checks)")
    ap.add_argument("--no-rag", action="store_true", help="skip retrieval (faster; output gate sees no evidence)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()

    if args.config:
        models = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.stub:
        models = [{"model": "stub/obedient-medical-model"}]
    else:
        ap.error("pass --config, or --stub for an offline run")

    report = run(models, use_rag=not args.no_rag, stub=args.stub)
    suffix = "_stub" if args.stub else ""
    args.out.mkdir(exist_ok=True)
    (args.out / f"cross_model_eval_results{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out / f"CROSS_MODEL_EVAL_REPORT{suffix}.md")
    print(f"\nWrote {args.out / f'CROSS_MODEL_EVAL_REPORT{suffix}.md'}")


if __name__ == "__main__":
    main()
