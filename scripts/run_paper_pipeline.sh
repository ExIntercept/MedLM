#!/usr/bin/env bash
# Runs the paper experiments after the main e2e run (scripts/run_benchmark_suite.py
# e2e --limit 30 --temperature 0) has finished: second judge, ablations on one
# model, second judge on the ablations, then the result tables.
# Every step resumes from its checkpoints, so the script can simply be re-run.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export GUARD_BASE_URL=http://localhost:11434 GUARD_MODEL=llama-guard3:8b PYTHONIOENCODING=utf-8
RUN="$PY scripts/run_benchmark_suite.py e2e --config scripts/ablation_model.json \
  --judge-config scripts/judge_local.json --limit 30 --subset 15 --temperature 0 \
  --conditions defended --out docs/paper/ablation_raw"

echo "### second judge on the main run $(date)"
$PY scripts/second_judge.py --judge-config scripts/judge2_local.json --pattern "e2e_*_limit30_temp0.jsonl"
$PY scripts/paper_tables.py

echo "### ablations $(date)"
$RUN --no-rag --tag lg-only --disable layer_triage layer_guardrails layer_prompt_guard \
     layer_template_matcher layer_harm_classifier layer_prompt_hardening
$RUN --tag no-llamaguard --disable layer_llama_guard layer_output_guard
$RUN --tag no-output-guard --disable layer_output_guard
$RUN --tag no-rag --no-rag
$RUN --tag no-input-detectors --disable layer_prompt_guard layer_template_matcher \
     layer_harm_classifier layer_llama_guard
$RUN --tag no-hardening --disable layer_prompt_hardening

echo "### second judge on the ablations $(date)"
$PY scripts/second_judge.py --judge-config scripts/judge2_local.json --pattern "e2e_*_limit30_temp0_*.jsonl"
$PY scripts/paper_tables.py
echo "### done $(date)"
