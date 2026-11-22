#!/usr/bin/env bash
# Runs the paper experiments after the main e2e run (scripts/run_benchmark_suite.py
# e2e --limit 30 --temperature 0) has finished. Order is by importance, so a run
# cut short still has the key results: ablations first (the Llama-Guard-only
# baseline is the main comparison), then the second judge on attack prompts
# (all its tables use), then the tables, failure analysis and figures.
# Every step resumes from its checkpoints, so the script can simply be re-run.
#
# Llama Guard: Ollama's stock llama-guard3 template always asks about the last
# *User* message, so (request, response) checks judged the request. The fixed
# copy is built from scripts/ollama/llama-guard3-agent.Modelfile and is used for
# judge 1 and for the gateway's input and output guard.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
ollama create llama-guard3-agent:8b -f scripts/ollama/llama-guard3-agent.Modelfile >/dev/null
export GUARD_BASE_URL=http://localhost:11434 GUARD_MODEL=llama-guard3-agent:8b PYTHONIOENCODING=utf-8
RUN="$PY scripts/run_benchmark_suite.py e2e --config scripts/ablation_model.json \
  --judge-config scripts/judge_local_agent.json --limit 30 --subset 10 --temperature 0 \
  --conditions defended --out docs/paper/ablation_raw"
J2="$PY scripts/second_judge.py --judge-config scripts/judge2_local.json --kind attack"

echo "### re-judge the main run with the fixed guard template $(date)"
$PY scripts/rejudge.py --judge-config scripts/judge_local_agent.json --pattern "e2e_*_limit30_temp0.jsonl"

echo "### ablations $(date)"
$RUN --no-rag --tag lg-only --disable layer_triage layer_guardrails layer_prompt_guard \
     layer_template_matcher layer_harm_classifier layer_prompt_hardening
$RUN --tag no-input-detectors --disable layer_prompt_guard layer_template_matcher \
     layer_harm_classifier layer_llama_guard
$RUN --tag no-llamaguard --disable layer_llama_guard layer_output_guard
$RUN --tag no-rag --no-rag
$PY scripts/paper_tables.py

echo "### second judge on attack prompts $(date)"
$J2 --pattern "e2e_*_limit30_temp0.jsonl"
$J2 --pattern "e2e_*_limit30_temp0_*.jsonl"

echo "### tables, failure analysis, figures $(date)"
$PY scripts/paper_tables.py
$PY scripts/failure_analysis.py
$PY scripts/paper_figures.py
echo "### done $(date)"
