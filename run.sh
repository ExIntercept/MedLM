#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and paste your ngrok URL into it."
  exit 1
fi

echo "Starting Medical RAG System on http://127.0.0.1:8600"
exec ./.venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8600 --reload
