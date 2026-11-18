$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Write-Error "No .env found. Copy .env.example to .env and configure the model endpoint."
}

Write-Host "Starting Medical RAG System on http://127.0.0.1:8600"
$env:HF_HUB_OFFLINE = if ($env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE } else { "1" }
$env:NLTK_DATA = if ($env:NLTK_DATA) { $env:NLTK_DATA } else { ".nltk_data" }
& ".\.venv\Scripts\python.exe" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8600 --reload
