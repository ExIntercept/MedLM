# Medical Multimodal RAG & Multi-Agent Verification System (MedGemma + Colab)

A production-grade clinical RAG (Retrieval-Augmented Generation) and multi-agent safety verification system. It pairs local hybrid retrieval (sparse BM25 + dense S-PubMedBERT embeddings + MedCPT cross-encoder reranking) with your **MedGemma** model running on Google Colab via an ngrok tunnel.

---

## 🏛️ System Architecture

```
                                  LOCAL WORKSPACE (Mac / Local Machine)
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                                  │
  │   ┌───────────────────────────┐         ┌────────────────────────────────────────────────────┐   │
  │   │  Web Clinical Console     │  HTTP   │  FastAPI Backend (src/api/server.py :8600)         │   │
  │   │  - Patient Triage Mode    │ ◄─────► │  - JWT Auth & Session Persistence (SQLite)         │   │
  │   │  - Clinician QA Mode      │         │  - Real-Time SSE Streaming (/api/chat/stream)      │   │
  │   │  - Real-time NER Profile  │         │  - Corpus Search & Medication Checker              │   │
  │   │  - Live Evidence Drawer   │         └─────────────────┬──────────────────────────────────┘   │
  │   └───────────────────────────┘                           │                                      │
  │                                                           ▼                                      │
  │   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
  │   │  Multi-Agent Safety & Verification Pipeline                                              │   │
  │   │  1. Emergency Red-Flag Triage   ─► Immediate escalation on acute cardiovascular / FAST    │   │
  │   │  2. Adversarial Guardrails      ─► Hard-rule contraindication & toxic dosage intercept    │   │
  │   │  3. Real-Time Intake Extraction ─► Extracts age, sex, duration, conditions, medications  │   │
  │   │  4. Hybrid RAG Retrieval        ─► BM25 + ChromaDB (S-PubMedBERT) + MedCPT Cross-Encoder │   │
  │   │  5. Faithfulness Audit          ─► Lexical overlap verification (0–100% groundedness)    │   │
  │   └───────────────────────────────────────────────┬──────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────┼──────────────────────────────────────────────┘
                                                      │
                                                      │ HTTPS (ngrok tunnel)
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │  Google Colab GPU Instance  │
                                       │  - vLLM / OpenAI Server     │
                                       │  - google/medgemma-4b-it    │
                                       │  - /v1/completions stream   │
                                       └─────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Configure `.env`
Ensure your `.env` file at the root contains your active Colab ngrok tunnel URL:

```bash
# Paste the URL Colab prints (no trailing slash)
COLAB_API_BASE=https://xxxx-xx-xx-xx-xx.ngrok-free.dev
COLAB_API_KEY=your-api-key-from-notebook

# MedGemma model configuration
MODEL_ID=google/medgemma-4b-it
PROMPT_STYLE=gemma
MAX_MODEL_LEN=8192
RESERVE_OUTPUT_TOKENS=640
```

### 2. Launch the System
```bash
./run.sh
```
Or with python directly:
```bash
./.venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8600 --reload
```

Open your browser at **[http://127.0.0.1:8600](http://127.0.0.1:8600)**.

---

## 🔍 Key Modules & Features

- **`src/retrieval/retriever.py`**:
  - Sparse lexical search over 30,980+ StatPearls, OpenFDA, and MedQuAD chunks (`corpus/bm25_index.pkl`).
  - Dense semantic retrieval with `pritamdeka/S-PubMedBert-MS-MARCO` via ChromaDB.
  - Reciprocal Rank Fusion (RRF) with medication entity amplification and source weighting (`StatPearls`: 1.3, `OpenFDA`: 1.1, `MedQuAD`: 0.8).
  - Neural Cross-Encoder Reranking via `ncbi/MedCPT-Cross-Encoder`.
- **`src/agents/emergency_triage.py`**:
  - Deterministic triage classifier across cardiovascular, FAST stroke, respiratory distress, anaphylaxis, GI bleeding/surgical abdomen, trauma, sepsis, and psychiatric red-flags.
- **`src/agents/guardrails.py`**:
  - Intercepts dangerous combinations (e.g. Warfarin in pregnancy, Metformin in severe renal impairment, IV potassium push, Acetaminophen overdose) before any retrieval/generation.
- **`src/agents/verifier.py`**:
  - Post-generation evidence auditing verifying claim overlap against retrieved guidelines.
- **`src/llm/client.py`**:
  - Remote streaming client communicating with your Colab instance over ngrok with `ngrok-skip-browser-warning` headers and MedGemma prompt formatting.
- **`src/api/server.py`**:
  - FastAPI endpoints for authentication, consultations, real-time SSE streaming (`/api/chat/stream`), corpus searching (`/api/evidence/search`), and medication checking (`/api/medications/check`).
- **`scripts/run_benchmark.py`**:
  - Benchmark evaluation suite running clinical test cases against the live pipeline.
