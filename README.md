# MedGemma Clinical Decision Support & Multi-Agent Safety System

A production-ready clinical AI consultation, triage, and multi-agent safety verification platform. The system pairs local hybrid retrieval (**BM25 + S-PubMedBERT + MedCPT**) and an end-to-end **Multi-Layer Input Defense Pipeline** with a **MedGemma** model running on a GPU instance (such as Google Colab via ngrok).

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [End-to-End Pipeline Explanation](#-end-to-end-pipeline-explanation)
   - [Stage 0: Sanitization & Normalization (Layer S)](#stage-0-sanitization--normalization-layer-s)
   - [Stage 1: Emergency Red-Flag Triage (Layer 0)](#stage-1-emergency-red-flag-triage-layer-0)
   - [Stage 2: Deterministic Adversarial Guardrails (Layer 1)](#stage-2-deterministic-adversarial-guardrails-layer-1)
   - [Stage 3: Multi-Layer Neural & Semantic Input Defense (Layer D)](#stage-3-multi-layer-neural--semantic-input-defense-layer-d)
   - [Stage 4: Real-Time Patient Intake Extraction (NER)](#stage-4-real-time-patient-intake-extraction-ner)
   - [Stage 5: Hybrid Clinical RAG Retrieval](#stage-5-hybrid-clinical-rag-retrieval)
   - [Stage 6: Prompt Construction & Delimiter Hardening (Layer C)](#stage-6-prompt-construction--delimiter-hardening-layer-c)
   - [Stage 7: Remote Inference Streaming (MedGemma)](#stage-7-remote-inference-streaming-medgemma)
   - [Stage 8: Post-Generation Evidence Grounding Audit](#stage-8-post-generation-evidence-grounding-audit)
3. [Repository Structure](#-repository-structure)
4. [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
5. [How to Start the Project Up](#-how-to-start-the-project-up)
   - [Step 1: Start the Remote MedGemma Server (Google Colab)](#step-1-start-the-remote-medgemma-server-google-colab)
   - [Step 2: Configure Environment Variables (`.env`)](#step-2-configure-environment-variables-env)
   - [Step 3: Launch Local FastAPI Backend](#step-3-launch-local-fastapi-backend)
   - [Step 4: Access the Clinical Web Console](#step-4-access-the-clinical-web-console)
6. [Security & Audit Console (UI Controls)](#-security--audit-console-ui-controls)
7. [Evaluation & Benchmarking Suites](#-evaluation--benchmarking-suites)
8. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🏛️ System Architecture

```
                                  LOCAL CLIENT & BACKEND (Mac / Linux)
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                                        │
  │   ┌───────────────────────────┐           HTTP / SSE            ┌──────────────────────────────────┐   │
  │   │  Web Clinical Console     │ ◄─────────────────────────────► │  FastAPI Server (Port 8600)      │   │
  │   │  - Patient & Clinician QA │                                 │  - JWT Auth (SQLite)             │   │
  │   │  - Live NER Patient Form  │                                 │  - Real-Time SSE Streamer        │   │
  │   │  - Security & Audit Modal │                                 │  - Thread-Safe Pipeline Executor │   │
  │   └───────────────────────────┘                                 └────────────────┬─────────────────┘   │
  │                                                                                  │                     │
  │                                                                                  ▼                     │
  │   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
  │   │                          Multi-Stage Clinical AI Safety Pipeline                               │   │
  │   │                                                                                                │   │
  │   │  [Stage 0] Layer S: Input Sanitizer (Unicode Cf strip, Gemma/ChatML control tag neutraliser)   │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 1] Layer 0: Emergency Red-Flag Triage ──(Critical?)──► Immediate Escalation (Exit)     │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 2] Layer 1: Adversarial Guardrails ────(Contraindicated?)──► Direct Intercept (Exit)   │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 3] Layer D: Neural & Semantic Input Defense                                            │   │
  │   │            - Layer A: Prompt Guard (DeBERTa-v3 / Llama-Prompt-Guard-2)                         │   │
  │   │            - Layer B: Template Matcher (SentenceTransformer + 9 Jailbreak Components)          │   │
  │   │            - Decision Engine: BLOCK (Exit) | FLAG | ALLOW                                      │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 4] Intake NER: Extracts age, sex, duration, conditions, medications                   │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 5] Hybrid RAG Retrieval: BM25 + S-PubMedBERT + MedCPT Reranking (StatPearls / FDA)     │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 6] Layer C: Nonce Delimiter Prompt Hardening + Server-side DB History                  │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 7] Remote Token Streaming via ngrok                                                   │   │
  │   │       │                                                                                        │   │
  │   │       ▼                                                                                        │   │
  │   │  [Stage 8] Post-Generation Faithfulness Audit (Grounding Score 0–100%)                         │   │
  │   └───────────────────────────────────────────────┬────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                                      │
                                                      │ HTTPS (ngrok tunnel with custom bypass header)
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │  Google Colab GPU Instance  │
                                       │  - vLLM / HuggingFace Engine│
                                       │  - google/medgemma-4b-it    │
                                       │  - Token streaming output   │
                                       └─────────────────────────────┘
```

---

## 🔬 End-to-End Pipeline Explanation

Every user query flows through an 8-stage clinical pipeline designed to prevent jailbreaks, ensure patient safety, ground deductions in verified medical literature, and verify factual consistency:

### Stage 0: Sanitization & Normalization (Layer S)
*File: `src/security/sanitize.py`*
- **Unicode Filtering**: Strips zero-width spaces (`U+200B`–`U+200F`), invisible joiners (`U+2060`–`U+2064`, `U+FEFF`), bidirectional override controls (`U+202A`–`U+202E`, `U+2066`–`U+2069`), and Unicode tag blocks (`U+E0000`–`U+E007F`). This completely neutralizes **ASCII smuggling** and invisible instruction injection.
- **Control Token Defanging**: Detects Gemma turn tokens (`<start_of_turn>`, `<end_of_turn>`) and ChatML tags (`<|im_start|>`, `<|im_end|>`) and replaces `< >` with single angle quotation marks `‹ ›` (`U+2039`/`U+203A`), rendering them harmless plain text to the LLM tokenizer.
- **Application Tag Defanging**: Neutralizes attempts to forge delimiter boundaries (e.g. `</user_message_...>`, `</reference_material>`).
- **Dual Representation**:
  - `model_text`: Retains original case and special clinical symbols (`µg`, `m²`, `½`).
  - `detector_text`: Normalizes via Unicode `NFKC` so downstream neural classifiers can detect obfuscated homoglyphs.

### Stage 1: Emergency Red-Flag Triage (Layer 0)
*File: `src/agents/emergency_triage.py`*
- Deterministic regex classifier identifying acute medical emergencies:
  - **Cardiovascular**: Crushing chest pain radiating to arm/jaw, acute diaphoresis.
  - **Neurological (FAST)**: Sudden facial drooping, arm weakness, slurred speech.
  - **Respiratory**: Severe stridor, inability to speak full sentences, cyanosis.
  - **Anaphylaxis**: Acute airway swelling with urticaria/hypotension.
  - **Surgical Abdomen & Hemorrhage**: Rigid abdomen, massive hematemesis.
  - **Psychiatric**: Active acute suicidal ideation with imminent intent.
- **Immediate Escalation**: If matched, immediately halts processing, instructs the patient to seek urgent emergency care (911/ER), and logs the event without making any LLM call.

### Stage 2: Deterministic Adversarial Guardrails (Layer 1)
*File: `src/agents/guardrails.py`*
- Catches lethal contraindications, lethal dosing combinations, or illicit chemical inquiries before retrieval:
  - **Absolute Contraindications**: E.g., Warfarin administration during pregnancy (teratogenicity), Metformin in severe renal impairment/metabolic acidosis.
  - **Lethal Dosages & Toxic Overdoses**: E.g., Acetaminophen > 4000 mg/day, rapid IV potassium push.
  - **Controlled Substance / Illicit Instructions**: Intercepts attempts to solicit illicit synthesis recipes.
- Immediately responds with a structured clinical refusal explaining the exact pharmacological contraindication.

### Stage 3: Multi-Layer Neural & Semantic Input Defense (Layer D)
*Files: `src/security/prompt_guard.py`, `src/security/template_matcher.py`, `src/security/decision.py`*
Evaluates three distinct input contexts: (1) the current query, (2) the intake profile text, and (3) a **multi-turn window** (last 2 user turns + current query):
- **Layer A (Prompt Injection Classifier)**:
  - Runs `ProtectAI/deberta-v3-base-prompt-injection-v2` (or `meta-llama/Llama-Prompt-Guard-2-86M`) in 510-token sliding windows (stride 384) to generate an injection probability score (`0.0`–`1.0`).
- **Layer B (Template Matcher)**:
  - Computes dense cosine similarity between query windows and a curated corpus of adversarial jailbreak templates using `sentence-transformers/all-MiniLM-L6-v2`.
  - Analyzes 9 jailbreak semantic components: `fictional_frame`, `amoral_character`, `unconditional_obedience`, `ethics_waiver`, `realism_demand`, `stakes_pressure`, `private_audience_claim`, `redaction_claim`, and `forced_opening`.
- **Decision Engine**:
  - `BLOCK`: If `pg_score >= 0.90`, `tm_max_sim >= 0.80`, or `tm_n_components >= 3`. Halts pipeline with an adversarial intercept message.
  - `FLAG`: If `pg_score >= 0.50`, `tm_max_sim >= 0.70`, `tm_n_components >= 2`, or control tokens were defanged. Proceeds to generation with hardened system constraints.
  - `ALLOW`: Query is deemed clean.

### Stage 4: Real-Time Patient Intake Extraction (NER)
*File: `src/agents/intake_extraction.py`*
- Real-time regular expression & heuristic NER parser extracting key clinical attributes:
  - **Age**: (e.g. `45yo`, `62 years old`)
  - **Sex / Gender**: (`male`, `female`)
  - **Duration**: (`for 3 days`, `since yesterday`)
  - **Conditions**: (`hypertension`, `type 2 diabetes`, `asthma`)
  - **Medications**: (`lisinopril`, `metformin`, `albuterol`)
- Emits real-time SSE event `profile_update` to dynamically update the intake form on the UI sidebar.

### Stage 5: Hybrid Clinical RAG Retrieval
*File: `src/retrieval/retriever.py`*
Retrieves verified clinical evidence across a curated corpus of 30,980+ chunks from **StatPearls**, **OpenFDA Drug Labels**, and **MedQuAD**:
- **Sparse Lexical Search**: BM25 keyword matching with medical synonym and acronym expansion (`corpus/bm25_index.pkl`).
- **Dense Semantic Search**: ChromaDB vector index embedded via `pritamdeka/S-PubMedBert-MS-MARCO`.
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense candidate rankings with corpus-specific authority multipliers (`StatPearls`: 1.3, `OpenFDA`: 1.1, `MedQuAD`: 0.8) and medication entity boosts.
- **Cross-Encoder Neural Reranking**: `ncbi/MedCPT-Cross-Encoder` computes cross-attention scores across the top candidates and delivers the top 3 evidence passages.

### Stage 6: Prompt Construction & Delimiter Hardening (Layer C)
*File: `src/agents/prompting.py`*
- Formats the MedGemma prompt with mode-specific instructions (`patient` or `clinical_qa`).
- **Nonce-Tagged Delimiters**: Encloses untrusted user input within random cryptographic nonces:
  ```xml
  <user_message_a3f89b>
  Patient input goes here
  </user_message_a3f89b>
  ```
- **Post-Rules Injection**: Reinforces system instructions directly after the user input block to prevent recency-bias jailbreaks.
- **Server-Side History**: Uses history loaded strictly from the local SQLite database; completely ignores client-supplied turn history to prevent **turn-forgery attacks**.

### Stage 7: Remote Inference Streaming (MedGemma)
*File: `src/llm/client.py`*
- Streams tokens from the remote Colab GPU instance running `google/medgemma-4b-it` via an ngrok tunnel.
- Includes `ngrok-skip-browser-warning: 1` headers to bypass ngrok intermediary landing pages.
- Uses Server-Sent Events (SSE) `/api/chat/stream` to stream tokens in real-time directly into the chat bubble.

### Stage 8: Post-Generation Evidence Grounding Audit
*File: `src/agents/verifier.py`*
- Evaluates the generated clinical response against the retrieved evidence passages.
- Computes claim overlap and emits a **Faithfulness / Grounding Score** (0–100%) displayed directly in the chat UI with a badge indicating `PASS` or `FLAGGED`.

---

## 📁 Repository Structure

```
MedGemmaLocal/
├── app/                        # Minimal standalone FastAPI prototype
├── corpus/                     # Clinical knowledge base & precomputed indices
│   ├── bm25_index.pkl          # Serialized BM25 index over clinical corpus
│   ├── bm25_metadata.json      # Metadata mapping for BM25 chunks
│   └── *.jsonl                 # StatPearls, OpenFDA, and MedQuAD chunk files
├── chroma_db/                  # ChromaDB vector store (S-PubMedBERT embeddings)
├── data/                       # Local SQLite DB & decision logs
│   ├── medical_history.db      # SQLite database (conversations, messages, users)
│   └── defense_log.jsonl       # Audit log of all defense decisions
├── docs/                       # Benchmark reports & evaluation summaries
├── scripts/                    # Automation & evaluation scripts
│   ├── build_template_corpus.py  # Assembles jailbreak templates from datasets
│   ├── calibrate_defenses.py     # Optimizes detector thresholds
│   ├── index_chroma.py           # Builds local ChromaDB vector store
│   ├── run_benchmark.py          # Runs clinical QA benchmark suite
│   ├── run_defense_eval.py       # Evaluates input defense pipeline
│   └── run_full_pipeline_safety_eval.py  # End-to-end multi-layer evaluation
├── src/                        # Core application source code
│   ├── agents/                 # Clinical agents & prompt generators
│   │   ├── emergency_triage.py # Layer 0: Emergency triage classifier
│   │   ├── guardrails.py       # Layer 1: Deterministic contraindication intercept
│   │   ├── intake_extraction.py# Real-time NER intake extractor
│   │   ├── prompting.py        # Layer C: Hardened prompt builder
│   │   └── verifier.py         # Post-generation faithfulness auditor
│   ├── api/                    # Web API & routing
│   │   └── server.py           # FastAPI application & SSE streaming endpoints
│   ├── auth/                   # Security & JWT user authentication
│   ├── db/                     # SQLModel ORM models and database engine
│   ├── llm/                    # Remote MedGemma streaming client
│   ├── retrieval/              # Hybrid RAG retriever & MedCPT reranker
│   └── security/               # 3-Layer Input Defense Pipeline
│       ├── decision.py         # Decision logic (BLOCK / FLAG / ALLOW)
│       ├── device.py           # Hardware accelerator selector (CUDA/MPS/CPU)
│       ├── logging.py          # Thread-safe defense logger
│       ├── prompt_guard.py     # Layer A: DeBERTa / Prompt Guard classifier
│       ├── sanitize.py         # Layer S: Text sanitizer & control token defanger
│       ├── template_matcher.py # Layer B: Semantic template matcher
│       └── thresholds.json     # Calibrated detection thresholds
├── static/                     # Frontend UI assets
│   ├── index.html              # Main single-page clinical console
│   ├── app.js                  # Frontend state, SSE handling, & security console
│   └── styles.css              # Dark-mode medical UI styles
├── run.sh                      # One-click startup script
├── requirements.txt            # Python dependencies
└── .env.example                # Example environment configuration
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. System Requirements
- **OS**: macOS (Apple Silicon supported via MPS), Linux (Ubuntu 20.04+), or Windows (WSL2 recommended).
- **Python**: Version `3.10` or `3.11`.
- **Hardware**:
  - Local machine: 8 GB+ RAM (loads DeBERTa-v3, MiniLM, and S-PubMedBERT).
  - Remote/Colab GPU: T4, V100, A100, or L4 GPU for running MedGemma (`google/medgemma-4b-it`).

### 2. Environment Setup
Clone the repository and create a Python virtual environment:

```bash
cd MedGemmaLocal
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Start the Project Up

### Step 1: Start the Remote MedGemma Server (Google Colab)
1. Open your Google Colab notebook serving `google/medgemma-4b-it` (using vLLM, HuggingFace TGI, or custom FastAPI wrapper).
2. Start an **ngrok** tunnel exposing the model API port (e.g. port `8000`).
3. Note the generated public URL (e.g., `https://guiding-ashy-campus.ngrok-free.dev`) and the API key configured in the notebook.

### Step 2: Configure Environment Variables (`.env`)
Create your `.env` file at the root of the project by copying `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and configure your credentials:

```ini
# Remote Colab ngrok tunnel endpoint (NO trailing slash, NO /v1 at the end)
COLAB_API_BASE=https://guiding-ashy-campus.ngrok-free.dev

# Must match the API key specified in your Colab notebook
COLAB_API_KEY=change-me-please

# MedGemma model configuration
MODEL_ID=google/medgemma-4b-it
PROMPT_STYLE=gemma
MAX_MODEL_LEN=8192
RESERVE_OUTPUT_TOKENS=640

# Defense pipeline configuration
DEFENSE_ENABLED=1
PROMPT_GUARD_MODEL_ID=ProtectAI/deberta-v3-base-prompt-injection-v2
```

> [!TIP]
> By default, the system uses the ungated, open-weights model `ProtectAI/deberta-v3-base-prompt-injection-v2` for Layer A, which runs locally without requiring Hugging Face token approval. If you wish to use `meta-llama/Llama-Prompt-Guard-2-86M`, set `HF_TOKEN=hf_...` with an account that has accepted the Llama 4 Community License.

### Step 3: Launch Local FastAPI Backend
Run the startup script:

```bash
./run.sh
```

Or run Uvicorn directly from your virtual environment:

```bash
./.venv/bin/uvicorn src.api.server:app --host 127.0.0.1 --port 8600 --reload
```

During startup, the server automatically initializes:
- SQLite database tables (`conversations`, `conversation_messages`, `api_users`).
- Local BM25 and ChromaDB clinical indices.
- Neural model warmups for DeBERTa, S-PubMedBERT, and MedCPT reranker.

### Step 4: Access the Clinical Web Console
Open your browser and navigate to:
```
http://127.0.0.1:8600
```
- Sign up or log in with any username and password (sessions are handled via secure JWTs stored in local storage).
- Check the health chips at the top navigation bar to confirm **Colab MedGemma** and **RAG Engine** display green **Ready** dots.

---

## 🛡️ Security & Audit Console (UI Controls)

Click the **🛡️ Security & Audit** button in the top navigation bar to access the security console:

1. **Defense Decision Logs**:
   - Live stream of all incoming requests and their defense evaluations.
   - Filter records by **All**, **Blocked**, **Flagged**, or **Allowed**.
   - Inspect neural injection scores (`pg_score`), template similarity (`tm_sim`), matched jailbreak components, and defanged sanitization tags.
2. **Pipeline Controls & Ablation**:
   - Toggle individual layers on or off dynamically without restarting the server:
     - *Layer 0*: Emergency Red-Flag Triage
     - *Layer 1*: Adversarial Guardrails
     - *Layer A*: Prompt Guard Neural Classifier
     - *Layer B*: Template Semantic Matcher
     - *Layer C*: Prompt Hardening & Delimiter Nonces
   - Adjust detection thresholds (`pg_block`, `pg_flag`, `tm_block`, `tm_flag`) on the fly.

---

## 🧪 Evaluation & Benchmarking Suites

The repository includes complete test suites for evaluating defense performance and clinical grounding:

### 1. Run Full Multi-Layer Adversarial Defense Evaluation
Evaluates Layer S, Layer A, Layer B, and full combination defense pipelines across 140+ diverse adversarial samples (jailbreaks, injections, benign medical queries):
```bash
python scripts/run_full_pipeline_safety_eval.py
```
*Results are saved to `docs/full_pipeline_safety_eval.json`.*

### 2. Calibrate Thresholds
Calculates optimal thresholds across the jailbreak corpus to maximize detection while keeping false positive rates on genuine medical questions below 1%:
```bash
python scripts/calibrate_defenses.py
```
*Updates `src/security/thresholds.json`.*

### 3. Run Clinical Retrieval Benchmark
Runs automated clinical QA cases against the live MedGemma pipeline to assess grounding faithfulness, triage sensitivity, and response accuracy:
```bash
python scripts/run_benchmark.py
```
*Outputs are saved to `docs/benchmark_results.json`.*

---

## ❓ Troubleshooting & FAQs

#### 1. Why does the top bar say "Colab MedGemma: Offline"?
- Ensure your Google Colab cell running ngrok and the model is currently active.
- If Colab restarted, ngrok assigned a new URL. Copy the new URL from the notebook and paste it into `.env` under `COLAB_API_BASE=https://...`, then restart the server.

#### 2. Chat renders text but seems to drop numbers or dosages?
- This issue was resolved in the SSE event handler. If you experience this, perform a hard refresh (**Cmd + Shift + R** on Mac or **Ctrl + Shift + R** on Windows/Linux) to clear cached JavaScript.

#### 3. How do I clear chat history and defense logs?
- In the UI, click **`+ New`** to start a clean consultation.
- To purge database history and audit logs from disk:
  ```bash
  python3 -c "
  import sqlite3, pathlib
  con = sqlite3.connect('data/medical_history.db')
  cur = con.cursor()
  cur.execute('DELETE FROM conversation_messages; DELETE FROM conversations;')
  con.commit(); cur.execute('VACUUM'); con.close()
  pathlib.Path('data/defense_log.jsonl').write_text('')
  print('Reset complete.')
  "
  ```
- Your user login credentials in `api_users` will remain intact.

#### 4. Can I run the entire pipeline offline without Colab?
- The RAG retrieval (BM25, ChromaDB), Triage, Guardrails, and Input Defense (Layer A, B, S) all run **100% locally**.
- If you have an Apple Silicon Mac (M1/M2/M3 with 16GB+ RAM) or a local NVIDIA GPU, you can point `COLAB_API_BASE` to a local vLLM or Ollama instance serving `google/medgemma-4b-it` or `epfl-llm/meditron-7b`.

---

## 📜 License & Acknowledgments
- Built with **Google MedGemma**, **Hugging Face Transformers**, **Sentence-Transformers**, and **FastAPI**.
- Medical knowledge derived from **StatPearls** (NCBI Bookshelf), **OpenFDA**, and **MedQuAD**.
