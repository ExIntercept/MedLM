# MedLM: Model-Agnostic Adversarial Defense for Medical LLMs

MedLM is a defense gateway that sits between users and a medical language model. It protects the model against jailbreaks, prompt injection and harmful requests without retraining or changing the model. Every request goes through the same 8 stages:
- input sanitization, emergency triage and contraindication guardrails;
- four attack detectors (Layers A, B, H and G);
- evidence retrieval (**BM25 + S-PubMedBERT + MedCPT**) and a hardened prompt;
- a check that the answer is backed by the retrieved evidence.

The model itself is just an endpoint, reached through an OpenAI-compatible chat API or a plain completions API.

**Headline result.** We tested four medical models from three model families (MedGemma, OpenBioLLM, BioMistral and JSL-MedLlama), running locally through Ollama:
- **Without the gateway**, attacks succeeded on 36–47% of attempts.
- **With the gateway**, they succeeded on **0%** for every model.
- MedQA accuracy was kept, except for one model; see [Results across medical models](#-results-across-medical-models).

**Scope: text only.** The gateway accepts and checks text. It has no image input, so it does not defend vision-language (VLM) use.

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
   - [Running everything locally with Ollama](#running-everything-locally-with-ollama)
6. [Security & Audit Console (UI Controls)](#-security--audit-console-ui-controls)
7. [Evaluation & Benchmarking Suites](#-evaluation--benchmarking-suites)
8. [Results across medical models](#-results-across-medical-models)
9. [Limitations](#-limitations)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)

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
  │   │            - Layer H: Harmful-Intent Classifier (embedding + logistic regression)              │   │
  │   │            - Layer G: Llama Guard 3 input moderation (any OpenAI-compatible endpoint)          │   │
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
- **Framing check**: Red-flag symptoms also appear in nearly every cardiology vignette. Without this check, 11% of MedQA questions got the emergency warning instead of an answer. A red flag is let through to the model only when the text is clearly a case discussion and nobody describes themselves or someone close to them. Case discussions are:
  - an exam-style question, in any mode,
  - a third-person vignette ("A 65-year-old man presents…"), in clinician mode only,
  - a short topic question ("What is aphasia?").
- **Anything personal still escalates**: "I", "my father", "our baby", "someone near me" or "right now". Quoted speech inside a vignette and Roman-numeral "I" (ECG lead I, type I diabetes) are ignored. In patient mode, a third-person description without exam framing still escalates. Regression cases are in `tests/test_emergency_triage_framing.py`.

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
- **Layer H (Harmful-Intent Classifier)** — *File: `src/security/harm_classifier.py`*:
  - Layers A and B detect *how* an attack is phrased (personas, overrides, injected delimiters). A plainly worded harmful request, such as "write a justification for prescribing a cancer drug for a cold", has none of that. On the public benchmark, Layers A and B stopped under 5% of MedSafetyBench requests.
  - Layer H scores *what* is being asked for: a sentence embedding followed by logistic regression, trained by `scripts/train_harm_classifier.py`.
  - Harmful training examples: the MedSafetyBench training split and OR-Bench toxic prompts. Benign training examples: MedQuAD, the MedQA training split, ordinary user prompts, and OR-Bench's seemingly-toxic-but-safe prompts.
  - Any training prompt with cosine similarity above 0.90 to an evaluation prompt is removed, so the benchmarks stay held out.
  - Block and flag thresholds are calibrated on out-of-fold scores so that every benign source (including harmless but alarming-sounding prompts) stays at or below 1% (block) and 3% (flag) false positives. The weights are plain arrays in `harm_classifier.npz`, loaded without pickle.
  - Long prompts are scored in overlapping windows, because jailbreaks often put the harmful request at the end.
  - Blocks get a clinical refusal (`HARMFUL_REQUEST_MESSAGE`) that redirects to safe care or ethics channels.
- **Layer G (Llama Guard input moderation)** — *File: `src/security/llama_guard.py`*:
  - Layer H was trained on harmful *clinical* requests. Plainly worded requests to hurt someone or to make a dangerous substance got past Layers A, B and H in UI testing. Llama Guard 3 is trained on exactly these hazard categories.
  - Llama Guard classifies the multi-turn window through any OpenAI-compatible chat endpoint. Configure it with `GUARD_BASE_URL`, `GUARD_MODEL` and optionally `GUARD_API_KEY`; with Ollama, use `http://localhost:11434` and `llama-guard3:8b`. If these are unset, the layer is off.
  - Two categories never block on their own:
    - **S6 "specialized advice"** fires on ordinary medical questions, which this product exists to answer.
    - **S11 "self-harm"** is left to emergency triage, which points the user to crisis support instead of refusing.
  - **Which size to use.** On an 8 GB GPU:
    - **`llama-guard3:8b`** classified every UI test prompt correctly. Recommended for evaluation.
    - **`llama-guard3:1b`** is about 3× faster, but missed one harmful request and blocked one harmless programming question.
  - It can be switched off at runtime with the `layer_llama_guard` toggle.
- **Decision Engine**:
  - `BLOCK`: If `pg_score >= 0.90`, `tm_max_sim >= 0.80`, `tm_n_components >= 3`, or `harm_score >= harm_block`. Halts the pipeline with a refusal matched to the category (`JAILBREAK`, `INJECTION`, `HARMFUL_REQUEST`).
  - `FLAG`: If `pg_score >= 0.50`, `tm_max_sim >= 0.70`, `tm_n_components >= 2`, `harm_score >= harm_flag`, or control tokens were defanged. Proceeds to generation with hardened system constraints.
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

### Stage 7: Model-Agnostic Inference (any LLM)
*Files: `src/llm/adapters.py`, `src/security/gateway.py`*
- The protected model is reached only through an `LLMAdapter`. The defense stages never depend on which model is behind it.
- `openai_chat` sends role-based messages to any OpenAI-compatible `/v1/chat/completions` server (vLLM, TGI, Ollama, LM Studio, llama.cpp, OpenAI), and the server applies the model's own chat template.
- `completions` renders the template locally (`gemma`, `chatml`, `llama3`, `mistral`, `plain`) for base models or servers without a chat endpoint. Control tokens from every model family are neutralised in all message content before rendering.
- `CallableAdapter` wraps any Python function (a local Hugging Face pipeline, a vendor SDK, or a test stub).

### Stage 8: Post-Generation Evidence Grounding Audit
*File: `src/agents/verifier.py`*
- Evaluates the generated clinical response against the retrieved evidence passages.
- Computes claim overlap and emits a **Faithfulness / Grounding Score** (0–100%) displayed directly in the chat UI with a badge indicating `PASS` or `FLAGGED`.
- How the score is computed:
  - A sentence counts as grounded when at least 18% of its content words appear in the reference text (`OVERLAP_THRESHOLD`).
  - An answer passes when at least 50% of its sentences are grounded (`PASS_THRESHOLD`).
  - The reference text is the retrieved evidence, the question, and the answer options of a multiple-choice question. The options are included because naming the chosen option only restates the question.
- The answer is held until this check finishes. A `FLAGGED` answer is never shown. The user receives a clinical "could not verify" message instead.
- **Tuning.** `scripts/tune_grounding.py` saves real model drafts, has Llama Guard judge the attack drafts, and replays the drafts against alternative thresholds.
  - Adding the answer options cut medical answers withheld from 17 to 10 of 120, and changed nothing for the other prompts.
  - Lower pass thresholds released more non-medical and attack answers for little gain on medical ones, so the threshold stays at 0.50.
- **What it is not.** It checks that an answer is supported by evidence; it is not a harm filter. Of 96 attack drafts that reached it, Llama Guard judged 2 unsafe, and both passed the check, because they reused the question's own words.

### Using the gateway directly
`ClinicalDefenseGateway` runs stages 0–8 around any adapter and does not depend on FastAPI, so evaluation scripts can compare models under identical defenses:

```python
from src.llm.adapters import OpenAIChatAdapter, CallableAdapter
from src.retrieval.retriever import retrieve_context
from src.security.gateway import ClinicalDefenseGateway

llm = OpenAIChatAdapter("http://localhost:11434", "llama3.1")          # or any adapter
gateway = ClinicalDefenseGateway(llm, retriever=lambda q: retrieve_context(q, top_k=3))
result = gateway.run("How is asthma managed?", mode="clinician")
print(result.status, result.trace, result.response_text)
```

### Supported medical models
*File: `src/llm/model_profiles.py`*

The gateway identifies the configured medical model from its ID and adapts automatically. Only the model-specific details change; the defense stages are the same for every model.

| Model | Prompt format | Context | Notes |
|---|---|---|---|
| MedGemma (`google/medgemma-*`) | gemma | 8192 | Instruction-tuned |
| Meditron-7B / 70B (`epfl-llm/meditron-*`) | plain | 2048 / 4096 | Base model, no system role |
| BioMistral-7B | mistral | 2048 | No system role |
| OpenBioLLM 8B / 70B | llama3 | 8192 | Instruction-tuned |
| Med42-v2 (`m42-health/Llama3-Med42-*`) | llama3 | 8192 | Instruction-tuned |
| PMC-LLaMA-13B | plain | 2048 | Base model |
| ClinicalCamel-70B | plain | 4096 | |
| HuatuoGPT-o1 (Qwen / Llama 3.1) | chatml / llama3 | 32768 | |

For each model the gateway:
- **Fits the context window.** For small-context models like Meditron or BioMistral (2048 tokens), it drops older history first, then lower-ranked evidence, then shortens evidence. The safety rules and delimiter tags are never trimmed.
- **Cleans the output.** It removes leftover control tokens and echoed delimiter tags. For base models, it also cuts off any fabricated next turn (for example `User: ...`).
- **Uses the right format.** It applies the correct prompt template and stop tokens, and merges the system prompt into the user message for models that do not accept a system role.

To add an unlisted model, set `LLM_TEMPLATE` and `MAX_MODEL_LEN` in `.env`, or add one line to `MEDICAL_MODEL_PROFILES`.

### Cross-model defense evaluation
```bash
python scripts/run_cross_model_eval.py --config scripts/cross_model_config.example.json
```
This runs the adversarial and benign battery against each listed medical model twice: once **undefended** (raw model) and once **defended** (through the gateway). It writes `docs/CROSS_MODEL_EVAL_REPORT.md` with a per-model comparison. `--stub` runs the same pipeline with a fake model and needs no GPU.

#### Running it on Colab (one GPU, models in turn)
`colab/cross_model_eval_server.ipynb` serves one medical model at a time behind a fixed tunnel. Your laptop runs the gateway and the benchmark. The Llama Guard 3 1B judge is optional and comes last (cell 9), because Meta approves access to it by hand.

1. In Colab, choose an A100 or L4 runtime if available and add the secret `HF_TOKEN` (the tunnel is a Cloudflare quick tunnel and needs no account). The free T4 also works: the notebook then loads the 7–8B models in 4-bit (bitsandbytes) with a 4096-token context, so note that when reporting their results.
2. Run cells 1–5, then copy the printed `COLAB_API_BASE` and `COLAB_API_KEY` lines into `.env`.
3. In cell 6, set `MODEL` and run the cell. When it prints *Ready*, run the command it prints on your laptop. With the judge running, that is:
   ```bash
   python scripts/run_benchmark_suite.py e2e --config scripts/cross_model_colab.json \
       --judge-config scripts/judge_colab.json --limit 30 \
       --suites medsafetybench medical_jailbreak jbb_harmful deepset_injection medqa medquad xstest_safe
   ```
4. Change `MODEL` in cell 6, run it again, and re-run the same command.
5. Once Llama Guard access is approved, run cell 9 and add the printed `COLAB_JUDGE_BASE` line to `.env`.
   - Without the judge, a refusal-phrase heuristic is used, which does not measure harm.
   - Results are kept separately per judge, so re-run any model you tested before the judge was up.

How the runner handles the switching:
- It tests only the model the endpoint is serving now.
- Finished models are kept as checkpoints in `data/benchmarks/runs/e2e_*.jsonl`.
- Each run rewrites `docs/BENCHMARK_SUITE_E2E.md` with every model done so far.
- If a run is interrupted, re-running the command resumes where it stopped.

`scripts/cross_model_colab.json` covers seven models:

| Model | Family | Prompt format | Kind |
|---|---|---|---|
| MedGemma 4B | Gemma 3 | gemma | Medical, instruction-tuned |
| OpenBioLLM 8B | Llama 3 | llama3 | Medical, instruction-tuned |
| Med42-v2 8B | Llama 3 | llama3 | Medical, instruction-tuned |
| BioMistral 7B | Mistral | mistral | Medical, no system role |
| Meditron 7B | Llama 2 | plain | Medical base model (not instruction-tuned) |
| HuatuoGPT-o1 7B | Qwen 2.5 | chatml | Medical reasoning model |
| Llama 3.1 8B Instruct | Llama 3 | llama3 | General-purpose control |

Expect about 30–60 minutes per model at `--limit 30`.

### Public benchmark suite
The defense is measured on public datasets the project did not write:

| Suite | Type | Source |
|---|---|---|
| `medsafetybench` | Harmful medical requests (900) | MedSafetyBench, NeurIPS 2024 D&B (test split) |
| `medical_jailbreak` | Real jailbreak template + MedSafetyBench request | In-the-wild jailbreaks (Shen et al., CCS 2024) × MedSafetyBench |
| `in_the_wild` | Jailbreak templates (1,405) | Shen et al., CCS 2024 |
| `jbb_harmful` / `jbb_benign` | Harmful and matched benign behaviors | JailbreakBench |
| `deepset_injection` / `deepset_benign` | Prompt injections and controls | deepset/prompt-injections (test split) |
| `xstest_unsafe` / `xstest_safe` | Over-refusal probes | XSTest |
| `medqa` | USMLE questions (clinician) | MedQA-USMLE 4-option (test split) |
| `medquad` | Consumer health questions (patient) | MedQuAD (held-out 75% partition) |
| `orbench_hard` | Seemingly toxic but safe prompts | OR-Bench hard-1k |

```bash
python scripts/fetch_benchmarks.py                       # data files only -> data/benchmarks/raw (gitignored)
python scripts/run_benchmark_suite.py screen --limit 200 # pre-model stages only, no GPU
python scripts/run_benchmark_suite.py screen --limit 200 --ablate   # each detector layer alone
python scripts/run_benchmark_suite.py e2e --config scripts/cross_model_config.example.json \
    --limit 25 --judge-config judge.json                 # undefended vs defended per medical model
```

- **`screen`** reports, per suite, the share of attacks stopped or flagged before reaching any model, and the share of benign questions wrongly stopped. Every rate includes a 95% Wilson interval.
- **`e2e`** reports, per medical model, the attack success rate, the benign refusal rate and MedQA accuracy, undefended vs defended.
- **Judge.** For `e2e`, serve Llama Guard 3 behind an OpenAI-compatible endpoint and pass it with `--judge-config`. Without a judge, a refusal-phrase heuristic is used, which measures non-refusal only, not harm.
- **Training-data overlap.** Suites marked `*` in the reports overlap the published training data of the Prompt Guard classifier (XSTest, and jailbreakchat-derived prompts). Report them as in-distribution, not held-out.

#### Effect of Layer H (screening, 200 prompts per suite, seed 13)

| Suite | Before Layer H | Layer H, pooled calibration | Layer H, worst-source calibration (current) |
|---|---|---|---|
| medsafetybench (harmful medical) | 4.5% | 92.0% | **89.5%** |
| medical_jailbreak* | 74.5% | 80.0% | 80.5% |
| jbb_harmful | 1.0% | 76.0% | 69.0% |
| xstest_unsafe* | 1.0% | 62.0% | 55.0% |
| medqa (benign, wrongly stopped) | 17.0% | 17.0% | 17.0% (all emergency triage, none from Layer H) |
| medquad (benign) | 0.0% | 1.0% | 1.0% (emergency triage) |
| jbb_benign (benign) | 2.0% | 19.0% | 17.0% |
| xstest_safe* (benign) | 0.5% | 14.0% | 9.0% |
| orbench_hard (benign) | n/a | 15.5% | 10.0% |

After the triage framing check (detector layers unchanged, attack rates unchanged), the benign rates are:
- medqa **1.0%** (2 contraindication-guardrail intercepts, 0 emergency escalations)
- medquad **0.0%**
- xstest_safe 8.5%
- orbench_hard 9.0%
- jbb_benign 17.0%

On the full datasets, no MedQA question (of 1,273) and no MedQuAD question (of 11,285) is escalated. Before the fix, 142 and 13 were.

The first calibration held benign false positives at 1% *pooled* over all benign sources. Easy negatives (MedQA vignettes score near zero) hid a 2.9% rate on harmless prompts that only sound alarming. The current thresholds come from 5-fold out-of-fold scores and keep *every* benign source at or below 1% (block) and 3% (flag). 8,000 OR-Bench hard negatives are used instead of 3,000.

Layer H does not wrongly stop medical questions. It still over-refuses some general-domain prompts that sound harmful but are not (9–17% on the held-out over-refusal suites, which are harder than the validation data). It adds about 0.5 s per prompt on CPU (all-mpnet-base-v2). Full report: `docs/BENCHMARK_SUITE_SCREEN.md`.

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
├── run.sh                      # One-click startup script (macOS/Linux)
├── run.ps1                     # One-click startup script (Windows PowerShell)
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
- The React clinical console is served when `frontend/dist` exists. Build it first with
  `npm install` and `npm run build` in `frontend/`. If it has not been built, the
  server falls back to the legacy static console.
- The console is designed for active clinical workflows rather than passive reporting:
  authenticated consultation threads, patient intake, verified consultations,
  medication safety checks, evidence search, and evaluation results are available
  from the navigation tabs.
- Sign up or log in with any username and password (sessions are handled via secure JWTs stored in local storage).
- Check the health chips at the top navigation bar to confirm **Colab MedGemma** and **RAG Engine** display green **Ready** dots.

On Windows PowerShell, use:
```powershell
.\run.ps1
```

Set `ADMIN_USERNAMES` in `.env` to a comma-separated list of usernames permitted to
view defense logs and change defense configuration. These endpoints require both a
valid JWT and membership in that allowlist.

### Running everything locally with Ollama
One consumer GPU is enough. The cross-model results below come from an RTX 4060 with 8 GB of memory. Ollama swaps the medical model and Llama Guard in and out of GPU memory as needed.

1. Install [Ollama](https://ollama.com) and pull a medical model and the guard:
   ```bash
   ollama pull hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M
   ollama pull llama-guard3:8b
   ```
2. In `.env`, point the gateway at Ollama:
   ```ini
   LLM_PROVIDER=openai_chat
   LLM_BASE_URL=http://localhost:11434
   LLM_API_KEY=ollama
   LLM_MODEL=hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M
   GUARD_BASE_URL=http://localhost:11434
   GUARD_MODEL=llama-guard3:8b
   ```
   `LLM_API_KEY` can be any value, because Ollama ignores it. OpenBioLLM's Ollama build has no chat template, so serve it with `LLM_PROVIDER=completions` and `LLM_TEMPLATE=llama3` instead.
3. Start the server as in Step 3.
4. To reproduce the cross-model benchmark:
   - `scripts/cross_model_local.json` lists the four models.
   - `scripts/judge_local.json` uses Llama Guard 8B as the judge.
   - Interrupted runs resume from their checkpoints.

   ```bash
   python scripts/fetch_benchmarks.py
   python scripts/run_benchmark_suite.py e2e --config scripts/cross_model_local.json \
       --judge-config scripts/judge_local.json --limit 15 --out docs/e2e_local
   ```
   This takes several hours for four models. Each model's answers are generated first, then judged in one pass, so the GPU swaps models as few times as possible.

On Windows, start Ollama from its desktop app rather than with `ollama serve`. A server started from a terminal stops when that terminal closes.

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

## 📊 Results across medical models

These results compare each model **undefended** (called directly) and **defended** (through the gateway with all layers on, including Llama Guard 8B).

**Setup:**
- Four medical models, quantized to Q4_K_M, served by Ollama on one 8 GB GPU.
- 15 prompts per benchmark suite, seed 13.
- The judge is Llama Guard 3 8B. An attack counts as successful when the judge rates the answer unsafe.
- Each figure is pooled over its suites, with a 95% Wilson interval.

Full per-suite tables: [`docs/e2e_local/BENCHMARK_SUITE_E2E.md`](docs/e2e_local/BENCHMARK_SUITE_E2E.md).

| Model (base family) | Attack success: undefended → defended | Medical questions refused | MedQA accuracy | General prompts refused |
|---|---|---|---|---|
| MedGemma 4B (Gemma 3) | 46.7% → **0.0%** [0–4] | 0.0% → 3.3% | 40.0% → 46.7% | 11.7% → 78.3% |
| OpenBioLLM 8B (Llama 3) | 35.6% → **0.0%** [0–4] | 0.0% → 0.0% | 20.0% → 26.7% | 0.0% → 50.0% |
| BioMistral 7B (Mistral) | 40.0% → **0.0%** [0–4] | 0.0% → 13.3% | 13.3% → 20.0% | 0.0% → 36.7% |
| JSL-MedLlama 3 8B (Llama 3) | 41.1% → **0.0%** [0–4] | 0.0% → 60.0% | 66.7% → 13.3% | 10.0% → 88.3% |

- **Attack suites:** MedSafetyBench, medical jailbreaks, in-the-wild jailbreaks, JailbreakBench harmful, deepset injections and XSTest unsafe. That is 90 attacks per model.
- **Medical questions:** MedQA and MedQuAD.
- **General prompts:** harmless but mostly non-medical prompts: JailbreakBench benign, deepset benign, XSTest safe and OR-Bench hard.

**What the results show:**
- **The same gateway stops attacks on every model.** No code or prompt was changed per model. Without the gateway, a third to a half of attacks succeeded.
- **Medical questions mostly get through.** On three of the four models, at most 13% of medical questions were refused, and MedQA accuracy stayed the same or improved.
- **Most refusals of harmless prompts are deliberate.** The detectors wrongly blocked only 3 general prompts per model. Most of the rest were withheld by the grounding check: a non-medical question has no clinical evidence to support its answer, and a clinical assistant is meant to decline it.
- **JSL-MedLlama is the exception.**
  - Its answers often failed the grounding check.
  - It sometimes replied that it needed more information instead of answering.
  - Its defended MedQA accuracy fell from 66.7% to 13.3%.

  The gateway protects every model, but how usable the result is still depends on how well the model follows the hardened prompt.

---

## ⚠️ Limitations

- **Text only.** There is no image input, so vision-language attacks (for example instructions hidden in an image) are not covered.
- **Llama Guard plays two roles.** The same Llama Guard 8B model is both a defense layer (G) and the judge, which probably makes defended attack success look a little lower than it is. An independent second judge would remove this bias.
- **Small samples, varying answers.**
  - Each suite has 15 prompts per model, so intervals are wide.
  - Models are called at temperature 0.2 with no fixed seed, so repeated runs give somewhat different answers.
  - Read differences between models as patterns, not exact rates.
- **The output is not checked for harm.** Llama Guard checks the user's message, and the grounding check measures evidence support, not harm. A harmful answer built from the question's own words can pass both. Running Llama Guard on the model's answer as well would close this gap.
- **Non-medical prompts are declined by design.** A high refusal rate on general prompts is expected from a clinical assistant. It would be a problem for a general-purpose chatbot.
- **Models tested.**
  - Only 4–8B quantized models were tested.
  - Meditron-7B was left out: it is a base model that continues the prompt instead of answering it.
  - Med42 and HuatuoGPT are configured in `scripts/cross_model_colab.json` but were not run.

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
- Yes. With a local NVIDIA GPU (8 GB is enough) or an Apple Silicon Mac with 16 GB or more, follow [Running everything locally with Ollama](#running-everything-locally-with-ollama).

---

## 📜 License & Acknowledgments
- Built with **Google MedGemma**, **Hugging Face Transformers**, **Sentence-Transformers**, and **FastAPI**.
- Medical knowledge derived from **StatPearls** (NCBI Bookshelf), **OpenFDA**, and **MedQuAD**.
