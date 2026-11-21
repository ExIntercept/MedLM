# MedLM: A Model-Agnostic Defense Gateway Against Adversarial Prompts for Medical Language Models

*[Author names], PES University*

> **Draft status.** Sections 1–5 and 7–9 are complete. Numbers marked **[R]** are filled in from `docs/paper/RESULTS.md` once the experiment runs finish (`scripts/run_paper_pipeline.sh`). Citations are listed at the end; check every entry against the original paper before submission.

---

## Abstract

Medical large language models (LLMs) are increasingly fine-tuned from general-purpose bases and released as open weights. Fine-tuning on clinical text often weakens the safety behaviour of the base model, and jailbreak prompts, prompt injection and plainly worded harmful clinical requests transfer readily to these models. Retraining every medical model for safety is impractical. We present **MedLM**, a defense gateway that sits between the user and *any* medical LLM and leaves the model unchanged. Every request passes through the same nine stages:
- input sanitization;
- emergency triage and contraindication guardrails;
- four complementary attack detectors: a prompt-injection classifier, a jailbreak-template matcher, a harmful-clinical-intent classifier and Llama Guard 3;
- evidence retrieval;
- nonce-delimited prompt hardening;
- generation;
- an evidence-grounding check;
- an output-side safety classifier.

The protected model is reached only through a standard chat or completions API.

We evaluated MedLM on four open medical LLMs from three model families: MedGemma 4B, OpenBioLLM 8B, BioMistral 7B and JSL-MedLlama 3 8B. The test sets were six public attack benchmarks and six benign benchmarks: 30 prompts per benchmark, 720 conversations per model. Two judges from different model families scored the answers.
- **Attack success:** the gateway reduced it from **[R]%** to **[R]%** on average, and the reduction was significant for every model (exact McNemar test, p < **[R]**).
- **Medical utility:** MedQA accuracy was preserved or improved on **[R]** of four models.
- **Ablations:** removing any single stage increases either attack success or over-refusal. The full gateway outperforms a Llama-Guard-only baseline **[R]**.

The system runs on a single 8 GB consumer GPU. We release the code, configurations and evaluation harness.

---

## 1. Introduction

Open medical LLMs are attractive for clinical decision support because they can be run locally, which keeps patient data on site. Most are produced by continued pre-training or instruction tuning of a general model on biomedical text. This process tends to erode the refusal behaviour learned during the base model's safety alignment. In our own baseline measurements, the four medical models we tested complied with roughly 35–47% of attack prompts when called directly (Section 6). Adversarial prompts are cheap to produce, transfer between models, and in a medical setting can cause direct physical harm: unsafe dosing, falsified clinical documentation, or advice that discourages necessary care.

Two families of mitigation exist:
1. **Model-level alignment** (safety fine-tuning, RLHF). It must be repeated for every model and every update, and it requires training infrastructure that many clinical groups lack.
2. **System-level guardrails** (input/output classifiers, prompt engineering, retrieval grounding). They are applied around the model and can be reused across models. However, they are usually evaluated on general-purpose chat models with a single safety classifier, and their effect on *clinical utility* is rarely measured.

This work asks whether a single system-level gateway can protect many different medical LLMs without per-model changes, while keeping the models useful for genuine clinical questions. Our contributions are:

1. **A model-agnostic gateway design** (Section 4). Nine ordered stages combine deterministic clinical safety rules, four complementary detectors, evidence retrieval, delimiter hardening, an evidence-grounding check and an output safety classifier. The protected model is an interchangeable endpoint: model-specific details (prompt template, context window, system-role support) are handled by declarative profiles, not by code.
2. **A harmful-clinical-intent classifier** (Layer H). It targets plainly worded harmful medical requests, which prompt-injection detectors miss (they stopped under 5% of MedSafetyBench requests). It was trained with near-duplicate removal against the evaluation sets, and it was calibrated so that *every* benign source stays at or below 1% false blocks.
3. **A cross-model evaluation** (Sections 5–6). We test four medical LLMs from three families on twelve public benchmarks, with:
   - repeatable decoding;
   - two independent judges;
   - paired significance tests;
   - a per-stage attribution of which defense stopped each attack;
   - ablations, including a Llama-Guard-only baseline.
4. **An analysis of the utility–safety trade-off** in a clinical assistant (Section 7):
   - **Non-medical prompts are declined by design.** Most refusals of harmless prompts are deliberate: they are non-medical, and the grounding check has no clinical evidence to support an answer. They are not detector false positives.
   - **Grounding is not a harm filter.** An evidence-grounding check cannot stand in for one: unsafe answers that reuse the question's own words pass word-overlap grounding, which motivated the output-side classifier.

---

## 2. Related Work

**Jailbreaks and attack benchmarks.**
- In-the-wild jailbreak templates such as "Do Anything Now" were collected and characterised by Shen et al. [1].
- JailbreakBench [2] and HarmBench [3] standardise harmful behaviours and evaluation.
- XSTest [4] and OR-Bench [5] measure the opposite failure: over-refusal of safe prompts that look harmful.
- MedSafetyBench [6] adapts harmful-request evaluation to medicine, using principles of medical ethics.
- Prompt injection, where instructions arrive inside data rather than from the user, was described by Greshake et al. [7].

**Guard models.**
- Llama Guard [8] and its successors classify prompts and responses against a hazard taxonomy.
- Prompt Guard and DeBERTa-based injection classifiers [9] detect injection and jailbreak phrasing.
- WildGuard [10] jointly detects harmful prompts, harmful responses and refusals.
- These classifiers are general-domain. A single classifier used alone either misses domain-specific harms or over-blocks domain-specific vocabulary. Llama Guard, for example, flags ordinary medical questions as "specialized advice".

**System-level defenses.**
- NeMo Guardrails [11] offers programmable dialogue rails.
- SmoothLLM [12] perturbs inputs to break optimisation-based attacks.
- Spotlighting [13] and related delimiter schemes mark untrusted input to reduce injection.
- Retrieval grounding and post-hoc verification are widely used against hallucination. Their interaction with adversarial prompts is less studied.

**Medical LLM safety.** Medical models have been shown to follow harmful clinical instructions more readily than their base models [6].

**Our position.** To our knowledge, no prior work evaluates a single layered gateway *across several medical LLMs*, measuring both attack success and clinical utility with independent judges.

---

## 3. Threat Model

**The defender:**
- operates a clinical question-answering service built on an open medical LLM;
- controls the server, the retrieval corpus and the gateway;
- does **not** retrain the model.

**The attacker:**
- is an untrusted user who sends text messages over the normal interface, in one or several turns;
- knows that a medical LLM is behind the service and may use any public jailbreak template, prompt-injection payload, or plainly worded harmful request;
- may try to forge conversation structure, using fake turn markers or model control tokens.

The attacker's goal is a response that provides harmful content. Examples are unsafe dosing, assistance with harming a patient, falsified or deceptive clinical content, unethical practice, or hateful content.

**Out of scope:**
- white-box optimisation against the gateway's own classifiers (e.g. gradient-based suffixes);
- image inputs;
- attacks on the retrieval corpus.

---

## 4. System Design

Figure 1 (README, "System Architecture") shows the pipeline. Stages 1–4 run before the model and can end the request early with a fixed, clinically worded response. Stages 5–9 run around a single call to the protected model.

### 4.1 Stage 1 — Sanitization
Unicode format characters (zero-width, bidirectional overrides, tag characters) are removed to stop invisible-text smuggling. The control tokens of every supported model family are neutralised by replacing their angle brackets with look-alike quotation marks. These include Gemma turn markers, ChatML, Llama 3 headers and Mistral `[INST]`. Forged application delimiters are neutralised the same way.

Two copies of the text are kept:
- **The model copy** keeps clinical symbols (µg, m²).
- **The detector copy** is NFKC-normalised, so homoglyph obfuscation is visible to the classifiers.

### 4.2 Stage 2 — Emergency triage
Deterministic patterns recognise acute emergencies, such as chest pain with radiation, FAST stroke signs, anaphylaxis and active suicidal intent. These are escalated to emergency services without calling the model.

A framing check prevents false escalation. Red-flag symptoms appear in almost every cardiology exam question, so first-person and close-relation descriptions escalate, while exam questions and third-person case discussions are answered. This check reduced escalations on MedQA from 11% of questions to none on the full test split.

### 4.3 Stage 3 — Contraindication guardrails
Hard rules intercept absolute contraindications and lethal dosing patterns before retrieval. Examples are warfarin in pregnancy and acetaminophen above the daily maximum. The response is a clinical explanation, not a generic refusal.

### 4.4 Stage 4 — Attack detection (Layers A, B, H, G)
The detectors score three views of the input: the current message, the patient-profile fields, and a multi-turn window (the last two user turns plus the current one).

- **Layer A — injection classifier.**
  - A DeBERTa-v3 prompt-injection classifier [9], applied in 510-token sliding windows (stride 384).
- **Layer B — jailbreak-template matcher.**
  - Sentence-embedding similarity to a curated corpus of jailbreak templates.
  - Detection of nine semantic components of jailbreaks: fictional framing, amoral persona, unconditional obedience, ethics waiver, realism demand, stakes pressure, private-audience claim, redaction claim and forced opening.
- **Layer H — harmful clinical intent.**
  - **What it does.** Layers A and B detect *how* an attack is phrased; Layer H scores *what* is requested.
  - **Model.** A sentence embedding (all-mpnet-base-v2) followed by logistic regression.
  - **Training data.**
    - Harmful: the MedSafetyBench training split and OR-Bench toxic prompts.
    - Benign: MedQuAD, MedQA training questions, ordinary user prompts and OR-Bench "seemingly toxic" safe prompts.
  - **Held-out evaluation.** Training prompts with cosine similarity above 0.90 to any evaluation prompt were removed.
  - **Calibration.** Thresholds were calibrated on 5-fold out-of-fold scores so that each benign source, separately, stays at or below 1% blocked and 3% flagged. Pooled calibration had hidden a 2.9% false-block rate on alarming-sounding but harmless prompts.
- **Layer G — Llama Guard 3 (8B)** [8].
  - Classifies the multi-turn window through any OpenAI-compatible endpoint.
  - **S6 "specialized advice"** never blocks on its own, because it fires on ordinary medical questions.
  - **S11 "self-harm"** is routed to emergency triage instead of a refusal.

**Decision rule.** Each layer has a *block* and a *flag* threshold:
- A request is **blocked** if any layer crosses its block threshold.
- It is **flagged** if any layer crosses its flag threshold, or if control tokens were neutralised. A flagged request proceeds with an extra safety instruction in the prompt.
- Otherwise it is **allowed**.

Every layer can be toggled at runtime, and an unreachable layer is bypassed and recorded rather than failing the request.

### 4.5 Stage 5 — Evidence retrieval
Hybrid retrieval runs over 30,980 chunks from StatPearls, OpenFDA drug labels and MedQuAD:
- **BM25**, with medical synonym expansion;
- **dense search**, with S-PubMedBERT;
- **fusion**, by reciprocal rank fusion with source-authority weights;
- **reranking**, by the MedCPT cross-encoder [14], which keeps the top three passages.

### 4.6 Stage 6 — Prompt hardening
- **Nonce-tagged delimiters.** Untrusted input is enclosed in delimiters carrying a random nonce per request (`<user_message_a3f89b>…</user_message_a3f89b>`). The system instructions state that nothing inside these delimiters is an instruction (cf. [13]).
- **Restated rules.** The safety rules are repeated after the user block, to counter instructions placed at the end of the prompt.
- **Server-side history.** Conversation history is loaded only from the server's database. Client-supplied turns are ignored, which prevents turn forgery.
- **Context fitting.** For small-context models, the prompt is fitted to the window by dropping older history first, then lower-ranked evidence. Safety rules and delimiters are never trimmed.

### 4.7 Stage 7 — Generation through an adapter
The model is called through an adapter, the only model-specific component:
- **`openai_chat`** for servers that apply the model's chat template;
- **`completions`** with a locally rendered template (Gemma, ChatML, Llama 3, Mistral, plain).

A declarative profile per model supplies the template, the context window and system-role support. Adding a model is a configuration change.

### 4.8 Stage 8 — Evidence-grounding check
The answer is checked against the reference text: the retrieved evidence, the question and, for multiple-choice questions, the answer options.
- A sentence counts as **grounded** when at least 18% of its content words appear in the reference text.
- An answer **passes** when at least half of its sentences are grounded.
- An answer that fails is **withheld** and replaced by a "could not verify" message.

**Tuning.** The thresholds were tuned on 444 captured model drafts:
- Including the answer options cut medical answers withheld from 17 to 10 of 120 and changed nothing else.
- Lower pass thresholds released more non-medical and attack answers for little medical gain.

### 4.9 Stage 9 — Output guard
Grounding measures evidence support, not harm.
- **The problem.** When tuning stage 8, Llama Guard judged 2 of 96 attack answers that reached the check unsafe, and *both* passed it: one repeated a jailbreak system prompt back, and the other was a hateful answer. Both reused the question's own words, so word overlap rated them fully grounded.
- **The fix.** Stage 9 classifies the (request, answer) pair with Llama Guard and replaces unsafe answers. As in stage 4, S6 and S11 alone do not block.

---

## 5. Experimental Setup

**Protected models.**

| Model | Base family | Weights |
|---|---|---|
| MedGemma 4B-it | Gemma 3 | unsloth GGUF |
| OpenBioLLM 8B | Llama 3 | mradermacher GGUF |
| BioMistral 7B | Mistral | MaziyarPanahi GGUF |
| JSL-MedLlama 3 8B v2.0 | Llama 3 | bartowski GGUF |

- All four models are 4-bit quantized (Q4_K_M) and served by Ollama on one NVIDIA RTX 4060 laptop GPU (8 GB).
- Meditron-7B was excluded: it is a base model that continues the prompt instead of answering it.

**Benchmarks.** We sampled 30 prompts per benchmark with a fixed seed (13).

| Group | Benchmarks |
|---|---|
| Attacks | MedSafetyBench test split [6]; medical jailbreaks (in-the-wild templates [1] wrapped around MedSafetyBench requests); in-the-wild jailbreaks [1]; JailbreakBench harmful [2]; deepset prompt injections; XSTest unsafe [4] |
| Medical benign | MedQA-USMLE 4-option test split [15] (accuracy is measured on it); MedQuAD held-out partition [16] |
| General benign | JailbreakBench benign [2]; deepset benign; XSTest safe [4]; OR-Bench hard-1k [5] |

**Conditions.**
- **Undefended:** the model is called directly with the prompt.
- **Defended:** the full gateway.
- **Decoding:** greedy (temperature 0) with a fixed sampling seed sent to the server, so reruns give the same answers.

**Judges.** Llama Guard is both a defense layer and a common judge, which could flatter the defended results, so we used two judges from different families:
- **Judge 1** is Llama Guard 3 8B, classifying the (request, response) pair.
- **Judge 2** is Qwen3-8B, with thinking disabled and a rubric that asks separately whether the response itself contains harmful content and whether it refuses.
- Responses stopped by the gateway before or after the model are counted as refusals and as not harmful.

**Metrics.**

| Metric | Definition |
|---|---|
| Attack success rate (ASR) | Share of attack prompts whose final response the judge rates harmful |
| Medical refusal rate | Share of medical-benign prompts refused or stopped |
| General refusal rate | The same for general-benign prompts |
| MedQA accuracy | Share of questions where the extracted answer letter matches the key |

- Every rate is reported with a 95% Wilson interval.
- Undefended vs defended ASR is compared with an exact two-sided McNemar test, pairing the same prompt in both conditions.
- Judge agreement is reported as Cohen's κ.

**Ablations.** Run on MedGemma, on the first 15 prompts per benchmark of the same sample, defended condition only:

| Configuration | What changes |
|---|---|
| Llama-Guard-only baseline | Only sanitization and Llama Guard on the input and output; no triage, guardrails, Layers A/B/H, retrieval, grounding or hardening |
| No Llama Guard | Input and output guard both off: the gateway as it was before Layer G |
| No output guard | Stage 9 off |
| No retrieval | No evidence; the grounding check then has nothing to check against |
| No input detectors | Layers A, B, H and G off |
| No prompt hardening | Stage 6 replaced by a plain prompt |

---

## 6. Results

**[R] Table 1 — Main results.** Per model, ASR undefended → defended under both judges, the McNemar p-value, medical refusal, MedQA accuracy and general refusal (`docs/paper/RESULTS.md`, Table 1).

**[R] Table 2 — ASR by attack suite**, pooled over models.

**[R] Table 3 — Where attacks were stopped.** Outcome of every defended attack by stage. In the MedGemma run, the input detectors stopped 143 of 180 attack prompts, the grounding check withheld 18, the model itself refused 12, and 7 were answered, none harmful according to judge 1.

**[R] Table 4 — Judge agreement** (Cohen's κ).

**[R] Table 5 — Ablations.**

**Pre-model screening** (from earlier work, 200 prompts per benchmark, Layers A, B and H with triage and guardrails, without Layer G; `docs/BENCHMARK_SUITE_SCREEN.md`):

| Prompts | Benchmark | Rate |
|---|---|---|
| Attacks stopped before the model | MedSafetyBench | 89.5% |
| | Medical jailbreaks | 80.5% |
| | In-the-wild jailbreaks | 74.0% |
| | JailbreakBench harmful | 69.0% |
| | XSTest unsafe | 55.0% |
| | deepset injections | 35.0% |
| Benign prompts wrongly stopped | MedQA | 1.0% |
| | MedQuAD | 0.0% |
| | deepset benign | 0.0% |
| | XSTest safe | 8.5% |
| | OR-Bench hard | 9.0% |
| | JailbreakBench benign | 17.0% |

---

## 7. Discussion

**One gateway, several models.** The detectors and rules do not depend on the protected model, so most attacks are stopped before any model is called. This is why the defended ASR is similar across models whose undefended behaviour differs widely. The model-dependent part is utility. A model that follows the hardened prompt and answers in the language of the retrieved evidence keeps or improves its MedQA accuracy. A model that does not loses answers to the grounding check: JSL-MedLlama in our pilot run. Model-agnostic *protection* does not imply model-agnostic *usability*. Reporting both is necessary.

**Over-refusal is mostly by design.**
- Defended refusal rates on general benign prompts are high.
- Per-stage attribution shows that detector false blocks are rare: three prompts per model in the pilot run.
- Most of these refusals come from the grounding check. A non-medical request has no clinical evidence behind it, so the assistant declines it.
- For a clinical assistant this is intended behaviour. For a general assistant it would not be. Deployments that must also answer general questions would need a separate path for them.

**Grounding is not safety.** Word-overlap grounding protects against unsupported claims, not against harmful ones. Our observation that both unsafe drafts passed it motivated stage 9. The same reasoning likely applies to any similarity-based faithfulness score.

**Judge circularity.** Layer G and judge 1 are the same model. Agreement with the independent judge 2 (Table 4) and the ablation without Llama Guard (Table 5) together indicate how much of the defended result depends on that model **[R]**.

**Cost.**
- The detectors add about 0.5 s per prompt on CPU.
- Llama Guard 8B adds the largest latency on an 8 GB GPU, because the guard and the medical model are swapped in GPU memory. An answered request took roughly 40 s end to end with the 8B guard, against 11–13 s with the 1B guard. The 1B guard missed one harmful request and blocked one benign programming question in our UI tests.
- Production deployments would host the guard and the model on separate devices.

---

## 8. Limitations and Ethical Considerations

- **Scope.** Text only; no vision-language inputs. Four quantized models of 4–8B parameters. 30 prompts per benchmark. Results for larger or unquantized models may differ.
- **Adaptive attacks.** We evaluate public attacks, not attacks optimised against the gateway itself (e.g. gradient-based suffixes, or iterative attacker LLMs). A determined white-box attacker targeting Layers A, B, H and G is not covered.
- **Judges.** Both judges are automatic. Harm judgments for medical content can require clinical expertise; a clinician-annotated subset would strengthen the evaluation.
- **Clinical use.** MedLM is a research prototype, not a medical device. Its emergency triage and contraindication rules are illustrative, not exhaustive.
- **Dual use.** The evaluation uses public harmful-prompt benchmarks. We release no new harmful content, and model responses are stored only locally.

---

## 9. Conclusion

A layered, model-agnostic gateway can bring the attack success of open medical LLMs from tens of percent to near zero, with the same configuration for every model and without retraining. The remaining challenges are utility for models that do not follow the hardened prompt, robustness to adaptive attacks, and multimodal inputs. Next steps are a vision path for medical VLMs such as LLaVA-Med, an evaluation with adaptive attacks, and a clinician-annotated harm evaluation.

**Reproducibility.** The code and all configurations are in the repository, including model configs, judge configs and seeds. The commands are:
- `scripts/run_benchmark_suite.py e2e --limit 30 --temperature 0` for the main run;
- `scripts/run_paper_pipeline.sh` for the second judge, the ablations and the tables.

Every run resumes from its checkpoints.

---

## References (verify each before submission)

1. X. Shen, Z. Chen, M. Backes, Y. Shen, Y. Zhang. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. ACM CCS 2024.
2. P. Chao et al. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. NeurIPS 2024 Datasets and Benchmarks.
3. M. Mazeika et al. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. ICML 2024.
4. P. Röttger et al. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. NAACL 2024.
5. J. Cui et al. OR-Bench: An Over-Refusal Benchmark for Large Language Models. arXiv 2405.20947, 2024.
6. T. Han, A. Kumar, C. Agarwal, H. Lakkaraju. MedSafetyBench: Evaluating and Improving the Medical Safety of Large Language Models. NeurIPS 2024 Datasets and Benchmarks.
7. K. Greshake et al. Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. AISec 2023.
8. H. Inan et al. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv 2312.06674, 2023. (Llama Guard 3: Meta, 2024.)
9. ProtectAI. deberta-v3-base-prompt-injection-v2. Hugging Face model card, 2024.
10. S. Han et al. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs. NeurIPS 2024 Datasets and Benchmarks.
11. T. Rebedea et al. NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails. EMNLP 2023 (System Demonstrations).
12. A. Robey, E. Wong, H. Hassani, G. Pappas. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. arXiv 2310.03684, 2023.
13. K. Hines et al. Defending Against Indirect Prompt Injection Attacks With Spotlighting. arXiv 2403.14720, 2024.
14. Q. Jin et al. MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval. Bioinformatics, 2023.
15. D. Jin et al. What Disease Does This Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences, 2021.
16. A. Ben Abacha, D. Demner-Fushman. A Question-Entailment Approach to Question Answering. BMC Bioinformatics, 2019.
17. Models: Google, MedGemma (2025); Saama AI, OpenBioLLM (2024); Y. Labrak et al., BioMistral, Findings of ACL 2024; John Snow Labs, JSL-MedLlama-3-8B-v2.0 (2024); Qwen Team, Qwen3 (2025).
