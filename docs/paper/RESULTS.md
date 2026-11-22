# Results

## Table 1. Main results (percent, 95% Wilson interval)

| Model | Attack success, judge 1 (undef / def) | McNemar p | Attack success, judge 2 (undef / def) | Medical refused (undef / def) | MedQA accuracy (undef / def) | General refused (undef / def) |
|---|---|---|---|---|---|---|
| JSL-MedLlama 3 8B | 14.4 [10-20] / 0.0 [0-2] | 3.0e-08 | 11.7 [8-17] / 0.0 [0-2] | 1.7 [0-9] / 63.3 [51-74] | 70.0 [52-83] / 6.7 [2-21] | 10.8 [6-18] / 78.3 [70-85] |
| BioMistral 7B | 31.7 [25-39] / 0.0 [0-2] | 1.4e-17 | 26.1 [20-33] / 0.6 [0-3] | 0.0 [0-6] / 8.3 [4-18] | 13.3 [5-30] / 26.7 [14-44] | 3.3 [1-8] / 50.0 [41-59] |
| OpenBioLLM 8B | 33.9 [27-41] / 0.6 [0-3] | 2.7e-17 | 26.7 [21-34] / 0.6 [0-3] | 0.0 [0-6] / 8.3 [4-18] | 13.3 [5-30] / 40.0 [25-58] | 0.8 [0-5] / 48.3 [40-57] |
| MedGemma 4B | 7.2 [4-12] / 0.0 [0-2] | 2.4e-04 | 7.2 [4-12] / 0.6 [0-3] | 0.0 [0-6] / 6.7 [3-16] | 60.0 [42-75] / 70.0 [52-83] | 15.0 [10-22] / 74.2 [66-81] |

Judge 1: Llama Guard 3 8B. Judge 2: qwen3 8B with a harm/refusal rubric (a different model family from the guard used inside the gateway). McNemar: exact two-sided test pairing each attack prompt undefended vs defended (judge 1).

## Table 2. Attack success by attack suite, pooled over models (judge 1 / judge 2)

| Suite | Undefended | Defended |
|---|---|---|
| medsafetybench | 17.5 [12-25] / 16.7 [11-24] | 0.0 [0-3] / 2.5 [1-7] |
| medical_jailbreak | 25.0 [18-33] / 14.2 [9-22] | 0.0 [0-3] / 0.0 [0-3] |
| in_the_wild | 9.2 [5-16] / 5.8 [3-12] | 0.0 [0-3] / 0.0 [0-3] |
| jbb_harmful | 42.5 [34-51] / 36.7 [29-46] | 0.0 [0-3] / 0.0 [0-3] |
| deepset_injection | 0.0 [0-3] / 0.0 [0-3] | 0.8 [0-5] / 0.0 [0-3] |
| xstest_unsafe | 36.7 [29-46] / 34.2 [26-43] | 0.0 [0-3] / 0.0 [0-3] |

## Table 3. Outcome of defended attack prompts by pipeline stage (counts)

| Model | emergency triage | guardrails | input detectors | grounding check | output guard | model refused | answered | Answered and harmful (judge 1) |
|---|---|---|---|---|---|---|---|---|
| JSL-MedLlama 3 8B | 0 | 0 | 143 | 23 | 0 | 12 | 2 | 0 |
| BioMistral 7B | 0 | 0 | 143 | 13 | 0 | 9 | 15 | 0 |
| OpenBioLLM 8B | 0 | 0 | 143 | 18 | 1 | 5 | 13 | 1 |
| MedGemma 4B | 0 | 0 | 143 | 18 | 0 | 12 | 7 | 0 |

## Table 4. Agreement between judges on answered attack responses

| Model | Responses compared | Both harmful | Judge 1 only | Judge 2 only | Cohen's kappa |
|---|---|---|---|---|---|
| JSL-MedLlama 3 8B | 194 | 19 | 7 | 2 | 0.78 |
| BioMistral 7B | 204 | 42 | 15 | 6 | 0.73 |
| OpenBioLLM 8B | 198 | 46 | 16 | 3 | 0.76 |
| MedGemma 4B | 199 | 11 | 2 | 3 | 0.80 |

## Table 5. Ablations (defended condition only)

| Model | Configuration | Prompts | Attack success, judge 1 | Attack success, judge 2 | Medical refused | MedQA accuracy | General refused |
|---|---|---|---|---|---|---|---|
| MedGemma 4B | full gateway | 120 | 0.0 [0-6] | 0.0 [0-6] | 0.0 [0-16] | 60.0 [31-83] | 75.0 [60-86] |
| MedGemma 4B | lg-only | 120 | 0.0 [0-6] | 0.0 [0-6] | 0.0 [0-16] | 50.0 [24-76] | 67.5 [52-80] |
| MedGemma 4B | no-rag | 120 | 0.0 [0-6] | 0.0 [0-6] | 0.0 [0-16] | 50.0 [24-76] | 70.0 [55-82] |
| MedGemma 4B | no-input-detectors | 120 | 3.3 [1-11] | 1.7 [0-9] | 0.0 [0-16] | 50.0 [24-76] | 72.5 [57-84] |
| MedGemma 4B | no-llamaguard | 120 | 0.0 [0-6] | 0.0 [0-6] | 0.0 [0-16] | 60.0 [31-83] | 75.0 [60-86] |
