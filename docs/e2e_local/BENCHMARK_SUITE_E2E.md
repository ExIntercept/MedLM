# Benchmark Suite - End-to-End Across Medical Models

Generated 2026-11-21 04:31:57. Judge: `guard:openai_chat:llama-guard3:8b`. 

## Overview (pooled over suites)

| Model | Attack success: undefended | defended | Medical benign refused: undefended | defended | General benign refused: undefended | defended | MedQA accuracy: undefended | defended |
|---|---|---|---|---|---|---|---|---|
| MedGemma | 46.7% [37-57] | 0.0% [0-4] | 0.0% [0-11] | 3.3% [1-17] | 11.7% [6-22] | 75.0% [63-84] | 40.0% [20-64] | 60.0% [36-80] |
| OpenBioLLM | 35.6% [26-46] | 1.1% [0-6] | 0.0% [0-11] | 10.0% [3-26] | 0.0% [0-6] | 45.0% [33-58] | 20.0% [7-45] | 26.7% [11-52] |
| BioMistral | 40.0% [30-50] | 0.0% [0-4] | 0.0% [0-11] | 10.0% [3-26] | 0.0% [0-6] | 46.7% [35-59] | 13.3% [4-38] | 13.3% [4-38] |
| Medical Llama 3 fine-tune | 41.1% [32-51] | 0.0% [0-4] | 0.0% [0-11] | 46.7% [30-64] | 10.0% [5-20] | 86.7% [76-93] | 66.7% [42-85] | 26.7% [11-52] |

Why defended harmless prompts were refused (counts):

| Model | Prompts | Grounding check withheld | Detector blocked | Model declined | Emergency triage |
|---|---|---|---|---|---|
| MedGemma | medical | 1 | 0 | 0 | 0 |
| MedGemma | general | 24 | 3 | 18 | 0 |
| OpenBioLLM | medical | 3 | 0 | 0 | 0 |
| OpenBioLLM | general | 23 | 3 | 1 | 0 |
| BioMistral | medical | 2 | 0 | 1 | 0 |
| BioMistral | general | 19 | 3 | 6 | 0 |
| Medical Llama 3 fine-tune | medical | 12 | 0 | 2 | 0 |
| Medical Llama 3 fine-tune | general | 37 | 3 | 12 | 0 |

Medical benign = MedQA + MedQuAD. General benign = JBB benign, deepset benign, XSTest safe and OR-Bench hard: mostly non-medical, which a clinical assistant is built to decline or withhold.

## hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M (MedGemma, `gemma` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 53.3% [30-75] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 13.3% [4-38] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 86.7% [62-96] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 86.7% [62-96] | 0.0% [0-20] |
| medqa | accuracy | 40.0% [20-64] | 60.0% [36-80] |
| medqa | refused / blocked | 0.0% [0-20] | 6.7% [1-30] |
| medquad | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| jbb_benign | refused / blocked | 6.7% [1-30] | 93.3% [70-99] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 73.3% [48-89] |
| xstest_safe | refused / blocked | 6.7% [1-30] | 60.0% [36-80] |
| orbench_hard | refused / blocked | 33.3% [15-58] | 73.3% [48-89] |

## hf.co/mradermacher/Llama3-OpenBioLLM-8B-GGUF:Q4_K_M (OpenBioLLM, `llama3` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 33.3% [15-58] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 0.0% [0-20] | 6.7% [1-30] |
| jbb_harmful | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 60.0% [36-80] | 0.0% [0-20] |
| medqa | accuracy | 20.0% [7-45] | 26.7% [11-52] |
| medqa | refused / blocked | 0.0% [0-20] | 20.0% [7-45] |
| medquad | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| jbb_benign | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 60.0% [36-80] |
| xstest_safe | refused / blocked | 0.0% [0-20] | 40.0% [20-64] |
| orbench_hard | refused / blocked | 0.0% [0-20] | 46.7% [25-70] |

## hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M (BioMistral, `mistral` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 20.0% [7-45] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 6.7% [1-30] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 93.3% [70-99] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| medqa | accuracy | 13.3% [4-38] | 13.3% [4-38] |
| medqa | refused / blocked | 0.0% [0-20] | 13.3% [4-38] |
| medquad | refused / blocked | 0.0% [0-20] | 6.7% [1-30] |
| jbb_benign | refused / blocked | 0.0% [0-20] | 53.3% [30-75] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 60.0% [36-80] |
| xstest_safe | refused / blocked | 0.0% [0-20] | 40.0% [20-64] |
| orbench_hard | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |

## hf.co/bartowski/JSL-MedLlama-3-8B-v2.0-GGUF:Q4_K_M (Medical Llama 3 fine-tune, `llama3` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 46.7% [25-70] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 6.7% [1-30] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 73.3% [48-89] | 0.0% [0-20] |
| medqa | accuracy | 66.7% [42-85] | 26.7% [11-52] |
| medqa | refused / blocked | 0.0% [0-20] | 60.0% [36-80] |
| medquad | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |
| jbb_benign | refused / blocked | 26.7% [11-52] | 100.0% [80-100] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 86.7% [62-96] |
| xstest_safe | refused / blocked | 6.7% [1-30] | 80.0% [55-93] |
| orbench_hard | refused / blocked | 6.7% [1-30] | 80.0% [55-93] |

