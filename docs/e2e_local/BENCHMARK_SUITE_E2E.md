# Benchmark Suite - End-to-End Across Medical Models

Generated 2026-11-21 09:42:25. Judge: `guard:openai_chat:llama-guard3:8b`. 

## Overview (pooled over suites)

| Model | Attack success: undefended | defended | Medical benign refused: undefended | defended | General benign refused: undefended | defended | MedQA accuracy: undefended | defended |
|---|---|---|---|---|---|---|---|---|
| MedGemma | 46.7% [37-57] | 0.0% [0-4] | 0.0% [0-11] | 3.3% [1-17] | 11.7% [6-22] | 78.3% [66-87] | 40.0% [20-64] | 46.7% [25-70] |
| OpenBioLLM | 35.6% [26-46] | 0.0% [0-4] | 0.0% [0-11] | 0.0% [0-11] | 0.0% [0-6] | 50.0% [38-62] | 20.0% [7-45] | 26.7% [11-52] |
| BioMistral | 40.0% [30-50] | 0.0% [0-4] | 0.0% [0-11] | 13.3% [5-30] | 0.0% [0-6] | 36.7% [26-49] | 13.3% [4-38] | 20.0% [7-45] |
| Medical Llama 3 fine-tune | 41.1% [32-51] | 0.0% [0-4] | 0.0% [0-11] | 60.0% [42-75] | 10.0% [5-20] | 88.3% [78-94] | 66.7% [42-85] | 13.3% [4-38] |

Why defended harmless prompts were refused (counts):

| Model | Prompts | Grounding check withheld | Detector blocked | Model declined | Emergency triage |
|---|---|---|---|---|---|
| MedGemma | medical | 1 | 0 | 0 | 0 |
| MedGemma | general | 26 | 3 | 18 | 0 |
| OpenBioLLM | medical | 0 | 0 | 0 | 0 |
| OpenBioLLM | general | 26 | 3 | 1 | 0 |
| BioMistral | medical | 3 | 0 | 1 | 0 |
| BioMistral | general | 14 | 3 | 5 | 0 |
| Medical Llama 3 fine-tune | medical | 13 | 0 | 5 | 0 |
| Medical Llama 3 fine-tune | general | 39 | 3 | 11 | 0 |

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
| medqa | accuracy | 40.0% [20-64] | 46.7% [25-70] |
| medqa | refused / blocked | 0.0% [0-20] | 6.7% [1-30] |
| medquad | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| jbb_benign | refused / blocked | 6.7% [1-30] | 100.0% [80-100] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 73.3% [48-89] |
| xstest_safe | refused / blocked | 6.7% [1-30] | 60.0% [36-80] |
| orbench_hard | refused / blocked | 33.3% [15-58] | 80.0% [55-93] |

## hf.co/mradermacher/Llama3-OpenBioLLM-8B-GGUF:Q4_K_M (OpenBioLLM, `llama3` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 33.3% [15-58] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 60.0% [36-80] | 0.0% [0-20] |
| medqa | accuracy | 20.0% [7-45] | 26.7% [11-52] |
| medqa | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| medquad | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| jbb_benign | refused / blocked | 0.0% [0-20] | 66.7% [42-85] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 53.3% [30-75] |
| xstest_safe | refused / blocked | 0.0% [0-20] | 46.7% [25-70] |
| orbench_hard | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |

## hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M (BioMistral, `mistral` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 20.0% [7-45] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 6.7% [1-30] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 93.3% [70-99] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| medqa | accuracy | 13.3% [4-38] | 20.0% [7-45] |
| medqa | refused / blocked | 0.0% [0-20] | 0.0% [0-20] |
| medquad | refused / blocked | 0.0% [0-20] | 26.7% [11-52] |
| jbb_benign | refused / blocked | 0.0% [0-20] | 46.7% [25-70] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 26.7% [11-52] |
| xstest_safe | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |
| orbench_hard | refused / blocked | 0.0% [0-20] | 40.0% [20-64] |

## hf.co/bartowski/JSL-MedLlama-3-8B-v2.0-GGUF:Q4_K_M (Medical Llama 3 fine-tune, `llama3` template)

| Suite | Metric | Undefended | Defended |
|---|---|---|---|
| medsafetybench | attack success (unsafe) | 40.0% [20-64] | 0.0% [0-20] |
| medical_jailbreak | attack success (unsafe) | 46.7% [25-70] | 0.0% [0-20] |
| in_the_wild | attack success (unsafe) | 6.7% [1-30] | 0.0% [0-20] |
| jbb_harmful | attack success (unsafe) | 80.0% [55-93] | 0.0% [0-20] |
| deepset_injection | attack success (unsafe) | 0.0% [0-20] | 0.0% [0-20] |
| xstest_unsafe | attack success (unsafe) | 73.3% [48-89] | 0.0% [0-20] |
| medqa | accuracy | 66.7% [42-85] | 13.3% [4-38] |
| medqa | refused / blocked | 0.0% [0-20] | 86.7% [62-96] |
| medquad | refused / blocked | 0.0% [0-20] | 33.3% [15-58] |
| jbb_benign | refused / blocked | 26.7% [11-52] | 86.7% [62-96] |
| deepset_benign | refused / blocked | 0.0% [0-20] | 80.0% [55-93] |
| xstest_safe | refused / blocked | 6.7% [1-30] | 93.3% [70-99] |
| orbench_hard | refused / blocked | 6.7% [1-30] | 93.3% [70-99] |

