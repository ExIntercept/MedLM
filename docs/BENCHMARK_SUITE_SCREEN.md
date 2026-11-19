# Benchmark Suite - Pre-Model Screening

Generated 2026-11-19 13:04:54. Each prompt goes through the gateway's pre-model stages only (sanitize, emergency triage, contraindication guardrails, Prompt Guard + Template Matcher). No model is called. Up to 200 prompts per suite, fixed seed. Rates show the 95% Wilson interval in brackets.

- **Stopped** = BLOCKED, emergency-escalated or contraindication-intercepted before the model.
- **Caught** = stopped, or FLAGGED (forwarded with the mitigation rule).
- Suites marked * overlap Prompt Guard's published training data; treat them as in-distribution, not held-out.

## Configuration: `full`

### Attacks (higher is better)

| Suite | n | Stopped | Caught (stopped + flagged) | p50 ms |
|---|---|---|---|---|
| medsafetybench | 200 | 92.0% [87-95] | 98.5% [96-99] | 565.1 |
| medical_jailbreak* | 200 | 80.0% [74-85] | 87.5% [82-91] | 7827.8 |
| in_the_wild* | 200 | 74.0% [68-80] | 77.0% [71-82] | 8145.2 |
| jbb_harmful | 100 | 76.0% [67-83] | 85.0% [77-91] | 553.0 |
| deepset_injection | 60 | 35.0% [24-48] | 40.0% [29-53] | 686.5 |
| xstest_unsafe* | 200 | 62.0% [55-68] | 79.0% [73-84] | 517.9 |

### Benign (lower is better)

| Suite | n | Wrongly stopped | Flagged | of which emergency-escalated | p50 ms |
|---|---|---|---|---|---|
| medqa | 200 | 17.0% [12-23] | 0.5% [0-3] | 32 | 4748.8 |
| medquad | 200 | 1.0% [0-4] | 0.0% [0-2] | 2 | 544.9 |
| jbb_benign | 100 | 19.0% [13-28] | 26.0% [18-35] | 1 | 546.4 |
| deepset_benign | 56 | 0.0% [0-6] | 1.8% [0-9] | 0 | 537.9 |
| xstest_safe* | 200 | 14.0% [10-19] | 15.0% [11-21] | 1 | 528.4 |
| orbench_hard | 200 | 15.5% [11-21] | 13.0% [9-18] | 4 | 611.3 |

