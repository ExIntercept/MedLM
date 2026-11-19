# Benchmark Suite - Pre-Model Screening

Generated 2026-11-19 18:13:57. Each prompt goes through the gateway's pre-model stages only (sanitize, emergency triage, contraindication guardrails, Prompt Guard + Template Matcher). No model is called. Up to 200 prompts per suite, fixed seed. Rates show the 95% Wilson interval in brackets.

- **Stopped** = BLOCKED, emergency-escalated or contraindication-intercepted before the model.
- **Caught** = stopped, or FLAGGED (forwarded with the mitigation rule).
- Suites marked * overlap Prompt Guard's published training data; treat them as in-distribution, not held-out.

## Configuration: `full`

### Attacks (higher is better)

| Suite | n | Stopped | Caught (stopped + flagged) | p50 ms |
|---|---|---|---|---|
| medsafetybench | 200 | 89.5% [84-93] | 95.5% [92-98] | 597.8 |
| medical_jailbreak* | 200 | 80.5% [74-85] | 86.0% [81-90] | 7875.1 |
| in_the_wild* | 200 | 74.0% [68-80] | 76.5% [70-82] | 8536.3 |
| jbb_harmful | 100 | 69.0% [59-77] | 82.0% [73-88] | 570.8 |
| deepset_injection | 60 | 35.0% [24-48] | 38.3% [27-51] | 760.9 |
| xstest_unsafe* | 200 | 55.0% [48-62] | 71.5% [65-77] | 536.7 |

### Benign (lower is better)

| Suite | n | Wrongly stopped | Flagged | of which emergency-escalated | p50 ms |
|---|---|---|---|---|---|
| medqa | 200 | 1.0% [0-4] | 0.0% [0-2] | 0 | 5712.9 |
| medquad | 200 | 0.0% [0-2] | 0.0% [0-2] | 0 | 548.2 |
| jbb_benign | 100 | 17.0% [11-26] | 12.0% [7-20] | 1 | 534.7 |
| deepset_benign | 56 | 0.0% [0-6] | 1.8% [0-9] | 0 | 527.0 |
| xstest_safe* | 200 | 8.5% [5-13] | 11.5% [8-17] | 0 | 514.0 |
| orbench_hard | 200 | 9.0% [6-14] | 7.5% [5-12] | 2 | 592.5 |

