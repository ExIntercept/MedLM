# Benchmark Suite - Pre-Model Screening

Generated 2026-11-19 10:38:45. Each prompt goes through the gateway's pre-model stages only (sanitize, emergency triage, contraindication guardrails, Prompt Guard + Template Matcher). No model is called. Up to 200 prompts per suite, fixed seed. Rates show the 95% Wilson interval in brackets.

- **Stopped** = BLOCKED, emergency-escalated or contraindication-intercepted before the model.
- **Caught** = stopped, or FLAGGED (forwarded with the mitigation rule).
- Suites marked * overlap Prompt Guard's published training data; treat them as in-distribution, not held-out.

## Configuration: `full`

### Attacks (higher is better)

| Suite | n | Stopped | Caught (stopped + flagged) | p50 ms |
|---|---|---|---|---|
| medsafetybench | 200 | 4.5% [2-8] | 6.0% [3-10] | 181.2 |
| medical_jailbreak* | 200 | 74.5% [68-80] | 79.0% [73-84] | 3396.3 |
| in_the_wild* | 200 | 74.0% [68-80] | 76.0% [70-81] | 5778.4 |
| jbb_harmful | 100 | 1.0% [0-5] | 1.0% [0-5] | 417.0 |
| deepset_injection | 60 | 33.3% [23-46] | 36.7% [26-49] | 247.3 |
| xstest_unsafe* | 200 | 1.0% [0-4] | 1.0% [0-4] | 155.8 |

### Benign (lower is better)

| Suite | n | Wrongly stopped | Flagged | of which emergency-escalated | p50 ms |
|---|---|---|---|---|---|
| medqa | 200 | 17.0% [12-23] | 0.0% [0-2] | 32 | 1969.0 |
| medquad | 200 | 0.0% [0-2] | 0.0% [0-2] | 0 | 154.7 |
| jbb_benign | 100 | 2.0% [1-7] | 0.0% [0-4] | 1 | 157.1 |
| deepset_benign | 56 | 0.0% [0-6] | 0.0% [0-6] | 0 | 153.0 |
| xstest_safe* | 200 | 0.5% [0-3] | 0.0% [0-2] | 1 | 155.4 |

