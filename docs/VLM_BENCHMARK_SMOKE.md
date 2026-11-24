# VLM Benchmark Smoke Report

Generated with MedGemma 4B through local Ollama.

## Summary

| Dataset | Direct MedGemma | Through MedLM |
|---|---:|---:|
| VQA-RAD | 60.0% (3/5) | 20.0% (1/5), 2 withheld |
| RadBench Radiopaedia subset | 80.0% (4/5) | 40.0% (2/5), 1 withheld |

These are smoke samples only. They are not clinical accuracy estimates and must not be presented as production metrics.

## VQA-RAD observations

- Direct MedGemma answered 3 of 5 questions correctly.
- Through MedLM, 1 of 5 answers remained correct.
- Two answers were withheld by evidence grounding.
- One harmless visible-text line, `Hx of Situs Inversus`, was quarantined.
- The gateway therefore changes the image-to-answer path from direct visual reasoning to:
  1. Vision description.
  2. Sentence quarantine.
  3. Text prompt with evidence.
  4. Grounding verification.

## RadBench observations

- Direct MedGemma answered 4 of 5 sampled questions correctly.
- Through MedLM, 2 of 5 remained correct.
- One answer was withheld because grounding evidence was unavailable.
- One defended result changed `yes` to `no` for a tension-pneumatothorax feature.

## Known limitations

- Only 5 examples per benchmark were run.
- RadBench MedPix image IDs could not be resolved through the current MedPix REST page.
- 3MAD was not completed within the submission-time limit.
- The Cloudflare hostname in `.env` was stale and DNS resolution failed. It cannot be recreated by this local process because the tunnel was started in Colab. Restart the Colab server/notebook and replace `COLAB_API_BASE` with the newly printed URL.
- The model is a 4B quantized GGUF, not a clinically validated diagnostic system.
