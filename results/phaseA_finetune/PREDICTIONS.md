# Predictions for the held-out fine-tuning re-evaluation (H1)

Copied verbatim from the team's private research log, where it was written on 2026-11-14 after the first fine-tuning results (on 100 `mlabonne/harmful_behaviors` test prompts) and **before** the held-out jobs ran. The log has no independent timestamp, so treat this as a good-faith record, not a cryptographic pre-registration.

> **Expectation written before results:** if the 100-prompt findings are real, the ordering (A1 breaks ~3–4× later than v2 at lr 2e-5; base ≤ v2 at lr 2e-4; A2 ≈ v2 or worse) should hold on HarmBench-193. A reversal means the 100-prompt results were set-specific.

## Outcome

All three predictions held (numbers in [README.md](README.md)):

| Prediction | Held-out result | Held? |
|---|---|---|
| A1 breaks ~3–4× later than v2 at lr 2e-5 | v2 collapses between steps 25 and 50; A1 and A1 (seed 1) between 100 and 175 | yes |
| base ≤ v2 at lr 2e-4 | step 5: base 3% refusal, v2 94% | yes |
| A2 ≈ v2 or worse | lr 2e-5: same break point as v2; lr 2e-4 step 5: 25% vs v2's 94% | yes (worse) |
