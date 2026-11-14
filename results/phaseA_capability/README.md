# Phase A capability cost: defended checkpoints A1/A2 on the frozen clean panel

Same harness, settings, prompts, parsers and scorers as the merged capability audit (PR #7). base and v2 rows are the frozen PR #7 artifacts re-read (`full-20261113-v2-{base,v2}`), so these comparisons stay exactly paired; A1 and A2 ran the unchanged pipeline on the pinned revisions below. All comparisons are exploratory clean-checkpoint measurements with paired Newcombe 95% intervals; they do not establish equivalence or post-tampering retention. No human validation of refusal labels was performed or planned.

| Benchmark | base | v2 | A1 | A2 | A1 - v2 [95% CI] | A2 - v2 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| IFEval prompt strict | 155/200 (77.5%) | 135/200 (67.5%) | 125/200 (62.5%) | 133/200 (66.5%) | -5.0 [-11.6, +1.7] | -1.0 [-6.8, +4.8] |
| GSM8K strict | 139/200 (69.5%) | 113/200 (56.5%) | 104/200 (52.0%) | 97/200 (48.5%) | -4.5 [-11.4, +2.5] | -8.0 [-14.3, -1.5] |
| GSM8K published numeric parser (secondary) | 143/200 (71.5%) | 141/200 (70.5%) | 118/200 (59.0%) | 129/200 (64.5%) | -11.5 [-18.0, -4.9] | -6.0 [-12.0, +0.1] |
| HumanEval pass@1 (isolated backend) | 72/164 (43.9%) | 70/164 (42.7%) | 37/164 (22.6%) | 64/164 (39.0%) | -20.1 [-27.4, -12.5] | -3.7 [-9.8, +2.5] |
| MMLU generated letter | 110/200 (55.0%) | 109/200 (54.5%) | 110/200 (55.0%) | 113/200 (56.5%) | +0.5 [-3.6, +4.6] | +2.0 [-1.2, +5.2] |
| MMLU forced choice (secondary) | 110/200 (55.0%) | 109/200 (54.5%) | 110/200 (55.0%) | 113/200 (56.5%) | +0.5 [-3.6, +4.6] | +2.0 [-1.2, +5.2] |
| CyberMetric generated letter | 391/500 (78.2%) | 384/500 (76.8%) | 363/500 (72.6%) | 382/500 (76.4%) | -4.2 [-6.8, -1.7] | -0.4 [-2.3, +1.5] |
| CyberMetric forced choice (secondary) | 392/500 (78.4%) | 384/500 (76.8%) | 363/500 (72.6%) | 382/500 (76.4%) | -4.2 [-6.8, -1.7] | -0.4 [-2.3, +1.5] |
| XSTest safe-prompt classifier refusal (provisional) | 43/250 (17.2%) | 107/250 (42.8%) | 85/250 (34.0%) | 136/250 (54.4%) | -8.8pp [-13.4, -4.1] | +11.6pp [+6.7, +16.4] |

Delta units are percentage points for accuracy endpoints and for the refusal rate. Full paired statistics (discordant counts, exact McNemar p, secondary deltas against base) are in `phaseA_capability_scores.json`.

The MMLU forced-choice secondary row is identical to the generated-letter row for all four models: the forced-choice scorer resolved to the same letter as the generated-letter parser on every one of the 200 items (the same behavior held in PR #7's frozen clean_results.json), so the rows match item-for-item rather than being copied.

## Models

- `base`: `CohereLabs/tiny-aya-global` @ `af89d219b53ed9b13b8a4645f9c8028973510324` (frozen PR #7 artifacts re-read)
- `v2`: `StarWarrior24x7/tiny-aya-global-xref-v2` @ `14b4675f635b48f670bd42bd8ccea69ed6dec6ad` (frozen PR #7 artifacts re-read)
- `a1`: `StarWarrior24x7/tiny-aya-global-phaseA-v2-repnoise` @ `c9cc2c739430c4597a766a8be9a0b529ec85c4f5` (new run, unchanged harness)
- `a2`: `StarWarrior24x7/tiny-aya-global-phaseA-v2-booster` @ `ac4c7c485ea0293e80fe149417b432b269af9ed0` (new run, unchanged harness)

## XSTest safe-prompt responses

`xstest_safe_responses.jsonl` holds every 250 safe-prompt response per model with truncation and classifier metadata. Prompts are safe by construction, so responses are shareable. Refusal labels are provisional ProtectAI classifier output under a 128-token generation cap; many answers truncate, and a classifier flag is not a human-validated withholding judgment.

## Limitations

- Exploratory paired comparisons; no preregistered noninferiority margin. A delta whose interval includes 0 is "not detected at this resolution", not evidence of no difference.
- Clean checkpoints only: post-tampering useful-task retention unmeasured.
- Refusal labels are ProtectAI classifier output under a 128-token generation cap; many answers truncate; classifier flags are not human-validated withholding judgments.
- CyberMetric is MCQ security knowledge; IFEval is instruction compliance.
- FP16 near-ties can vary with hardware/batching; recorded settings are bound per run.
- base and v2 rows are the frozen PR #7 artifacts re-read, not new generations.
