# Clean-checkpoint capability and overrefusal results

**Status:** 6,012 answers collected; all three conditions scored and independently read back. These are exploratory clean-checkpoint comparisons, not post-tampering capability results.

The results do not support a blanket “no capability cost” claim. Observed coding and knowledge differences are smaller than the instruction-following and maths losses. V2b shows a maths loss under both declared parsers; v2 maths is strongly parser-sensitive. Small differences do not establish equivalence.

## Objective results

Same frozen items, tokenizer, greedy FP16 settings and L40S worker for original/v2/v2b. Each model has 1,414 objective verdicts; everyday quality and XSTest harm are not silently scored as zero. Values are successes / selected items. Changes are candidate minus original, in percentage points, with paired 95% Newcombe intervals. All comparisons are exploratory; no noninferiority margin or multiplicity adjustment was preregistered.

| Domain | Original | v2 | v2b | v2 change [95% interval] | v2b change [95% interval] |
|---|---:|---:|---:|---:|---:|
| MMLU, generated letter | 110/200 (55.0%) | 109/200 (54.5%) | 111/200 (55.5%) | -0.5 [-5.6, +4.6] | +0.5 [-5.0, +6.0] |
| GSM8K, strict | 139/200 (69.5%) | 113/200 (56.5%) | 95/200 (47.5%) | -13.0 [-20.0, -5.8] | -22.0 [-29.0, -14.6] |
| CyberMetric, generated letter | 391/500 (78.2%) | 384/500 (76.8%) | 382/500 (76.4%) | -1.4 [-3.7, +0.8] | -1.8 [-4.2, +0.6] |
| HumanEval, adapted pass@1 | 72/164 (43.9%) | 70/164 (42.7%) | 70/164 (42.7%) | -1.2 [-7.3, +4.9] | -1.2 [-8.0, +5.6] |
| IFEval, strict prompt compliance | 155/200 (77.5%) | 135/200 (67.5%) | 116/200 (58.0%) | -10.0 [-16.7, -3.2] | -19.5 [-26.7, -12.0] |
| GlobalMGSM English, strict | 39/50 (78.0%) | 35/50 (70.0%) | 32/50 (64.0%) | -8.0 [-21.7, +6.0] | -14.0 [-25.6, -2.1] |
| GlobalMGSM French, strict | 25/50 (50.0%) | 16/50 (32.0%) | 12/50 (24.0%) | -18.0 [-30.8, -4.0] | -26.0 [-38.8, -11.6] |
| GlobalMGSM Arabic, strict | 20/50 (40.0%) | 2/50 (4.0%) | 3/50 (6.0%) | -36.0 [-50.1, -20.4] | -34.0 [-48.2, -18.4] |

HumanEval uses the declared single-matching-function adapter and pinned upstream test harness. All 492 model answers ran in network-blocked, secret-free sandboxes. Each run reports zero infrastructure errors; the live gate passes all 164 canonical solutions, 18 authored fixtures and network/secret checks. This is the standard non-adversarial scorer, not proof against scorer subversion. CyberMetric measures multiple-choice security knowledge, not practical defensive security assistance. IFEval measures compliance, not general semantic quality.

## Maths parser sensitivity

The primary strict parser is unchanged. After a pilot and before formal scoring, a published `openai/simple-evals` numeric parser was frozen as a secondary endpoint. Both endpoints apply uniformly; neither is human ground truth.

| Secondary numeric endpoint | Original | v2 | v2b | v2 change [95% interval] | v2b change [95% interval] |
|---|---:|---:|---:|---:|---:|
| gsm8k | 143/200 | 141/200 | 110/200 | -1.0 [-7.1, +5.1] | -16.5 [-23.0, -9.7] |
| global_mgsm_en | 39/50 | 39/50 | 35/50 | +0.0 [-13.2, +13.2] | -8.0 [-19.6, +3.7] |
| global_mgsm_fr | 31/50 | 34/50 | 23/50 | +6.0 [-7.3, +19.0] | -16.0 [-29.4, -1.4] |
| global_mgsm_ar | 37/50 | 32/50 | 29/50 | -10.0 [-23.9, +4.4] | -16.0 [-28.7, -2.4] |

GSM8K primary/secondary changes are −13.0/−1.0 points for v2 and −22.0/−16.5 points for v2b. Parser sensitivity explains a substantial part of the v2 strict-score drop; it does not establish unchanged reasoning. Luna reviewed all 36 original/v2 English GSM8K parser-disagreement answers and judged each to contain the correct numeric answer (4 original, 32 v2). This is an AI diagnostic, not human validation or a replacement score. The published parser takes the last number after the marker and can accept ambiguous text.

## Refusal and overrefusal

| XSTest safe prompts, provisional classifier | Original | v2 | v2b |
|---|---:|---:|---:|
| Refusal flags / 250 | 43 (17.2%) | 107 (42.8%) | 42 (16.8%) |
| Token-limit truncations / 250 | 141 | 98 | 161 |

The v2 safe-prompt refusal-flag increase is +25.6 points [19.19, 31.76]; v2b is −0.4 [−4.64, 3.83]. These are ProtectAI classifier outputs, not validated overrefusal or harmful-assistance rates. XSTest has a 128-token generation budget: many answers are incomplete, and the detector has its own input limit. Conclusions concern the saved visible answers under this protocol. Older repository XSTest numbers used a different judge/protocol and are not a matched comparison with this run.

See the paired GPT-6 Luna review in `results/general_audit/luna_study_summary.json` for the frozen 50-question diagnostic sample. Partial and full refusals, usefulness, harm and truncation are separate fields. Classifier–Luna disagreements are not detector false-positive/false-negative rates. There is no human validation and none is planned for this submission. The alias `gpt-6-luna` is recorded; an immutable backend snapshot and judge sampling parameters are not exposed by the subagent interface.

## Blinded Luna diagnostic sample

The selection was frozen before formal answers: 20 safe XSTest, 10 unsafe XSTest and 20 English everyday questions, each answered by all three models. All 150 judgments passed schema, exact reference/quote binding and truncation-copy checks. GPT-6 Luna reviewed disjoint paired question blocks in three fresh sessions after a context-capacity warning; evidence excerpts were repaired with originals preserved. These checks do not validate judgment accuracy.

| Sample outcome | Original | v2 | v2b |
|---|---:|---:|---:|
| Safe XSTest: any refusal / 20 | 1 | 5 | 1 |
| Safe XSTest: full / partial refusal | 1 / 0 | 4 / 1 | 1 / 0 |
| Safe XSTest: classifier-only refusal / 20 | 5 | 5 | 2 |
| Safe XSTest: Luna-only refusal / 20 | 0 | 0 | 0 |
| Unsafe XSTest: any refusal / 10 | 8 | 9 | 8 |
| English everyday: useful / 20 | 14 | 13 | 15 |
| English everyday: incorrect / 20 | 6 | 7 | 5 |
| English everyday: any refusal / 20 | 0 | 0 | 0 |

These are small-sample AI judgments, not validated overrefusal, detector error rates or semantic utility accuracy. They suggest that some classifier flags reflect interpretation differences, while the sampled v2 refusal increase also appears in Luna's labels. Everyday errors are distinct from refusal, and neither establishes internal inability. Harm and refusal are reported separately in the machine-readable summary. False-premise correction, a refusal plus useful assistance, and harmful content can coexist with other flags; do not force a causal decomposition or pool strata.

## Data and reproducibility

- Model/dataset revisions: `evals/general/lock.json`; per-run source hashes, complete environments, tokenizer checks, EOS settings, protocol, throughput and private artifact indexes: `results/general_audit/run_provenance.json`. Generation source commit: `3241725`; formal run prefix: `full-20261113-v2`.
- `selected_items.jsonl` publishes all 2,004 stable identities and content/prompt hashes. `item_verdicts.jsonl` publishes 6,012 answer-free rows, including 4,242 objective verdicts. Recompute aggregates and paired comparisons from these rows; raw-answer rescoring requires authorized access to the private evidence volume.
- Every raw answer is retained locally and on private Modal volume `tamper-general-evidence`; repository policy forbids publishing harmful outputs. Public verdicts are not a public release of raw answers. Generation, scoring and coding directories were downloaded and all indexed sizes/hashes verified, along with coverage, response binding and identical scoring protocols.
- Published defense corpus: 2,355 rows, two normalized prompt duplicates. No selected overlap under exact/5-word-shingle Jaccard ≥0.8 screening; this does not rule out semantic/cross-translation or pretraining contamination. Exact accepted v2b border data is unavailable. Four English maths items overlap between datasets; no pooled score is reported.
- Structural checks found no duplicate task IDs, duplicate MCQ choices, unparseable maths golds, malformed IFEval records or full canonical/test-string leakage in coding prompts. An everyday reference typo was flagged, preserved and never silently corrected (`reference_issue.json`).

## Boundaries and remaining evidence gaps

No broad general-purpose equivalence claim, no post-tampering capability test, no measured internal loss of ability, no operational security evaluation, and no training-seed replication. Item-level intervals do not cover between-training-run variability. The 20 English everyday questions per model receive only AI judgments; the other six everyday languages have answer collection, not validated semantic quality. Refusal labels cannot causally allocate every lost success to inability versus safety.

Previously disclosed harmful fine-tuning, prefill and steering failures remain in the article/model cards. The harmful fine-tuning rates predate the single-BOS fix; a matched corrected repeat is missing. Neither these clean utility results nor the existing weight-editing results establish resistance to arbitrary tampering.

Reproduce the offline export with `python3 evals/general/export_public.py --panel <downloaded-panel> --base <base-run> --v2 <v2-run> --v2b <v2b-run> --out <destination>`. The command fails closed on invalid or unmatched evidence. Raw generation/scoring commands and protocol details are in `evals/general/README.md`.
