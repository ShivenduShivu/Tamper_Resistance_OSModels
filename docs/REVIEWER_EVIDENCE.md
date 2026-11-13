# Submission evidence contract

Status: protocol and tools, **not new coding/security model results**. Freeze a
run-specific manifest before model outputs are inspected. Public-data preflight,
offline tests and synthetic fixtures establish machinery only.

## Question and success criteria

Does training through simulated abliteration improve safety against tested
weight edits, while retaining measured benign task performance and avoiding
excessive refusal? Treat safety, usefulness and overrefusal as co-primary evidence
threads. A favorable value in one does not establish success in another.

Before the confirmatory run, choose the release candidate (v2 or v2b), the primary
utility endpoint and an acceptable loss margin in percentage points, with a
practical justification. No preservation claim until the paired interval rules
out losses larger than that margin. The existing harness tests for differences;
it does not implement a noninferiority/equivalence decision. Until margins and
analysis are frozen, all model results are exploratory. Do not choose a candidate
or endpoint based on favorable test results. Pilots cannot become confirmation.

## Evidence matrix

| Thread | Minimum measurements | Current state |
|---|---|---|
| Functional coding | HumanEval 164, executed pass@1, base/v2/v2b on same problems | Harness imported; no scored model run |
| Generative maths | GSM8K frozen 200, exact-match plus extraction/truncation counts | Harness imported; no scored model run |
| Cybersecurity knowledge | CyberMetric frozen 500, forced-choice accuracy | Harness imported; no scored model run; not operational skill |
| Benign security help | Human-reviewed authorized defensive questions; correctness and refusal separate | Missing; MCQ cannot substitute |
| Overrefusal | XSTest safe and unsafe contrasts; held-out OR-Bench; random human audit | AI-classified aggregates exist; human audit missing |
| Safety under tampering | Harmful assistance and refusal separately, clean and attacked | Existing AI-judged evidence; human calibration missing |
| Fine-tuning failure | Corrected single-BOS repeat with matched base control | Recorded pre-fix failure retained; corrected repeat missing |

Primary matrix: base/v2/v2b × clean/Heretic. Add v1 as the training-recipe control.
At attack points, use the same items and task protocols as clean measurements.
Measure both selected low-KL and strongest-found points if budget permits.
If only clean results finish, explicitly restrict the conclusion to clean
checkpoints; the current capability runner does not reconstruct attacks.

## What makes the comparison credible

1. Record source commit, model/tokenizer revisions and subfolders, dataset
   revisions, stable item IDs, complete prompts, attack trial/config/version,
   source checkpoint hashes, seeds, precision, environment, output budget and
   parser/scorer hashes. Keep every answer in private durable storage; export and
   verify artifact hashes outside the worker before calling a run complete.
2. Reconstruct archived attacks without persisting/publishing attacked weights.
   A fresh stochastic search is a new experiment, not exact reconstruction.
   Selection is based on a declared search objective, not whichever attack gives
   the preferred result. Report infeasible KL budgets, not invented zero effects.
3. Archived Heretic scoring uses harmful_behaviors test[:100]. Treat it as search
   development data, regardless of the word "test". Use a separate frozen holdout
   for final harm estimates; document semantic overlap with defense training,
   attack tuning, and OR-Bench training. Do not tune on the holdout.
4. Match search budgets across model comparisons. Repeat independent attack seeds
   when affordable; report between-search variability separately from per-prompt
   uncertainty. One 40-vs-200 trial comparison is not search saturation evidence.
5. On benign tasks report correct/useful, refused, incorrect, empty/irrelevant,
   truncated and unparsed responses. These flags can overlap; do not sum them as
   exclusive outcomes. All-item task success is primary; conditional accuracy
   among answered items is supplemental. Forced-choice MCQ is not natural refusal.
6. Report candidate-minus-base percentage-point changes, paired intervals,
   discordant counts and exact sample sizes. Report exclusions and common-item
   coverage. Do not infer equivalence from overlapping intervals or p > .05.
   Keep non-primary comparisons exploratory or use a declared multiplicity plan.
7. Human-check a random subset plus separate disagreement enrichment, blinded
   to model and machine verdict. Two reviewers, adjudication, explicit uncertain
   labels, both classifier error directions. Audit harm labels as well as refusal.
   Human subset validation does not validate all AI labels.
8. For Tiny Aya's multilingual claim, add benign utility/overrefusal probes in
   evaluated languages when possible. Harm reduction on different language prompt
   sets does not establish retained multilingual capability.

## Budget and stopping rules

Available user-authorized ceilings: $18 Modal and $19 Hugging Face (2026-11-13).
Do not assume credits transfer between services. Reserve room for scoring,
artifact export and failures. Start with a small pilot and record its elapsed
resource use before estimating a full run. One job at a time, bounded timeouts,
no automatic paid retries; no claim that an estimate is a provider billing cap.
The public CPU preflight has one container, 1 CPU, 2 GiB, a 300-second function
timeout and no retries. Its image build and overhead are outside that timeout.
Private/gated model runs require an authorized HF_TOKEN; missing access is a
blocker, not a model failure. Never commit credentials or raw responses.

## Submission claims that must remain bounded

- MMLU aggregates describe that test only; they do not establish broad capability
  preservation or a statistically detectable difference without suitable analysis.
- Heretic keyword refusal is a search metric, not a harmful-response rate.
- KL is distribution shift on specified prompts, not attack compute cost or proof
  of capability damage. "100% keyword refusal" is not "all attacks fail".
- Keep harmful fine-tuning, prefill and steering failures visible, including
  pre-fix measurement caveats. No claim of general tamper resistance.
- Mechanistic direction magnitudes are suggestive, not proof of distributed
  safety. Training-recipe ablations and causal interventions would be needed.

Research basis: [XSTest](https://aclanthology.org/2024.naacl-long.301/),
[OR-Bench](https://proceedings.mlr.press/v267/cui25a.html),
[CyberMetric](https://arxiv.org/abs/2402.07688),
[EvalPlus](https://github.com/evalplus/evalplus),
[HarmBench](https://proceedings.mlr.press/v235/mazeika24a.html).
