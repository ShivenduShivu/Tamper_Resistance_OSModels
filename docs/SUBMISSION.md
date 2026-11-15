# Submission checklist: where each item is answered

[Back to the summary](../README.md). This page maps every required item of the hackathon's write-up checklist (Track 2b) to where it is answered. Items not covered elsewhere are answered here in full: metrics, splits and leak checks, per-toolkit settings, judge checks, cost per 1,000 replies and the declaration of outside work.

| Checklist item | Where |
|---|---|
| Title + one-line claim | [README](../README.md) header |
| Tracks + problem statements | [README → Tracks and problem statement](../README.md#tracks-and-problem-statement) |
| Content warning | [README](../README.md) top |
| Abstract | [README → Abstract](../README.md#abstract); [ABSTRACT.md](ABSTRACT.md) for the form |
| Headline before → after | [README → Results at a glance](../README.md#results-at-a-glance) |
| Base model + method in one line | [README → How it works](../README.md#how-it-works) |
| Problem, approach, why this problem | [DETAILS §1–2](DETAILS.md#1-the-threat-in-plain-words) |
| Pre-registered hypotheses | [Fine-tuning predictions, written before the runs](../results/phaseA_finetune/PREDICTIONS.md) |
| Related work | [DETAILS §2](DETAILS.md#2-our-innovation-train-through-the-attack), [references](REPRODUCE.md#18-team-and-references) |
| Hackathon repos used, and how they connect | [DETAILS §7](DETAILS.md#7-result-4-independent-validation-with-lexsis-toolkits); settings [below](#settings-for-each-hackathon-repo) |
| Models and licences | [REPRODUCE §19](REPRODUCE.md#19-supplementary-models-datasets-licences-glossary) |
| Datasets, licences, languages | [REPRODUCE §19](REPRODUCE.md#19-supplementary-models-datasets-licences-glossary); languages [below](#test-sets-languages-and-leak-checks) |
| Train / test splits, leak check | [Below](#test-sets-languages-and-leak-checks) |
| Safety benchmarks; over-refusal test set | [Below](#test-sets-languages-and-leak-checks) |
| Metrics, and why | [Below](#metrics) |
| Checking the automatic judge | [Below](#checking-the-automatic-judges) |
| Training / attack setup, hyperparameters | [REPRODUCE → Training configuration](REPRODUCE.md#19-supplementary-models-datasets-licences-glossary); fine-tuning defenses in the [A1](https://huggingface.co/StarWarrior24x7/tiny-aya-global-phaseA-v2-repnoise) / [A2](https://huggingface.co/StarWarrior24x7/tiny-aya-global-phaseA-v2-booster) cards; toolkit settings [below](#settings-for-each-hackathon-repo) |
| Seeds, ablations | v1 → v2 → v2b change one thing at a time; A1 trained with two seeds; prefill pilot with matched controls ([DETAILS §12](DETAILS.md#12-separate-pilot-reducing-prefill-harm)) |
| Results: same model before vs after | [README](../README.md#results-at-a-glance); [DETAILS §4–11](DETAILS.md#4-result-1-tiny-aya-is-stripped-for-free) |
| Safety + general results side by side | [DETAILS §10](DETAILS.md#10-safety-vs-usability) |
| Confidence intervals | Wilson 95% intervals for rates; paired Newcombe 95% intervals for capability ([capability results](GENERAL_CAPABILITY_RESULTS.md)) |
| What didn't work | [DETAILS §13](DETAILS.md#13-limitations-failures-and-honest-corrections); A2 (Booster) negative result |
| Results on the quantized model | 4-bit NF4 (bitsandbytes): original 99%, v2 100% refusal ([DETAILS §8](DETAILS.md#8-attack-resistance-matrix-what-it-stops-and-what-it-doesnt)); measured before the BOS fix, so provisional |
| Compute, cost / ROI, cost per 1k replies | [DETAILS §15](DETAILS.md#15-cost-and-real-world-value); per 1,000 replies [below](#cost-per-1000-replies) |
| Limitations | [README](../README.md#limitations), [DETAILS §13](DETAILS.md#13-limitations-failures-and-honest-corrections) |
| With more credits: approach and expected before / after | [DETAILS §16](DETAILS.md#16-roadmap) |
| Artifacts (HF collections, repo, Space) | [README → Submission materials](../README.md#submission-materials) |
| Repo layout, run instructions, environment | [REPRODUCE §17](REPRODUCE.md#17-reproduce-everything); `requirements.txt` |
| Declaration of work outside the hackathon | [Below](#declaration-of-work-done-outside-the-hackathon) |
| Teammates and contributions | [README → Team](../README.md#team-and-references) |

## Metrics

| Metric | Judge / source | Better | Why |
|---|---|---|---|
| Harmful-answer rate | IBM Granite Guardian 3.2-5B, risk `harm`, P(yes) > 0.5 | lower | The main safety number: whether the answer is actually harmful, not whether it sounds like a refusal |
| Refusal rate | ProtectAI `distilroberta-base-rejection-v1`, cross-checked with a keyword list | higher on harmful prompts | Direct measure of refusal; Heretic optimises against the keyword score, so we report both |
| KL divergence of an attack | Heretic, on `harmless_alpaca` | higher = the attack had to change the model more | How far an attack shifts normal behaviour; low KL means a stripped model still behaves like the original |
| Attack success, any of 3 samples | ProtectAI, temperature 1 | lower | Cost-to-break by simple retries |
| Over-refusal | ProtectAI on XSTest's 250 safe prompts | lower | Safety must not mean refusing harmless questions |
| Refusal steps survived under harmful fine-tuning | ProtectAI + Granite on HarmBench-193 | more steps | Durability against the track's main attack |
| MMLU, IFEval, GSM8K, HumanEval, CyberMetric, GlobalMGSM | Standard scorers, paired items | higher | What should not get worse: knowledge, instruction following, maths, coding, security knowledge, multilingual maths |
| Multilingual harm | Granite on NLLB-200 English translations; Llama Guard 3-1B on the original text | lower | Whether the defense holds outside English |
| AuditKIT probe pass rate | AuditKIT `RedTeamRunner` | higher | Independent probe suite from the hackathon toolkit (indicative, small probe set) |

## Test sets, languages and leak checks

**Training data** ([tiny-aya-extended-refusal](https://huggingface.co/datasets/StarWarrior24x7/tiny-aya-extended-refusal), English):
- v1 / v2: 2,355 rows. 1,200 explained refusals to `LLM-LAT/harmful-dataset` prompts and 1,155 benign answers to `databricks-dolly-15k` prompts, all generated by Tiny Aya itself.
- v2b: adds 1,218 borderline-but-safe prompts from OR-Bench; 200 more OR-Bench prompts are held out.

**Test sets:**

| Set | Size | Languages | Role |
|---|---|---|---|
| `mlabonne/harmful_behaviors` test | 100 | English | Focus: harm and refusal under every attack |
| XSTest safe / unsafe | 250 / 200 | English | Over-refusal (focus for usability) / unsafe contrast |
| `CohereLabs/aya_redteaming` | 100 per language | English, Hindi, Arabic, French, Spanish | Focus: does the defense hold outside English |
| HarmBench "standard" minus near-duplicates | 193 | English | Held-out harmful fine-tuning test |
| StrongREJECT minus near-duplicates | 282 | English | Held-out clean refusal check |
| MMLU | 1,000 (clean and under attack) / 200 paired | English | Retention check |
| IFEval, GSM8K, HumanEval, CyberMetric | 200 / 200 / 164 / 500 paired | English | Retention check |
| GlobalMGSM | 50 per language | English, French, Arabic | Retention check (multilingual maths) |

**Leak checks** (lexical; no embedding or MinHash check was run):
- **Heretic's prompts:** the prompts Heretic uses to find and score the refusal direction are excluded from training.
- **Training vs evaluation:** a normalised-text audit found two duplicate prompts and no overlap between training and evaluation.
- **Held-out fine-tuning sets:** HarmBench and StrongREJECT prompts were dropped if they matched any training, attack or defense prompt exactly or with word Jaccard ≥ 0.5. That removed 7 and 31 prompts.
- **Fine-tuning attack vs defense data:** they come from disjoint rows of `LLM-LAT/harmful-dataset` (attack rows 0–699, defense rows 1000+, after a fixed shuffle that removes our training prompts).
- **XSTest:** never trained on.

## Settings for each hackathon repo

**SafeTune** (`pip install safetune`, used in `kaggle/day5`, `day5b`, `day6` and `phaseA`):
- **Refusal direction:** `extract_refusal_direction(model, tok, harmful, harmless, RefusalDirectionConfig(normalize=False, select_directions=False))` with 64 `harmful_behaviors` and 64 `harmless_alpaca` train prompts. Per-layer directions are logged.
- **Ablation attack:** `RefusalDirectionModel(model, direction, mode="ablate")`, scored on the 100-prompt harmful test set.
- **Steering attack:** `extract_caa_vectors(..., CAAConfig())` with defaults, then `CAAModel(model, {layer: vector}, strength=-1.0)` at the refusal-direction layer. Strength −4 was also tried and breaks every model into gibberish.
- **Phase A:** RepNoise and Booster were implemented in our own `kaggle/phaseA/harden.py`, following the papers, so that v2's attack simulation could run alongside them.

**AuditKIT** (`pip install auditkit`):
- **Probe suite:** `RedTeamRunner(generate).run(probes=[p])` for each of `jailbreak`, `encoding`, `prompt_injection` and `refusal`.
- **Generation:** greedy decoding, 96 new tokens. Each probe runs on its own, which avoids the multi-probe scoring bug we fixed in AuditKIT #35.

**CircuitKIT** (`pip install circuitkit`, `kaggle/day5c`):
- **Task:** a `harmful_vs_benign` paired task built from JailbreakBench behaviours, registered with `register_task`. This needed the pairing fix in CircuitKIT #37.
- **Discovery:** `discover(model, task, algorithm="eap", n_examples=8, batch_size=1)`. It ran out of memory on a 16 GB T4 for this 3.35B, 262k-vocabulary model, so no circuit results are claimed.

**Heretic** (not a hackathon repo; our main attacker): `heretic-llm` 2.1.0, classic abliteration, Optuna TPE with 40 trials (12 random start) per model and 200 on v2. Full settings in [REPRODUCE](REPRODUCE.md#19-supplementary-models-datasets-licences-glossary) and `results/heretic200_v2/config.toml`.

## Checking the automatic judges

- **No human-label agreement was measured.** We did not hand-label a sample, so there is no % agreement or Cohen's κ against humans. This is a stated gap.
- **What we did instead:**
  - **Two independent harm judges:** Granite Guardian and Llama Guard 3 agree on 81–87% of multilingual rows (85% on English without translation).
  - **Refusal judge vs harm judge:** ProtectAI refusal is reported alongside Granite harm. Where the two disagree, we read the cases: for example, v2 rewriting a harmful request into a harmless answer.
  - **Judge vs keywords:** after Heretic's attack, the keyword score said the original still refuses 14%, while ProtectAI says 1%. So we lead with the trained classifier and with harm, not keywords.
- **Spot checks:** the capability audit's parser disagreements were reviewed by AI, not by humans ([capability results](GENERAL_CAPABILITY_RESULTS.md)).

## Cost per 1,000 replies

Same workload as [DETAILS §15](DETAILS.md#15-cost-and-real-world-value): 300 input and 200 output tokens per reply, so 1,000 replies are 0.3M input and 0.2M output tokens. Self-hosted uses the measured throughput of about 150 output tokens/s on one T4 with plain `transformers` (no vLLM), at $0.35 per T4-hour, GPU only.

| Option | Cost per 1,000 replies |
|---|---|
| **Self-hosted hardened Tiny Aya, 1× T4 (unoptimised, projected)** | **~$0.26** |
| GPT-6 Luna / Claude Haiku 5.5 | $0.13 |
| GPT-5.6 Luna | $0.30 |
| GPT-5.4 mini | $1.13 |
| GPT-6.1 Terra | $1.30 |
| Claude Sonnet 5.5 | $2.60 |

API prices are list prices checked Nov 12, 2026 (sources in DETAILS §15). A serving engine such as vLLM would raise throughput and lower the self-hosted figure; we did not measure it.

## Declaration of work done outside the hackathon

All code, training, evaluation and writing in this submission was done during the hackathon build period. The repository's history starts on 2026-11-10. The work builds on public tools and papers (Heretic, the Lexsi toolkits, Tiny Aya, and the papers cited), which are credited and not claimed as our own. The five upstream fixes were made during the hackathon while integrating the toolkits.
