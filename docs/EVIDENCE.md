# Evidence index

[Back to the summary](../README.md). Choose a result, then open its score file or protocol directly. JSON and JSONL links below are downloadable without a GitHub account.

| Result family | Readable tables | Machine-readable scores | Protocol / config / code |
|---|---|---|---|
| Heretic harm and SafeTune | [Attack tables](DETAILS.md#8-attack-resistance-matrix-what-it-stops-and-what-it-doesnt) | [Granite harm / refusal summary](../results/guard_judge_day6/summary.jsonl); [Per-item harm / refusal scores](../results/guard_judge_day6/scores.jsonl) | [day6_bos_reeval.py](../kaggle/day6/day6_bos_reeval.py); [guard_judge.py](../kaggle/guard_judge/guard_judge.py) |
| Heretic search / Pareto fronts | [Heretic table](DETAILS.md#5-result-2-v2-resists-heretic) | [Day6 search + clean summary](../results/day6/summary.jsonl); [200-trial search summary](../results/heretic200_v2/summary.jsonl) | [pareto.py](../scripts/pareto.py); [config.toml](../results/heretic200_v2/config.toml) |
| Multilingual harm | [Language table](DETAILS.md#9-result-5-it-holds-outside-english) | [Language-level summary](../results/multilingual/summary.jsonl); [Per-item multilingual scores](../results/multilingual/scores.jsonl) | [multilingual_eval.py](../kaggle/multilingual/multilingual_eval.py) |
| Clean capability and safe-prompt refusal | [Full tables](GENERAL_CAPABILITY_RESULTS.md) | [clean_results.json](../results/general_audit/clean_results.json); [item_verdicts.jsonl](../results/general_audit/item_verdicts.jsonl); [selected_items.jsonl](../results/general_audit/selected_items.jsonl) | [Capability protocol + commands](../evals/general/README.md); [run_provenance.json](../results/general_audit/run_provenance.json) |
| Harmful fine-tuning / A1 delay | [Full tables](../results/phaseA_finetune/README.md) | [heldout_jobA_v2_A1.summary.jsonl](../results/phaseA_finetune/heldout_jobA_v2_A1.summary.jsonl); [heldout_jobB_A1s1_A2_base.summary.jsonl](../results/phaseA_finetune/heldout_jobB_A1s1_A2_base.summary.jsonl); [granite_rejudge.summary.jsonl](../results/phaseA_finetune/granite_rejudge.summary.jsonl) | [Attack job A: v2 / A1](../kaggle/heldout_a/heldout.py); [Attack job B: base / A1s1 / A2](../kaggle/heldout_b/heldout.py); [heldout_judge.py](../kaggle/heldout_judge/heldout_judge.py) |
| A1/A2 capability cost | [Full tables](../results/phaseA_capability/README.md) | [phaseA_capability_scores.json](../results/phaseA_capability/phaseA_capability_scores.json) | [Capability protocol + commands](../evals/general/README.md) |
| Single-BOS fine-tuning check (A0) | [Summary JSONL](../results/phaseA_A0/summary.jsonl) | [A0 config + scores](../results/phaseA_A0/summary.jsonl) | Config recorded in summary; see [reproduction notes](REPRODUCE.md). |

## What can be inspected

- Every link above points to public, score-only evidence in this repository. No new data is published by this index.
- Clean audit item verdicts include model/task/item identities and score metadata; selected-item files supply stable IDs and hashes. These support checking aggregates and paired comparisons, not rescoring unpublished raw answers.
- Harm and refusal are separate labels. Read each linked protocol for the prompt set, judge, model revisions and limitations. The 1,000-question MMLU aggregate is not the paired 200-item audit.
- Harmful outputs, attacked weights and refusal-direction tensors are deliberately withheld. Clean audit raw answers are retained privately. Availability of score files is not certification of full reproduction.
- The HF results collection covers earlier runs; use the links above for the later clean audit and Phase A evidence.
