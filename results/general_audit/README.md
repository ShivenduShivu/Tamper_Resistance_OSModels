# General evaluation evidence

Start with [the results and limitations](../../docs/GENERAL_CAPABILITY_RESULTS.md). All comparisons are exploratory clean-checkpoint measurements.

- `clean_results.json`: objective scores, paired intervals/discordances and diagnostic flags.
- `selected_items.jsonl`: 2,004 stable item identities plus prompt/content hashes.
- `item_verdicts.jsonl`: 6,012 answer-free rows, including 4,242 objective outcomes.
- `run_provenance.json`: exact model/data/source/runtime/scorer details and private artifact indexes.
- `public_artifact_index.json`: hashes for the four files produced by `export_public.py`.
- `luna_study_summary.json`, `luna_item_labels.jsonl`: GPT-6 Luna sample counts and answer-free judgments; **no human validation**. Disagreement with the classifier is not detector accuracy.
- `luna_study_quality.json`, `luna_practice_audit.json`, `luna_calibration_audit.json`, `math_parser_ai_audit.json`: AI-review limitations, repairs and bounded checks.
- `data_audit.json`, `independent_data_checks.json`, `reference_issue.json`, `coding_backend.json`: data and live-infrastructure audits. These do not establish semantic reference correctness or pretraining independence.

Raw answers, review excerpts/rationales and blinded keys are private. They are preserved on Modal volume `tamper-general-evidence`, run directories `full-20261113-v2-{base,v2,v2b}` and `ai-review-20261113`. Public verdicts permit aggregate reproduction; raw-answer rescoring requires authorized private access. No attacked weights or harmful outputs are published.

See `evals/general/lock.json`, `ai_review_plan.json`, `luna_judge_contract.json`, and the per-run hashes. Frozen historical protocol text mentioning pending human review records the original plan; the final decision was AI-only review, explicitly disclosed.
