# Tamper Resistance for Open-Weight Models: Heretic-Resistant Tiny Aya

Lexsi Labs × Cohere × Hugging Face AI Safety & Alignment Hackathon, **Track 2b: Tamper Resistance**.

## Scope
- **Current scope: defend against data-free weight-editing attacks.** We target abliteration and its automated form, [Heretic](https://github.com/p-e-w/heretic). This is the free, one-command method behind most "abliterated" models on Hugging Face.
- **Next scope: harmful fine-tuning attacks.** Our current models do **not** resist them yet (see Results). This is the open problem we are extending towards, on a separate branch.

## Results so far (Tiny Aya Global, 3.35B)
Each Heretic attack below is a fresh 40-trial search run on that checkpoint, so the attacker adapts to each model. Refusal is measured on `mlabonne/harmful_behaviors` test[:100]. KL is measured on harmless prompts; lower KL means a cheaper, less visible attack.

| Model | Over-refusal (XSTest safe) | Refusal under Heretic, KL ≤ 0.01 | KL ≤ 0.05 | KL ≤ 0.1 | Lowest refusal reached | MMLU (1k) |
|---|---|---|---|---|---|---|
| Original | 12% | **14%** | 14% | 14% | 14% | 52.9% |
| v1: extended-refusal LoRA | 17.6% | 70% | 43% | 17% | 17% | 50.7% |
| v2: v1 + trained through simulated abliteration | 46% | **100%** | **100%** | 96% | 33% | 50.5% |
| v2b: v2 + borderline-benign data (OR-Bench) | 20.8% | 99% | 52% | 47% | 47% | 50.1% |

- **Harmful LoRA fine-tune (100 to 1,000 examples, lr 2e-4):** strips **every** model, ours included (judge-scored refusal 1–10%). This is the limitation the next phase addresses.
- **Caveat:** Heretic refusals are keyword-scored. On clean models, a classifier judge (ProtectAI rejection model) agrees with the keyword counts.
- **Harm judge on attacked outputs (Granite Guardian 3.2-5B):** Heretic's strongest attack makes the original **93% unsafe** and v2 **44%** (XSTest unsafe prompts: 60% vs 7%). Per-file scores in `results/guard_judge/`; Llama Guard pending access.

![Pareto fronts](docs/build/pareto_chart.png)

## Method
1. **Baseline attack:** Heretic (classic abliteration, Optuna TPE, 40 trials) on the original model.
2. **v1:** LoRA SFT (r=32, all linear layers) on 1,200 self-generated *extended refusals* plus 1,155 self-distilled Dolly answers (after arXiv 2505.19056).
3. **v2:** same data, but on 50% of micro-steps forward hooks project a refusal direction out of every layer. The direction is re-estimated from the current model every 100 steps, with randomised layer profiles and strengths. The model must still refuse harmful prompts and answer benign ones (ART / Fool's Gold-style adversarial training, with a refusal target and no hazardous data).
4. **v2b:** v2 plus 1,218 borderline-benign OR-Bench prompts, answered by the original model, to reduce over-refusal.
5. **Evaluation:** a fresh Heretic run on each checkpoint; XSTest over-refusal; OR-Bench held-out over-refusal; MMLU; and a harmful LoRA fine-tune attack at 3 budgets.

## Repository layout
| Path | Contents |
|---|---|
| `kaggle/<job>/` | One folder per Kaggle GPU job (`kernel-metadata.json` + script). Push with `kaggle kernels push -p kaggle/<job>` |
| `results/<job>/` | Run outputs: `summary.jsonl`, Heretic study journals (`ckpt/*.jsonl`), logs |
| `results/pareto_summary.json` | Pareto fronts for all models (`python scripts/pareto.py`) |
| `scripts/` | Pareto parser, guard-judge summary, Kaggle run watchers, output fetcher |
| `notes/LOG.md` | Experiment log: every run, its config, its result and the decision taken |
| `docs/` | Beginner guide to tamper resistance, team brief; `docs/build/` holds the generators |

## Reproduce
- Jobs run on Kaggle T4 GPUs (FP16).
- Each job reads an HF token from a private Kaggle dataset, and the training data from a private Kaggle dataset.
- Hardened weights live in private HF repos.
- Pareto fronts: `pip install optuna && python scripts/pareto.py`.

## Responsible release
- Stripped or attacked checkpoints are never saved or published; only metrics are.
- Harmful prompts come only from public benchmarks.

## References
- Arditi et al. 2024, *Refusal in LLMs is mediated by a single direction*
- p-e-w, *Heretic*
- *Extended refusal* defense, arXiv 2505.19056
- Kuo, Yadav, Smith 2026, *ART*, arXiv 2605.26526
- *Fool's Gold*, arXiv 2608.17202
- Tamirisa et al. 2024, *TAR*
- *TamperBench* (KDD 2026)
- Qi et al. 2023, *Fine-tuning aligned LMs compromises safety*
- Röttger et al. 2024, *XSTest*
- Cui et al. 2024, *OR-Bench*
