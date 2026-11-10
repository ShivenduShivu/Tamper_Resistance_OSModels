# Contributing

`main` is protected. Nobody pushes to it directly; all changes arrive through reviewed pull requests.

## Workflow
1. Branch from the latest `main`:
   `git switch main && git pull && git switch -c ft-defense/<short-topic>`
   The prefix `ft-defense/` is for fine-tuning-attack defenses; use `exp/` for other experiments.
2. Commit small, logical steps. Each new Kaggle job gets its own folder under `kaggle/<job>/`. Its outputs go in `results/<job>/`.
3. Add every run to `notes/LOG.md`: config, result and decision. Negative results count.
4. Open a pull request into `main` only when there is a result worth merging. Include the numbers and a link to the run.
5. The repo owner reviews and merges. Before asking for review, rebase on `main` or merge `main` into your branch.

## Never commit
- Tokens or keys (HF, Kaggle, W&B). Use Kaggle datasets or secrets instead.
- Model weights (`*.safetensors`, `*.bin`, `*.gguf`), or any stripped or attacked checkpoint.
- Generated training data (it lives in private HF or Kaggle datasets).
- Harmful model outputs. Commit metrics only.
