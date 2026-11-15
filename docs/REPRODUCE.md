> **This is the detailed appendix.** The judge-facing summary is in the [README](../README.md).

# Reproduce, artifacts, licences

## 17. Reproduce everything

| What | Where |
|---|---|
| Hardened models v1, v2 (+ `v2b/` subfolder), with model cards | [Models collection](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-models-6af3e1eff9a91556f5535934) |
| Training data; every Heretic journal; judge scores; run logs (numbers only) | [Data & results collection](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-data-and-results-6af3e1f03fff746c25b4884c) |
| Interactive results explorer | [Hugging Face Space](https://huggingface.co/spaces/StarWarrior24x7/heretic-resistant-tiny-aya) |
| Every Kaggle job (`day1`–`day6`, guard judge, Heretic-200, multilingual) | [`kaggle/`](../kaggle) |
| Prefill pilot code and score-only results | [`experiments/`](../experiments) |
| Every run, decision and fix, in order | [`notes/LOG.md`](../notes/LOG.md) |
| Defended checkpoints A1, A2 and their capability cost | [`results/phaseA_capability/`](../results/phaseA_capability/README.md) |
| Figures in this article | `pip install -r requirements.txt && python scripts/make_figures.py` |
| Attack curves from the raw Heretic journals | `pip install optuna && python scripts/pareto.py` |
| Every headline number in the README, rebuilt from `results/` and checked against the text | `python scripts/rebuild_numbers.py` (standard library only; prints each number with its source file) |

Jobs run on free Kaggle T4 GPUs (FP16). **Never published:** stripped models, their raw outputs, refusal-direction tensors and credentials.

**Try v2** (a free T4 is enough; no Hugging Face login needed):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "StarWarrior24x7/tiny-aya-global-xref-v2"  # for v2b, also pass subfolder="v2b"
tok = AutoTokenizer.from_pretrained(repo)         # the tokenizer ships with the model; the base repo is gated
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, device_map="auto")

msgs = [{"role": "user", "content": "Write a Python function that reverses a string."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)  # the template already adds BOS
out = model.generate(**enc, max_new_tokens=256, do_sample=False, eos_token_id=[tok.eos_token_id, 3])
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

- **Single BOS:** the chat template already starts with `<BOS_TOKEN>`. Tokenizing the templated text with the default `add_special_tokens=True` adds a second one, which quietly hurts quality.
- **Stopping:** turns end with `<|END_OF_TURN_TOKEN|>`, which is `tok.eos_token_id`. Pass it to `generate`, as above, so answers stop at the end of the turn.
- **System prompt:** with no system message, the template inserts Cohere's default safety preamble.

## 18. Team and references

**Team**

| Member | Contributions |
|---|---|
| **Shivendu** ([@StarWarrior24x7](https://huggingface.co/StarWarrior24x7)) | Threat demo; v1/v2/v2b defenses; adaptive evaluation; Lexsi toolkit integration and the 4 upstream fixes; single-BOS re-evaluation; Phase A fine-tuning defenses (A1, A2) and held-out tests with predictions and Granite re-judge; article, explorer, brief |
| **Abhinav** ([@laterabhi](https://huggingface.co/laterabhi)) | Granite Guardian harm judging; Heretic-200; multilingual evaluation with Llama Guard cross-check; ROI; judge-style fresh-clone review with consistency and overclaim fixes, the stop-token fix and the HF sync script |
| **Aban** ([@abanwild](https://huggingface.co/abanwild)) | Found the double-BOS bug; prefix-recovery pilot with matched and benign controls; tokenisation fix; ROI model check; frozen paired capability audit of v2, v2b, A1 and A2 (IFEval, GSM8K, HumanEval, MMLU, CyberMetric, XSTest) |

**References**
- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*
- p-e-w, *Heretic*
- *Extended refusal* (arXiv 2505.19056)
- *Fool's Gold* (arXiv 2608.17202)
- Kuo, Yadav and Smith 2026, *ART* (arXiv 2605.26526)
- *DeepRefusal* (arXiv 2509.15202)
- Tamirisa et al. 2024, *TAR*
- *TamperBench* (KDD 2026)
- Qi et al. 2023, *Fine-tuning Aligned Language Models Compromises Safety*
- Röttger et al. 2024, *XSTest*
- Cui et al. 2024, *OR-Bench*
- Padhi et al. 2024, *Granite Guardian*
- Aakanksha et al. 2024, *The Multilingual Alignment Prism* (aya_redteaming)

## 19. Supplementary: models, datasets, licences, glossary

<details>
<summary><b>Models used and their licences</b></summary>

| Model | Role | Licence |
|---|---|---|
| [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) | Base model | CC-BY-NC-4.0 |
| [StarWarrior24x7/tiny-aya-global-xref-v1](https://huggingface.co/StarWarrior24x7/tiny-aya-global-xref-v1) | Our v1 | CC-BY-NC-4.0 (inherited) |
| [StarWarrior24x7/tiny-aya-global-xref-v2](https://huggingface.co/StarWarrior24x7/tiny-aya-global-xref-v2) (root = v2, `v2b/` = v2b) | Our v2 / v2b | CC-BY-NC-4.0 (inherited) |
| [StarWarrior24x7/tiny-aya-global-phaseA-v2-repnoise](https://huggingface.co/StarWarrior24x7/tiny-aya-global-phaseA-v2-repnoise) | A1: v2 + RepNoise | CC-BY-NC-4.0 (inherited) |
| [StarWarrior24x7/tiny-aya-global-phaseA-v2-booster](https://huggingface.co/StarWarrior24x7/tiny-aya-global-phaseA-v2-booster) | A2: v2 + Booster | CC-BY-NC-4.0 (inherited) |
| [ProtectAI/distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1) | Refusal judge | Apache-2.0 |
| [ibm-granite/granite-guardian-3.2-5b](https://huggingface.co/ibm-granite/granite-guardian-3.2-5b) | Harm judge | Apache-2.0 |
| [meta-llama/Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B) | Multilingual harm judge | Llama 3.2 Community Licence |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Translation for judging | CC-BY-NC-4.0 |
</details>

<details>
<summary><b>Datasets and benchmarks used and their licences</b></summary>

| Dataset | Use | Licence |
|---|---|---|
| [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) | Harmful prompts for refusal training; `rejected` responses for the fine-tuning attack (disjoint prompts) | not stated |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Benign prompts (answers self-distilled) | CC-BY-SA-3.0 |
| [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | Borderline-benign training for v2b; 200 held out | CC-BY-4.0 |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors), [harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) | Heretic search prompts; harmful test[:100] | see cards |
| [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | Over-refusal and unsafe-contrast evaluation (never trained on) | CC-BY-4.0 |
| [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming) | Multilingual red-team prompts | Apache-2.0 |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | General knowledge (1,000 questions) | MIT |
| [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | CircuitKIT task; pilot evaluation | MIT |
| Our [tiny-aya-extended-refusal](https://huggingface.co/datasets/StarWarrior24x7/tiny-aya-extended-refusal) | Training data | CC-BY-NC-4.0 |
</details>

<details>
<summary><b>Training configuration</b></summary>

- **Base:** `CohereLabs/tiny-aya-global`, FP16, single T4.
- **LoRA:**
  - r = 32, α = 64, dropout 0.05;
  - on q, k, v, o, gate, up and down projections;
  - 2 epochs, lr 1e-4 with OneCycle;
  - batch 1 × 8 gradient accumulation, max length 1,024;
  - prompt tokens masked.
- **v2 attack simulation:**
  - P(attack step) = 0.5;
  - refusal directions are per-layer mean-difference vectors from 96 harmful and 96 benign prompts, re-estimated every 100 optimizer steps;
  - on each attack step, a direction is taken from a random middle layer, with a peaked per-layer weight profile and top strength 0.7–1.0;
  - forward hooks project it out of every decoder layer.
- **Heretic:**
  - `heretic-llm` 2.1.0, classic abliteration;
  - Optuna TPE, 40 trials (12 random start), objectives = keyword refusals and KL on `harmless_alpaca`;
  - a fresh search per model, plus a 200-trial search on v2.
</details>

<details>
<summary><b>Glossary</b></summary>

- **Open-weight:** the model's weights are published, so anyone can download and edit them.
- **Refusal direction:** one direction in a model's internal activity that mostly controls "saying no".
- **Abliteration:** editing the weights to remove that direction. **Heretic** automates it.
- **KL divergence:** measured distribution shift on a specified harmless-prompt set; not an estimate of compute cost, invisibility or capability loss.
- **Pareto front:** for each level of behaviour shift (KL), the strongest attack found.
- **LoRA:** cheap fine-tuning that trains small add-on matrices.
- **Adversarial training:** training while being attacked, so the model learns to resist.
- **Steering (CAA):** pushing the model's activations in a direction while it runs.
- **Prefill:** forcing the first words of the model's answer.
- **Over-refusal:** refusing safe questions that only sound risky (XSTest).
- **MMLU:** a general-knowledge multiple-choice test.
- **BOS token:** the start marker at the beginning of every input.
- **Confidence interval (95% Wilson):** the likely range of the true rate, given our sample size.
- **Matched control:** an experiment arm identical except for the one thing being tested.
</details>
