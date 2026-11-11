# Heretic-Resistant Tiny Aya: training a small open model to keep its safety when someone tries to edit it out

*Lexsi Labs × Cohere × Hugging Face AI Safety & Alignment Hackathon, Track 2b: Tamper Resistance*

> **TL;DR.** With [Heretic](https://github.com/p-e-w/heretic), a free one-command tool, we stripped Cohere's Tiny Aya Global of its safety: judged refusal went from 99% to **1%**, with no loss on MMLU. We then trained Tiny Aya *while simulating that attack*. Against our model, the same attack at the same damage budget leaves refusal at **100%**. Heretic's strongest attack still only reaches 23%, and needs about 16× more distortion of the model's normal behaviour. Two Lexsi toolkits (SafeTune and AuditKIT) independently confirm the effect. The limits are real, and we report them: harmful fine-tuning and prefilling still break the model.

---

## 1. Problem and approach in brief

**The problem.** Open-weight models ship with safety training, but the weights are public. Arditi et al. (2024) showed that refusal lives mostly in **one direction** inside the model's activations. Remove that direction from the weights, a technique called *abliteration*, and the model stops refusing. **Heretic** automates this. It searches for the abliteration that removes the most refusals while changing normal behaviour the least, measured by KL divergence on harmless prompts. It needs no data and no expertise, and it runs in about an hour on a free GPU. Many "abliterated" models made this way are already on the Hub.

**Our question.** Can a small open model be made to keep its refusals against the attack people actually use, without a big hit to usefulness?

**Approach.**
1. **Measure the threat.** We ran Heretic against Tiny Aya Global, with 40 optimisation trials.
2. **v1: extended refusals.** We trained the model on long, explained refusals, which spreads refusal over more directions (arXiv 2505.19056).
3. **v2: train through the attack.** On half of the training steps, we remove a refusal direction from every layer *during the forward pass*, and the model must still refuse harmful prompts and answer benign ones. The direction is re-computed from the current model every 100 steps, so the simulated attacker keeps up with training. This is adversarial training in the style of Fool's Gold and ART, adapted to a refusal target with no hazardous data.
4. **v2b: reduce over-refusal.** v2 plus 1,218 borderline-but-safe prompts, so the model doesn't over-refuse.
5. **Attack every model adaptively.** We ran a fresh Heretic search on each checkpoint, plus several other attacks (section 7).

## 2. Hackathon repos used

| Repo | What we used it for | Features used |
|---|---|---|
| **SafeTune** | An independent second implementation of the attack, steering attacks, and the mechanism measurement | `extract_refusal_direction` (per-layer refusal directions); `RefusalDirectionModel(mode="ablate")` (inference-time direction ablation as an attack); `extract_caa_vectors` + `CAAModel` with negative strength (activation-steering attacks) |
| **AuditKIT** | Independent red-team probes on every model | `RedTeamRunner` with the jailbreak, encoding, prompt-injection and refusal probes. While using it we found and fixed two bugs: [AuditKIT #35](https://github.com/Lexsi-Labs/AuditKIT/pull/35) and [#36](https://github.com/Lexsi-Labs/AuditKIT/pull/36) |
| **CircuitKIT** *(attempted)* | Refusal-circuit comparison, original vs v2 | `SafetyPromptAdapter` with `pairing_mode="harmful_vs_benign"` (built correctly), `load_model` (Cohere port), `discover` (EAP / EAP-IG). Discovery ran out of memory on a 16 GB T4 for the 3.35B model with its 262k vocabulary. We also found that the public `register_*` helpers only build degenerate prompt pairs; see section 7.4 |

SafeTune and AuditKIT form one connected chain with our pipeline. SafeTune extracts the refusal direction, which is the same object our training-time attack removes, and then attacks each checkpoint with ablation and steering. AuditKIT and the classifier judge score the outputs.

## 3. Models used and their licences

| Model | Role | Licence |
|---|---|---|
| [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) | Base model | CC-BY-NC-4.0 |
| [StarWarrior24x7/tiny-aya-global-xref-v1](https://huggingface.co/StarWarrior24x7/tiny-aya-global-xref-v1) | Our v1 | CC-BY-NC-4.0 (inherited) |
| [StarWarrior24x7/tiny-aya-global-xref-v2](https://huggingface.co/StarWarrior24x7/tiny-aya-global-xref-v2) (root = v2, `v2b/` = v2b) | Our v2 / v2b | CC-BY-NC-4.0 (inherited) |
| [ProtectAI/distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1) | Refusal judge | Apache-2.0 |

## 4. Datasets and benchmarks used, and their licences

| Dataset | Use | Licence |
|---|---|---|
| [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) | Harmful prompts for the refusal training data; `rejected` responses for the fine-tuning attack (prompts disjoint from training) | not stated on the card |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Benign prompts (answers self-distilled from Tiny Aya) | CC-BY-SA-3.0 |
| [bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench) | Borderline-benign training prompts for v2b; 200 held out for testing | CC-BY-4.0 |
| [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors), [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) | Heretic's search prompts; harmful test[:100] | see cards |
| [natolambert/xstest-v2-copy](https://huggingface.co/datasets/natolambert/xstest-v2-copy) | Over-refusal (safe) and unsafe-contrast evaluation; never trained on | CC-BY-4.0 |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | General ability (1,000-question sample) | MIT |
| [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | CircuitKIT circuit discovery | MIT |

## 5. Model artifacts (HF collection)

**Collection:** [Heretic-Resistant Tiny Aya: Models](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-models-6af3e1eff9a91556f5535934). It contains xref-v1, xref-v2 (with v2b) and their model cards. **Stripped or attacked models are never published.** They are exactly the harm this project defends against, and we never saved them beyond the evaluation run.

## 6. Data artifacts (HF collection)

**Collection:** [Heretic-Resistant Tiny Aya: Data & Results](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-data-and-results-6af3e1f03fff746c25b4884c). It contains:
- the training data ([tiny-aya-extended-refusal](https://huggingface.co/datasets/StarWarrior24x7/tiny-aya-extended-refusal));
- [results, attack logs and benchmarks](https://huggingface.co/datasets/StarWarrior24x7/tiny-aya-tamper-resistance-results): all Heretic study journals (every trial, with its parameters, refusal and KL);
- run summaries and logs, and the evaluation code.

Raw responses from attacked models are kept private and are available to judges on request. We publish scores, not harmful text.

## 7. Experiments, findings, and what else we tried

### 7.1 The threat is real: Tiny Aya is stripped for free
| | Refusal (judge) | MMLU |
|---|---|---|
| Tiny Aya Global | 99% | 52.9% |
| After Heretic's best attack (KL 0.009) | **1%** | 52.7% |

Heretic's own keyword metric reported 14%. Our classifier judge shows the strip is almost total: keyword matching counts phrases like "illegal" as refusals even when the model complies.

### 7.2 Training through the attack works against direction-removal attacks
Lowest refusal Heretic reached within a given KL budget (Heretic's keyword scorer, 40-trial search per model):

| Model | KL ≤ 0.01 | KL ≤ 0.05 | KL ≤ 0.1 | Best attack found |
|---|---|---|---|---|
| Original | 14% | 14% | 14% | 14% (judge: **1%**) |
| v1 (extended refusal) | 70% | 43% | 17% | 17% |
| **v2 (trained through abliteration)** | **100%** | **100%** | **96%** | 33% (judge: **23%**) |
| v2b (+ borderline data) | 99% | 52% | 47% | 47% (judge: 24%) |

![Pareto fronts](https://raw.githubusercontent.com/ShivenduShivu/Tamper_Resistance_OSModels/main/docs/build/pareto_chart.png)

**Is the attacker just paying more KL?** Yes, and that is the honest framing. At its strongest attack on v2, Heretic needs about 16× the KL. MMLU stays at 51.0%, so general knowledge survives, but the model still refuses 23% of harmful prompts and 62% of XSTest's unsafe prompts (the attacked original: 1% and 17%). The defense raises the cost of a strip and makes it less complete. It does not make the model impossible to strip.

### 7.3 Independent confirmation with the Lexsi toolkits, plus other attacks
Judged refusal on 100 harmful prompts (higher is safer):

| Attack | Original | v2 | v2b |
|---|---|---|---|
| SafeTune refusal-direction ablation | 28% | **100%** | **100%** |
| SafeTune CAA steering, strength −1 | 4% | **42%** | 30% |
| SafeTune CAA steering, strength −4 | 0% | 0% | 0% |
| Sampling attack (T=1, success in any of 3 tries) | 9% succeed | **0% succeed** | 2% succeed |
| 4-bit quantised (bitsandbytes NF4) | 99% | **100%** | **100%** |
| Prefill "Sure, here is" | 31% | 38% | 41% |
| Harmful LoRA fine-tune, 100 examples | 1% | 3% | 1% |
| AuditKIT prompt-injection probes (pass rate) | 62.5% | **87.5%** | 75% |

**What generalises:** attacks that work by removing or steering the refusal direction (Heretic, SafeTune ablation, mild CAA), plus sampling and quantisation.
**What doesn't:** harmful fine-tuning, prefilling and strong steering. These are the known open problems in tamper resistance (Qi et al. 2023; TamperBench), and they are where we go next.

### 7.4 Mechanism
- **Refusal direction (SafeTune):** v2's refusal direction is 10–20% weaker in the middle and late layers than the original's (layer 24: 40.2 vs 46.4). It peaks at the same layer (18). Our reading is that v2 stops relying on a single dominant direction, so removing it no longer removes refusal.
- **CircuitKIT (attempted):**
  - **Building the task:** CircuitKIT's public safety-dataset helpers only produce prompt pairs whose corrupt prompt equals the clean prompt, which `discover()` rejects. We built the task directly with `harmful_vs_benign` pairing instead.
  - **Running discovery:** it still needed more than the 16 GB of a free T4 for a 3.35B model with a 262k-token vocabulary. Circuit-level analysis is first on our list if we get more compute.

### 7.5 What else we tried or considered
- **v1 alone** delays Heretic, but it still reaches 17%. Extended refusals are not enough by themselves.
- **v2 over-refused** safe prompts (46% on XSTest). v2b fixes most of this (20.8%) but gives up some low-budget robustness. We report v2 and v2b as two points on a safety vs usability curve.
- **Dropped ideas:**
  - A multilingual/Hinglish attack angle: we found no evidence it reflects real-world attacks.
  - A straight Fool's Gold replication: it failed at 9B, needs hazardous data, and the code is withheld.

## 8. With more credits: approach and expected before/after
- **Fine-tuning resistance.** Simulate a few LoRA attack steps inside training, the same way v2 simulates abliteration (a lightweight version of TAR). Our teammates are already exploring this on a separate branch.
- **Prefilling robustness:** train on "recovery" examples, where the model starts complying and then stops and refuses.
- **Scale and multilinguality:** repeat on Aya Expanse 8B, and evaluate in the 70+ Tiny Aya languages.
- **Longer Heretic searches** (200+ trials) and a stronger judge (Llama Guard / Granite Guardian, being run by a teammate).

**Cost so far:** about 25 GPU-hours on free Kaggle T4s. One defense costs about 2–3 GPU-hours to train, and one adaptive Heretic evaluation about 1 GPU-hour.

## 9. Team
*Tag teammates here: @…, @…, @…*

---

### Threat model
The attacker has the full weights and a free GPU, and uses public tools: Heretic, steering and fine-tuning. The defender controls only the released weights. We claim robustness to **data-free direction-removal attacks** (abliteration/Heretic and inference-time ablation). We do **not** claim robustness to fine-tuning or prefilling.

### Responsible release
- Only hardened models, scores and code are public.
- Stripped models were created only inside the evaluation runs and deleted.
- Raw outputs of attacked models are private.
- All harmful prompts come from existing public benchmarks.

### Reproduce
Code, configs, Heretic journals and run logs: [github.com/ShivenduShivu/Tamper_Resistance_OSModels](https://github.com/ShivenduShivu/Tamper_Resistance_OSModels). Every Kaggle job is in `kaggle/<day>/`, and `notes/LOG.md` records every run and decision.

### References
- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*
- p-e-w, *Heretic*
- *Extended refusal* (arXiv 2505.19056)
- *Fool's Gold* (arXiv 2608.17202)
- Kuo, Yadav, Smith 2026, *ART* (arXiv 2605.26526)
- Tamirisa et al. 2024, *TAR*
- *TamperBench* (KDD 2026)
- Qi et al. 2023, *Fine-tuning Aligned Language Models Compromises Safety*
- Röttger et al. 2024, *XSTest*
- Cui et al. 2024, *OR-Bench*
