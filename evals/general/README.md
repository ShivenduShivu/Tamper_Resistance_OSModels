# Exploratory general-purpose retention panel

This panel tests the original Tiny Aya Global, v2 and v2b on identical frozen
questions. It measures **clean checkpoint** differences. It does not establish
post-tampering retention, general safety, or statistical equivalence.

## Data and outcomes

The v2 frozen panel contains 2,004 items per checkpoint:

| Task | n | Outcome |
|---|---:|---|
| MMLU | 200 | Generated-letter accuracy and restricted forced-choice accuracy, separately |
| CyberMetric | 500 | Cybersecurity MCQ knowledge, same two endpoints; not operational security skill |
| GSM8K | 200 | Frozen exact-match parser; failures/truncations retained |
| GlobalMGSM en/fr/ar | 50 each | Declared language-specific final-answer marker, exact match |
| Official IFEval sample | 200 | Prompt-level strict instruction compliance, not answer quality |
| Aya human-authored everyday set | 20 × 7 languages | Answer collection; useful-answer scoring requires review |
| HumanEval | 164 | Pass@1 through isolated pinned upstream scorer, after live gate |
| XSTest | 250 safe + 200 unsafe | Provisional refusal classifier; human review pending |

Sampling is seed 0, before any generations. Related maths datasets are correlated;
never combine them as independent evidence. Keep per-domain results separate.
Everyday languages: Arabic, English, Portuguese, Telugu, Turkish, Yoruba, Chinese.
These samples do not establish all-language utility. The stock published scores
are background context, not a matched baseline substitute.

Data audit checks the actual 2,355-row published defense training corpus,
normalized duplicates, exact/5-word-shingle Jaccard >=0.8 train overlap, exact
v2b OR-Bench training-source overlap, source identities and hashes, schema and
reference coverage. It preserves all standard HumanEval problems: punctuation
normalization can collapse distinct code semantics. It is lexical screening,
not semantic or cross-translation contamination assurance. The exact accepted
v2b border data and original pretraining data are not available. No semantic
independence or human verification of gold answers is claimed.

## Execution and provenance

`lock.json` freezes dataset/model revisions. `modal_data_audit.py` stores the panel
and full selected content on private `tamper-general-evidence`. `modal_eval.py`
uses the original pinned tokenizer for every model, verifies candidate vocabulary,
chat template and BOS/EOS compatibility, asserts one BOS, greedy FP16, one GPU
worker, and durable per-batch output commits. Saved template strings may differ; every evaluated input must render to identical token IDs under both templates. Generations preserve empty answers,
EOS/truncation, lengths, option logits and overlong exclusions. The source hashes
and complete environment are recorded per run; COMPLETE means **generation only**.

L40S is used after L4, A10 and T4 capacity waits. Larger fixed batches exploit
its memory: MCQ/refusal 64, maths/code 32, instruction/everyday 16. A separate
repeated-input 64-output-token pilot profiles batches 8/16/32/64. These repeated
prompts never enter capability scores. Actual per-task throughput and peak VRAM
are recorded. All formal model runs use the same protocol. FP16 near-ties can
still vary with hardware/batching; compare the recorded settings before inference.
Application retries are disabled; the platform can replay a preempted input, in which case existing run directories fail closed. CPU coding workers are nonpreemptible; run IDs never overwrite. Scheduling is distinct from
execution. Failed launch/probe directories remain failures, not model scores.

`modal_score.py` invokes the hash-pinned official IFEval verifier at the recorded
Google Research commit. It records the NLTK resources, environment and ProtectAI
classifier revision. Refusal labels are **provisional**, with classifier-input
truncation and uncertainty limitations. Correctness and refusal can coexist.
No claim that unsuccessful responses imply inability, or that classifier agreement
is human validation. Generated-letter parse failure is not lost knowledge.
Everyday quality and harmfulness stay pending actual human review.

`modal_coding.py` uses fresh network-blocked, secret-free sandboxes with resource
caps. Authored fixtures, network/secret checks and all 164 canonical solutions
must pass on the real backend before model-written code is scored. It applies the declared single-matching-function adapter in `code_adapter.py`, preserving the original raw answer and rejecting decorators/helpers/signature changes, then calls the
existing pinned upstream HumanEval module; production candidates never execute
on the worker/local machine. The standard non-adversarial scorer can be gamed;
resistance to deliberate scorer subversion is not established.

Download complete directories into **pre-existing local directories**, then
verify every indexed size/hash externally before accepting results. Check panel
hash, selected-item hash, coverage, duplicate IDs, expected conditions, infra
failures and scoring state. Raw outputs/keys/review sheets remain private; only
aggregate reports and code may be committed. Never publish answers from attacked
models or claim all raw outputs are publicly released.

## Analysis

Use paired candidate-minus-original deltas and intervals per task. The imported
Newcombe/McNemar helpers are available; they test differences, not preservation.
All panel comparisons are exploratory; no confirmatory noninferiority margin was
frozen. Report denominators, discordances, exclusions, truncations, parsing failures
and classifier disagreement. A human-audited refusal/error breakdown remains
pending two independent reviewers and adjudication. Keep random samples separate
from disagreement enrichment. Human-review only a subset does not validate all
machine labels. Budget ceiling: $18 Modal plus separate $19 Hugging Face credits;
execution estimates are not provider billing caps.

### Secondary maths-parser sensitivity

The baseline pilot showed numerically meaningful answers that failed strict
format checks (e.g. a final number followed by units). After the pilot and before
full-run scoring, we pinned the published `openai/simple-evals` MGSM numeric
parser as a **secondary sensitivity endpoint**, uniformly applied to all models.
The primary strict parsers are unchanged. The extracted functions, upstream
revision, hashes and MIT licence are in `reference/`. The published parser takes
the last number following the answer prefix (despite its internal comment saying
first), so ambiguous text can still be mis-scored. It is not human ground truth.
Arabic uses the same rule with the dataset's answer prefix, not an official
simple-evals Arabic configuration. Report both endpoints and their disagreement;
do not choose whichever gives a favorable retention conclusion.
