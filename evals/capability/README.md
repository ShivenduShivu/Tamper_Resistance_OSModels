# Capability eval suite (bounded, UNRUN DRAFT)

Status: the imported scaffold and live backend were exercised by the [general panel](../general/README.md). [Completed clean-checkpoint results](../../docs/GENERAL_CAPABILITY_RESULTS.md) include all 164 HumanEval problems per model. The older public-data preflight alone remains infrastructure evidence, not model performance. The standalone legacy entrypoint is not the formal general-panel run.

Why this exists. The official Track 2b statement is: **"Tamper resistance: make safety survive later harmful fine-tuning by someone else, and demonstrate it with attack fine-tunes."** It asks for safety survival, nothing about benchmarks. The capability side enters only through our own article's claim *"without becoming less useful"*. This suite exists to make THAT claim defensible with measured, paired evidence - it is not a general model leaderboard.

## Scope rule

Bounded by the claim being defended:

| Pack | What it answers | Default n |
|---|---|---|
| `mmlu` | Anchor at the article's n=1000, letter-logit MCQ, now with per-item logging and stable item IDs | 1000 |
| `gsm8k` | Generative math under a declared exact-match protocol - a capability letter-logit MCQ cannot see | 200 |
| `ifeval_custom` | Instruction-format following - closest cheap probe to "useful assistant" | up to 200 eligible rows |
| `global_mmlu_<lang>` | Multilingual knowledge in hi/ar/fr/es (official lowercase configs) - the model's niche; the defense trained English-only | 200/lang |
| `cybermetric` | Cybersecurity knowledge MCQ (tihanyin/CyberMetric, pinned) - knowledge recall, not operational skill | 500 |
| `humaneval` | Code generation pass@1 (openai/openai_humaneval, pinned) - functional correctness, non-adversarial scope (owner-approved) | 164 |

**Coding pack status: STANDARD ROUTE, pinned upstream module,
offline-validated, no run yet.** The owner approved the standard
openai/human-eval runner on 2026-11-13 (WhatsApp reply to the exact
proposal), with no claim that the scorer resists deliberate cheating. Every
verdict is computed by the vendored, hash-pinned upstream module
(reference/human_eval_execution.py @ commit 37c4dd63, MIT, sha256 79901c6f...
- verified at import and again inside the sandbox), called via its own
check_correctness(), inside a fresh Modal Sandbox per item (digest-pinned base
image with no unpinned layers, enforced SDK pin, no network, no secrets,
hard cpu/memory caps; the final runtime is recorded at the first run). The non-adversarial
scoring scope is an ASSUMPTION/LIMITATION, not a finding - in
tamper-resistance research, a model's propensity to subvert measurement is
an open question, and any claim touching it is out of scope here. Driver
phase separation keeps infrastructure faults (no startup frame) out of
candidate outcomes. Offline validation is green (110 unit tests, 18 fidelity
fixtures incl. every review repro, 164/164 canonical solutions scored by the
pinned module itself). No scored run: run scope and cost come back for
approval before any spend, and the first authorized run re-validates
canonical + backend probes on the actual 3.13 image BEFORE any model
answers (see CODING_PACK_DESIGN.md).

Every model answers the SAME items (mandatory pinned dataset revisions in `dataset_lock.json` - loaders raise on a missing revision, "latest" is never used), fixed seed, stable source-row identity (`split#index` under the pinned revision, or the dataset's own key), so differences are analyzed paired: `paired_vs_base.json` reports each candidate's accuracy delta vs base (**candidate minus base**: positive = candidate better) with a Newcombe method-10 paired CI and a full-precision exact McNemar p-value.

## Naming honesty

- `ifeval_custom` is a set of **custom deterministic format checks on IFEval prompts**, not the official IFEval verifier registry. Word counting, keyword frequency and bullet matching differ from the official implementations (the end-phrase check mirrors it). Cite it as "custom instruction-format checks on IFEval prompts", never as IFEval. The run manifest reports how many of the 541 rows are eligible and which instruction types survive.
- MMLU here is a **new measurement**, not a re-run of the article's number. The article's 52.3/53.0 came from a run whose item manifest was never persisted, so exact comparability is unrecoverable; base is re-run inside this harness and all comparisons are internal to it.
- Letter-logit MCQ scoring is **restricted forced-choice** accuracy on the first generated position, not natural assistant answers. Per item we also record the model's actual top-1 token and whether it was a letter, so format drift is visible instead of hidden.
- **Clean task accuracy is not benign helpfulness.** An irrelevant answer can pass keyword/format checks; a safe refusal can fail them. The existing over-refusal evidence (XSTest safe-prompt refusal 39% v2 vs 16% base, v2b 18%) stays the helpfulness-cost thread and MUST be reported alongside any numbers from this suite, not replaced by them.

## Pre-registered reading (fix BEFORE running)

- One **primary comparison** is declared up front via `--primary <task>:<model>` (e.g. `mmlu:v2`). Everything else is labeled `exploratory` in the output; selecting one primary does not turn the remaining unadjusted comparisons into confirmatory conclusions.
- The decision rule is directional: **detected_loss** = exact McNemar p < 0.05 AND the Newcombe CI excludes 0 AND delta < 0; **detected_gain** = same with delta > 0. Anything else is reported as **"not detected at this resolution (n=..., discordant=...)"**, never as preservation. No equivalence claim without an equivalence test.
- "Not detected" is not "no difference". Resolution depends on sample size, observed discordance and the effect size you care about, not just a marginal interval width: as rough orientation, a 50% rate at n=200 carries about +/-7 points of marginal uncertainty, and at n=1000 about +/-3.1 points; the paired design sharpens this when the models agree often, which hardened variants of the same base usually do.
- The extraction/scoring parser is FROZEN before any measurement run: one identical protocol applied to every model, declared in `scoring.py`. A failed extraction is its own reported outcome, never silently relaxed because one model trips it more, and never read as math incapability. Changing the protocol after seeing outputs requires a new pre-registration applied uniformly, and the preserved raw responses are what make any such change auditable. A smoke run is a pilot of the machinery - it validates execution and exposes counter/parser behaviour, and confirms nothing about capability.

## Run layout and provenance

Each invocation creates a NEW unique directory `<out>/run-<utc>-<rand>/` and fails rather than overwrite; there is no resume. It contains:

- `manifest.json` - CLI args, seed, pinned revisions, sha256 of the suite's own source files, environment versions (python/torch/transformers/datasets), generation budgets, batch sizes, skip policy, per-task eligible/excluded/selected counts and per-model counters (extraction failures/fallbacks, empty generations, no-EOS truncations, overlong skips). Also rewritten on failure, so a partial run is auditable.
- `items_manifest_<task>.json` - the frozen selected rows with source identity and full content, written before any model loads.
- `summary.jsonl` - metrics only. `items_*.jsonl` - per-item ids + correctness + MCQ option scores/top-token + generative token counts and eos_reached. `responses/` - raw generations, LOCAL ONLY, never published (same rule as the safety evals).
- `COMPLETE` written only when everything finished; `FAILED` with the error otherwise. A directory without `COMPLETE` is not a result.
- Empty packs fail closed: a task with zero scored items raises instead of reporting 0%.
- Durable storage is REQUIRED: the output directory must be a mounted volume or equivalent persistent path, never container-local scratch. A run whose artifacts vanish with the container cannot be audited - raw responses are what make every extraction decision and counter checkable after the fact. (Lesson from the first cloud smoke: metrics printed to stdout survived, the container-local responses/ did not.)

External readback procedure (mandatory before any run is called done): the launcher writes artifact_index.json (sha256 + size per file, fail-closed evidence state) and commits the volume. After the FunctionCall returns, the operator downloads the run directory from OUTSIDE the container (`modal volume get capability-eval-artifacts <run-dir>`) and re-verifies every file against artifact_index.json with export_check.verify_against_index. A directory listing (modal volume ls) is presence-only and never sufficient. The run's evidence verdict is artifact_index.json's state field: complete_valid, complete_invalid, partial_failure, or invalid.

Conventions inherited from the repo: FP16, left padding, greedy decoding, `add_special_tokens=False` everywhere (the double-BOS lesson, notes/LOG.md) - enforced by a one-BOS assertion on every run, not just in smoke. MCQ scoring reads logits straight from the model's own forward pass (`logits_to_keep=1`, ~2 MB per batch); there is no reimplemented lm_head path to verify, and non-finite logits abort the run. (An earlier reimplemented lm_head path twice failed the suite's full-vs-fast equality guard on GPU; the exact diff was never captured and fp16 GEMM-shape numerics are only a hypothesis, so the shortcut was removed rather than re-patched.)

## Preflight first (free, no GPU)

```
HF_TOKEN=... python run.py --smoke
```

Checks the tokenizer invariants (one BOS, single-token letters, pad), the live dataset schemas for every pack, and that every pinned lock revision resolves. The tokenizer download needs an HF token because `CohereLabs/tiny-aya-global` is gated. Then a real run:

```
python run.py --models base v2 v2b --tasks mmlu gsm8k ifeval_custom \
    --primary mmlu:v2 --out /kaggle/working/capability
```

## What this suite does not do

- It measures CLEAN models only. The stripped/attacked checkpoints from the defense runs were never archived. Capability at ABLITERATION attack points would mean re-applying the archived Heretic trial parameters exactly as `kaggle/day6` does - a separate scoped plan with its own approval and budget, and not the same thing as the statement's harmful fine-tuning attack, which is its own open evidence gap. This suite commits to neither.
- It does not establish broad usefulness, does not fix over-refusal, and does not answer the official harmful-fine-tune-survival requirement. Those are separate evidence threads.
- "Not detected" is never reported as "no capability loss".

## Tests

`python3 test_suite.py` - 111 offline tests: extraction edge cases (fractions, leading-dot decimals, exponents on every path, ambiguous/negated marker lines, mixed numbers, exact rational equality), fail-closed verifier validation (bad types, non-positive counts, empty instruction lists), empty-pack behavior, and the paired statistics (the published Newcombe reference example, boundary 100/0 case, identical/opposite/small-discordance counterexamples, sign convention, full-precision p). No GPU or network.
