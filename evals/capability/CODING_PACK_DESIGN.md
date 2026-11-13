# Coding pack design (v4 STANDARD route, pinned upstream module, owner-approved 2026-11-13)

Status: STANDARD ROUTE implemented and offline-validated (95/95 unit tests,
15/15 fidelity fixtures, 164/164 canonical solutions pass). The owner
approved this route on 2026-11-13 (WhatsApp reply to the exact proposal):
the standard openai/human-eval harness semantics with NO claim that the
scorer resists deliberate cheating. No paid run yet - run scope and cost
come back for approval before any spend. No publication.

## What v4 is

The ACTUAL pinned upstream harness, not a reimplementation: the vendored
human_eval/execution.py from openai/human-eval commit
37c4dd63798c3c9ba32fa69a2fb49c5e2c43a181 (MIT; sha256
79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c; see
reference/PROVENANCE.md). coding_exec re-hashes it at import; the per-item
driver re-hashes its in-sandbox copy BEFORE importing it. The verdict for
every item comes from that module's own check_correctness() - assembly,
reliability guard, multiprocessing child, outcome classes are upstream's.

v3 (a hand copy of the same semantics) was discarded after independent
review showed two concrete divergences, both now fixtures:
- a correct completion importing multiprocessing PASSES upstream but failed
  the copy (the copy never pre-imported multiprocessing, so the guard's
  os.getcwd=None broke the import) - multiprocessing_import_ok;
- an invalid-Python DRIVER (infrastructure fault before candidate code ran)
  was scored candidate timed_out/correct 0 - driver_startup_error_infra.
Phase separation fixes the second class, with STRICT frame validation
(second review round): the driver emits HESTARTUP:: only after its own
imports, input parse and module-hash check succeed; trusted-side failures
around the exec phase are caught and emitted as explicit HEINFRA:: frames;
HEVERDICT:: follows check_correctness. The parser requires: a valid startup
frame BEFORE any verdict, attesting EXACTLY the pinned module hash and the
submitted item hash plus a plausible runtime version; upstream's exact
verdict strings with result/passed consistency; no conflicting or
duplicated-distinct frames. Any violation is INFRA. No startup frame ->
INFRA; startup without verdict (hard-killed driver) -> INFRA: driver/sandbox
death is NOT an upstream candidate timeout - only a check_correctness
VERDICT ("passed" / "timed out" / "failed: ...") is a candidate outcome.
Conflicting verdict frames -> timed_out (confusion class, never trusted).

Harness-level deviations from upstream defaults, all outside the module and
frozen in coding_exec.py:
- soft timeout 5s (upstream default 3.0s); hardware-dependent; local
  validation ran on Python 3.10 while the image is 3.13, so timeout
  calibration is NOT established locally - the first authorized run
  re-validates all 164 canonical solutions on the actual image BEFORE any
  model answers (pre-run protocol below).
- the frozen raw-generation extraction rule (documented in coding_exec).
- sandbox hard caps: cpu (0.125, 0.25), memory (128, 512) MiB, lifetime 30s,
  block_network=True, secrets=None, no mounts, fresh sandbox PER ITEM.

## Pre-run protocol (required before ANY model answers, part of the first
   authorized paid check)

1. Backend bring-up probes, each producing recorded evidence: network-block
   probe (outbound connect attempt must fail), resource-cap probe (observed
   cpu/memory limits), cleanup probe (sandbox terminate leaves nothing).
2. Canonical validation ON THE ACTUAL pinned image (Python 3.13): all 164
   canonical solutions must pass; timeout calibrated there, not locally.
3. Modal SDK import/version check against requirements-coding-exec.txt.
Only then do model generations score.

## The accepted assumption (made explicit, owner-approved)

NON-ADVERSARIAL SCORING SCOPE. The same process that runs candidate code
computes the outcome, exactly as the reference harness does. A completion
that deliberately manipulates the interpreter CAN influence its verdict -
fixture ASSUMPTION_forge_passes demonstrates this by forging a pass through
the test's own serializer, and it is expected to pass. This is an
ASSUMPTION/LIMITATION chosen for this measurement, not evidence: in
tamper-resistance research, a model's propensity to subvert measurement is
an open question. Any claim touching that propensity is out of scope for
this pack and needs a separate validated workstream (the v1/v2 custom-runner
review trail documents what that requires).

## Why HumanEval and not MBPP

- openai/openai_humaneval (MIT, arXiv:2107.03374), 164 hand-written Python
  problems, public, not gated. Pinned revision: 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544.
- pass@1 against the dataset's own unit tests IS the standard HumanEval
  protocol. The earlier objection ("disclosed-test pass@1 is not a standard
  MBPP protocol") was MBPP-specific; it does not apply here.
- n = 164 fixed (whole set, no sampling); seeds and sample sizes do not enter.
- Multilingual caveat: problems are English-only. tiny-aya-global's niche is
  multilingual; an English coding score measures capability RETENTION after
  weight edits, not the model's best coding ability. Label results accordingly.

## Measurement protocol (pre-registered shape, sizes are the user's call)

All Wilson intervals reported by this pack are DESCRIPTIVE over the fixed
item set under an item-superpopulation assumption; they are NOT
decoding/model-resampling uncertainty (review note 2026-11-13).

1. Prompt = the dataset's own prompt field (signature + docstring), chat
   template with add_special_tokens=False (the double-BOS rule), greedy
   decoding (temperature 0) so reruns are deterministic, max new tokens 512,
   truncation counted via the existing no-EOS counter, never hidden.
2. Completion extraction: cut at the first line that dedents to column 0 and
   is not part of the function body (standard HumanEval stop rule); the
   extraction code is frozen with tests. The raw generation is length-capped
   BEFORE extraction; an over-cap raw generation scores 0 as
   finish_reason=raw_over_cap (an input-cap rule, not a verdict).
3. Execution: two-process trust boundary below. pass@1 = binary 0/1 per item.
   Timeout/crash/exception = 0, counted, never fatal to the run; per-item
   backend exceptions degrade to infra_error for that item, not a batch abort.
4. Reporting: pass@1 with Wilson 95 CI per model, paired per-item comparison
   vs base on the identical 164 items (same McNemar path as other packs).
   Only metrics and per-item 0/1 are logged for publication; raw generations
   stay local (repo rule, unchanged).

## Two-process trust boundary (v2 history - SUPERSEDED)

The v2 two-process RPC runner was abandoned after independent review: it
could not preserve ordinary Python semantics, and each verdict-integrity
patch opened a new class (shared-FS module injection forged a PASS). The
standard route replaces it in full; the git history retains the trail.

Details live in git history (v1 332e7fb, v2 76c1a07) and in the review
trail: v1 forged a verdict frame in-process; v2 moved the verdict to a
parent process but still diverged from Python semantics (in-place mutation,
codec collisions) and fell to a shared-filesystem module-injection forgery.
Both are superseded; neither ships.

## Isolation boundary (each point maps to a threat)

Generated code is untrusted input. Every layer below must hold; "sandbox" alone
is not the boundary.

| Boundary | Mechanism | Threat addressed |
|---|---|---|
| Network | modal.Sandbox.create(..., block_network=True) - docs: drops ALL outbound traffic; sandboxes have no incoming access and no Modal resource access by default | exfiltration, callbacks, fetching payloads |
| Secrets | the sandbox gets NO env vars, NO HF token, NO Modal token; the assembled program is passed as a literal string | credential theft |
| Filesystem | fresh ephemeral sandbox per item; NO volume mounts, NO repo checkout; /work created with mkdir -p -m 700 and a checked return code before any copy; candidate runs in a fresh tempdir | reading run artifacts, poisoning other items, persistence |
| Compute | cpu=(0.125, 0.25) / memory=(128, 512) MiB (request, HARD limit) at sandbox creation | resource exhaustion billed to us |
| Time | 5s soft SIGALRM limit in the driver (reference uses 3.0s), sandbox lifetime backstop 30s | infinite loops hanging the run |
| Process | the canonical reliability_guard None's destructive os/shutil/subprocess functions and builtins exit/quit before exec (it is NOT a security boundary - it keeps generated code from interfering with the test process); sb.terminate(wait=True) in finally for every item | interference with the test process, orphaned processes |
| Input size | raw generation capped (20000 chars) BEFORE extraction | pathological inputs |

LocalBackend exists ONLY for the authored-fixture self-check
(fixture_mode=True is mandatory) and is not an isolation boundary; a real run
must never route model-written code there.

## Scoring durability (coding_score.py)

- Manifest binding: response ids must match the pinned dataset ids exactly
  (duplicates, extras, or missing all reject); per-item fields_sha256 is
  RECOMPUTED over the manifest's prompt/test/entry_point fields and compared
  against the pinned dataset (a genuine stored label next to altered fields
  is rejected), and scoring uses the trusted pinned-dataset rows.
- APPEND-RESUME checkpointing per item (review-fixed 2026-11-13): a restart
  loads existing rows and validates EACH against the strict outcome schema
  AND this run's full context - model, items-manifest/responses/dataset
  hashes, protocol version, driver sha, runtime image, SDK pin. A stale,
  other-model, other-runtime, malformed or duplicate row refuses closed.
  Completed spend is never rerun; the cell labels n_resumed,
  n_executed_this_invocation and cost_estimate_this_invocation_usd
  separately (cumulative spend = the sum across invocations' records).
- complete_valid requires EVERY intended item scored with a valid candidate
  outcome (n_scored == n_items, zero infra, not aborted, all rows schema-
  clean) - a length-only check previously let a zero-scored file pass.
- Paired comparison is emitted ONLY when BOTH cells are complete_valid with
  identical id sets, identical protocol/driver/image/SDK/manifest/dataset
  binding, and the base scored file's bytes still hash to its own cell
  record; anything else is skipped with a stated reason.
- NaN/inf and concurrency (1-8) validated up front; real ThreadPoolExecutor
  concurrency (default 1 - deterministic order).
- A scored cell is complete_valid only when every item has a candidate
  outcome; infra_error items make it complete_invalid, never silently absent.
- Strict outcome schema (review round 3, 2026-11-13): `correct` must be the
  exact int 0/1 (bool rejected) and MUST equal int(status == "passed") -
  failed/1, passed/0 and True are all refused; infra rows must OMIT correct;
  candidate rows must carry phase/provenance evidence (only the explicit
  raw_over_cap pre-exec case is exempt); each row binds the expected
  per-item input hash (recomputed from the trusted pinned-dataset row) and
  the expected Python target (3.13, with an explicit local-fixture
  exception). A wrong module hash or python target is INFRA, never a
  candidate outcome.
- Immutable measurement context (review round 3): every row and resume
  binds the FULL context - model id + model revision + generation manifest
  sha + items/responses/dataset hashes, protocol, driver sha, extractor
  (coding_exec) sha, scorer sha, measurement-config sha (cap, cpu, memory,
  lifetime; timeout is covered by the driver sha, not duplicated), runtime
  image, SDK pin. Paired comparison requires the same binding on BOTH
  cells.
- Sandbox identity (review round 6): the exact Modal sandbox object ID is
  ledgered (state=created) the MOMENT Sandbox.create returns, before any
  exec; billing reconciliation binds to THAT sandbox. If the create
  response is lost the item stays pending = unknown and is reconciled,
  never blindly recreated.
- Lock-before-snapshot (review round 5 TOCTOU): the flock is taken BEFORE
  any mutable read - the resume snapshot (load_resume) and the ledger fold
  (reconcile_ledger) both happen UNDER the held flock. Previously a second
  invocation could snapshot an empty scored file pre-lock, watch the first
  finish, then acquire and rerun paid items (deterministic full-main race
  repro: 4 persisted duplicate rows from 2 items). Now a concurrent second
  is refused as a live writer; a later one resumes the finished rows. The
  flock is released in a finally on EVERY exit so an abortive SystemExit
  cannot leak it in-process; ledger obligations persist regardless.
- Writer lock + billing ledger (review round 4, rewritten): the lock is an
  OS-level flock held for the WHOLE run, so a later invocation can always
  tell a LIVE writer (flock fails -> refuse; --ack-interrupted NEVER
  overrides a live writer, under any context) from a DEAD one (OS released
  the flock -> reconciliation path). The immutable run context is ALSO
  persisted in a durable .context.json that survives lock-file removal and
  is compared before any acquire or ack: a different-context invocation
  refuses. The billing ledger is read BEFORE the lock is touched, so
  manually removing the lock file cannot erase an unresolved
  pending-without-done obligation - it still refuses until
  --ack-interrupted (recorded in the ledger). Malformed ledger lines fail
  CLOSED (a skipped line could hide a pending obligation). Each item is
  ledgered pending BEFORE create and done only after the result row is
  checkpointed. RESIDUAL WINDOWS, stated honestly: between sandbox create
  and the done-append an interruption leaves a created, possibly-billed
  item marked pending (that is exactly what reconciliation before rerun is
  for), and no exactly-once network guarantee is claimed - the mechanism
  narrows the lost-binding window to item granularity and forces an
  explicit decision instead of automatic rebilling.
- Finalization consistency (review round 4): the trusted per-item expected
  input mapping (recomputed from the pinned-dataset row + bound generation
  + current extractor) is built ONCE and used by BOTH the resume loader
  and the final whole-file validation - the final pass previously called
  the schema check without it, rejecting every valid phase-evidenced row
  and marking 164 successes complete_invalid.
- Generation-manifest validation (review rounds 4-5): scoring refuses
  BEFORE any costly step unless the manifest is finalized, COMPLETED and
  self-consistent: the expected model entry with repo + 40-hex revision
  ("unknown"/empty no longer passes), the tokenizer pin (repo + 40-hex
  revision), the environment block, humaneval task coverage, the
  completion markers written only after generation (the model's
  effective_eos_ids and per-model generation counters), reconciled config
  (positive gen_max_new_tokens; the seed read from argv.seed, matching
  run.py's actual writer schema), and the embedded lock's
  dataset pin equal to the scorer's pinned revision with the lock's model
  revision equal to the model entry's. A bounded preflight fixture
  manifest carries the same markers explicitly when authored.
- Scored output writes items_humaneval_scored__<model>.jsonl and never
  overwrites the generation-time items file.
- Runtime pinning (2026-11-13): the BASE image is pinned by DIGEST
  (python:3.13-slim@sha256:48d88481..., linux/amd64, Docker Hub tag API
  2026-11-13) with NO add_python layer (it would be unpinned), and the Modal
  SDK pin (modal==1.6.4, requirements-coding-exec.txt) is ENFORCED - the
  scorer and the backend both refuse a mismatched SDK before any sandbox is
  created. The FINAL runtime (base plus any Modal-injected layers) is NOT
  pinned or observed yet: it is recorded and verified at the first
  authorized run, along with the pulled-digest match and the 3.13-vs-3.10
  canonical re-validation (pre-run protocol).

## Offline validation (all pass as of v4 round 5)

- 110 unit tests (test_suite.py), including fully mocked MAIN-level
  runs (positive, resume, interrupt+reconcile, live-writer refusal,
  concurrency, staggered two-invocation TOCTOU regression) - not
  helper-only: extraction, phase-aware frame parsing
  (startup/verdict precedence, strict evidence binding, order/schema/result
  consistency, confusion class, HEINFRA mapping), strict resume schema +
  run-context binding (incl. the reviewer's bare-row and stale-row repros),
  payload shape, cost bound, loader validation, binding (including the
  altered-fields repro), reference-module/image/SDK pins (source-checked:
  no add_python layer, enforced version compare), export-check tiers,
  scoring edge cases.
- 18 fidelity fixtures (coding_score.py --self-check) against LocalBackend:
  pass/fail/exception/hang, in-place mutation and tag-shaped dict fidelity
  (RPC-era repros), exit classes, child segfault, the reviewer's exact v3
  repros (multiprocessing-import correct completion PASSES; invalid driver
  is INFRA; trusted exec-phase fault emits HEINFRA and is INFRA),
  candidate-kills-driver post-startup (INFRA - driver death is not a
  candidate outcome), output flood swallowing, tempdir file write, network
  attempt -
  plus ASSUMPTION_forge_passes, which demonstrates the accepted
  non-adversarial scope (a wrong answer passing via interpreter manipulation
  under the REAL upstream module) and is EXPECTED to pass.
- Canonical self-check: all 164 dataset canonical solutions pass under the
  harness caps, scored by the pinned upstream module itself. This validates
  harness fidelity against ground truth ON LOCAL PYTHON 3.10; it says
  nothing about any model, it is not evidence of adversarial robustness,
  and it does not calibrate the 3.13 image (pre-run protocol).

## Execution cost shape

Modal Sandbox rates are 3x the function rates ($0.00003942/core-s CPU,
$0.00000667/GiB-s memory; reviewer-verified against modal.com/pricing). The
compute-time bound per item is 0.25 core x 30s + 0.5 GiB x 30s = $0.0003957;
492 items (164 x 3 models) bounds compute at $0.1947. This is a compute-only
bound, NOT a spend ceiling: sandbox startup, network transfer, and image
retention overheads come on top and are accounted before any authorized run.
Generation still runs on the panel's T4; execution is a separate,
network-free phase that can run after the GPU is released.

## Explicitly out of scope

- MBPP (protocol objection stands until a reviewed design addresses it).
- Multi-sample pass@k (k>1) or temperature sweeps.
- Executing anything beyond the dataset's own tests.
- Any claim about operational cybersecurity ability: this pack plus the
  CyberMetric MCQ pack measure code-generation correctness and security
  KNOWLEDGE recall respectively; neither measures operational security skill.
