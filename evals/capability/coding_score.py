"""Coding-pack scorer: executes HumanEval completions in isolated sandboxes.

Separate phase from generation (run.py's GPU job). Reads the run dir's frozen
items_manifest_humaneval.json + responses/humaneval__<model>.jsonl, verifies the
binding (manifest field hashes == pinned dataset recomputation; response ids ==
manifest ids exactly; duplicates/extras rejected), then scores each item via
coding_exec (fresh sandbox per item; the pinned upstream human_eval module
computes every verdict) and APPEND-checkpoints every result atomically as it
lands - a restart resumes completed items, never reruns them. A final cell is written in a finally-safe closeout even on abort, with
state complete_valid only when every item has a candidate outcome and zero infra
errors. Can run after the GPU is released.

Usage:
  python coding_score.py --run-dir RUN --model base
  python coding_score.py --self-check                  # fixtures, LOCAL, offline
  python coding_score.py --self-check --full-canonical # + all 164 canonical, LOCAL

Self-check is the pre-scoring gate and runs on the LOCAL backend with authored
fixtures only (never model output). It evidences harness LOGIC; the Modal binding
is live-verified at the first authorized run.

Spend: --max-cost-usd is a PER-INVOCATION budget tripwire checked before each
sandbox starts, using the caps-based compute-only estimate (see coding_exec);
it is not an aggregate spend ceiling (provisioning/overhead excluded, and it
resets on restart). Aggregate spend is governed by the approved scope only.
NO retries of anything.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time

import coding_exec as cx
import scoring
import tasks

INFRA = "infra_error"
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_lock.json")

# Coding scoring endpoint status: the owner approved the STANDARD
# openai/human-eval runner route on 2026-11-13 (WhatsApp reply to the exact
# proposal), with NO claim that the scorer resists deliberate cheating. v4
# executes the ACTUAL pinned upstream module (reference/human_eval_execution.py,
# commit 37c4dd63, sha-verified at import and in-sandbox). Paid compute and PR
# posting remain separately gated (run scope/cost first).
CODING_ENDPOINT_STATUS = ("STANDARD-ROUTE owner-approved 2026-11-13; pinned "
                          "upstream module; offline-validated only - no scored run yet")

# ---------------------------------------------------------------- fixtures
# Authored by us for the local self-check; never model output. These validate
# harness FIDELITY to the standard semantics. They are NOT adversarial-
# integrity tests: the last fixture documents the accepted non-adversarial
# assumption by showing that a deliberate in-process forgery DOES pass.

FIXTURE_PROBLEM = {
    "task_id": "fixture/0",
    "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    "test": ("def check(candidate):\n"
             "    assert candidate(1, 2) == 3\n"
             "    assert candidate(2, 2) == 4\n"),
    "entry_point": "add",
    "canonical_solution": "    return a + b\n",
}

MUT_PROBLEM = {
    "task_id": "fixture/mut",
    "prompt": "def mutate(xs):\n    \"\"\"Append 1 in place.\"\"\"\n",
    "test": ("def check(candidate):\n"
             "    x = []\n"
             "    candidate(x)\n"
             "    assert x == [1]\n"),
    "entry_point": "mutate",
    "canonical_solution": "    xs.append(1)\n    return None\n",
}

DICT_PROBLEM = {
    "task_id": "fixture/dict",
    "prompt": "def value():\n    \"\"\"Return the marker dict.\"\"\"\n",
    "test": "def check(candidate):\n    assert candidate() == {'__tuple__': [1]}\n",
    "entry_point": "value",
    "canonical_solution": "    return {'__tuple__': [1]}\n",
}

FORGE_PROBLEM = {
    "task_id": "fixture/forge",
    "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    "test": ("def check(candidate):\n"
             "    import json\n"
             "    got = candidate(1, 2)\n"   # mutation happens inside the call
             "    assert json.dumps(got) == json.dumps(3)\n"
             "    assert json.dumps(candidate(2, 2)) == json.dumps(4)\n"),
    "entry_point": "add",
    "canonical_solution": "    return a + b\n",
}

FIXTURES = {
    # name: (problem, completion, (kind, status)); status None = any candidate status
    "pass_canonical": (FIXTURE_PROBLEM, "    return a + b\n", ("candidate", "passed")),
    # control: plain wrong answer must fail
    "fail_wrong_answer": (FIXTURE_PROBLEM, "    return a - b\n", ("candidate", "failed")),
    "fail_exception": (FIXTURE_PROBLEM, "    raise ValueError('boom')\n", ("candidate", "failed")),
    "fail_hang": (FIXTURE_PROBLEM, "    while True:\n        pass\n", ("candidate", "timed_out")),
    # standard-semantics fidelity: in-place argument mutation propagates (this
    # exact shape FAILED under the abandoned RPC runner - review fixture)
    "pass_inplace_mutation": (MUT_PROBLEM, MUT_PROBLEM["canonical_solution"],
                              ("candidate", "passed")),
    # a literal dict shaped like the old codec's tuple tag round-trips natively
    "pass_tag_shaped_dict": (DICT_PROBLEM, DICT_PROBLEM["canonical_solution"],
                             ("candidate", "passed")),
    # canonical guard None's builtins exit -> TypeError -> failed
    "fail_builtin_exit": (FIXTURE_PROBLEM, "    exit(0)\n", ("candidate", "failed")),
    # SystemExit is caught by the BaseException handler -> failed
    "fail_sys_exit": (FIXTURE_PROBLEM, "    import sys\n    sys.exit(0)\n", ("candidate", "failed")),
    # os._exit(0) kills the upstream child: empty result -> "timed out"
    # (the reference's own crash class)
    "fail_os__exit": (FIXTURE_PROBLEM, "    import os\n    os._exit(0)\n",
                      ("candidate", "timed_out")),
    # segfault kills the upstream child: same reference crash class -> timed_out
    "fail_child_segfault": (FIXTURE_PROBLEM,
                            "    import ctypes\n    ctypes.string_at(0)\n    return a + b\n",
                            ("candidate", "timed_out")),
    # reviewer's exact v3 repro: an ordinary correct completion importing
    # multiprocessing PASSES upstream (the module is already in sys.modules);
    # the abandoned hand copy failed it. Must pass.
    "multiprocessing_import_ok": (FIXTURE_PROBLEM,
                       "    import multiprocessing\n    return a + b\n",
                       ("candidate", "passed")),
    # a candidate that kills the DRIVER post-startup (libc kill bypasses the
    # guard's os.kill None-ing) -> startup frame but no verdict -> INFRA
    # (review fix: driver/sandbox death is not an upstream candidate timeout;
    # only a check_correctness VERDICT is a candidate outcome)
    "driver_killed_post_startup": (FIXTURE_PROBLEM,
                       "    import ctypes, os\n"
                       "    ctypes.CDLL(None).kill(os.getppid(), 9)\n"
                       "    return a + b\n",
                       ("infra", INFRA)),
    # the reviewer's v4 repro: a trusted-side failure INSIDE the exec phase
    # (machinery around the candidate, not the candidate) must be INFRA via an
    # explicit HEINFRA frame, never candidate timed_out. run_fixtures
    # special-cases the name to patch the driver source.
    "trusted_exec_fault_infra": (FIXTURE_PROBLEM, "    return a + b\n",
                                 ("infra", INFRA)),
    # the reviewer's second repro: invalid-Python DRIVER (infrastructure fault
    # before candidate code runs) must be INFRA, never candidate timed_out.
    # run_fixtures special-cases the name to monkeypatch driver_source.
    "driver_startup_error_infra": (FIXTURE_PROBLEM, "    return a + b\n",
                                   ("infra", INFRA)),
    # prints are swallowed (swallow_io); a bounded flood does not change verdicts
    "flood_swallowed": (FIXTURE_PROBLEM,
                        "    for _ in range(2000):\n        print('x' * 1024)\n    return a + b\n",
                        ("candidate", "passed")),
    # locally network is open; verdict must still be produced (any candidate
    # status: connect to TEST-NET times out -> exception -> failed). The real
    # block is the sandbox's block_network=True, live-verified at first run.
    "network_attempt_recorded": (FIXTURE_PROBLEM,
                                 "    import socket\n    s = socket.socket()\n"
                                 "    s.settimeout(2)\n    s.connect(('192.0.2.1', 9))\n"
                                 "    return a + b\n",
                                 ("candidate", None)),
    # writes land only in the ephemeral tempdir; verdict unaffected
    "file_write_tempdir": (FIXTURE_PROBLEM,
                           "    open('pwned.txt', 'w').write('x')\n    return a + b\n",
                           ("candidate", "passed")),
    # THE ACCEPTED ASSUMPTION, MADE VISIBLE: under the standard non-adversarial
    # harness, a completion that monkeypatches the test's own serializer DOES
    # forge a pass. This fixture must PASS. It is the documented scope limit
    # the owner approved - adversarial scorer integrity is separate work.
    "ASSUMPTION_forge_passes": (FORGE_PROBLEM,
                                "    import json\n"
                                "    json.dumps = lambda obj: '0'\n"
                                "    return a - b\n",
                                ("candidate", "passed")),
}


# driver-source transforms for infrastructure-fault fixtures (review repros)
_DRIVER_OVERRIDES = {
    # invalid-Python driver: no frames at all -> INFRA
    "driver_startup_error_infra": lambda src: "THIS IS INVALID PYTHON !\n",
    # trusted RuntimeError between startup and check_correctness -> the driver
    # catches it and emits an explicit HEINFRA frame -> INFRA
    "trusted_exec_fault_infra": lambda src: src.replace(
        'res = he_upstream.check_correctness(item["problem"], item["completion"], TIMEOUT)',
        'raise RuntimeError("trusted driver setup failure before check_correctness")'),
}


def run_fixtures(backend, full_canonical=False):
    failures = []
    for name, (problem, completion, (want_kind, want_status)) in FIXTURES.items():
        if name in _DRIVER_OVERRIDES:
            real = cx.driver_source
            cx.driver_source = lambda: _DRIVER_OVERRIDES[name](real())
            try:
                rec = cx.score_completion(problem, completion, backend)
            finally:
                cx.driver_source = real
        else:
            rec = cx.score_completion(problem, completion, backend)
        got_kind = "infra" if rec["status"] == INFRA else "candidate"
        ok = got_kind == want_kind and (want_status is None or rec["status"] == want_status)
        print(f"fixture {name:34s} -> {rec['status']:10s} "
              f"{'OK' if ok else 'MISMATCH (wanted ' + str(want_status) + ')'}")
        if not ok:
            failures.append(name)
    if full_canonical:
        rows = tasks.humaneval_load(_rev())
        npass = 0
        for r in rows:
            prob = {"task_id": r["id"], "prompt": r["prompt"], "test": r["test"],
                    "entry_point": r["entry_point"],
                    "canonical_solution": r["canonical_solution"]}
            rec = cx.score_completion(prob, r["canonical_solution"], backend)
            npass += rec.get("correct", 0)
            if rec.get("correct") != 1:
                print(f"canonical FAIL {r['id']}: {rec['status']} {rec.get('detail','')[:80]}")
                failures.append(r["id"])
        print(f"canonical solutions: {npass}/{len(rows)} pass under caps")
    return failures


def _rev():
    import run as runner
    return runner._require_rev(runner.load_lock(LOCK_PATH), "datasets",
                               "openai/openai_humaneval")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_bind(run_dir, model):
    """Bind scoring to the exact frozen generation: the responses file must
    cover exactly the pinned dataset ids, no duplicates, no extras, and the
    manifest's per-item fields must hash to the pinned dataset's values.
    Scoring uses the TRUSTED pinned-dataset rows, never the manifest's field
    copies (review: a genuine stored fields_sha256 next to altered fields was
    previously accepted, binding the label instead of the content)."""
    rev = _rev()
    rows = tasks.humaneval_load(rev)
    truth = {r["id"]: r["fields_sha256"] for r in rows}
    truth_rows = {r["id"]: r for r in rows}
    im_path = os.path.join(run_dir, "items_manifest_humaneval.json")
    with open(im_path, encoding="utf-8") as f:
        frozen = json.load(f)
    frozen_by_id = {}
    for r in frozen:
        if r["id"] in frozen_by_id:
            raise SystemExit(f"items_manifest duplicate id {r['id']}")
        frozen_by_id[r["id"]] = r
    if set(frozen_by_id) != set(truth):
        raise SystemExit("items_manifest ids != pinned dataset ids")
    for i, r in frozen_by_id.items():
        recomputed = hashlib.sha256(
            (r.get("prompt", "") + r.get("test", "")
             + r.get("entry_point", "")).encode()).hexdigest()
        if recomputed != truth[i] or r.get("fields_sha256") != truth[i]:
            raise SystemExit(f"items_manifest {i}: fields hash != pinned dataset - "
                             f"refusing to score against a different problem")
    resp_path = os.path.join(run_dir, "responses", f"humaneval__{model}.jsonl")
    gens = {}
    with open(resp_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in gens:
                raise SystemExit(f"duplicate response id {d['id']}")
            gens[d["id"]] = d["response"]
    if set(gens) != set(truth):
        missing = sorted(set(truth) - set(gens))
        extra = sorted(set(gens) - set(truth))
        raise SystemExit(f"response ids != manifest ids (missing={missing[:3]}... "
                         f"extra={extra[:3]}...): incomplete, refusing")
    binding = {"items_manifest_sha256": _sha256_file(im_path),
               "responses_sha256": _sha256_file(resp_path),
               "dataset_revision": rev}
    # score the TRUSTED pinned-dataset rows, not the manifest's field copies
    return [truth_rows[i] for i in sorted(truth_rows)], gens, binding


CANDIDATE_STATUSES = ("passed", "failed", "timed_out")
# Every scored row carries this IMMUTABLE measurement context; resume and
# pairing validate it field-by-field so a stale, other-model, other-run,
# other-code or other-config row can never be silently absorbed (review).
# The exec timeout is inside driver_sha256 and deliberately not duplicated.
ROW_CONTEXT_KEYS = ("model", "model_revision", "generation_manifest_sha256",
                    "items_manifest_sha256", "responses_sha256",
                    "dataset_revision", "protocol", "driver_sha256",
                    "coding_exec_sha256", "scorer_sha256",
                    "measurement_config_sha256", "sandbox_image",
                    "modal_sdk_required", "expected_python")


def measurement_config_sha256():
    """Frozen measurement config hash: input cap + sandbox cpu/memory/lifetime.
    (Exec timeout rides inside driver_sha256.) Change = new measurement."""
    cfg = {"max_completion_chars": cx.MAX_COMPLETION_CHARS,
           "sandbox_cpu": list(cx.SANDBOX_CPU),
           "sandbox_memory": list(cx.SANDBOX_MEMORY),
           "sandbox_lifetime_sec": cx.SANDBOX_LIFETIME_SEC}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def validate_scored_row(rec, all_ids, ctx, expected_input_sha=None,
                        expected_over_cap=None):
    """Strict outcome schema + immutable-context binding for one row. Returns
    an error string, or None when the row is valid for THIS run/context.

    Candidate rows: correct is EXACTLY int (bool rejected) and must equal
    int(status == "passed"); phase/provenance evidence is REQUIRED except
    the explicit pre-exec cap case (raw_over_cap), which must NOT carry it.
    Infra rows OMIT correct entirely.
    """
    if not isinstance(rec, dict):
        return "row is not an object"
    rid = rec.get("id")
    if rid not in all_ids:
        return f"id {rid!r} not in the current binding"
    for k in ROW_CONTEXT_KEYS:
        if rec.get(k) != ctx.get(k):
            return f"context field {k} mismatch (row={rec.get(k)!r}, this run={ctx.get(k)!r})"
    status = rec.get("status")
    if status not in CANDIDATE_STATUSES + (INFRA,):
        return f"invalid status {status!r}"
    if status == INFRA:
        if "correct" in rec:
            return "infra row must omit correct entirely"
        return None
    # candidate outcome
    c = rec.get("correct")
    if type(c) is not int:  # bool is not acceptable even though bool == int
        return f"correct must be int 0/1 exactly, got {type(c).__name__}"
    if c != int(status == "passed"):
        return f"correct={c} inconsistent with status {status!r}"
    if not isinstance(rec.get("detail", ""), str):
        return "detail must be a string"
    finish = rec.get("finish_reason")
    if not isinstance(finish, str):
        return "finish_reason must be a string"
    pe = rec.get("phase_evidence")
    if finish == "raw_over_cap":
        # explicit pre-exec case: never reached a driver, no phase evidence
        if pe is not None:
            return "raw_over_cap row must not carry phase evidence"
        if expected_over_cap is False:
            return "raw_over_cap row but the bound generation is NOT over cap"
        return None
    if not (isinstance(pe, dict) and pe.get("startup") is True):
        return "candidate row missing required phase/provenance evidence"
    if pe.get("module_sha256") != cx.REFERENCE_SHA256:
        return "phase evidence module hash != pinned upstream module"
    if expected_input_sha is not None and pe.get("input_sha256") != expected_input_sha:
        return "phase evidence input hash != recomputed expected input"
    if expected_input_sha is None:
        return "row carries phase evidence but recomputation is the cap case"
    py = str(pe.get("driver_python", ""))
    if not ctx.get("allow_local_python") and not py.startswith(ctx["expected_python"] + "."):
        return f"driver python {py!r} != expected target {ctx['expected_python']}.x"
    return None


def load_resume(out_items, rows, gens, ctx):
    """APPEND-RESUME loader (review rounds 2-3): never truncate prior scored
    rows. Every existing row must pass the strict outcome schema, match this
    run's full immutable measurement context, and - per item - attest the
    input hash recomputed from the CURRENT trusted row + bound generation +
    current extractor (raw_over_cap rows re-checked against the cap).
    Anything else refuses closed rather than mixing runs or losing completed
    spend. Returns (done_ids, records)."""
    done_ids, recs = set(), []
    if not os.path.exists(out_items):
        return done_ids, recs
    row_by_id = {r["id"]: r for r in rows}
    expected = _expected_inputs(rows, gens)
    with open(out_items, encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"resume: malformed row {ln_no} in {out_items}; "
                                 "refusing to mix - archive or remove the file")
            rid = old.get("id") if isinstance(old, dict) else None
            expected_sha, over_cap = expected.get(rid, (None, None))
            err = validate_scored_row(old, set(row_by_id), ctx,
                                      expected_input_sha=expected_sha,
                                      expected_over_cap=over_cap)
            if err:
                raise SystemExit(f"resume: row {ln_no} in {out_items}: {err}; "
                                 "refusing to mix - archive or remove the file")
            if old["id"] in done_ids:
                raise SystemExit(f"resume: duplicate id {old['id']} in {out_items}; "
                                 "refusing to mix")
            done_ids.add(old["id"])
            recs.append(old)
    return done_ids, recs


# ---- single-writer lock + pending/done ledger (review round 3) ------------
# Practical mechanism, NOT an exactly-once guarantee: the lock serializes
# writers; the ledger records pending (submitted, before any sandbox create
# for the item) and done (result checkpointed) states. An INTERRUPTED run
# leaves pending-without-done items UNKNOWN (they may have been created and
# billed): resume refuses to rerun them until the operator acknowledges the
# possible double-bill with --ack-interrupted (recorded in the ledger). The
# residual window (crash between sandbox create and the pending write, and
# sandbox-side execution after client death) cannot be closed offline; that
# is what the bounded, separately approved preflight probes measure.


def _ledger_states(ledger_path):
    """id -> latest state ('pending'/'done'), folded in append order.
    A malformed line fails CLOSED (review round 4): silently skipping it
    could drop a pending obligation and rebill an item."""
    states = {}
    if not os.path.exists(ledger_path):
        return states
    with open(ledger_path, encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"ledger {ledger_path} line {ln_no} is malformed; "
                                 "failing closed - reconcile the billing ledger "
                                 "manually before any rerun")
            if not isinstance(e, dict):
                raise SystemExit(f"ledger {ledger_path} line {ln_no} is not an "
                                 "object; failing closed")
            if e.get("state") in ("pending", "done") and "id" in e:
                states[e["id"]] = e["state"]
    return states


def acquire_writer_lock(out_items, ctx):
    """Take the single-writer flock BEFORE any mutable scored/ledger read
    (review round 5 TOCTOU: a resume snapshot or ledger read taken pre-lock
    goes stale when another invocation finishes in between, and paid items
    get rerun). Returns (token, fd); the flock is held for the whole run
    and released by the OS if the process dies:
    - flock fails -> a LIVE writer owns this output: refuse. Nothing -
      including --ack-interrupted - overrides a live writer.
    - flock succeeds -> any prior owner is dead. The DURABLE binding
      (.context.json, separate from the lock, survives lock-file removal)
      must equal this run's immutable context, ack or not.
    Resume loading and ledger reconciliation happen under this lock, in
    main(), via load_resume + reconcile_ledger."""
    import fcntl
    import uuid
    lock_path = out_items + ".lock"
    ctx_path = out_items + ".context.json"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit(f"{lock_path} is held by a LIVE writer - refusing to "
                         "run concurrently. --ack-interrupted never overrides "
                         "a live writer; stop the other run first.")
    # flock held: any previous owner is dead. Compare the durable binding.
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding="utf-8") as f:
            durable = json.load(f).get("ctx")
        if durable != ctx:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise SystemExit("durable run context on disk != this invocation's "
                             "immutable measurement context - a different run "
                             "owns this output; --ack-interrupted does not apply. "
                             "Score to a different output or archive the old one.")
    token = uuid.uuid4().hex
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "w", encoding="utf-8") as f:
        json.dump({"token": token, "ctx": ctx, "state": "active",
                   "started": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump({"ctx": ctx, "token": token, "state": "active",
                   "ts": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    return token, fd


def reconcile_ledger(out_items, done_ids, ack_interrupted):
    """Interruption reconciliation, called UNDER the writer flock with the
    freshly loaded done_ids (review round 5): the ledger is read only here,
    never pre-lock, so its pending/done fold cannot go stale. Items ledgered
    pending but never done AND not checkpointed in the scored file are
    UNKNOWN (possibly created and billed): refuse until the operator
    reconciles and reruns with --ack-interrupted (recorded in the ledger).
    Manually removing the lock file cannot erase the obligation - the
    ledger is read under the fresh flock regardless."""
    states = _ledger_states(out_items + ".ledger.jsonl")
    unknown = sorted(i for i, s in states.items()
                     if s == "pending" and i not in done_ids)
    if unknown and not ack_interrupted:
        raise SystemExit(
            f"interrupted run: {len(unknown)} items were submitted but never "
            f"checkpointed (e.g. {unknown[:3]}). They may have been created "
            "and BILLED; rerunning them may double-bill. Reconcile first, "
            "then rerun with --ack-interrupted (recorded in the ledger).")
    if unknown:
        ledger_append(out_items, {"state": "reconciled", "unknown_rerun": unknown,
                                  "ack_interrupted": True})


def ledger_append(out_items, entry):
    ledger_path = out_items + ".ledger.jsonl"
    entry = dict(entry, ts=time.time())
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def release_writer_lock(out_items, token, fd):
    """Mark the run released in the lock AND the durable context, then
    unlock. A released run keeps its files: the durable binding and the
    ledger obligations survive for the next invocation to compare."""
    import fcntl
    lock_path = out_items + ".lock"
    ctx_path = out_items + ".context.json"
    try:
        with open(lock_path, encoding="utf-8") as f:
            owner = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        owner = {}
    if owner.get("token") != token:
        return  # not ours - never touch another writer's lock
    ended = time.time()
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "w", encoding="utf-8") as f:
        json.dump({"token": token, "ctx": owner.get("ctx"),
                   "state": "released", "ended": ended}, f)
        f.flush()
        os.fsync(f.fileno())
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump({"ctx": owner.get("ctx"), "state": "released",
                   "ts": ended}, f)
        f.flush()
        os.fsync(f.fileno())
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _expected_inputs(rows, gens):
    """id -> (expected_input_sha256 or None, over_cap). The ONE trusted
    per-item mapping (recomputed from the pinned-dataset row + bound
    generation + current extractor) used by resume AND final validation
    alike (review round 4: the final pass previously called
    validate_scored_row without it, rejecting every valid row)."""
    out = {}
    for r in rows:
        prob = {"task_id": r["id"], "prompt": r["prompt"], "test": r["test"],
                "entry_point": r["entry_point"]}
        sha = cx.expected_input_sha256(prob, gens[r["id"]])
        out[r["id"]] = (sha, sha is None)
    return out


def validate_gen_manifest(gen_manifest, model, expected_rev):
    """Refuse to spend on an unfinalized or inconsistent generation
    manifest (review round 4): an empty {} with a default 'unknown'
    revision previously passed. Requires the finalized manifest: the
    expected model entry (repo + 40-hex revision), the tokenizer pin, the
    environment block, humaneval task coverage, and agreement with the
    scorer's pinned dataset lock."""
    import re as _re
    if not isinstance(gen_manifest, dict) or not gen_manifest:
        raise SystemExit("generation manifest is empty - not a finalized run")
    entry = (gen_manifest.get("models") or {}).get(model)
    if not isinstance(entry, dict):
        raise SystemExit(f"generation manifest has no model entry for {model!r}")
    rev = entry.get("revision", "")
    if not (isinstance(rev, str) and _re.fullmatch(r"[0-9a-f]{40}", rev)):
        raise SystemExit(f"generation manifest model {model!r} revision "
                         f"{rev!r} is not a pinned 40-hex commit")
    if not entry.get("repo"):
        raise SystemExit(f"generation manifest model {model!r} has no repo")
    tok = gen_manifest.get("tokenizer") or {}
    trev = tok.get("revision", "")
    if not (tok.get("repo") and isinstance(trev, str)
            and _re.fullmatch(r"[0-9a-f]{40}", trev)):
        raise SystemExit("generation manifest missing the finalized tokenizer "
                         "pin (repo + 40-hex revision)")
    if not gen_manifest.get("environment"):
        raise SystemExit("generation manifest has no environment block - "
                         "not a finalized run")
    if "humaneval" not in (gen_manifest.get("tasks") or {}):
        raise SystemExit("generation manifest does not cover task 'humaneval'")
    # completion state (review round 5): a manifest from a run that never
    # finished generation lacks the post-generation markers. A bounded
    # preflight fixture manifest carries them explicitly when authored.
    eos = entry.get("effective_eos_ids")
    if not (isinstance(eos, list) and eos and
            all(isinstance(x, int) for x in eos)):
        raise SystemExit(f"generation manifest model {model!r} lacks "
                         "effective_eos_ids - generation for this model never "
                         "completed (or an incomplete manifest)")
    counters = ((gen_manifest.get("tasks") or {}).get("humaneval", {})
                .get("counters") or {})
    if model not in counters:
        raise SystemExit("generation manifest lacks generation counters for "
                         f"model {model!r} on humaneval - generation for this "
                         "model never completed")
    # reconciled config: the generation knobs the scoring side assumes
    if not (isinstance(gen_manifest.get("gen_max_new_tokens"), int)
            and gen_manifest["gen_max_new_tokens"] > 0):
        raise SystemExit("generation manifest lacks a positive "
                         "gen_max_new_tokens - config not reconciled")
    # run.py records the seed at argv.seed (manifest["argv"] = vars(args))
    seed = gen_manifest.get("seed",
                            (gen_manifest.get("argv") or {}).get("seed"))
    if not isinstance(seed, int):
        raise SystemExit("generation manifest lacks the seed (argv.seed) - "
                         "config not reconciled")
    lock = gen_manifest.get("lock") or {}
    ds = (lock.get("datasets") or {}).get("openai/openai_humaneval")
    if ds != expected_rev:
        raise SystemExit(f"generation manifest dataset pin {ds!r} != the "
                         f"scorer's pinned revision {expected_rev!r}")
    lmr = (lock.get("models") or {}).get(model)
    if lmr != rev:
        raise SystemExit(f"generation manifest lock model revision {lmr!r} "
                         f"!= model entry revision {rev!r}")
    return rev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--model")
    ap.add_argument("--max-cost-usd", type=float, default=0.25,
                    help="per-invocation budget tripwire (caps-based estimate; NOT "
                         "an aggregate spend ceiling)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="sandboxes in flight (serial default); hard-capped at 8")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--ack-interrupted", action="store_true",
                    help="reconcile an interrupted run: acknowledge that "
                         "pending-but-never-checkpointed items may have been "
                         "created and billed, and authorize rerunning them "
                         "(recorded in the ledger)")
    ap.add_argument("--full-canonical", action="store_true")
    args = ap.parse_args(argv)

    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be a positive finite number")
    if args.concurrency < 1 or args.concurrency > 8:
        raise SystemExit("concurrency must be 1..8")

    if args.self_check:
        failures = run_fixtures(cx.LocalBackend(fixture_mode=True), args.full_canonical)
        print("SELF-CHECK", "FAIL: " + ",".join(failures) if failures else "OK")
        sys.exit(1 if failures else 0)

    if not (args.run_dir and args.model):
        raise SystemExit("--run-dir and --model are required for scoring")

    rows, gens, binding = load_and_bind(args.run_dir, args.model)
    # preserve the generation-phase counters from the frozen generation records
    gen_items_path = os.path.join(args.run_dir, f"items_humaneval__{args.model}.jsonl")
    gen_counters = {}
    if os.path.exists(gen_items_path):
        with open(gen_items_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gen_counters[d["id"]] = {k: v for k, v in d.items()
                                         if k in ("finish_reason", "truncated_no_eos",
                                                  "eos_reached", "skipped")}

    bound = cx.cost_bound_per_item_usd()
    print(f"per-item caps-based compute estimate ${bound:.7f} (Sandbox list rates; "
          f"excludes provisioning/overhead); tripwire ${args.max_cost_usd:.4f} "
          f"-> at most {int(args.max_cost_usd / bound)} items this invocation")
    backend = cx.SandboxBackend()
    try:
        import modal as _modal
        modal_version = getattr(_modal, "__version__", "unknown")
    except Exception:
        modal_version = "unknown"
    # enforce the pinned SDK BEFORE any sandbox starts (review fix: the pin
    # was recorded but never compared)
    if modal_version != cx.REQUIRED_MODAL_VERSION:
        raise SystemExit(f"modal SDK {modal_version} != pinned "
                         f"{cx.REQUIRED_MODAL_VERSION} (requirements-coding-exec.txt); "
                         "refusing to run on an unpinned runtime")
    # generation-phase binding: the run manifest pins model revision,
    # tokenizer, environment and generation config; scoring refuses without it
    gen_manifest_path = os.path.join(args.run_dir, "manifest.json")
    if not os.path.exists(gen_manifest_path):
        raise SystemExit("generation manifest.json missing - cannot bind the "
                         "measurement context (review: immutable full context)")
    with open(gen_manifest_path, encoding="utf-8") as f:
        gen_manifest = json.load(f)
    # refuse an unfinalized/inconsistent manifest BEFORE any costly scoring
    model_revision = validate_gen_manifest(gen_manifest, args.model,
                                           binding["dataset_revision"])
    ctx = {"model": args.model,
           "model_revision": model_revision,
           "generation_manifest_sha256": _sha256_file(gen_manifest_path),
           "items_manifest_sha256": binding["items_manifest_sha256"],
           "responses_sha256": binding["responses_sha256"],
           "dataset_revision": binding["dataset_revision"],
           "protocol": cx.CODING_PROTOCOL_VERSION,
           "driver_sha256": cx.driver_sha256(),
           "coding_exec_sha256": _sha256_file(cx.__file__),
           "scorer_sha256": _sha256_file(os.path.abspath(__file__)),
           "measurement_config_sha256": measurement_config_sha256(),
           "sandbox_image": cx.SANDBOX_IMAGE,
           "modal_sdk_required": cx.REQUIRED_MODAL_VERSION,
           "expected_python": "3.13"}
    if isinstance(backend, cx.LocalBackend):
        # explicit offline-fixture exception: the local driver runs the
        # host interpreter, not the pinned 3.13 image (recorded in ctx)
        ctx["allow_local_python"] = True

    out_items = os.path.join(args.run_dir, f"items_humaneval_scored__{args.model}.jsonl")
    results, spent_est, infra = [], 0.0, 0
    t0 = time.time()
    aborted = None

    # SINGLE-WRITER FLOCK FIRST (review round 5 TOCTOU): the resume snapshot
    # and the ledger fold are mutable state - reading them before the lock
    # lets a finishing invocation invalidate them, rerunning paid items.
    # Everything mutable is read under the flock.
    all_ids = {r["id"] for r in rows}
    lock_token, lock_fd = acquire_writer_lock(out_items, ctx)
    try:
        # APPEND-RESUME (review fix): never truncate prior scored rows. Existing
        # rows must belong to THIS binding; anything else refuses closed rather
        # than mixing runs or losing completed spend.
        done_ids, resumed = load_resume(out_items, rows, gens, ctx)
        # interruption reconciliation under the same lock, over the fresh snapshot
        reconcile_ledger(out_items, done_ids, args.ack_interrupted)
        results.extend(resumed)
        if done_ids:
            infra += sum(1 for r in resumed if r.get("status") == INFRA)
            print(f"resuming: {len(done_ids)} items already scored, "
                  f"{len(rows) - len(done_ids)} remaining")
        rows_todo = [r for r in rows if r["id"] not in done_ids]

        def one(r):
            prob = {"task_id": r["id"], "prompt": r["prompt"], "test": r["test"],
                    "entry_point": r["entry_point"], "canonical_solution": r["canonical_solution"]}
            # persist the exact sandbox ID as soon as create returns (review
            # round 6): reconciliation binds to THIS sandbox; a lost create
            # response leaves the item pending = unknown, never recreated
            def _created(sandbox_id, _i=r["id"]):
                ledger_append(out_items, {"state": "created", "id": _i,
                                          "sandbox_id": sandbox_id})
            try:
                rec = cx.score_completion(prob, gens[r["id"]], backend,
                                          on_create=_created)
            except Exception as e:
                # backend/provisioning failure is an INFRA outcome for this item,
                # never a run abort that loses accumulated results
                rec = {"id": r["id"], "status": INFRA,
                       "detail": f"{type(e).__name__}: {str(e)[:150]}",
                       "finish_reason": "unscored"}
            rec.update(ctx)  # bind the row to THIS run/model/driver/runtime
            return rec

        # checkpoint every result as it lands (APPEND + fsync): an abort or a
        # restart never loses completed items, and the file state always reflects
        # real progress
        with open(out_items, "a", encoding="utf-8") as out:
            try:
                with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    pending = {}
                    it = iter(rows_todo)
                    exhausted = False
                    while True:
                        while not exhausted and len(pending) < args.concurrency:
                            if spent_est + bound > args.max_cost_usd:
                                aborted = (f"budget tripwire before next item "
                                           f"(est ${spent_est:.4f})")
                                exhausted = True
                                break
                            try:
                                r = next(it)
                            except StopIteration:
                                exhausted = True
                                break
                            ledger_append(out_items, {"state": "pending", "id": r["id"]})
                            pending[pool.submit(one, r)] = r
                            spent_est += bound
                        if not pending:
                            break
                        done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for fut in done:
                            rec = fut.result()
                            pending.pop(fut)
                            results.append(rec)
                            infra += rec["status"] == INFRA
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            out.flush()
                            os.fsync(out.fileno())
                            ledger_append(out_items, {"state": "done", "id": rec["id"]})
                            if len(results) % 20 == 0:
                                print(f"{len(results)}/{len(rows)} scored "
                                      f"({len(done_ids)} resumed), infra={infra}, "
                                      f"est ${spent_est:.4f}, {time.time()-t0:.0f}s")
            except BaseException as e:
                aborted = f"{type(e).__name__}: {str(e)[:150]}"

        # complete_valid requires EVERY intended item scored with a valid
        # candidate outcome (review fix: a length check alone let zero-scored
        # rows pass). Schema-validate all rows, resumed or fresh.
        expected = _expected_inputs(rows, gens)
        schema_errors = []
        for r in results:
            sha, over_cap = expected.get(r.get("id"), (None, None))
            err = validate_scored_row(r, all_ids, ctx, expected_input_sha=sha,
                                      expected_over_cap=over_cap)
            if err:
                schema_errors.append(f"{r.get('id')}: {err}")
        if schema_errors:
            aborted = (aborted or "") + f" | {len(schema_errors)} rows failed the strict "                                   f"outcome schema (first: {schema_errors[0]})"
        scored = [r["correct"] for r in results
                  if r.get("status") in CANDIDATE_STATUSES and r.get("correct") in (0, 1)]
        n_valid_scored = len(scored)
        complete = (n_valid_scored == len(rows) and infra == 0 and aborted is None
                    and len(results) == len(rows))
        cell = {"task": "humaneval", "model": args.model,
                "state": "complete_valid" if complete else "complete_invalid",
                "n_items": len(rows),
                "n_resumed": len(done_ids),
                "n_executed_this_invocation": len(results) - len(done_ids),
                "n_scored": n_valid_scored, "infra_errors": infra,
                "aborted": aborted,
                "items_file_sha256": _sha256_file(out_items),
                "cost_estimate_this_invocation_usd": round(spent_est, 6),
                "cost_estimate_basis": "caps x lifetime x Sandbox list rates; compute "
                                       "only; excludes provisioning/build/overhead; "
                                       "THIS INVOCATION only - resumed items are not "
                                       "re-billed; cumulative spend is the sum across "
                                       "invocations' records",
                "runtime_final": "NOT yet observed/pinned - the digest pins the base "
                                 "image; the final runtime (base plus any Modal-injected "
                                 "layers) is recorded at the first authorized run",
                "run_context": ctx,
                "protocol": cx.CODING_PROTOCOL_VERSION,
                "driver_sha256": cx.driver_sha256(),
                "sandbox_image": cx.SANDBOX_IMAGE,
                "modal_sdk_version": modal_version,
                "modal_sdk_required": cx.REQUIRED_MODAL_VERSION,
                "image_digest": ("image pinned by digest (see sandbox_image); actual "
                                 "pulled image + 3.13 canonical re-validation verified "
                                 "at the first authorized run - local validation was "
                                 "on Python " + ".".join(map(str, sys.version_info[:2]))),
                **binding}
        if scored:
            rate, lo, hi, nsc = scoring.rate_with_ci(scored)
            cell.update(pass_at_1=round(rate, 4), wilson95=[round(lo, 4), round(hi, 4)],
                        wilson95_basis="descriptive CI over the fixed 164-item set "
                                       "under an item-superpopulation assumption; NOT "
                                       "decoding/model-resampling uncertainty")
        # merge the generation-phase counters for the audited record
        cell["generation_counters_by_id"] = gen_counters
        cell_path = os.path.join(args.run_dir, f"cell_humaneval__{args.model}.json")
        with open(cell_path, "w", encoding="utf-8") as f:
            json.dump(cell, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        ledger_append(out_items, {"state": "cell", "complete": complete})
    finally:
        # release on EVERY exit (review round 5): an abortive
        # SystemExit must not leak the flock in-process; the
        # ledger obligations persist for the next invocation
        release_writer_lock(out_items, lock_token, lock_fd)
    print(json.dumps({k: v for k, v in cell.items()
                      if k != "generation_counters_by_id"}, indent=1))

    # paired analysis vs base, ONLY when BOTH cells are complete_valid and
    # bound to the same manifest + dataset revision + protocol + driver +
    # runtime image + SDK pin (review fix: an aborted-but-full run, a stale
    # or differently-run base, or a scored file whose bytes drifted from its
    # cell record must never pair). Anything else is skipped with a stated
    # reason - never silently intersected.
    if args.model != "base":
        base_cell_path = os.path.join(args.run_dir, "cell_humaneval__base.json")
        base_path = os.path.join(args.run_dir, "items_humaneval_scored__base.jsonl")
        paired_skip = None
        base_c = None
        if not complete:
            paired_skip = (f"candidate cell state is not complete_valid "
                           f"(aborted={aborted!r}, infra={infra})")
        elif not os.path.exists(base_cell_path):
            paired_skip = "no base cell file"
        else:
            with open(base_cell_path, encoding="utf-8") as f:
                base_cell = json.load(f)
            base_ctx = base_cell.get("run_context", base_cell)
            same_run_keys = ("generation_manifest_sha256",
                             "items_manifest_sha256", "dataset_revision",
                             "protocol", "driver_sha256", "coding_exec_sha256",
                             "scorer_sha256", "measurement_config_sha256",
                             "sandbox_image", "modal_sdk_required",
                             "expected_python")
            if base_cell.get("state") != "complete_valid":
                paired_skip = (f"base cell state={base_cell.get('state')!r} "
                               "(need complete_valid)")
            elif not os.path.exists(base_path):
                paired_skip = "base cell present but scored items file missing"
            else:
                mismatch = [k for k in same_run_keys
                            if base_ctx.get(k) != ctx.get(k)]
                if mismatch:
                    paired_skip = (f"base cell differs on {mismatch} - not the "
                                   "same run/protocol/runtime")
                elif base_cell.get("items_file_sha256") != _sha256_file(base_path):
                    paired_skip = ("base scored items file hash drifted from its "
                                   "cell record - refusing to pair")
                else:
                    with open(base_path, encoding="utf-8") as f:
                        base_c = {json.loads(l)["id"]: json.loads(l).get("correct")
                                  for l in f}
                    if set(base_c) != all_ids:
                        paired_skip = "base scored ids != the binding's exact id set"
        if paired_skip:
            print(f"paired: SKIPPED ({paired_skip})")
        else:
            cand_c = {r["id"]: r.get("correct") for r in results}
            if set(cand_c) != all_ids:
                print("paired: SKIPPED (candidate scored ids != exact id set)")
            else:
                r = scoring.mcnemar([cand_c[i] for i in sorted(all_ids)],
                                    [base_c[i] for i in sorted(all_ids)])
                paired_path = os.path.join(args.run_dir,
                                           f"paired_humaneval__{args.model}_minus_base.json")
                with open(paired_path, "w") as f:
                    json.dump(r, f, indent=1)
                print("paired:", json.dumps(r))
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()
