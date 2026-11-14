"""Isolated execution boundary for the coding pack (HumanEval pass@1), v4.

ROUTE (owner-approved 2026-11-13): the STANDARD openai/human-eval harness -
the ACTUAL pinned upstream module, not a reimplementation. The vendored copy
reference/human_eval_execution.py is byte-verified against upstream commit
37c4dd63798c3c9ba32fa69a2fb49c5e2c43a181 (human_eval/execution.py, MIT;
sha256 79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c)
and re-hashed at import here. The per-item driver calls the module's own
check_correctness(); assembly, guard, outcome classes and multiprocessing
child semantics are upstream's. v1/v2's custom RPC runner and v3's hand copy
were abandoned after independent review (v3 diverged: a correct completion
importing multiprocessing passes upstream but failed the copy; a driver
fault was scored as a candidate failure).

What this pack CLAIMS:
- pass@1 under the standard, non-adversarial HumanEval harness: the pinned
  upstream check_correctness, run inside a fresh Modal Sandbox per item for
  HOST isolation. Ordinary Python semantics are exactly upstream's.

What it does NOT claim (stated assumption/limitation, owner-approved):
- NO adversarial verdict integrity. Under upstream semantics the candidate's
  code shares the child process and namespace with the test's check(), so a
  completion that deliberately manipulates the interpreter CAN influence its
  verdict (fixture ASSUMPTION_forge_passes demonstrates this and must PASS).
  The non-adversarial scoring scope is an ASSUMPTION, not evidence: in
  tamper-resistance research a model's propensity to subvert measurement is
  an open question, and no claim here touches it.

PHASE SEPARATION (review 2026-11-13, two rounds): the driver emits a
HESTARTUP:: frame only AFTER its own imports, input parse and upstream-module
hash check all succeed; trusted-side faults around the exec phase are caught
and emitted as explicit HEINFRA:: frames; HEVERDICT:: follows
check_correctness. The parser strictly binds the startup evidence to the
pinned module hash, the submitted item hash and a plausible runtime version,
enforces frame order and verdict schema/result consistency. No valid startup
frame, an explicit HEINFRA frame, a protocol violation, or a hard-killed
driver (startup but no verdict) are ALL infrastructure: driver/sandbox death
is NOT an upstream candidate timeout. Only a check_correctness VERDICT is a
candidate outcome. Multiple conflicting verdict frames -> timed_out
(confusion class; pass-to-fail only; a completion could write to fd 1
directly, so conflicting frames are never trusted).

SECURITY boundary (protects the host/accounts, NOT the verdict): fresh Modal
Sandbox PER ITEM, block_network=True, secrets=None, no mounts, hard cpu/memory
caps, lifetime timeout, terminate(wait=True) in finally. Model-written code
never executes in the GPU/credential/volume process. The LOCAL backend exists
ONLY for offline authored fixtures and is not an isolation boundary.

Candidate vs infra outcomes: passed/failed/timed_out are CANDIDATE outcomes
(correct 1/0) and come ONLY from an upstream check_correctness verdict.
Sandbox API failures, driver faults (pre-startup, exec-phase or death), and
frame/evidence violations -> INFRA errors (item incomplete; cell invalid,
never silently absent). NO retries: one sandbox, one verdict.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

# ---- frozen protocol constants (change = new measurement, bump the version) ----

CODING_PROTOCOL_VERSION = "he-suite-4-upstream"
MAX_COMPLETION_CHARS = 20000          # input cap on the RAW generation, before extraction
EXEC_TIMEOUT_SEC = 5                  # soft per-item limit; upstream default is 3.0.
                                      # Hardware-dependent; NOT calibrated by local
                                      # checks - recalibrated on the actual image at
                                      # the first authorized run before model answers
SANDBOX_CPU = (0.125, 0.25)           # (request, HARD limit) physical cores
SANDBOX_MEMORY = (128, 512)           # (request, HARD limit) MiB
SANDBOX_LIFETIME_SEC = 30             # sandbox-level backstop timeout
# Base image pinned by DIGEST (linux/amd64), resolved 2026-11-13 from the Docker
# Hub tag API (tag last_updated 2026-11-08T02:07:18Z). The base ships python3.13;
# no add_python layer (it would be unpinned). The FINAL runtime (base plus any
# Modal-injected layers) is NOT pinned or observed yet: the actual pulled image
# and runtime are recorded and verified at the first authorized run; local
# validation ran on Python 3.10, NOT on this image.
SANDBOX_IMAGE = ("python:3.13-slim"
                 "@sha256:48d884813cf44e927cf64c5529dd630dd3b18338c6dc727991eef9b71e934a20")
FRAME_STARTUP = "HESTARTUP::"
FRAME_VERDICT = "HEVERDICT::"
FRAME_INFRA = "HEINFRA::"

# The pinned upstream module (vendored; see reference/PROVENANCE.md).
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "reference", "human_eval_execution.py")
REFERENCE_SHA256 = "79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c"
REQUIRED_MODAL_VERSION = "1.4.2"      # requirements-coding-exec.txt; live gate recorded in results/general_audit
                                      # offline - import-check at first authorized run

# Modal SANDBOX per-second list rates (modal.com/pricing, Sandbox section,
# non-preemptible, verified 2026-11-13; ~3x the regular-function rates):
RATE_CPU_PER_CORE_S = 0.00003942
RATE_MEM_PER_GIB_S = 0.00000667


def _load_reference_module_bytes():
    with open(REFERENCE_PATH, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != REFERENCE_SHA256:
        raise RuntimeError(f"vendored upstream module hash mismatch: {actual} "
                           f"!= pinned {REFERENCE_SHA256} - refusing to run")
    return data


_UPSTREAM_BYTES = _load_reference_module_bytes()


def cost_bound_per_item_usd():
    """COMPUTE-ONLY estimate for ONE item's sandbox at the hard caps and Sandbox
    list rates (billed max(request, actual) per docs). NOT a total bill ceiling:
    image build/provisioning, service overhead and actual instantiated
    limits/lifetime are not live-verified, and this per-invocation arithmetic is
    not an aggregate spend control (aggregate spend = the approved scope only)."""
    cpu_s = SANDBOX_CPU[1] * SANDBOX_LIFETIME_SEC
    gib_s = (SANDBOX_MEMORY[1] / 1024) * SANDBOX_LIFETIME_SEC
    return cpu_s * RATE_CPU_PER_CORE_S + gib_s * RATE_MEM_PER_GIB_S


def extract_completion(raw):
    """Frozen extraction. Input: raw model generation. Output: (body, finish_reason).

    Rules (fixed before any score):
    1. strip one surrounding ```python ... ``` fence pair if present (both fences,
       else left as-is);
    2. cut at the first line that starts a NEW top-level statement
       (non-indented def/class/if/print(/#/@/import/from);
    3. finish_reason reports what happened: "fence_stripped"/"raw" for (1) and
       "stop_cut"/"no_stop" for (2). no-EOS truncation is tracked by the GENERATION
       counters (truncated_no_eos), separately from extraction.
    """
    text = raw.strip("\n")
    fence = "raw"
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
        fence = "fence_stripped"
    body = []
    cut = "no_stop"
    for ln in lines:
        if ln and not ln[0] in " \t" and (
                ln.startswith(("def ", "class ", "if ", "print(", "#", "@",
                               "import ", "from "))):
            cut = "stop_cut"
            break
        body.append(ln)
    return "\n".join(body), f"{fence}+{cut}"


# ---- driver (the ONLY process inside the sandbox; calls the pinned module) ----

_DRIVER_TEMPLATE = r'''
import hashlib
import json
import sys

EXPECTED_MODULE_SHA256 = "__MODULE_SHA__"
TIMEOUT = __TIMEOUT__
WORK = "/work"


def _emit(prefix, obj):
    sys.stdout.write(prefix + json.dumps(obj) + "\n")
    sys.stdout.flush()


# ---- STARTUP PHASE: imports, input parse, module-hash verification.
# Anything failing BEFORE the startup frame is infrastructure failure; the
# caller maps "no startup frame" to an INFRA error, never a candidate outcome.
with open(WORK + "/item.json", encoding="utf-8") as f:
    raw = f.read()
item = json.loads(raw)
with open(WORK + "/he_upstream.py", "rb") as f:
    module_bytes = f.read()
module_sha = hashlib.sha256(module_bytes).hexdigest()
if module_sha != EXPECTED_MODULE_SHA256:
    _emit("HEINFRA::", {"phase": "startup",
                        "error": "upstream module hash mismatch inside sandbox"})
    raise SystemExit(2)
sys.path.insert(0, WORK)
try:
    import he_upstream  # the pinned vendored human_eval/execution.py
except Exception as e:
    _emit("HEINFRA::", {"phase": "startup",
                        "error": type(e).__name__ + ": " + str(e)[:200]})
    raise SystemExit(2)
_emit("HESTARTUP::", {"ok": True, "module_sha256": module_sha,
                      "input_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                      "python": sys.version.split()[0]})

# ---- EXEC PHASE: the actual pinned upstream module computes the verdict.
# Candidate code runs in upstream's multiprocessing child; a crashed child
# yields upstream's "timed out" VERDICT. A trusted-side failure here (the
# driver, the Manager, the machinery AROUND the candidate) is INFRASTRUCTURE:
# it is caught and emitted as an explicit HEINFRA frame, never a candidate
# outcome. A hard-killed driver (no frames at all) is mapped to infra by the
# caller: driver/sandbox death is not an upstream candidate timeout.
try:
    res = he_upstream.check_correctness(item["problem"], item["completion"], TIMEOUT)
except Exception as e:
    _emit("HEINFRA::", {"phase": "exec",
                        "error": type(e).__name__ + ": " + str(e)[:200]})
    raise SystemExit(2)
_emit("HEVERDICT::", {"result": res["result"], "passed": bool(res["passed"])})
'''


def driver_source():
    return (_DRIVER_TEMPLATE
            .replace("__MODULE_SHA__", REFERENCE_SHA256)
            .replace("__TIMEOUT__", repr(EXEC_TIMEOUT_SEC)))


def driver_sha256():
    """Pin over driver template + vendored upstream module bytes."""
    h = hashlib.sha256()
    h.update(driver_source().encode())
    h.update(_UPSTREAM_BYTES)
    return h.hexdigest()


import re as _re

_PY_VERSION_RE = _re.compile(r"^\d+\.\d+\.\d+$")
_VALID_RESULTS = ("passed", "timed out")


def _valid_verdict(v):
    """Strict verdict schema: upstream's exact result strings and result/passed
    consistency. Anything else is not a verdict."""
    if not isinstance(v, dict):
        return False
    result, passed = v.get("result"), v.get("passed")
    if not isinstance(passed, bool) or not isinstance(result, str):
        return False
    if result in _VALID_RESULTS or result.startswith("failed: "):
        return passed == (result == "passed")
    return False


def parse_driver_output(rc, stdout, expected_module_sha, expected_input_sha):
    """Phase-aware frame parsing with STRICT evidence validation (review fix:
    forged/mismatched startup evidence was previously accepted). Returns
    (kind, payload):
    ("verdict", (startup, verdict)) - exactly one valid, consistent verdict
    ("confused", startup)           - conflicting verdict frames (never trusted)
    ("infra", detail)               - no valid startup, explicit HEINFRA frame,
                                      protocol violation, or post-startup death:
                                      driver/sandbox death is NOT an upstream
                                      candidate timeout
    """
    startups, verdicts, infras = [], [], []
    first_verdict_line = None
    for i, ln in enumerate(stdout.splitlines()):
        if ln.startswith(FRAME_STARTUP):
            try:
                s = json.loads(ln[len(FRAME_STARTUP):])
            except json.JSONDecodeError:
                return ("infra", "malformed startup frame")
            if first_verdict_line is not None:
                return ("infra", "startup frame after verdict frame (order violation)")
            startups.append(s)
        elif ln.startswith(FRAME_VERDICT):
            if first_verdict_line is None:
                first_verdict_line = i
            try:
                verdicts.append(json.loads(ln[len(FRAME_VERDICT):]))
            except json.JSONDecodeError:
                return ("infra", "malformed verdict frame")
        elif ln.startswith(FRAME_INFRA):
            try:
                infras.append(json.loads(ln[len(FRAME_INFRA):]))
            except json.JSONDecodeError:
                infras.append({"phase": "unknown", "error": "malformed infra frame"})
    if infras:
        f0 = infras[0]
        return ("infra", f"driver infra frame (phase={f0.get('phase')}): "
                         f"{f0.get('error')} (rc={rc})")
    if not startups or not all(s.get("ok") is True for s in startups):
        return ("infra", f"no valid startup frame (rc={rc}); driver failed before "
                         "candidate code ran (startup/imports/module check)")
    distinct_starts = {json.dumps(s, sort_keys=True) for s in startups}
    if len(distinct_starts) > 1:
        return ("infra", "conflicting startup frames (protocol violation)")
    startup = startups[0]
    # evidence binding: the startup frame must attest EXACTLY the module and
    # input this caller supplied, plus a plausible runtime version
    if startup.get("module_sha256") != expected_module_sha:
        return ("infra", f"startup module_sha256 {startup.get('module_sha256')} "
                         f"!= expected {expected_module_sha} - evidence not trusted")
    if startup.get("input_sha256") != expected_input_sha:
        return ("infra", "startup input_sha256 does not match the submitted item - "
                         "evidence not trusted")
    if not _PY_VERSION_RE.match(str(startup.get("python", ""))):
        return ("infra", f"implausible driver python {startup.get('python')!r} - "
                         "evidence not trusted")
    verdicts = [v for v in verdicts if _valid_verdict(v)]
    distinct = {json.dumps(v, sort_keys=True) for v in verdicts}
    if len(distinct) > 1:
        return ("confused", startup)
    if not verdicts:
        return ("infra", f"driver died after startup with no verdict (rc={rc}); "
                         "driver/sandbox death is not a candidate outcome")
    return ("verdict", (startup, verdicts[0]))


def build_item_payload(problem, completion_raw):
    """Frozen item construction, shared by scoring AND resume revalidation.
    Returns (item_json, finish_reason); item_json is None in the explicit
    pre-exec cap case (raw_over_cap), which then has no phase evidence."""
    if len(completion_raw) > MAX_COMPLETION_CHARS:
        return None, "raw_over_cap"
    body, finish = extract_completion(completion_raw)
    item = {"problem": {"task_id": problem["task_id"], "prompt": problem["prompt"],
                        "test": problem["test"], "entry_point": problem["entry_point"]},
            "completion": body}
    return json.dumps(item), finish


def expected_input_sha256(problem, completion_raw):
    """The input hash a row's phase evidence MUST attest for this problem +
    raw generation under the current extractor, or None for the pre-exec
    cap case. Resume revalidation uses this to bind rows per-item."""
    item_json, _ = build_item_payload(problem, completion_raw)
    if item_json is None:
        return None
    return hashlib.sha256(item_json.encode("utf-8")).hexdigest()


def score_completion(problem, completion_raw, backend, on_create=None):
    """One item: cap -> extract -> run the pinned-upstream driver -> verdict.

    Returns a record: id, finish_reason, status (passed/failed/timed_out/
    infra_error), correct (only for candidate outcomes), detail, plus phase
    and provenance evidence (module/input hashes, driver python version).
    """
    rec = {"id": problem["task_id"]}
    item_json, finish = build_item_payload(problem, completion_raw)
    if item_json is None:
        rec.update(status="failed", correct=0, finish_reason="raw_over_cap",
                   detail=f"raw generation exceeded {MAX_COMPLETION_CHARS} chars "
                          "(capped before extraction; conservative fail)")
        return rec
    rec["finish_reason"] = finish
    expected_input_sha = hashlib.sha256(item_json.encode("utf-8")).hexdigest()
    try:
        rc, stdout = backend.run_driver(driver_source(), _UPSTREAM_BYTES,
                                        item_json, on_create=on_create)
    except Exception as e:
        rec.update(status="infra_error",
                   detail=f"backend failure: {type(e).__name__}: {str(e)[:150]}")
        return rec
    kind, payload = parse_driver_output(rc, stdout, REFERENCE_SHA256,
                                        expected_input_sha)
    if kind == "infra":
        rec.update(status="infra_error", detail=payload)
        return rec
    if kind == "confused":
        rec.update(status="timed_out", correct=0,
                   detail="conflicting verdict frames (rc=%s); confusion class - "
                          "never trusted, conservative fail" % rc,
                   phase_evidence=_phase_evidence(payload))
        return rec
    startup, verdict = payload
    rec["phase_evidence"] = _phase_evidence(startup)
    result = verdict["result"]
    if result == "passed":
        rec.update(status="passed", correct=1, detail="")
    elif result == "timed out":
        rec.update(status="timed_out", correct=0, detail="upstream: timed out")
    else:
        rec.update(status="failed", correct=0, detail=str(result)[:200])
    return rec


def _phase_evidence(startup):
    return {"startup": True, "module_sha256": startup.get("module_sha256"),
            "input_sha256": startup.get("input_sha256"),
            "driver_python": startup.get("python")}


# ---- backends -----------------------------------------------------------------

class LocalBackend:
    """Runs the same driver + pinned module as a local subprocess. ONLY for the
    authored-fixture self-check (fixture_mode=True is mandatory); NOT an
    isolation boundary, never for model-written code in a real run."""

    def __init__(self, fixture_mode=False):
        if not fixture_mode:
            raise RuntimeError("LocalBackend is fixture-only; real runs use "
                               "SandboxBackend (fresh Modal Sandbox per item)")

    def run_driver(self, driver_src, module_bytes, item_json, on_create=None):
        if on_create is not None:
            on_create("local-fixture")
        with tempfile.TemporaryDirectory() as d:
            work = os.path.join(d, "work")
            os.mkdir(work)
            # the driver hardcodes WORK=/work for the sandbox; local runs patch it
            with open(os.path.join(work, "driver.py"), "w", encoding="utf-8") as f:
                # Match the Linux Python 3.13 target's fork semantics for authored
                # fixtures on macOS (whose default is spawn). Never used for
                # model output; the real SandboxBackend is unchanged.
                local_driver = driver_src.replace('WORK = "/work"', f'WORK = {work!r}')
                local_driver = local_driver.replace('import json', "import multiprocessing\nmultiprocessing.set_start_method('fork', force=True)\nimport json", 1)
                f.write(local_driver)
            with open(os.path.join(work, "he_upstream.py"), "wb") as f:
                f.write(module_bytes)
            with open(os.path.join(work, "item.json"), "w", encoding="utf-8") as f:
                f.write(item_json)
            p = subprocess.Popen([sys.executable, os.path.join(work, "driver.py")],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, start_new_session=True)
            try:
                rc = p.wait(timeout=EXEC_TIMEOUT_SEC + 20)
            except subprocess.TimeoutExpired:
                rc = None
            finally:
                # reap the whole group: a killed/hung driver leaves the
                # upstream child and the multiprocessing Manager holding
                # the stdout pipe open, which would block the read
                import signal as _sig
                try:
                    os.killpg(p.pid, _sig.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            out = p.stdout.read() if p.stdout else ""
            return rc, out


class SandboxBackend:
    """Fresh Modal Sandbox per item. The driver and the pinned upstream module
    are written as literal files; the driver verifies the module hash in-sandbox
    before importing it (runtime pin enforcement)."""

    def __init__(self, app_name="he-suite"):
        self.app_name = app_name

    def run_driver(self, driver_src, module_bytes, item_json, on_create=None):
        import modal
        # enforce the pinned SDK at every use, not just at the scorer's entry
        actual = getattr(modal, "__version__", "unknown")
        if actual != REQUIRED_MODAL_VERSION:
            raise RuntimeError(f"modal SDK {actual} != pinned "
                               f"{REQUIRED_MODAL_VERSION} - refusing to run on "
                               "an unpinned runtime")
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        # NO add_python: the digest-pinned base already ships python3.13, and an
        # added layer would be unpinned (review fix). The FINAL runtime (base +
        # anything Modal injects) is not observed/pinned until the first
        # authorized run records it.
        image = modal.Image.from_registry(SANDBOX_IMAGE)
        # create the sandbox IDLE (sleep); the driver runs via exec only after
        # all three files are in place. The Modal binding (Sandbox.create exec/
        # open semantics, caps, block_network, terminate) is code-reviewed but
        # Live backend gate verifies this path before accepting model results.
        sb = modal.Sandbox.create(
            "sleep", str(SANDBOX_LIFETIME_SEC),
            image=image, app=app,
            cpu=SANDBOX_CPU, memory=SANDBOX_MEMORY,
            timeout=SANDBOX_LIFETIME_SEC,
            block_network=True,
            secrets=None,
        )
        try:
            # persist the exact sandbox ID the MOMENT create returns (review:
            # billing reconciliation binds to this ID; if the create response
            # is lost the item stays unknown - never blindly recreated).
            # INSIDE the try (review round 6b): a ledger callback failure must
            # still hit the explicit terminate in finally, not just the 30s
            # server-side backstop.
            if on_create is not None:
                on_create(getattr(sb, "object_id", None) or "unknown-id")
            p = sb.exec("mkdir", "-p", "-m", "700", "/work")
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"sandbox mkdir rc={p.returncode}")
            sb.filesystem.write_text(driver_src, "/work/driver.py")
            sb.filesystem.write_text(module_bytes.decode("utf-8"), "/work/he_upstream.py")
            sb.filesystem.write_text(item_json, "/work/item.json")
            proc = sb.exec("python3", "/work/driver.py")
            out = proc.stdout.read()
            proc.wait()
            return proc.returncode, out
        finally:
            sb.terminate(wait=True)
