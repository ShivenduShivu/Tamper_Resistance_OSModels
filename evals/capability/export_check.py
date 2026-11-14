"""Durable export + verification for run artifacts. Stdlib only, offline-testable.

Three distinct tiers, never conflated:
  presence   - expected files exist (expected_files vs index)
  integrity  - existing files match recorded size+sha256 (verify_against_index)
  semantic   - contents parse and cross-check: summary cells == tasks x models,
               n_scored + skipped == selected, item/response row counts and
               id joins against items_manifest (validate_semantics)

Fail-closed: complete_valid requires one COMPLETE marker, a strictly
well-formed manifest, zero expected-missing, AND zero semantic problems.
Checksum integrity alone NEVER implies completeness. A FAILED run is
preserved partial evidence, explicitly distinct. Invalid manifests produce
reports, never exceptions.
"""
import hashlib
import json
import os
import shutil

INDEX_NAME = "artifact_index.json"

COMPLETE_VALID = "complete_valid"
COMPLETE_INVALID = "complete_invalid"
PARTIAL_FAILURE = "partial_failure"
INVALID = "invalid"


def sha256_file(path, _buf=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _walk_files(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            fp = os.path.join(dirpath, name)
            rel = os.path.relpath(fp, root)
            if rel == INDEX_NAME:
                continue
            out.append((rel, fp))
    return sorted(out)


def build_index(root):
    return {rel: {"size": os.path.getsize(fp), "sha256": sha256_file(fp)}
            for rel, fp in _walk_files(root)}


def _is_str_list(x):
    return isinstance(x, list) and x and all(isinstance(v, str) for v in x)


def validate_manifest(manifest):
    """(ok, problems[]) - strict shapes; never raises."""
    problems = []
    if not isinstance(manifest, dict):
        return False, ["manifest is not an object"]
    argv = manifest.get("argv")
    if not isinstance(argv, dict):
        return False, ["argv missing or not an object"]
    models = argv.get("models")
    tasks = argv.get("tasks")
    if not _is_str_list(models):
        problems.append("argv.models missing/empty/not a string list")
    if not _is_str_list(tasks):
        problems.append("argv.tasks missing/empty/not a string list")
    if problems:
        return False, problems
    mt = manifest.get("tasks")
    if not isinstance(mt, dict):
        return False, ["manifest.tasks missing or not an object"]
    for t in tasks:
        entry = mt.get(t)
        if not isinstance(entry, dict):
            problems.append("manifest.tasks.%s missing or not an object" % t)
            continue
        if entry.get("kind") not in ("mcq", "generative"):
            problems.append("manifest.tasks.%s.kind missing/invalid" % t)
        sel = entry.get("selected")
        if sel is not None and not (isinstance(sel, int) and sel >= 0):
            problems.append("manifest.tasks.%s.selected invalid" % t)
    return (not problems, problems)


def expected_files(manifest):
    """Derived by task kind and model count. Assumes a valid manifest."""
    argv = manifest["argv"]
    models = list(argv["models"])
    tasks = list(argv["tasks"])
    kinds = {t: manifest["tasks"][t]["kind"] for t in tasks}
    exp = {"manifest.json", "summary.jsonl"}
    if len(models) > 1:
        exp.add("paired_vs_base.json")
    for t in tasks:
        exp.add("items_manifest_%s.json" % t)
        for m in models:
            exp.add("items_%s__%s.jsonl" % (t, m))
            if kinds[t] == "generative":
                exp.add(os.path.join("responses", "%s__%s.jsonl" % (t, m)))
    return exp


def _read_jsonl(path):
    """(rows, problem) - parses every line; empty file is zero rows, not an error."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    return None, "%s line %d unparseable" % (
                        os.path.basename(path), i + 1)
    except OSError as e:
        return None, "%s unreadable: %s" % (os.path.basename(path), e)
    return rows, None


def _checked_ids(rows, label, problems):
    """Unique nonempty-string ids from dict rows, or None. Never raises
    (malformed ids such as lists/empties are reported, not crashed on)."""
    ids = []
    bad = 0
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) \
                or not r.get("id"):
            bad += 1
            continue
        ids.append(r["id"])
    if bad:
        problems.append("%s: %d rows with missing/malformed id" % (label, bad))
    if len(set(ids)) != len(ids):
        problems.append("%s: duplicate ids" % label)
    if bad or len(set(ids)) != len(ids):
        return None
    return set(ids)


def validate_semantics(run_dir, manifest):
    """Content-level cross-checks. Assumes a valid manifest. Never raises."""
    problems = []
    tasks = list(manifest["argv"]["tasks"])
    models = list(manifest["argv"]["models"])
    kinds = {t: manifest["tasks"][t]["kind"] for t in tasks}
    selected = {t: manifest["tasks"][t].get("selected") for t in tasks}

    # summary.jsonl: one JSON object per line; metric cells <task>__<model>
    srows, err = _read_jsonl(os.path.join(run_dir, "summary.jsonl"))
    if err:
        problems.append(err)
        cells = {}
    else:
        cells = {}
        for r in srows:
            if isinstance(r, dict):
                for k, v in r.items():
                    cells[k] = v
    manifest_ids = {}
    for t in tasks:
        # items_manifest: frozen selected rows with unique ids
        im_path = os.path.join(run_dir, "items_manifest_%s.json" % t)
        try:
            rows = json.load(open(im_path, encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("not a list")
        except (OSError, ValueError) as e:
            problems.append("items_manifest_%s.json unparseable: %s" % (t, e))
            manifest_ids[t] = None
            continue
        manifest_ids[t] = _checked_ids(
            rows, "items_manifest_%s.json" % t, problems)
        if selected[t] is not None and len(rows) != selected[t]:
            problems.append("items_manifest_%s.json: %d rows != selected %d"
                            % (t, len(rows), selected[t]))
        if not rows:
            problems.append("items_manifest_%s.json empty selection" % t)

        for m in models:
            # summary cell
            cell = cells.get("%s__%s" % (t, m))
            if not isinstance(cell, dict):
                problems.append("summary.jsonl missing metric cell %s__%s" % (t, m))
                n_scored = None
            else:
                n_scored = cell.get("n_scored")
                if not isinstance(n_scored, int):
                    problems.append("summary cell %s__%s n_scored missing/invalid"
                                    % (t, m))
                    n_scored = None
            # items file
            irows, err = _read_jsonl(
                os.path.join(run_dir, "items_%s__%s.jsonl" % (t, m)))
            if err:
                problems.append(err)
                continue
            iids = _checked_ids(irows, "items_%s__%s.jsonl" % (t, m), problems)
            if manifest_ids[t] is not None and iids is not None:
                if iids != manifest_ids[t]:
                    problems.append("items_%s__%s.jsonl: id set does not match "
                                    "items_manifest" % (t, m))
                if len(irows) != len(manifest_ids[t]):
                    problems.append("items_%s__%s.jsonl: %d rows != selection %d"
                                    % (t, m, len(irows), len(manifest_ids[t])))
            # skip/scored schema cross-check: a scored row must carry a valid
            # correctness field; a skipped row must not carry one
            n_items_scored = 0
            bad_correct = 0
            bad_skipped = 0
            for r in irows:
                if not isinstance(r, dict):
                    continue
                if "skipped" in r:
                    if "correct" in r:
                        bad_skipped += 1
                else:
                    n_items_scored += 1
                    if not (isinstance(r.get("correct"), int)
                            and r.get("correct") in (0, 1)):
                        bad_correct += 1
            if bad_correct:
                problems.append("items_%s__%s.jsonl: %d scored rows missing "
                                "valid correctness field" % (t, m, bad_correct))
            if bad_skipped:
                problems.append("items_%s__%s.jsonl: %d skipped rows carrying "
                                "correctness field" % (t, m, bad_skipped))
            if n_scored is not None and n_items_scored != n_scored:
                problems.append("items_%s__%s.jsonl: %d scored rows != summary "
                                "n_scored %d" % (t, m, n_items_scored, n_scored))
            # responses file (generative only)
            if kinds[t] == "generative":
                rrows, err = _read_jsonl(os.path.join(
                    run_dir, "responses", "%s__%s.jsonl" % (t, m)))
                if err:
                    problems.append(err)
                else:
                    rids = _checked_ids(rrows, "responses/%s__%s.jsonl"
                                        % (t, m), problems)
                    if manifest_ids[t] is not None and rids is not None \
                            and rids != manifest_ids[t]:
                        problems.append("responses/%s__%s.jsonl: id set does "
                                        "not match items_manifest" % (t, m))
                    if len(rrows) != len(irows):
                        problems.append("responses/%s__%s.jsonl: %d rows != items "
                                        "%d" % (t, m, len(rrows), len(irows)))
                    # the response field must be PRESENT on every row;
                    # an empty string is a valid (failed) generation
                    miss = sum(1 for r in rrows
                               if isinstance(r, dict) and "response" not in r)
                    if miss:
                        problems.append("responses/%s__%s.jsonl: %d rows missing "
                                        "response field" % (t, m, miss))
                    badt = sum(1 for r in rrows
                               if isinstance(r, dict) and "response" in r
                               and not isinstance(r["response"], str))
                    if badt:
                        problems.append("responses/%s__%s.jsonl: %d rows with "
                                        "non-string response" % (t, m, badt))
    # summary must not carry cells outside the declared grid; "DONE" is the
    # runner's own terminal log record (run.py writes {"DONE": true} last)
    grid = {"%s__%s" % (t, m) for t in tasks for m in models}
    for k in cells:
        if k in ("paired_vs_base", "DONE") or k.endswith("_sec") or k in grid:
            continue
        problems.append("summary.jsonl unexpected cell %r" % k)
    return problems


def write_index(run_dir):
    """Write artifact_index.json into run_dir. Returns the report dict."""
    index = build_index(run_dir)
    complete = os.path.exists(os.path.join(run_dir, "COMPLETE"))
    failed = os.path.exists(os.path.join(run_dir, "FAILED"))

    manifest = None
    manifest_problems = []
    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mpath):
        try:
            manifest = json.load(open(mpath, encoding="utf-8"))
        except (OSError, ValueError) as e:
            manifest_problems.append("manifest unparseable: %s" % e)
    else:
        manifest_problems.append("manifest.json missing")
    if manifest is not None:
        ok, problems = validate_manifest(manifest)
        manifest_problems.extend(problems)
    else:
        ok = False

    exp = expected_files(manifest) if ok else set()
    missing = sorted(f for f in exp if f not in index)
    semantic = validate_semantics(run_dir, manifest) if ok else []

    if complete and failed:
        state = INVALID
    elif failed:
        state = PARTIAL_FAILURE
    elif not complete:
        state = INVALID
    elif not ok:
        state = COMPLETE_INVALID
    elif missing:
        state = COMPLETE_INVALID
    elif semantic:
        state = COMPLETE_INVALID
    else:
        state = COMPLETE_VALID

    report = {
        "run_dir": os.path.basename(run_dir.rstrip("/")),
        "state": state,
        "n_files": len(index),
        "complete_marker": complete,
        "failed_marker": failed,
        "manifest_ok": ok,
        "manifest_problems": manifest_problems,
        "expected_checked": ok,
        "expected_missing": missing,
        "semantic_ok": ok and not semantic,
        "semantic_problems": semantic,
        "files": index,
    }
    with open(os.path.join(run_dir, INDEX_NAME), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def export_tree(src, dst):
    """Copy every file under src into dst (which must not equal src)."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("export destination must differ from source")
    for rel, fp in _walk_files(src):
        target = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(fp, target)
    idx = os.path.join(src, INDEX_NAME)
    if os.path.exists(idx):
        shutil.copy2(idx, os.path.join(dst, INDEX_NAME))


def verify_against_index(root, index):
    """Integrity tier only: every indexed file exists with matching
    size+sha256. Says nothing about presence-completeness or semantics."""
    problems = []
    for rel, meta in sorted(index.items()):
        fp = os.path.join(root, rel)
        if not os.path.exists(fp):
            problems.append("missing: " + rel)
            continue
        if os.path.getsize(fp) != meta["size"]:
            problems.append("size mismatch: " + rel)
            continue
        if sha256_file(fp) != meta["sha256"]:
            problems.append("checksum mismatch: " + rel)
    return (not problems, problems)


def export_and_verify(run_dir, dst):
    """Index, export, integrity-verify. The report 'state' is the evidence
    verdict across all three tiers; 'export.verified' is integrity of the copy."""
    report = write_index(run_dir)
    export_tree(run_dir, dst)
    ok, problems = verify_against_index(dst, report["files"])
    report["export"] = {"dst": dst, "verified": ok, "problems": problems}
    return report
