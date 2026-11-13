"""Task packs for the capability eval suite.

Each pack answers one gap from the capability-claim review:
  mmlu            - anchor at the article's n=1000, letter-logit MCQ, per-item logging
  gsm8k           - generative math (declared exact-match protocol), not letter logits
  ifeval_custom   - custom instruction checks on IFEval rows (NOT official IFEval
                    scoring; see the honesty note on CUSTOM_CHECKS below)
  global_mmlu_*   - multilingual knowledge (the model's niche; defense trained
                    English-only), lowercase official configs hi/ar/fr/es
  cybermetric     - cybersecurity knowledge MCQ (tihanyin/CyberMetric, Apache-2.0,
                    arXiv:2402.07688), letter-logit like mmlu; knowledge recall
                    only, NO model-written code is executed

Coding generation uses the pinned HumanEval pack. Execution is a separate
phase in coding_score.py using the isolated backend described in
CODING_PACK_DESIGN.md; run.py never executes model-written code.

Design rules inherited from the repo:
  - add_special_tokens=False everywhere (the double-BOS lesson, notes/LOG.md).
  - Only metrics and per-item 0/1 correctness are logged for publication; raw
    generations stay local (same rule as the safety evals).
  - Stable item IDs (source row index under the pinned revision, or the
    dataset's own key), MANDATORY pinned revisions, fixed seeds and explicit
    sample sizes, so runs are comparable across models and audits can recover
    the exact source rows.
"""
import json
import random
import re
from fractions import Fraction


def _dedupe_check(rows):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate stable ids in selected rows; sample is not item-identifiable")


def _load(dataset, config, split, revision):
    """Pinned load ONLY: a missing or malformed revision raises. 'latest' is never
    an implicit option - that would silently unpin the measurement."""
    if not (isinstance(revision, str) and len(revision) == 40 and
            all(c in "0123456789abcdef" for c in revision)):
        raise ValueError(f"{dataset}: a 40-hex pinned revision is required, got {revision!r}")
    from datasets import load_dataset
    return load_dataset(dataset, config, split=split, revision=revision)


def _with_src_index(ds):
    """Attach the original split row index BEFORE any shuffle: under the pinned
    revision, split#index is the exact, audit-recoverable source row identity."""
    return ds.map(lambda _r, i: {"_src_idx": i}, with_indices=True)

# ---------------------------------------------------------------- MMLU (anchor)

def mmlu_load(n=1000, seed=0, revision=None, meta_out=None):
    """cais/mmlu config 'all', test split, n matching the article's MMLU run.

    NOTE on comparability: the article's 52.3/53.0 numbers came from a run whose
    exact item manifest was never persisted (aggregate-only logging). This harness
    therefore defines a NEW measurement: base and defended models are BOTH run
    here, and comparisons are made within this harness only. Do not compare these
    numbers against the article's old aggregates.
    """
    ds = _with_src_index(_load("cais/mmlu", "all", "test", revision))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    rows = [{"id": f"test#{r['_src_idx']}", "question": r["question"],
             "choices": list(r["choices"]), "answer": int(r["answer"])} for r in ds]
    _dedupe_check(rows)
    if meta_out is not None:
        meta_out.update(dataset="cais/mmlu", config="all", split="test",
                        revision=revision, selected=len(rows))
    return rows


def mmlu_prompt(row):
    return (row["question"] + "\n" + "\n".join(
        f"{l}. {c}" for l, c in zip("ABCD", row["choices"])) + "\nAnswer with the letter only.")

# ---------------------------------------------------------------- HumanEval

HUMANEVAL_N = 164  # fixed whole set; no sampling


def humaneval_load(revision=None, meta_out=None):
    """openai/openai_humaneval (MIT, arXiv:2107.03374): 164 hand-written Python
    problems. ALL rows, no sampling - n is the set size. Stable ids are the
    dataset's own task_id; per-item sha256 over prompt+test+entry_point binds the
    scored fields into the manifest. Fail-closed: count != 164, duplicate ids, or
    missing/empty fields all raise. The canonical_solution field is loaded for the
    harness's pre-run self-check and is NEVER shown to the model."""
    ds = _load("openai/openai_humaneval", "openai_humaneval", "test", revision)
    if len(ds) != HUMANEVAL_N:
        raise ValueError(f"HumanEval: expected {HUMANEVAL_N} rows, got {len(ds)} - refusing")
    import hashlib
    rows = []
    for r in ds:
        for k in ("task_id", "prompt", "test", "entry_point", "canonical_solution"):
            if not (isinstance(r.get(k), str) and r[k].strip()):
                raise ValueError(f"HumanEval {r.get('task_id')!r}: missing/empty field {k}")
        rows.append({"id": r["task_id"],
                     "prompt": r["prompt"], "test": r["test"],
                     "entry_point": r["entry_point"],
                     "canonical_solution": r["canonical_solution"],
                     "fields_sha256": hashlib.sha256(
                         (r["prompt"] + r["test"] + r["entry_point"]).encode()).hexdigest()})
    _dedupe_check(rows)
    if meta_out is not None:
        meta_out.update(dataset="openai/openai_humaneval", config="openai_humaneval",
                        split="test", revision=revision, selected=len(rows))
    return rows


def humaneval_prompt(row):
    return ("Complete this Python function. Reply with ONLY the function body "
            "(the indented lines that complete it). No explanation, no test code.\n\n"
            + row["prompt"])

# ---------------------------------------------------------------- CyberMetric

CYBERMETRIC_FILE = "CyberMetric-10000-v1.json"


def _cybermetric_fetch(revision):
    """Pinned raw-file download ONLY. Kept as a seam so offline tests can inject a
    fixture without network; the 40-hex revision rule is enforced by the caller."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="tihanyin/CyberMetric", repo_type="dataset",
                           filename=CYBERMETRIC_FILE, revision=revision)


def cybermetric_load(n=500, seed=0, revision=None, meta_out=None):
    """tihanyin/CyberMetric: 10k human-verified cybersecurity MCQs compiled from
    standards, certifications and research papers (Apache-2.0, arXiv:2402.07688).
    Raw JSON {'questions': [{question, answers:{A..D}, solution}]}. Same row shape
    and letter-logit measurement as mmlu/global_mmlu. Fail-closed: bad revision,
    missing 'questions', or any malformed row raises rather than scores garbage."""
    if not (isinstance(revision, str) and len(revision) == 40 and
            all(c in "0123456789abcdef" for c in revision)):
        raise ValueError(f"tihanyin/CyberMetric: a 40-hex pinned revision is required, got {revision!r}")
    with open(_cybermetric_fetch(revision), encoding="utf-8") as f:
        payload = json.load(f)
    if not (isinstance(payload, dict) and isinstance(payload.get("questions"), list)):
        raise ValueError("tihanyin/CyberMetric: schema changed, expected top-level {'questions': [...]}")
    raw = payload["questions"]
    if len(raw) == 0:
        raise ValueError("tihanyin/CyberMetric: 0 questions; refusing to score")
    rows = []
    for i, r in enumerate(raw):  # source index BEFORE shuffle: audit-recoverable identity
        ans = r.get("answers") if isinstance(r, dict) else None
        sol = r.get("solution") if isinstance(r, dict) else None
        if not (isinstance(r.get("question"), str) and r["question"].strip() and
                isinstance(ans, dict) and set(ans) == {"A", "B", "C", "D"} and
                all(isinstance(ans[k], str) and ans[k].strip() for k in "ABCD") and
                sol in ("A", "B", "C", "D")):
            raise ValueError(f"tihanyin/CyberMetric row {i} malformed: {str(r)[:120]!r}")
        rows.append({"id": f"cm#{i}", "_src_idx": i, "question": r["question"],
                     "choices": [ans["A"], ans["B"], ans["C"], ans["D"]],
                     "answer": "ABCD".index(sol)})
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows = [rows[i] for i in order[:min(n, len(rows))]]
    for r in rows:
        del r["_src_idx"]
    _dedupe_check(rows)
    if meta_out is not None:
        meta_out.update(dataset="tihanyin/CyberMetric", config=CYBERMETRIC_FILE,
                        split="questions", revision=revision, selected=len(rows))
    return rows

# ---------------------------------------------------------------- GSM8K

def gsm8k_load(n=200, seed=0, revision=None, meta_out=None):
    """openai/gsm8k main/test. {id, question, answer} with answer the final number."""
    ds = _with_src_index(_load("openai/gsm8k", "main", "test", revision))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    rows = [{"id": f"test#{r['_src_idx']}", "question": r["question"],
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in ds]
    _dedupe_check(rows)
    if meta_out is not None:
        meta_out.update(dataset="openai/gsm8k", config="main", split="test",
                        revision=revision, selected=len(rows))
    return rows


def gsm8k_prompt(row):
    return (
        "Solve this math problem. Think briefly, then end your reply with a line "
        "containing only 'Answer: <number>'.\n\n" + row["question"]
    )

# Declared extraction protocol (fixed BEFORE any run):
#  - If the text contains 'Answer:' (any case), the answer is the FIRST LINE
#    after the LAST such marker, and that line must consist ENTIRELY of one
#    number: optional leading '$', optional minus, digits with optional commas,
#    optional decimal part ('.5' allowed), OPTIONAL simple fraction 'a/b',
#    optional single trailing '.'. Anything else on the line ('42or43',
#    'not 42', '42 bananas', '42%') is a failed extraction - it never awards a
#    prefix, a substring or a negation.
#  - Scientific notation is REJECTED everywhere (returns None): '1e5' is a
#    failed extraction, never silently parsed as 1 or 5.
#  - If there is no marker, the fallback is the last number-like token in the
#    text that does NOT overlap an exponent span ('1e5', '1e+5'); fallback hits
#    and marker misses are counted as extraction_fallbacks /
#    extraction_failures by the runner so truncation damage is visible.
#  - Scoring is EXACT rational equality (fractions.Fraction) with the gold
#    number. There is no tolerance.
_NUM_RE = re.compile(r"-?(?:\d[\d,]*\.?\d*|\.\d+)")
_NUM_LINE_RE = re.compile(
    r"^\s*\$?\s*(-?(?:\d[\d,]*\.?\d*|\.\d+))(?:\s*/\s*(\d[\d,]*\.?\d*|\.\d+))?\s*\.?\s*$")
_EXP_RE = re.compile(r"\d\s*[eE][+-]?\d")
_EXP_AFTER_RE = re.compile(r"[eE][+-]?\d")


def _to_number(s):
    """Parse a string as ONE complete number (the whole line), as an exact
    fractions.Fraction, else None. '42 or 43', 'not 42', '42 bananas', '42%'
    and mixed numbers like '1 1/3' all fail - never a prefix or substring."""
    s = s.strip()
    if not s:
        return None
    if _EXP_RE.search(s):
        return None  # exponent notation: declared unsupported, fail rather than misparse
    m = _NUM_LINE_RE.match(s)
    if not m:
        return None
    try:
        num = Fraction(m.group(1).replace(",", ""))
        if m.group(2) is not None:
            den = Fraction(m.group(2).replace(",", ""))
            return num / den if den else None
        return num
    except (ValueError, ZeroDivisionError):
        return None


def _fallback_number(text):
    """Last number-like token that does not overlap a scientific-notation span.
    'therefore 1e5' and 'therefore 1e+5' both yield None - the signed exponent
    cannot smuggle its trailing digits in as a separate token."""
    exp_spans = [m.span() for m in _EXP_RE.finditer(text)]
    def _in_exponent(m):
        return any(m.start() < e and m.end() > s for s, e in exp_spans)
    for m in reversed(list(_NUM_RE.finditer(text))):
        if _in_exponent(m):
            continue
        return Fraction(m.group(0).replace(",", ""))
    return None


def gsm8k_extract(text):
    """Returns (value_or_None, route) where route is 'marker' | 'fallback' | 'none'."""
    if not text:
        return None, "none"
    parts = re.split(r"answer\s*:", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        tail = parts[-1].strip()
        line = tail.splitlines()[0] if tail else ""
        v = _to_number(line)
        return (v, "marker") if v is not None else (None, "none")
    v = _fallback_number(text)
    return (v, "fallback") if v is not None else (None, "none")


def gsm8k_score_detail(row, generation):
    """Exact-match scoring: the extracted Fraction must equal the gold Fraction
    EXACTLY. There is no float tolerance in the declared protocol."""
    value, route = gsm8k_extract(generation)
    try:
        gold = Fraction(str(row["answer"]).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return {"id": row.get("id"), "correct": 0, "extracted": value, "route": route}
    correct = int(value is not None and value == gold)
    return {"id": row.get("id"), "correct": correct, "extracted": value, "route": route}


def gsm8k_score(row, generation):
    return gsm8k_score_detail(row, generation)["correct"]

# ------------------------------------------------- Custom instruction checks
# HONESTY NOTE: these are CUSTOM, deterministic checks on IFEval rows - they are
# NOT the official IFEval verifier registry and this pack must be cited as
# "custom instruction-following checks on IFEval prompts", never as IFEval.
# Known semantic differences from the official verifiers (google-research
# instruction_following_eval/instructions.py):
#  - word counting here is str.split(); the official counter differs.
#  - keyword frequency here is a literal substring count; the official is regex.
#  - bullet matching here accepts '* ' and '- ' lines only.
# The end-phrase check DOES mirror the official one: surrounding quotes stripped,
# case-insensitive endswith.

def _c_keywords_existence(text, kw):
    return all(k.lower() in text.lower() for k in kw["keywords"])


def _c_keywords_frequency(text, kw):
    cnt = text.lower().count(kw["keyword"].lower())
    return cnt < kw["frequency"] if kw["relation"] == "less than" else cnt >= kw["frequency"]


def _c_number_words(text, kw):
    n_words = len(text.split())
    return n_words < kw["num_words"] if kw["relation"] == "less than" else n_words >= kw["num_words"]


def _c_number_bullets(text, kw):
    bullets = [l for l in text.splitlines() if re.match(r"^\s*(\*|-)\s+", l)]
    return len(bullets) == kw["num_bullets"]


def _c_end_checker(text, kw):
    phrase = str(kw["end_phrase"]).strip().strip('"').casefold()
    return text.rstrip().strip('"').casefold().endswith(phrase)


def _c_title(text, kw):
    return bool(re.search(r"<<[^\n]+>>", text))


CUSTOM_CHECKS = {
    "keywords:existence": _c_keywords_existence,
    "keywords:frequency": _c_keywords_frequency,
    "length_constraints:number_words": _c_number_words,
    "detectable_format:number_bullet_lists": _c_number_bullets,
    "startend:end_checker": _c_end_checker,
    "detectable_format:title": _c_title,
}

# Required parameters per check, used to validate rows at load time. A row whose
# kwargs lack a required key, or carry an invalid relation, is EXCLUDED from the
# subset - it is never scored, and never silently passed.
_REQUIRED_PARAMS = {
    "keywords:existence": {"keywords": list},
    "keywords:frequency": {"keyword": str, "relation": str, "frequency": int},
    "length_constraints:number_words": {"relation": str, "num_words": int},
    "detectable_format:number_bullet_lists": {"num_bullets": int},
    "startend:end_checker": {"end_phrase": str},
    "detectable_format:title": {},
}
_VALID_RELATIONS = {"less than", "at least"}


def params_ok(instruction_id, kwargs):
    """Pure validation of one instruction's kwargs. Fail-closed: any defect
    returns False and the whole row is excluded by the loader."""
    spec = _REQUIRED_PARAMS.get(instruction_id)
    if spec is None or not isinstance(kwargs, dict):
        return False
    for key, typ in spec.items():
        if key not in kwargs or not isinstance(kwargs[key], typ):
            return False
        if typ is int and not kwargs[key] > 0:        # counts must be positive
            return False
    if "relation" in spec and kwargs["relation"] not in _VALID_RELATIONS:
        return False
    if "keywords" in spec and (not kwargs["keywords"]
                               or not all(isinstance(k, str) and k for k in kwargs["keywords"])):
        return False
    return True


def ifeval_load(n=200, seed=0, revision=None, meta_out=None):
    """IFEval rows (google/IFEval, split 'train' - the benchmark's only split, 541
    rows) where EVERY instruction has a custom check above AND valid params.

    Rows are keyed by the dataset's own 'key' field (stable source identity).
    Eligible/excluded/selected counts are reported through meta_out: if fewer
    than n rows qualify, the shortfall is explicit in the manifest, not silent.
    """
    ds = _load("google/IFEval", "default", "train", revision)
    rows = []
    excluded = {"shape_mismatch": 0, "unsupported_instruction": 0, "bad_params": 0}
    for r in ds:
        ids, kws = r["instruction_id_list"], r["kwargs"]
        if (not ids or not isinstance(kws, (list, tuple)) or len(kws) != len(ids)
                or not all(isinstance(k, dict) for k in kws)):
            excluded["shape_mismatch"] += 1
            continue
        if not all(i in CUSTOM_CHECKS for i in ids):
            excluded["unsupported_instruction"] += 1
            continue
        if not all(params_ok(i, k) for i, k in zip(ids, kws)):
            excluded["bad_params"] += 1
            continue
        rows.append({"id": str(r["key"]), "prompt": r["prompt"],
                     "instruction_id_list": list(ids), "kwargs": [dict(k) for k in kws]})
    random.Random(seed).shuffle(rows)
    sel = rows[:n]
    _dedupe_check(sel)
    if not sel:
        raise ValueError("ifeval_custom: zero eligible rows; refusing to score an empty pack")
    if meta_out is not None:
        type_counts = {}
        for r in sel:
            for i in r["instruction_id_list"]:
                type_counts[i] = type_counts.get(i, 0) + 1
        meta_out.update(dataset="google/IFEval", config="default", split="train",
                        revision=revision, eligible=len(rows),
                        excluded=dict(excluded), requested=n, selected=len(sel),
                        instruction_type_counts=type_counts)
    return sel


def ifeval_score(row, generation):
    ids, kws = row["instruction_id_list"], row["kwargs"]
    if not ids:
        raise ValueError("empty instruction list is malformed, not a vacuous pass")
    if not generation or not generation.strip():
        return 0
    if len(ids) != len(kws):
        return 0  # shape mismatch: row should have been excluded at load; never silently skip
    for iid, kw in zip(ids, kws):
        try:
            ok = CUSTOM_CHECKS[iid](generation, kw)
        except (KeyError, TypeError):
            return 0  # missing verifier params: strict fail, never silent pass
        if not ok:
            return 0
    return 1

# ---------------------------------------------------------------- Global MMLU

GLOBAL_MMLU_LANGS = ["hi", "ar", "fr", "es"]  # official lowercase configs; the 4 languages of section 9
GLOBAL_REQUIRED_COLS = {"sample_id", "question", "option_a", "option_b",
                        "option_c", "option_d", "answer"}


def global_mmlu_load(lang, n=200, seed=0, revision=None, meta_out=None):
    """CohereLabs/Global-MMLU for one language. Fail-closed: unknown language,
    missing schema columns, zero rows, or a bad answer label all raise - this
    loader can never silently score wrong-language or malformed rows. Items are
    keyed by the dataset's own sample_id."""
    lang = lang.lower()
    if lang not in GLOBAL_MMLU_LANGS:
        raise ValueError(f"Global-MMLU language '{lang}' not in allow-list {GLOBAL_MMLU_LANGS}")
    ds = _load("CohereLabs/Global-MMLU", lang, "test", revision)
    missing = GLOBAL_REQUIRED_COLS - set(ds.column_names)
    if missing:
        raise ValueError(f"Global-MMLU '{lang}' schema changed, missing columns {sorted(missing)}")
    if len(ds) == 0:
        raise ValueError(f"Global-MMLU config '{lang}' returned 0 rows; refusing to score")
    ds = _with_src_index(ds).shuffle(seed=seed).select(range(min(n, len(ds))))
    rows = []
    for r in ds:
        if not (isinstance(r["answer"], str) and r["answer"] in ("A", "B", "C", "D")):
            raise ValueError(f"Global-MMLU '{lang}' row {r['sample_id']} has bad answer {r['answer']!r}")
        rows.append({"id": f"{lang}-{r['sample_id']}", "src_index": r["_src_idx"],
                     "question": r["question"],
                     "choices": [r["option_a"], r["option_b"], r["option_c"], r["option_d"]],
                     "answer": "ABCD".index(r["answer"])})
    _dedupe_check(rows)
    if meta_out is not None:
        meta_out.update(dataset="CohereLabs/Global-MMLU", config=lang, split="test",
                        revision=revision, selected=len(rows))
    return rows
