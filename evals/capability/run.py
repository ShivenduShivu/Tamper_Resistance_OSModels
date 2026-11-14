"""Capability eval runner for the tamper-resistance project.

Purpose (grounded in the official Track 2b statement, "Tamper resistance: make
safety survive later harmful fine-tuning by someone else, and demonstrate it with
attack fine-tunes"): the statement asks for safety survival; the capability side
enters only through the article's own claim "without becoming less useful".
This suite exists to defend THAT claim with bounded, pre-registered evidence -
not to be a general model leaderboard. It measures CLEAN models only.

Usage (GPU box, e.g. Kaggle T4, same environment conventions as kaggle/day*):
    python run.py --models base v2 v2b --tasks mmlu gsm8k ifeval_custom \
        --primary mmlu:v2 --out /kaggle/working/capability

Available packs: mmlu, gsm8k, ifeval_custom, global_mmlu_{hi,ar,fr,es},
cybermetric (cybersecurity knowledge MCQ; letter-logit, no code execution).

Free preflight (no model, no GPU, no paid anything):
    python run.py --smoke
Checks: exactly one BOS after chat template + add_special_tokens=False,
single-token A/B/C/D, pad token; dataset schemas on the live datasets-server;
pinned lock revisions resolvable. Needs an HF_TOKEN env var only because
CohereLabs/tiny-aya-global is a gated repo (tokenizer download).

Output layout (new unique directory per run; the runner FAILS if it exists -
no silent overwrite, no resume):
    <out>/run-<timestamp>-<rand>/
        manifest.json          - CLI args, seed, pinned revisions, suite source
                                 hashes, environment versions, generation/token
                                 budgets, skip policy, per-task eligible/excluded/
                                 selected counts, counters, letter-token ids
        items_manifest_<task>.json - the FROZEN selected rows: stable id, source
                                 row index under the pinned revision, and the full
                                 source content (question/choices/answer). Written
                                 before any model loads; this is the audit record.
        summary.jsonl          - metrics only
        items_<task>__<model>.jsonl  - per-item ids + correctness; MCQ items also
                                 carry the four option scores, the top-1 token id
                                 and whether it was a letter; generative items
                                 carry output token count and eos_reached
        responses/<task>__<model>.jsonl - raw generations, LOCAL ONLY, never
                                 published (same rule as the safety evals)
        paired_vs_base.json    - candidate-minus-base paired analysis with
                                 detected_loss / detected_gain direction labels
        COMPLETE / FAILED      - terminal markers; a directory without COMPLETE
                                 is not a result. manifest.json is also (re)written
                                 on failure with the counters collected so far.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import time
import uuid

import tasks
import scoring

MODELS = {
    "base": ("CohereLabs/tiny-aya-global", None),
    "v1": ("StarWarrior24x7/tiny-aya-global-xref-v1", None),
    "v2": ("StarWarrior24x7/tiny-aya-global-xref-v2", None),
    "v2b": ("StarWarrior24x7/tiny-aya-global-xref-v2", "v2b"),
    "a1": ("StarWarrior24x7/tiny-aya-global-phaseA-v2-repnoise", None),
    "a2": ("StarWarrior24x7/tiny-aya-global-phaseA-v2-booster", None),
}
DEFAULT_N = {"mmlu": 1000, "gsm8k": 200, "ifeval_custom": 200, "global_mmlu": 200,
             "cybermetric": 500, "humaneval": 164}
# gsm8k 320->512 for the panel run: gate-1 left an unfinished answer at 320;
# truncation stays counted via the no-EOS counter, not hidden.
GEN_TOKENS = {"gsm8k": 512, "ifeval_custom": 512, "humaneval": 512}
BATCH = {"mcq": 4, "gen": 8}

MAX_PROMPT_TOKENS = 1536
REMOVED_TASKS = {"mbpp": "removed: disclosed-test pass@1 is not a standard MBPP "
                         "protocol. Coding coverage ships as the HumanEval pack "
                         "(isolated execution phase; see CODING_PACK_DESIGN.md)."}

_LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_lock.json")
_TASK_DATASET = {"mmlu": "cais/mmlu", "gsm8k": "openai/gsm8k", "ifeval_custom": "google/IFEval",
                 "global_mmlu": "CohereLabs/Global-MMLU", "cybermetric": "tihanyin/CyberMetric",
                 "humaneval": "openai/openai_humaneval"}


def load_lock(path=None):
    with open(path or _LOCK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _require_rev(lock, kind, key):
    rev = lock.get(kind, {}).get(key)
    if not (isinstance(rev, str) and len(rev) == 40 and
            all(c in "0123456789abcdef" for c in rev)):
        raise ValueError(f"dataset_lock.json is missing a valid 40-hex revision for "
                         f"{kind}:{key}; pinning is mandatory, 'latest' is never used")
    return rev


def resolve_task(name):
    if name in REMOVED_TASKS:
        raise ValueError(f"task '{name}' {REMOVED_TASKS[name]}")
    if name in ("mmlu", "gsm8k", "ifeval_custom", "cybermetric", "humaneval"):
        return name
    if name.startswith("global_mmlu_"):
        lang = name.rsplit("_", 1)[-1]
        if lang in tasks.GLOBAL_MMLU_LANGS:
            return name
    raise ValueError(f"unknown task {name!r}; valid: mmlu, gsm8k, ifeval_custom, cybermetric, humaneval, "
                     f"{', '.join('global_mmlu_' + l for l in tasks.GLOBAL_MMLU_LANGS)}")


def is_mcq(task):
    return task in ("mmlu", "cybermetric") or task.startswith("global_mmlu_")


def load_rows(task, limit, seed, lock, meta):
    if task == "mmlu":
        return tasks.mmlu_load(limit, seed, _require_rev(lock, "datasets", "cais/mmlu"), meta)
    if task == "gsm8k":
        return tasks.gsm8k_load(limit, seed, _require_rev(lock, "datasets", "openai/gsm8k"), meta)
    if task == "ifeval_custom":
        return tasks.ifeval_load(limit, seed, _require_rev(lock, "datasets", "google/IFEval"), meta)
    if task.startswith("global_mmlu_"):
        return tasks.global_mmlu_load(task.rsplit("_", 1)[-1], limit, seed,
                                      _require_rev(lock, "datasets", "CohereLabs/Global-MMLU"), meta)
    if task == "cybermetric":
        return tasks.cybermetric_load(limit, seed, _require_rev(lock, "datasets", "tihanyin/CyberMetric"), meta)
    if task == "humaneval":
        return tasks.humaneval_load(_require_rev(lock, "datasets", "openai/openai_humaneval"), meta)
    raise ValueError(task)


def build_prompt(task, row):
    if is_mcq(task):
        return tasks.mmlu_prompt(row)
    if task == "gsm8k":
        return tasks.gsm8k_prompt(row)
    if task == "ifeval_custom":
        return row["prompt"]
    if task == "humaneval":
        return tasks.humaneval_prompt(row)
    raise ValueError(task)


def atomic_write(path, text):
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def log(run_dir, k, v):
    print(k, json.dumps(v) if isinstance(v, (dict, list)) else v, flush=True)
    with open(os.path.join(run_dir, "summary.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")


def suite_source_hashes():
    """sha256 of the suite's own source files: the software protocol is part of
    the measurement, so it is recorded in every manifest."""
    out = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ["run.py", "tasks.py", "scoring.py", "dataset_lock.json"]:
        p = os.path.join(here, f)
        if os.path.exists(p):
            out[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def check_tokenizer(tok):
    """The one-BOS invariant and single-token letters. Runs in --smoke AND at the
    start of every real run; a run that violates the invariant does not start."""
    sample = tok.apply_chat_template([{"role": "user", "content": "Reply with A."}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok.encode(sample, add_special_tokens=False)
    n_bos = ids.count(tok.bos_token_id) if tok.bos_token_id is not None else 0
    if n_bos != 1:
        raise RuntimeError(f"expected exactly 1 BOS in rendered+encoded prompt, found {n_bos} "
                           "(double-BOS lesson, notes/LOG.md)")
    letters = {}
    for x in "ABCD":
        lids = tok.encode(x, add_special_tokens=False)
        if len(lids) != 1:
            raise RuntimeError(f"letter {x!r} is not a single token on this tokenizer; "
                               "letter-logit scoring is not portable")
        letters[x] = lids[0]
    if tok.pad_token is None and tok.eos_token is None:
        raise RuntimeError("no pad or eos token to pad with")
    return {"bos_count": n_bos, "letter_token_ids": letters,
            "sample_rendered_prompt": sample}


def smoke(lock):
    """Free preflight: tokenizer invariants (needs HF_TOKEN - gated repo), live
    dataset schemas, and lock revisions. No model, no GPU, no datasets download."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["base"][0],
                                        revision=_require_rev(lock, "models", "base"))
    info = check_tokenizer(tok)
    print("tokenizer: 1 BOS OK; letters", info["letter_token_ids"], "; pad OK")

    import urllib.parse
    import urllib.request
    schema_req = {"cais/mmlu#all": {"question", "choices", "answer"},
                  "openai/gsm8k#main": {"question", "answer"},
                  "google/IFEval#default": {"key", "prompt", "instruction_id_list", "kwargs"},
                  "CohereLabs/Global-MMLU#hi": tasks.GLOBAL_REQUIRED_COLS,
                  "CohereLabs/Global-MMLU#ar": tasks.GLOBAL_REQUIRED_COLS,
                  "CohereLabs/Global-MMLU#fr": tasks.GLOBAL_REQUIRED_COLS,
                  "CohereLabs/Global-MMLU#es": tasks.GLOBAL_REQUIRED_COLS,
                  "openai/openai_humaneval#openai_humaneval": {"task_id", "prompt", "test",
                                                               "entry_point", "canonical_solution"}}
    # CyberMetric raw JSON is not served by datasets-server (its viewer fails the
    # cast), so its schema check happens on the pinned file itself, below.
    cm_rev = _require_rev(lock, "datasets", "tihanyin/CyberMetric")
    with open(tasks._cybermetric_fetch(cm_rev), encoding="utf-8") as f:
        cm = json.load(f)
    cm_rows = cm.get("questions") if isinstance(cm, dict) else None
    if not (isinstance(cm_rows, list) and len(cm_rows) > 0 and
            all(k in cm_rows[0] for k in ("question", "answers", "solution"))):
        raise SystemExit("SMOKE FAIL: tihanyin/CyberMetric pinned file schema changed")
    print(f"schema: tihanyin/CyberMetric#{tasks.CYBERMETRIC_FILE} OK ({len(cm_rows)} rows)")
    for dc, req in schema_req.items():
        ds, cfg = dc.split("#")
        url = ("https://datasets-server.huggingface.co/first-rows?dataset="
               + urllib.parse.quote(ds, safe="") + f"&config={cfg}&split="
               + ("train" if ds == "google/IFEval" else "test"))
        with urllib.request.urlopen(url, timeout=30) as r:
            feats = {f["name"] for f in json.loads(r.read())["features"]}
        missing = req - feats
        if missing:
            raise SystemExit(f"SMOKE FAIL: {dc} schema missing {sorted(missing)} "
                             f"(note: datasets-server reflects the default branch; the run "
                             f"itself still loads the pinned revision)")
        print(f"schema: {dc} OK")

    from huggingface_hub import HfApi
    api = HfApi()
    for ds, rev in lock["datasets"].items():
        got = api.dataset_info(ds, revision=rev).sha
        if got != rev:
            raise SystemExit(f"SMOKE FAIL: {ds} pinned revision {rev} does not resolve ({got})")
        print(f"revision: {ds}@{rev[:8]} resolves OK")
    if os.environ.get("HF_TOKEN"):
        for m, rev in lock["models"].items():
            repo, sub = MODELS[m]
            got = api.model_info(repo, revision=rev).sha
            if got != rev:
                raise SystemExit(f"SMOKE FAIL: model {m} pinned revision {rev} != {got}")
            print(f"revision: model {m}@{rev[:8]} resolves OK")
    else:
        print("model revisions: skipped (HF_TOKEN not set; dataset revisions verified)")
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "v2", "v2b"])
    ap.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k", "ifeval_custom"])
    ap.add_argument("--limit", type=int, default=None,
                    help="override per-task sample size (default: per-task DEFAULT_N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary", default=None,
                    help="the ONE pre-registered primary comparison, e.g. 'mmlu:v2'. "
                         "Every other comparison is labeled exploratory.")
    ap.add_argument("--revisions", default=None, help="path to a dataset_lock.json override")
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--run-id", default=None,
                    help="unique caller-supplied run identifier; becomes the run "
                         "directory suffix so the caller knows exactly which "
                         "directory it owns (default: random hex)")
    ap.add_argument("--smoke", action="store_true", help="free tokenizer+schema preflight, then exit")
    args = ap.parse_args()

    lock = load_lock(args.revisions)
    if args.smoke:
        smoke(lock)
        return

    if not args.out:
        ap.error("--out is required for a run")
    # validate everything cheap BEFORE any expensive load
    if len(set(args.models)) != len(args.models) or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("duplicate model or task names are rejected")
    for m in args.models:
        if m not in MODELS:
            raise ValueError(f"unknown model {m!r}; valid: {list(MODELS)}")
        _require_rev(lock, "models", m)
    for t in args.tasks:
        resolve_task(t)
        _require_rev(lock, "datasets",
                     _TASK_DATASET["global_mmlu" if t.startswith("global_mmlu_") else t])
    if len(args.models) > 1 and "base" not in args.models:
        raise ValueError("paired analysis requires 'base' in --models; run base or run a single model")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.primary:
        pt, pm = args.primary.split(":") if ":" in args.primary else (None, None)
        if pt not in args.tasks or pm not in args.models or pm == "base":
            raise ValueError("--primary must be '<task>:<non-base model>' from --tasks/--models")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_suffix = args.run_id or uuid.uuid4().hex[:6]
    if not run_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--run-id must be alphanumeric with optional - or _")
    run_dir = os.path.join(args.out, f"run-{stamp}-{run_suffix}")
    os.makedirs(run_dir)  # fails if it somehow exists: no silent overwrite, no resume
    manifest = {
        "created_utc": stamp, "argv": vars(args), "lock": lock,
        "suite_source_sha256": suite_source_hashes(),
        "sign_convention": "candidate_minus_base (positive delta = candidate better)",
        "decision_rule": "exact McNemar p<0.05 AND Newcombe-10 CI excludes 0; "
                         "direction reported as detected_loss (delta<0) or detected_gain (delta>0)",
        "primary_comparison": args.primary,
        "non_primary_comparisons_are": "exploratory",
        "skip_policy": f"prompts over {MAX_PROMPT_TOKENS} tokens are skipped and counted, "
                       "never truncated",
        "gen_max_new_tokens": GEN_TOKENS, "batch_sizes": BATCH,
        "tasks": {}, "models": {m: {"repo": MODELS[m][0], "subfolder": MODELS[m][1],
                                    "revision": lock["models"][m]} for m in args.models},
    }

    try:
        # 1. Load every task ONCE, before any model load; freeze the item manifest.
        rows_by_task = {}
        for t in args.tasks:
            meta = {}
            n = args.limit or DEFAULT_N["global_mmlu" if t.startswith("global_mmlu_") else t]
            rows_by_task[t] = load_rows(t, n, args.seed, lock, meta)
            if len(rows_by_task[t]) == 0:
                raise ValueError(f"task {t} produced 0 rows")
            meta["kind"] = "mcq" if is_mcq(t) else "generative"
            manifest["tasks"][t] = meta
            atomic_write(os.path.join(run_dir, f"items_manifest_{t}.json"),
                         json.dumps(rows_by_task[t], ensure_ascii=False, indent=1))
        atomic_write(os.path.join(run_dir, "manifest.json"), json.dumps(manifest, indent=2))

        import torch
        import transformers
        import datasets as hf_datasets
        from transformers import AutoTokenizer, AutoModelForCausalLM
        manifest["environment"] = {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": hf_datasets.__version__}

        tok = AutoTokenizer.from_pretrained(MODELS["base"][0],
                                            revision=lock["models"]["base"])
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.padding_side = "left"
        manifest["tokenizer"] = {"repo": MODELS["base"][0], "revision": lock["models"]["base"],
                                 **{k: v for k, v in check_tokenizer(tok).items()
                                    if k != "sample_rendered_prompt"}}
        LET = [manifest["tokenizer"]["letter_token_ids"][x] for x in "ABCD"]

        def effective_eos_ids(model):
            """The model's EFFECTIVE stop set: tokenizer EOS plus every id in
            generation_config.eos_token_id (int or list). Trimming must use this
            set - a model whose generation_config stops on extra ids would
            otherwise look truncated on every item."""
            ids = set()
            if tok.eos_token_id is not None:
                ids.add(int(tok.eos_token_id))
            gc_eos = getattr(model.generation_config, "eos_token_id", None)
            if isinstance(gc_eos, int):
                ids.add(gc_eos)
            elif isinstance(gc_eos, (list, tuple)):
                ids.update(int(e) for e in gc_eos)
            return ids

        @torch.no_grad()
        def gen(model, eos_ids, task, rows, items, counters):
            """Greedy generation with the overlong-skip policy (same as MCQ) and
            per-item output token counts + eos_reached, so truncation is visible
            instead of silently degrading extraction."""
            n_new = GEN_TOKENS.get(task, 320)
            pending = []
            for idx, r in enumerate(rows):
                pending.append((idx, r))
            i = 0
            while i < len(pending):
                chunk = pending[i:i + BATCH["gen"]]
                i += BATCH["gen"]
                txt = [tok.apply_chat_template([{"role": "user", "content": build_prompt(task, r)}],
                                               tokenize=False, add_generation_prompt=True)
                       for _, r in chunk]
                enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False)
                lens = enc["attention_mask"].sum(1).tolist()
                keep = [j for j, L in enumerate(lens) if L <= MAX_PROMPT_TOKENS]
                counters["skipped_overlong_prompt"] += len(chunk) - len(keep)
                for j, (idx, r) in enumerate(chunk):
                    if j not in keep:
                        items[idx] = {"id": r["id"], "skipped": "prompt_overlong"}
                if not keep:
                    continue
                Lmax = max(lens[j] for j in keep)
                sub = {k: v[keep][:, -Lmax:].to(model.device) for k, v in enc.items()}
                o = model.generate(**sub, max_new_tokens=n_new, do_sample=False)
                kept_chunk = [c for j, c in enumerate(chunk) if j in keep]
                for j, (idx, r) in enumerate(kept_chunk):
                    gen_ids = o[j][sub["input_ids"].shape[1]:]
                    # real generated length: cut at the first EFFECTIVE EOS (HF
                    # pads the rest of the batch row after an early stop; padding
                    # is not output). eos_reached=False means a true
                    # max_new_tokens finish - i.e. truncation, counted as such.
                    eos_hit = False
                    if eos_ids:
                        is_eos = torch.isin(gen_ids, torch.tensor(list(eos_ids),
                                                                  device=gen_ids.device))
                        hits = is_eos.nonzero()
                        if len(hits):
                            gen_ids = gen_ids[:int(hits[0][0]) + 1]
                            eos_hit = True
                    if tok.pad_token_id is not None:
                        while len(gen_ids) and int(gen_ids[-1]) == int(tok.pad_token_id):
                            gen_ids = gen_ids[:-1]
                    items[idx] = {"id": r["id"],
                                  "response": tok.decode(gen_ids, skip_special_tokens=True),
                                  "n_output_tokens": int(gen_ids.shape[0]),
                                  "eos_reached": eos_hit}
            return items

        @torch.no_grad()
        def mcq(model, task, rows, items, counters):
            """Restricted-MCQ scoring: compares the FIRST generated-position logits
            of the four letter tokens (forced-choice MCQ accuracy, not natural
            assistant answers - see README). The full lm_head row is computed
            anyway, so the top-1 token id and whether it was a letter are also
            recorded: that shows when the model's actual first token is NOT one of
            the four letters (contextual-tokenization risk made visible).
            Overlong prompts are SKIPPED and counted, never silently truncated."""
            for i in range(0, len(rows), BATCH["mcq"]):
                batch = rows[i:i + BATCH["mcq"]]
                txt = [tok.apply_chat_template([{"role": "user", "content": build_prompt(task, r)}],
                                               tokenize=False, add_generation_prompt=True)
                       for r in batch]
                enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False)
                lens = enc["attention_mask"].sum(1).tolist()
                keep = [j for j, L in enumerate(lens) if L <= MAX_PROMPT_TOKENS]
                counters["skipped_overlong_prompt"] += len(batch) - len(keep)
                for j, r in enumerate(batch):
                    if j not in keep:
                        items.append({"id": r["id"], "skipped": "prompt_overlong"})
                if not keep:
                    continue
                Lmax = max(lens[j] for j in keep)
                sub = {k: v[keep][:, -Lmax:].to(model.device) for k, v in enc.items()}
                # CANONICAL path: the model's own forward, logits trimmed to the
                # final position by the architecture itself (logits_to_keep=1).
                # A previous reimplemented lm_head path twice failed the
                # full-vs-fast equality guard on GPU; the exact diff was never
                # captured, and fp16 GEMM-shape numerics are only a HYPOTHESIS
                # for it. Rather than re-patch a redundant shortcut, scoring
                # uses the canonical forward directly. batch x 1 x vocab fp16
                # is ~2 MB.
                full_lg = model(**sub, logits_to_keep=1).logits[:, -1]
                if not torch.isfinite(full_lg).all():
                    raise RuntimeError("non-finite logits from the model's own "
                                       "forward pass; refusing to score")
                top_ids = full_lg.argmax(-1).cpu()
                lg = full_lg[:, LET].float().cpu()
                for row_idx, r in zip(range(lg.shape[0]), [batch[j] for j in keep]):
                    scores = lg[row_idx].tolist()
                    pred = max(range(4), key=lambda k: scores[k])
                    tid = int(top_ids[row_idx])
                    items.append({"id": r["id"], "correct": int(pred == r["answer"]),
                                  "chosen": "ABCD"[pred], "option_scores": scores,
                                  "top_token_id": tid, "top_is_letter": tid in LET})
            return items

        per_model_correct = {}
        for name in args.models:
            repo, sub = MODELS[name]
            kw = {"subfolder": sub} if sub else {}
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float16, device_map="cuda",
                revision=lock["models"][name], **kw).eval()
            eos_ids = effective_eos_ids(model)
            manifest["models"][name]["effective_eos_ids"] = sorted(eos_ids)
            per_model_correct[name] = {}
            for task in args.tasks:
                t0 = time.time()
                rows = rows_by_task[task]
                counters = {"skipped_overlong_prompt": 0, "extraction_failures": 0,
                            "extraction_fallbacks": 0, "empty_generations": 0,
                            "truncated_no_eos": 0}
                if is_mcq(task):
                    items = mcq(model, task, rows, [], counters)
                else:
                    items = gen(model, eos_ids, task, rows, [None] * len(rows), counters)
                    assert all(it is not None for it in items), "every row must produce an item"
                    assert len(items) == len(rows)
                    rows_by_id = {r["id"]: r for r in rows}
                    resp_dir = os.path.join(run_dir, "responses")
                    os.makedirs(resp_dir, exist_ok=True)
                    atomic_write(os.path.join(resp_dir, f"{task}__{name}.jsonl"),
                                 "".join(json.dumps({"id": it["id"], "response": it.get("response", "")},
                                                    ensure_ascii=False) + "\n" for it in items))
                    for it in items:
                        if "skipped" in it:
                            continue
                        o = it.pop("response")
                        if not o.strip():
                            counters["empty_generations"] += 1
                        if not it["eos_reached"]:
                            counters["truncated_no_eos"] += 1
                        r = rows_by_id[it["id"]]
                        if task == "gsm8k":
                            d = tasks.gsm8k_score_detail(r, o)
                            if d["route"] == "none":
                                counters["extraction_failures"] += 1
                            elif d["route"] == "fallback":
                                counters["extraction_fallbacks"] += 1
                            it["correct"] = d["correct"]
                            it["extracted"] = None if d["extracted"] is None else float(d["extracted"])
                            it["route"] = d["route"]
                        elif task == "ifeval_custom":
                            it["correct"] = tasks.ifeval_score(r, o)
                        elif task == "humaneval":
                            # execution is a SEPARATE isolated phase (coding_score.py);
                            # generation-only here: no correctness is claimed inline
                            it["execution_pending"] = True
                        else:
                            raise ValueError(task)
                assert len(items) == len(rows), "scored item count must equal row count"
                atomic_write(os.path.join(run_dir, f"items_{task}__{name}.jsonl"),
                             "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
                scored = [it["correct"] for it in items if "correct" in it]
                if task == "humaneval":
                    # the pass@1 cell is written by coding_score.py after isolated
                    # execution; the generation phase only proves item coverage
                    log(run_dir, f"{task}__{name}",
                        {"state": "execution_pending", "n_generated": len(items),
                         "coding_endpoint": "standard_route_pinned_upstream_offline_validated", **counters})
                else:
                    rate, lo, hi, nsc = scoring.rate_with_ci(scored)  # raises if all skipped
                    log(run_dir, f"{task}__{name}",
                        {"acc": round(rate, 4), "wilson95": [round(lo, 4), round(hi, 4)],
                         "n_scored": nsc, **counters})
                log(run_dir, f"{task}__{name}_sec", round(time.time() - t0, 1))
                manifest["tasks"][task].setdefault("counters", {})[name] = counters
                per_model_correct[name][task] = {it["id"]: it["correct"]
                                                 for it in items if "correct" in it}
            del model
            torch.cuda.empty_cache()

        # 2. Paired analysis: every non-base model vs base, candidate MINUS base,
        #    on the intersection of items both models scored (stable ids).
        if len(args.models) > 1:
            paired = {}
            for task in args.tasks:
                if task == "humaneval":
                    continue  # no inline correctness; paired analysis belongs to coding_score.py
                base_c = per_model_correct["base"][task]
                for name in args.models:
                    if name == "base":
                        continue
                    cand_c = per_model_correct[name][task]
                    common = [i for i in base_c if i in cand_c]
                    if not common:
                        raise ValueError(f"{task}:{name} vs base share no scored items")
                    r = scoring.mcnemar([cand_c[i] for i in common], [base_c[i] for i in common])
                    r["role"] = "primary" if args.primary == f"{task}:{name}" else "exploratory"
                    paired[f"{task}:{name}_minus_base"] = r
            atomic_write(os.path.join(run_dir, "paired_vs_base.json"), json.dumps(paired, indent=2))
            log(run_dir, "paired_vs_base", paired)
        else:
            manifest["paired_analysis"] = "skipped: single-model run"
        atomic_write(os.path.join(run_dir, "manifest.json"), json.dumps(manifest, indent=2))
        atomic_write(os.path.join(run_dir, "COMPLETE"), "ok\n")
        log(run_dir, "DONE", True)
    except Exception as e:
        # flush whatever metadata exists so a failed run is still auditable
        try:
            atomic_write(os.path.join(run_dir, "manifest.json"), json.dumps(manifest, indent=2))
        finally:
            atomic_write(os.path.join(run_dir, "FAILED"), f"{type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
