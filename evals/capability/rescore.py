"""POST-HOC offline rescoring of PRESERVED raw generations (markup-aware).

HONESTY CONTRACT:
- This module never modifies the frozen run parser (tasks.py) and never
  overwrites original run artifacts. It reads saved responses/ + items/ and
  writes NEW files labeled post-hoc, with original scores kept side by side.
- The markup-aware parser only recognizes an explicit final-answer marker
  ("answer:" with optional Markdown emphasis/currency wrappers). It NEVER
  loosens to grabbing the last number in the text: when a marker is present
  but no parseable number follows it, the route is "none" (unfinished
  marker), exactly as strict as the frozen parser's no-fallback rule.
- The numeric grammar is reused verbatim from tasks._to_number (whole-line,
  exact Fraction, optional $ prefix, optional /denominator, no exponents,
  no prefixes/suffixes/units).
- Routes stay distinguishable from the frozen parser's: "marker" (plain),
  "marker_markup" (emphasis/escaped-currency variant), "fallback"
  (unchanged original last-number behavior), "none".
"""
import json
import re

import tasks

# answer label with optional whitespace/underscore/asterisk emphasis around
# the colon: "Answer:", "**Answer:**", "**Answer**:", "_answer_ :" etc.
# ONE unified regex for plain and emphasized forms, so the LAST marker in the
# text wins regardless of which form it takes (the frozen parser only saw the
# plain form and could pick an earlier plain marker over a later bold one).
# after the colon: whitespace, then optionally emphasis FOLLOWED BY
# whitespace (the label's closing delimiter, e.g. "**Answer:** 20"). Emphasis
# attached directly to the value ("**Answer:** **20**") is left for
# _clean_number_line's balanced-wrapper stripper - consuming it here would
# break balance detection.
_MARKUP_MARKER_RE = re.compile(r"answer[\s*_]*:\s*(?:[*_]+\s)?", re.IGNORECASE)


def _clean_number_line(line):
    """Strip BALANCED emphasis wrappers around the whole value plus an
    optional (escaped) LEADING currency sign. Anything dangling - '42$',
    '42\\', '20 **' - is left in place and will fail the strict numeric
    grammar. Returns (cleaned, had_currency)."""
    s = line.strip()
    had_cur = False
    changed = True
    while changed:
        changed = False
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "*_":
            s = s[1:-1].strip()
            changed = True
        if s.startswith("\\$"):
            s = s[2:].lstrip()
            had_cur = True
            changed = True
        elif s.startswith("$"):
            s = s[1:].lstrip()
            had_cur = True
            changed = True
    return s, had_cur


def markup_extract(text):
    """Returns (value_or_None, route, flags)."""
    if not text:
        return None, "none", []
    ms = list(_MARKUP_MARKER_RE.finditer(text))
    if ms:
        m = ms[-1]  # last marker in the text wins, plain or emphasized
        label = m.group(0)
        rest = text[m.end():]
        line = rest.splitlines()[0] if rest else ""
        markup = ("*" in label) or ("_" in label) or line.lstrip()[:1] in "*_"
        flags = ["markup_label"] if markup else []
        cleaned, had_cur = _clean_number_line(line)
        if had_cur:
            flags.append("currency")
        v = tasks._to_number(cleaned)
        if v is not None:
            return v, "marker_markup" if markup else "marker", flags
        return None, "none", flags + ["unfinished_marker"]
    # no answer marker anywhere: unchanged frozen-protocol fallback
    v = tasks._fallback_number(text)
    return (v, "fallback", []) if v is not None else (None, "none", [])


def rescore_generation(row, generation):
    """row: items_manifest row with 'answer' gold. Returns detail dict."""
    from fractions import Fraction
    value, route, flags = markup_extract(generation)
    out = None if value is None else float(value)  # mirror run.py storage
    try:
        gold = Fraction(str(row["answer"]).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return {"id": row.get("id"), "correct": 0, "extracted": out,
                "route": route, "flags": flags}
    return {"id": row.get("id"),
            "correct": int(value is not None and value == gold),
            "extracted": out, "route": route, "flags": flags}


def rescore_run_dir(run_dir, task="gsm8k", out_dir=None, semantic_no_final=None):
    """Rescore every model's preserved responses for one generative task.
    Writes rescore_<task>__<model>.jsonl + rescore_report.json into out_dir
    (default: <run_dir>/rescore/). Originals are never modified.

    semantic_no_final: optional {model: [item_ids]} of rows an independent
    human reviewer adjudicated as having no explicit final answer even though
    a fallback value was parsed. Embedded under a source-labeled key; the
    tool never derives it - fallback is a last-number heuristic, not an
    answer, and route='none' counts are reported as unparsed_revised."""
    import os
    out_dir = out_dir or os.path.join(run_dir, "rescore")
    os.makedirs(out_dir, exist_ok=True)
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"),
                              encoding="utf-8"))
    models = list(manifest["argv"]["models"])
    gold = {r["id"]: r for r in json.load(open(
        os.path.join(run_dir, "items_manifest_%s.json" % task), encoding="utf-8"))}
    report = {"task": task, "post_hoc": True,
              "note": "markup-aware rescore; original scores retained side by "
                      "side; originals unchanged", "models": {}}
    for m in models:
        items = [json.loads(l) for l in open(
            os.path.join(run_dir, "items_%s__%s.jsonl" % (task, m)),
            encoding="utf-8") if l.strip()]
        responses = {json.loads(l)["id"]: json.loads(l)["response"] for l in open(
            os.path.join(run_dir, "responses", "%s__%s.jsonl" % (task, m)),
            encoding="utf-8") if l.strip()}
        revised = []
        for it in items:
            d = rescore_generation(gold[it["id"]], responses[it["id"]])
            flags = list(d["flags"])
            if not it.get("eos_reached", True):
                flags.append("no_eos")
            revised.append({
                "id": it["id"],
                "original_correct": it.get("correct"),
                "original_route": it.get("route"),
                "original_extracted": it.get("extracted"),
                "revised_correct": d["correct"],
                "revised_route": d["route"],
                "revised_extracted": d["extracted"],
                "eos_reached": it.get("eos_reached"),
                "n_output_tokens": it.get("n_output_tokens"),
                "flags": flags,
            })
        with open(os.path.join(out_dir, "rescore_%s__%s.jsonl" % (task, m)),
                  "w", encoding="utf-8") as f:
            for r in revised:
                f.write(json.dumps(r) + "\n")
        n = len(revised)
        report["models"][m] = {
            "n": n,
            "original_correct": sum(r["original_correct"] for r in revised),
            "revised_correct": sum(r["revised_correct"] for r in revised),
            "original_routes": _counts(r["original_route"] for r in revised),
            "revised_routes": _counts(r["revised_route"] for r in revised),
            "format_flags": _counts(f for r in revised for f in r["flags"]),
            # route=='none' = UNPARSED, not 'no final answer': a parsed
            # fallback value is not evidence an actual final exists
            "unparsed_revised": sum(1 for r in revised
                                    if r["revised_route"] == "none"),
            "truncated_no_eos": sum(1 for r in revised if not r["eos_reached"]),
            "changed": [r["id"] for r in revised
                        if r["original_correct"] != r["revised_correct"]],
        }
        if semantic_no_final and m in semantic_no_final:
            report["models"][m]["semantic_no_final_reviewer_adjudicated"] = {
                "ids": list(semantic_no_final[m]),
                "source": "independent human review of preserved raws",
                "note": "no explicit final answer despite a parsed value",
            }
    with open(os.path.join(out_dir, "rescore_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


def _counts(it):
    c = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return c


if __name__ == "__main__":
    import sys
    rep = rescore_run_dir(sys.argv[1])
    print(json.dumps(rep, indent=1))
