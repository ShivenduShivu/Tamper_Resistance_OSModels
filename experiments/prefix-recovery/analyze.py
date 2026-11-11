import json, sys, random
arms = sys.argv[1:]; D = {a: json.load(open(f"/tmp/{a}.json")) for a in arms}
R = random.Random(0)
def rate(v): return sum(v) / len(v)
def boot(x, y, n=4000):
    """paired bootstrap over prompts of mean(y)-mean(x); returns (diff, lo, hi, n)"""
    k = len(x); ds = []
    for _ in range(n):
        idx = [R.randrange(k) for _ in range(k)]; ds.append(sum(y[i] - x[i] for i in idx) / k)
    ds.sort(); return round(rate(y) - rate(x), 3), round(ds[int(.025 * n)], 3), round(ds[int(.975 * n)], 3), k
def metrics(d):
    m = {"clean_harm_kw": d["clean_harm"]["kw"], "clean_harm_clf": d["clean_harm"]["clf"], "jbb_kw": d["clean_jbb"]["kw"], "jbb_clf": d["clean_jbb"]["clf"],
         "sure_here_is_kw": d["prefill_sure_here_is"]["kw"], "sure_here_is_clf": d["prefill_sure_here_is"]["clf"], "xstest_over_kw": d["xstest_overrefusal"]["kw"], "xstest_over_clf": d["xstest_overrefusal"]["clf"], "mmlu_acc": d["mmlu_correct"]}
    ops = list(d["prefill_heldout"]); 
    for t in ("kw", "clf"):   # pooled held-out openers: prompt x opener cells, paired
        m[f"heldout_pooled_{t}"] = [x for o in ops for x in d["prefill_heldout"][o][t]]
    return m
M = {a: metrics(D[a]) for a in arms}
print("RATES (n):"); 
for k in M[arms[0]]: print(f"  {k:24s}", {a: (round(rate(M[a][k]), 3), len(M[a][k])) for a in arms})
print("PAIRED DIFFS (second minus first), 95% paired bootstrap over prompts; held-out pooled cells are NOT independent (6 openers per prompt), treated as exploratory:")
pairs = [("B", "C")] + [("A", "B"), ("A", "C")] if "A" in arms else [("B", "C")]
for x, y in pairs:
    print(f" {y} - {x}")
    for k in M[x]: print(f"   {k:24s}", boot(M[x][k], M[y][k]))
