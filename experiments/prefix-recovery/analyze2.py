import json, sys, random
arms = sys.argv[1:]; D = {a: json.load(open(f"/tmp/{a}.json")) for a in arms}; R = random.Random(0)
rate = lambda v: sum(v) / len(v)
def boot_cluster(x, y, n=4000):
    """x,y: list over prompts of list over conditions (paired). Resample PROMPTS; each keeps all its conditions."""
    k = len(x); per = [sum(b) / len(b) - sum(a) / len(a) for a, b in zip(x, y)]; ds = []
    for _ in range(n): ds.append(sum(per[R.randrange(k)] for _ in range(k)) / k)
    ds.sort(); return round(sum(per) / k, 3), round(ds[int(.025 * n)], 3), round(ds[int(.975 * n)], 3), k
def single(d, t): return {"clean_harm": d["clean_harm"][t], "jbb": d["clean_jbb"][t], "sure_here_is": d["prefill_sure_here_is"][t], "xstest_over": d["xstest_overrefusal"][t], "mmlu": d["mmlu_correct"]}
OPS = list(D[arms[0]]["prefill_heldout"])
def pooled(d, t): return [[d["prefill_heldout"][o][t][i] for o in OPS] for i in range(len(d["prefill_heldout"][OPS[0]][t]))]  # per prompt: 6 conditions
print("n_unique_prompts_heldout:", len(pooled(D[arms[0]], "kw")), "conditions:", len(OPS))
for t in ("kw", "clf"):
    print(f"== scorer {t} rates"); 
    for a in arms:
        s = single(D[a], t); print(" ", a, {k: (round(rate(v), 3), len(v)) for k, v in s.items()}, "heldout_pooled", round(sum(map(sum, pooled(D[a], t))) / (6 * 100), 3))
    print(f"== per-opener {t} counts of refusal-like continuations out of 100 (arm: [per opener]):")
    for a in arms: print("  ", a, [sum(D[a]["prefill_heldout"][o][t]) for o in OPS])
    pairs = [("B", "C")] + ([("A", "B"), ("A", "C")] if "A" in arms else [])
    for x, y in pairs:
        print(f"== {y} - {x} ({t}), prompt-cluster paired bootstrap 95%")
        sx, sy = single(D[x], t), single(D[y], t)
        for k in sx: print(f"   {k:14s}", boot_cluster([[v] for v in sx[k]], [[v] for v in sy[k]]))
        print(f"   {'heldout_6pool':14s}", boot_cluster(pooled(D[x], t), pooled(D[y], t)), "(100 prompt clusters x 6 openers)")
        for j, o in enumerate(OPS): print(f"   opener{j+1} paired b/c(y>x)/(x>y):", sum(1 for a, b in zip(D[x]["prefill_heldout"][o][t], D[y]["prefill_heldout"][o][t]) if b > a), sum(1 for a, b in zip(D[x]["prefill_heldout"][o][t], D[y]["prefill_heldout"][o][t]) if a > b))
