import json, sys, random
# Benign-opener control analysis: WITHIN-pool paired comparisons only (250 XSTest-safe prompts).
# Paired across arms (same prompt set per arm) and across conditions within an arm (same prompts).
arms = sys.argv[1:]
D = {a: json.load(open(f"/tmp/{a}_benign.json")) for a in arms}
R = random.Random(0)
rate = lambda v: sum(v) / len(v)
OPS = list(D[arms[0]]["per_opener"])
NP = len(D[arms[0]]["no_opener"]["kw"])

def boot_pair(x, y, n=4000):
    """x,y: per-prompt lists of equal-length condition vectors. Resample prompts; each keeps all its conditions."""
    k = len(x); per = [sum(b) / len(b) - sum(a) / len(a) for a, b in zip(x, y)]; ds = []
    for _ in range(n): ds.append(sum(per[R.randrange(k)] for _ in range(k)) / k)
    ds.sort(); return round(sum(per) / k, 3), round(ds[int(.025 * n)], 3), round(ds[int(.975 * n)], 3), k

def pooled(d, t):  # per prompt: vector over 6 openers
    return [[d["per_opener"][o][t][i] for o in OPS] for i in range(NP)]
def noop(d, t): return [[v] for v in d["no_opener"][t]]

print("benign pool: n =", NP, "prompts; conditions: no-opener +", len(OPS), "held-out openers; max_new_tokens 48")
for t in ("kw", "clf"):
    print(f"== scorer {t} rates (benign prompts; refusal-like = potential over-refusal)")
    for a in arms:
        d = D[a]
        print(" ", a, "no_opener", round(rate(d["no_opener"][t]), 3),
              "heldout_pooled", round(sum(map(sum, pooled(d, t))) / (len(OPS) * NP), 3),
              "per_opener", [sum(d["per_opener"][o][t]) for o in OPS])
    pairs = [(x, y) for x in arms for y in arms if arms.index(x) < arms.index(y)]
    for x, y in pairs:
        print(f"== {y} - {x} ({t}), paired by prompt over the SAME benign pool")
        print(f"   {'no_opener':16s}", boot_pair(noop(D[x], t), noop(D[y], t)))
        print(f"   {'heldout_6pool':16s}", boot_pair(pooled(D[x], t), pooled(D[y], t)), f"({NP} prompt clusters x {len(OPS)} openers)")
        for j, o in enumerate(OPS):
            print(f"   opener{j+1} paired (y>x)/(x>y):", sum(1 for a, b in zip(D[x]["per_opener"][o][t], D[y]["per_opener"][o][t]) if b > a), sum(1 for a, b in zip(D[x]["per_opener"][o][t], D[y]["per_opener"][o][t]) if a > b))
    for a in arms:  # within-arm: opener minus no-opener on the same prompts
        d = D[a]
        print(f"== {a}: heldout opener pool - no_opener ({t}), same prompts",
              boot_pair(noop(d, t), pooled(d, t)))
