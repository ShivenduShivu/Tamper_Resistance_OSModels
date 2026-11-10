"""Parse Heretic optuna journals -> Pareto fronts (refusal rate, KL). Usage: python scripts/pareto.py
values[0] = refusals/100 on harmful_behaviors test[:100]; values[1] = KL on harmless_alpaca test[:100]."""
import json, glob, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
optuna.logging.set_verbosity(optuna.logging.ERROR)
RUNS = {"base": "results/day2_baseline/heretic_base/ckpt/*.jsonl",
        "v1": "results/day2/heretic_v1/ckpt/*.jsonl",
        "v2": "results/day3/heretic_v2/ckpt/*.jsonl"}
def front(pts):
    f, best = [], 9
    for r, kl in sorted(pts, key=lambda p: (p[1], p[0])):
        if r < best: f.append((r, kl)); best = r
    return f
out = {}
for tag, pat in RUNS.items():
    st = JournalStorage(JournalFileBackend(glob.glob(pat)[0]))
    s = optuna.load_study(study_name=st.get_all_studies()[0].study_name, storage=st)
    pts = [(t.values[0], t.values[1]) for t in s.trials if t.values]
    f = front(pts)
    def min_r(k): return min([r for r, kl in pts if kl <= k], default=None)
    out[tag] = {"front": f, "min_refusal_at_kl<=0.01": min_r(0.01), "<=0.05": min_r(0.05),
                "<=0.1": min_r(0.1), "<=0.5": min_r(0.5), "min_any": min(r for r, _ in pts)}
    print(tag, {k: v for k, v in out[tag].items() if k != "front"})
    print("   front:", [(round(r, 2), round(kl, 3)) for r, kl in f])
json.dump(out, open("results/pareto_summary.json", "w"), indent=1)
