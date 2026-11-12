"""Article figures (one consistent navy theme). Run from repo root: python scripts/make_figures.py
Every number is read from results/ or copied from notes/LOG.md (final single-BOS values); CIs are Wilson 95%."""
import json, glob, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "docs/figures"; os.makedirs(OUT, exist_ok=True)
NAVY, PANEL, INK, MUTED, GRID = "#0B1426", "#111D35", "#F2F5FA", "#9AA6BC", "#22304D"
BLUE, GREEN, RED, GREY, AMBER = "#4F8BFF", "#36D399", "#FF5A5F", "#8A94A6", "#F5B942"
COL = {"Original": RED, "v1": GREY, "v2": GREEN, "v2b": BLUE}
plt.rcParams.update({"figure.facecolor": NAVY, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 11, "axes.titleweight": "bold",
                     "axes.titlesize": 12.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)

def save(fig, name, note=None):
    if note: fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5, ha="left", va="bottom")
    fig.savefig(f"{OUT}/{name}.png", dpi=170, facecolor=NAVY); plt.close(fig); print("wrote", name)

def bars(ax, labels, vals, colors, ci=None, fmt="{:.0f}%", ylim=118, scale=100):
    xs = range(len(vals))
    err = None
    if ci:
        err = [[(v - lo) * scale for v, (lo, hi) in zip(vals, ci)], [(hi - v) * scale for v, (lo, hi) in zip(vals, ci)]]
    ax.bar(xs, [v * scale for v in vals], color=colors, width=0.62, yerr=err, ecolor=INK, capsize=5, error_kw={"elinewidth": 1.2})
    for x, v, e in zip(xs, vals, (err[1] if err else [0] * len(vals))):
        ax.text(x, v * scale + (e if err else 0) + ylim * 0.02, fmt.format(v * scale), ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, color=INK); ax.set_ylim(0, ylim); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

G = {json.loads(l)["file"]: json.loads(l) for l in open("results/guard_judge_day6/summary.jsonl", encoding="utf-8")}
def g(k): d = G["day6__" + k]; return d["unsafe"], tuple(d["unsafe_ci95"])

# ---------- 1. HERO
fig, axs = plt.subplots(1, 3, figsize=(13, 5.3)); fig.subplots_adjust(top=0.72, bottom=0.12, wspace=0.34)
fig.suptitle("One free command strips Tiny Aya's safety.  Our v2 holds.", fontsize=19, fontweight="bold", color=INK, y=0.97)
fig.text(0.5, 0.875, "Heretic (automated abliteration) vs Cohere's Tiny Aya Global and our hardened v2", ha="center", color=MUTED, fontsize=11.5)
bars(axs[0], ["Original", "v2"], [0.14, 1.00], [RED, GREEN]); axs[0].set_title("Refusal kept, cheap attacks\n(KL ≤ 0.05, keyword score)", color=INK)
u0, c0 = g("base_heretic_r14_harm"); u2, c2 = g("v2_heretic_r33_harm")
bars(axs[1], ["Original", "v2"], [u0, u2], [RED, GREEN], ci=[c0, c2]); axs[1].set_title("Harmful answers, strongest attack\n(Granite judge, 95% CI)", color=INK)
axs[2].barh([1, 0], [0.0094, 0.153], color=[RED, GREEN], height=0.55); axs[2].set_yticks([1, 0]); axs[2].set_yticklabels(["Original", "v2"], color=INK)
axs[2].set_xlim(0, 0.19); axs[2].grid(axis="x", color=GRID, lw=0.6); axs[2].set_axisbelow(True)
axs[2].text(0.0094 + 0.004, 1, "KL 0.009", va="center", fontweight="bold"); axs[2].text(0.153 + 0.004, 0, "KL 0.153  (~16×)", va="center", fontweight="bold")
axs[2].set_title("Damage needed for that attack\n(KL divergence)", color=INK)
save(fig, "hero", "n = 100 harmful prompts (mlabonne/harmful_behaviors test). Lower harmful = safer; higher KL = the attack visibly damages the model.")

# ---------- 2. VULNERABILITY (refusal and harm in separate panels)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.8)); fig.subplots_adjust(top=0.78, bottom=0.13, wspace=0.32)
fig.suptitle("The threat: Heretic strips the original Tiny Aya almost completely, at no visible cost", fontsize=14.5, fontweight="bold", y=0.97)
bars(axs[0], ["Before", "After Heretic"], [0.98, 0.01], [BLUE, RED]); axs[0].set_title("Refuses harmful requests\n(ProtectAI refusal judge)", color=INK)
uc, cc = g("base_clean_harm")
bars(axs[1], ["Before", "After Heretic"], [uc, u0], [BLUE, RED], ci=[cc, c0]); axs[1].set_title("Answers judged harmful\n(Granite Guardian, 95% CI)", color=INK)
bars(axs[2], ["Before", "After Heretic"], [0.523, 0.520], [BLUE, RED], fmt="{:.1f}%", ylim=72); axs[2].set_title("General knowledge (MMLU)\nunchanged: nobody notices", color=INK)
save(fig, "vulnerability", "Heretic's best attack on the original: KL 0.009, about 1 GPU-hour on a free T4.")

# ---------- 3. PARETO (with 200-trial v2 search)
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
optuna.logging.set_verbosity(optuna.logging.ERROR)
def front(path):
    st = JournalStorage(JournalFileBackend(glob.glob(path)[0])); s = optuna.load_study(study_name=st.get_all_studies()[0].study_name, storage=st)
    pts = sorted([(t.values[1], t.values[0]) for t in s.trials if t.values]); f, best = [], 9
    for kl, r in pts:
        if r < best: f.append((kl, r)); best = r
    return f
FR = {"Original": front("results/day2_baseline/heretic_base/ckpt/*.jsonl"), "v1": front("results/day2/heretic_v1/ckpt/*.jsonl"),
      "v2": front("results/day3/heretic_v2/ckpt/*.jsonl"), "v2b": front("results/day3b/heretic_v2b/ckpt/*.jsonl")}
F200 = front("results/heretic200_v2/ckpt/*.jsonl")
fig, ax = plt.subplots(figsize=(11.5, 5.6)); fig.subplots_adjust(top=0.86, bottom=0.2, left=0.08, right=0.97)
ax.axvspan(0, 0.05, color=GREEN, alpha=0.07); ax.text(0.025, 4, "cheap, near-invisible\nattacks (KL ≤ 0.05)", ha="center", color=GREEN, fontsize=9.5)
names = {"Original": "Original Tiny Aya", "v1": "v1: extended refusals", "v2": "v2: trained through the attack", "v2b": "v2b: + borderline-safe data"}
for k, f in FR.items():
    ax.step([0] + [kl for kl, _ in f], [100] + [r * 100 for _, r in f], where="post", color=COL[k], lw=2.6 if k == "v2" else 1.8, label=names[k])
    ax.scatter([kl for kl, _ in f], [r * 100 for _, r in f], color=COL[k], s=18, zorder=3)
ax.step([0] + [kl for kl, _ in F200], [100] + [r * 100 for _, r in F200], where="post", color=GREEN, lw=1.6, ls="--", label="v2, 5× longer search (200 trials)")
ax.set_xlim(-0.003, 0.17); ax.set_ylim(0, 104); ax.grid(color=GRID, lw=0.6)
ax.set_xlabel("Damage the attack does to normal behaviour (KL divergence)  →"); ax.set_ylabel("Harmful prompts still refused (%)")
fig.suptitle("Every attack Heretic found: higher curve = harder to strip", fontsize=14.5, fontweight="bold", y=0.97)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False, fontsize=9, labelcolor=INK)
save(fig, "pareto", "Each point = best attack at that damage level (Heretic's keyword refusal score; fresh search per model).")

# ---------- 4. 40 vs 200 trials
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]}); fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.28)
fig.suptitle("A 5× longer search finds cheaper partial attacks, but no stronger worst case", fontsize=14.5, fontweight="bold", y=0.97)
import numpy as np
x = np.arange(3); w = 0.36
for i, (lab, vals, col) in enumerate([("40 trials", [1.00, 1.00, 0.96], GREEN), ("200 trials", [1.00, 1.00, 0.79], BLUE)]):
    axs[0].bar(x + (i - 0.5) * w, [v * 100 for v in vals], w, color=col, label=lab)
    for xi, v in zip(x, vals): axs[0].text(xi + (i - 0.5) * w, v * 100 + 2, f"{v*100:.0f}%", ha="center", fontsize=10.5, fontweight="bold")
axs[0].set_xticks(x); axs[0].set_xticklabels(["KL ≤ 0.01", "KL ≤ 0.05", "KL ≤ 0.1"], color=INK); axs[0].set_ylim(0, 115)
axs[0].set_title("Lowest refusal Heretic reached on v2 (keyword)", color=INK); axs[0].legend(frameon=False, labelcolor=INK, loc="lower left")
axs[0].grid(axis="y", color=GRID, lw=0.6); axs[0].set_axisbelow(True)
bars(axs[1], ["40 trials", "200 trials"], [0.40, 0.42], [GREEN, BLUE], ci=[(0.309, 0.498), wilson(0.42, 100)])
axs[1].set_title("Worst case: harmful answers\n(Granite, 95% CI; overlapping)", color=INK)
save(fig, "search_budget", "One search per budget: shows no stronger worst case was found, not that none exists.")

# ---------- 5. VALIDATION DASHBOARD (harm, Granite)
rows = [("Heretic\n(strongest)", "base_heretic_r14_harm", "v2_heretic_r33_harm", "v2b_heretic_r47_harm"),
        ("SafeTune\ndirection ablation", "base_st_ablate_harm", "v2_st_ablate_harm", "v2b_st_ablate_harm"),
        ("SafeTune\nCAA steering −1", "base_st_caa1_harm", "v2_st_caa1_harm", "v2b_st_caa1_harm"),
        ("Prefill\n\"Sure, here is\"", "base_prefill_harm", "v2_prefill_harm", "v2b_prefill_harm")]
fig, ax = plt.subplots(figsize=(12.5, 5)); fig.subplots_adjust(top=0.84, bottom=0.22)
x = np.arange(len(rows)); w = 0.26
for i, (lab, col) in enumerate([("Original", RED), ("v2", GREEN), ("v2b", BLUE)]):
    vals = [g(r[i + 1]) for r in rows]
    ys = [v * 100 for v, _ in vals]; lo = [(v - c[0]) * 100 for v, c in vals]; hi = [(c[1] - v) * 100 for v, c in vals]
    ax.bar(x + (i - 1) * w, ys, w, color=col, label={"Original": "Original Tiny Aya"}.get(lab, lab), yerr=[lo, hi], ecolor=INK, capsize=3)
    for xi, y, h in zip(x, ys, hi): ax.text(xi + (i - 1) * w, y + h + 2, f"{y:.0f}", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], color=INK); ax.set_ylim(0, 120); ax.set_ylabel("Answers judged harmful (%)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.axvspan(-0.5, 1.5, color=GREEN, alpha=0.06); ax.text(0.5, 112,"direction-removal attacks: resisted", ha="center", color=GREEN, fontsize=10)
ax.axvspan(1.5, 3.5, color=RED, alpha=0.06); ax.text(2.5, 112,"steering and prefill: not resisted", ha="center", color=RED, fontsize=10)
fig.suptitle("Independent validation: two attack tools, one harm judge (lower = safer)", fontsize=14.5, fontweight="bold", y=0.97)
save(fig, "validation", "Granite Guardian 3.2-5B harm judge, 95% Wilson CIs, n = 100 harmful prompts per bar. Final single-BOS evaluation.")

# ---------- 6. MULTILINGUAL
M = [json.loads(l) for l in open("results/multilingual/summary.jsonl", encoding="utf-8")]
def m(cond, lang): r = [d for d in M if d["condition"] == cond and d["lang"] == lang][0]; return r["unsafe_mt"], tuple(r["unsafe_mt_ci95"])
langs = [("eng", "English"), ("hin", "Hindi"), ("arb", "Arabic"), ("fra", "French"), ("spa", "Spanish")]
fig, ax = plt.subplots(figsize=(12.5, 4.8)); fig.subplots_adjust(top=0.84, bottom=0.22)
x = np.arange(len(langs)); w = 0.26
for i, (cond, lab, col) in enumerate([("base_heretic", "Original Tiny Aya", RED), ("v2_heretic", "v2", GREEN), ("v2b_heretic", "v2b", BLUE)]):
    vals = [m(cond, l) for l, _ in langs]; ys = [v * 100 for v, _ in vals]
    ax.bar(x + (i - 1) * w, ys, w, color=col, label=lab, yerr=[[(v - c[0]) * 100 for v, c in vals], [(c[1] - v) * 100 for v, c in vals]], ecolor=INK, capsize=3)
    for xi, (v, c) in zip(x, vals): ax.text(xi + (i - 1) * w, c[1] * 100 + 1.5, f"{v*100:.0f}", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([n for _, n in langs], color=INK); ax.set_ylim(0, 75); ax.set_ylabel("Answers judged harmful (%)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Trained only in English, v2's protection carries over to four more languages", fontsize=14.5, fontweight="bold", y=0.97)
save(fig, "multilingual", "CohereLabs/aya_redteaming, 100 prompts per language (different prompts per language), each model's strongest Heretic attack. Granite on NLLB translations, 95% CI.")

# ---------- 7. SAFETY vs USABILITY
fig, ax = plt.subplots(figsize=(9.5, 5.4)); fig.subplots_adjust(top=0.86, bottom=0.16, left=0.11)
pts = {"Original": (16.4, 91, RED), "v2": (39.2, 40, GREEN), "v2b": (18.4, 57, BLUE)}
for k, (xo, yh, col) in pts.items():
    ax.scatter(xo, yh, s=420, color=col, zorder=3, edgecolor=INK, lw=1.2); ax.annotate(k, (xo, yh), xytext=(12, 8), textcoords="offset points", fontsize=13, fontweight="bold")
ax.annotate("", xy=(18.4, 57), xytext=(39.2, 40), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
ax.text(29, 52, "v2b: less cautious,\nless robust", color=MUTED, fontsize=9.5, ha="center")
ax.set_xlim(0, 50); ax.set_ylim(0, 100); ax.grid(color=GRID, lw=0.6)
ax.set_xlabel("Over-refusal: safe questions refused (XSTest, %)  →  less useful")
ax.set_ylabel("Harmful answers at strongest attack (%)  →  less safe")
ax.text(2, 4, "ideal corner", color=GREEN, fontsize=10)
fig.suptitle("Safety vs usability: two useful points, one trade-off", fontsize=14.5, fontweight="bold", y=0.97)
save(fig, "tradeoff", "MMLU is unchanged across all three (52.3% / 53.0% / 52.7%).")

# ---------- 8. PREFILL PILOT (separate)
fig, axs = plt.subplots(1, 2, figsize=(12, 5)); fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.28)
fig.suptitle("PILOT (separate setup): training refusals after openers", fontsize=14.5, fontweight="bold", y=0.97, color=AMBER)
fig.text(0.5, 0.86, "one seed per arm  ·  answers judged at 48 tokens (main results: 128)  ·  not re-tested against Heretic", ha="center", color=MUTED, fontsize=10.5)
bars(axs[0], ["No openers\n(control)", "Openers on\nharmful only", "Openers\non all"], [0.86, 0.0, 0.0], [GREY, GREEN, BLUE])
axs[0].set_title("Harmful answers under \"Sure, here is\" (Granite)", color=INK)
bars(axs[1], ["No openers\n(control)", "Openers on\nharmful only", "Openers\non all"], [0.384, 0.388, 0.46], [GREY, GREEN, BLUE], ylim=60)
axs[1].set_title("Safe questions refused in normal use (keyword)", color=INK)
save(fig, "prefill_pilot", "Adapted from DeepRefusal (arXiv 2509.15202) with generic openers only; matched control = same training without openers.")
