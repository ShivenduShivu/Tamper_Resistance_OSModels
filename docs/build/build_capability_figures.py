# Rebuilds docs/figures/tradeoff.png and docs/figures/capability_cost.png (README section 10).
# Numbers: harm (y) from Granite, Day-6 (single BOS); over-refusal (x) and paired capability changes from
# results/general_audit/clean_results.json (PR #7): ProtectAI classifier on 250 safe XSTest prompts (43 / 107 / 42).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, GRID, TXT, MUTED = "#0b1326", "#111c34", "#22304d", "#f2f5fa", "#9aa6bd"
RED, GREEN, BLUE = "#ff5a5f", "#34d399", "#4f8bff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

def frame(fig, ax):
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- 1. safety vs usability, with the capability cost written on each defended point
fig, ax = plt.subplots(figsize=(16.15, 9.18), dpi=100); frame(fig, ax)
pts = [("Original", 17.2, 91, RED, ""), ("v2", 42.8, 40, GREEN, "IFEval −10"),
       ("v2b", 16.8, 57, BLUE, "IFEval −19.5 · GSM8K −16.5")]
for name, x, y, c, cost in pts:
    ax.scatter(x, y, s=1300, color=c, edgecolor="white", linewidth=2.5, zorder=3)
    ax.annotate(name, (x, y), xytext=(22, 14), textcoords="offset points", fontsize=22, fontweight="bold", zorder=4)
    if cost:
        off = (-60, -52) if name == "v2b" else (22, -24)
        ax.annotate(cost, (x, y), xytext=off, textcoords="offset points", fontsize=15, color="#ffb86b", zorder=4)
ax.annotate("", xy=(17.4, 56.6), xytext=(42.2, 40.4), arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
ax.text(31.5, 36, "v2b: less cautious,\nless robust, larger capability cost", ha="center", fontsize=15, color=MUTED)
ax.text(2, 4, "ideal corner", fontsize=17, color=GREEN)
ax.set_xlim(0, 50); ax.set_ylim(0, 100); ax.tick_params(labelsize=17)
ax.set_xlabel("Over-refusal: safe XSTest questions flagged as refused (%, n = 250)  →  less useful", fontsize=18)
ax.set_ylabel("Harmful answers at strongest attack (%)  →  less safe", fontsize=18)
ax.set_title("Safety vs usability: no free point", fontsize=24, fontweight="bold", pad=24)
fig.text(0.01, 0.015, "Orange = capability change vs original (percentage points, paired 95% CI excludes zero). "
         "No change detected for MMLU, coding or security knowledge.\n"
         "Over-refusal = provisional ProtectAI classifier labels, 128-token answers. Details: section 10.", fontsize=13.5, color=MUTED)
fig.subplots_adjust(left=0.11, right=0.9, top=0.86, bottom=0.16)
fig.savefig("docs/figures/tradeoff.png", facecolor=BG); plt.close(fig)

# ---- 2. capability cost: paired change vs original with 95% intervals
tasks = [("Coding (HumanEval)", (-1.2, -7.3, 4.9), (-1.2, -8.0, 5.6)),
         ("General knowledge (MMLU)", (-0.5, -5.6, 4.6), (0.5, -5.0, 6.0)),
         ("Security knowledge (CyberMetric)", (-1.4, -3.7, 0.8), (-1.8, -4.2, 0.6)),
         ("Maths (GSM8K), standard parser", (-1.0, -7.1, 5.1), (-16.5, -23.0, -9.7)),
         ("Maths (GSM8K), strict format", (-13.0, -20.0, -5.8), (-22.0, -29.0, -14.6)),
         ("Instruction following (IFEval)", (-10.0, -16.7, -3.2), (-19.5, -26.7, -12.0))]
fig, ax = plt.subplots(figsize=(16.15, 8.6), dpi=100); frame(fig, ax); ax.grid(axis="y", visible=False)
ax.axvspan(-35, 0, color="#ff5a5f", alpha=0.05, zorder=0)
ax.axvline(0, color=TXT, lw=1.6, zorder=1)
for i, (label, a, b) in enumerate(tasks):
    for (m, lo, hi), dy, c, nm in [(a, 0.17, GREEN, "v2"), (b, -0.17, BLUE, "v2b")]:
        sig = hi < 0 or lo > 0
        ax.plot([lo, hi], [i + dy] * 2, color=c, lw=3.2, alpha=1 if sig else 0.55, zorder=2, solid_capstyle="round")
        ax.scatter(m, i + dy, s=170, color=c if sig else PANEL, edgecolor=c, linewidth=2.5, zorder=3)
        ax.text(hi + 0.6, i + dy, f"{m:+.1f}", va="center", fontsize=13, color=c if sig else MUTED)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[0] for t in tasks], fontsize=16)
ax.set_xlim(-32, 12); ax.tick_params(axis="x", labelsize=15)
ax.set_xlabel("Change vs original (percentage points, paired 95% interval)   ←  worse     better  →", fontsize=16)
ax.set_title("Capability cost: instruction following and maths drop", fontsize=21, fontweight="bold", pad=20)
ax.scatter([], [], s=170, color=GREEN, label="v2"); ax.scatter([], [], s=170, color=BLUE, label="v2b")
ax.scatter([], [], s=170, color=PANEL, edgecolor=MUTED, linewidth=2.5, label="interval includes zero")
leg = ax.legend(loc="upper right", fontsize=14, facecolor=BG, edgecolor=GRID, labelcolor=TXT)
fig.text(0.01, 0.015, "Clean checkpoints, same frozen items per task (HumanEval 164, MMLU 200, CyberMetric 500, GSM8K 200, IFEval 200).\n"
         "Exploratory, one training seed. Source: results/general_audit/clean_results.json.", fontsize=12.5, color=MUTED)
fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.17)
fig.savefig("docs/figures/capability_cost.png", facecolor=BG); plt.close(fig)
print("ok")
