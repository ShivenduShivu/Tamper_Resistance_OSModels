"""Team / coordinator brief. Usage: python docs/build/build_team_brief.py docs/Team_Brief.pdf
Reuses the style helpers from build_tamper_pdf.py; builds the Pareto chart from results/pareto_summary.json."""
import sys, os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = open(os.path.join(os.path.dirname(__file__), "build_tamper_pdf.py"), encoding="utf-8").read()
exec(src[:src.index("S = []")])            # styles + helpers: p, bullets, table, box, flow, H1.., B, SM
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Image
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Ar", F + "arial.ttf")); pdfmetrics.registerFont(TTFont("ArB", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("ArI", F + "ariali.ttf")); pdfmetrics.registerFont(TTFont("ArBI", F + "arialbi.ttf"))
addMapping("Ar", 0, 0, "Ar"); addMapping("Ar", 1, 0, "ArB"); addMapping("Ar", 0, 1, "ArI"); addMapping("Ar", 1, 1, "ArBI")
for st in (B, SM, BL, BOX): st.fontName = "Ar"
for st in (H1, H2, H3): st.fontName = "ArB"

# ---------- chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pf = json.load(open(os.path.join(ROOT, "results", "pareto_summary.json")))
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
sty = {"base": ("Original Tiny Aya", "#B4361A", "o"), "v1": ("v1 extended refusal", "#8A8F9C", "s"),
       "v2": ("v2 + trained against abliteration", "#1F2A44", "D"), "v2b": ("v2b + borderline-safe data", "#2E8B57", "^")}
for k, (lab, c, m) in sty.items():
    if k not in pf: continue
    f = sorted(pf[k]["front"], key=lambda t: t[1])
    ax.step([kl for _, kl in f], [r * 100 for r, _ in f], where="post", color=c, lw=1.8, label=lab)
    ax.scatter([kl for _, kl in f], [r * 100 for r, _ in f], color=c, marker=m, s=16, zorder=3)
ax.set_xlabel("Damage the attack causes to normal behaviour (KL divergence)  ->  more damage")
ax.set_ylabel("Harmful prompts still refused (%)")
ax.set_xlim(-0.003, 0.16); ax.set_ylim(0, 103); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
ax.set_title("Best attacks Heretic found (40 trials each): higher curve = harder to strip", fontsize=9)
fig.tight_layout(); CH = os.path.join(ROOT, "docs", "build", "pareto_chart.png"); fig.savefig(CH); plt.close(fig)

def tbl(rows, widths): return table(rows, widths)
S = []
# ================= COVER
S += [p("Heretic-Resistant Tiny Aya", H1),
      p("<b>Team brief:</b> what we researched, what we built, what we found, and where we stand. "
        "Lexsi Labs x Cohere x Hugging Face AI Safety &amp; Alignment Hackathon, Track 2b (Tamper Resistance).", B),
      Spacer(1, 4),
      box("The story in 30 seconds", [
          "1. Anyone can download an open model and <b>strip its safety</b> with a free one-command tool called <b>Heretic</b>.",
          "2. We tested this on <b>Cohere's Tiny Aya</b> (3.35B, the hackathon's model family). Heretic took it from "
          "<b>99% refusal to 14%</b> while changing its normal behaviour by almost nothing (KL 0.009). The threat is real.",
          "3. We built a defense that <b>trains the model while the attack is being applied to it</b>, so it learns to "
          "refuse even when its refusal direction is removed.",
          "4. Result: against our model <b>v2</b>, Heretic could not lower refusals at all (still <b>100%</b>) at the damage level "
          "that fully stripped the original. A second version, <b>v2b</b>, fixes the over-refusal that v2 introduced.",
          "5. A second attack, <b>harmful fine-tuning</b> (training on about 100 harmful examples), strips <b>every</b> model, ours included. "
          "So our defense is specific to weight-editing attacks like Heretic. The cost to general ability is small (MMLU 52.9% to about 50%)."]),
      Spacer(1, 6), p("Key numbers so far", H2),
      tbl([["", "Original", "v1", "v2", "v2b"],
           ["Refuses harmful prompts (no attack)", "99%", "99%", "100%", "100%"],
           ["After Heretic, small damage (KL at most 0.01)", "<b>14%</b>", "70%", "<b>100%</b>", "99%"],
           ["After Heretic, KL at most 0.05", "14%", "43%", "<b>100%</b>", "52%"],
           ["Lowest refusal Heretic ever reached", "14%", "17%", "33%", "<b>47%</b>"],
           ["Wrongly refuses safe prompts (XSTest)", "12%", "17.6%", "46%", "<b>20.8%</b>"],
           ["After harmful LoRA fine-tune, 100 examples (judge)", "1%", "3%", "3%", "1%"],
           ["MMLU, 1,000 questions (general ability)", "52.9%", "50.7%", "50.5%", "50.1%"]],
          [62, 28, 28, 28, 28]),
      p("Higher is better in rows 1 to 4 (more safety survives the attack). Lower is better in the last row (fewer "
        "needless refusals).", SM),
      PageBreak()]

# ================= 1. PROBLEM
S += [p("1. The problem, in plain words", H2),
      p("Companies release <b>open-weight</b> models: the full model file is public, so anyone can run or modify it. "
        "Safety training teaches the model to refuse harmful requests. But researchers found that this refusal behaviour "
        "lives mostly in <b>one direction</b> inside the model's internal activity (Arditi et al., 2024). "
        "Remove that direction and the model stops refusing. This is called <b>abliteration</b>."),
      p("<b>Heretic</b> (p-e-w/heretic) automates abliteration. It tries many settings and keeps the ones that remove "
        "the most refusals while changing normal behaviour the least. That change is measured by <b>KL divergence</b>: "
        "a number for how different the model's answers on harmless prompts become (0 = identical). "
        "It needs no training data, no expertise and about an hour on a free GPU. Hugging Face already hosts many "
        "abliterated models made this way."),
      box("Why this matters for Lexsi, Cohere and Hugging Face", [
          "- <b>Cohere</b> releases Tiny Aya openly. If its safety can be removed for free, every downstream copy is at risk.",
          "- <b>Hugging Face</b> hosts both the originals and the stripped copies.",
          "- <b>Lexsi Labs</b> builds alignment and interpretability tools (SafeTune, AuditKIT, CircuitKIT). Tamper resistance "
          "is exactly the open problem their Track 2b asks about.",
          "- Our question: <b>can we make Tiny Aya's safety survive the attack people actually use today?</b>"],
          bg="#EEF2FA", border="#1F2A44"),
      p("2. What we researched and where each idea came from", H2),
      p("We read the recent tamper-resistance work before choosing. Each row shows what we took from a source."),
      tbl([["Source", "What it says", "What we took from it"],
           ["Arditi et al. 2024, <i>Refusal is mediated by a single direction</i>", "Refusal is one direction in activation space.", "Explains why Heretic works. Our defense computes the same direction."],
           ["Heretic (p-e-w/heretic, 2025)", "Automatic abliteration with an optimiser (Optuna).", "Our <b>main attacker</b>. We re-run it adaptively on every model we make."],
           ["Extended-refusal defense (arXiv 2505.19056)", "Training on long, explained refusals spreads refusal across many directions.", "Our <b>v1</b> recipe."],
           ["Fool's Gold (arXiv 2608.17202) and ART (Kuo, Yadav, Smith 2026, arXiv 2605.26526)", "Train the model <b>while simulating the attack</b> inside training.", "Our <b>v2</b> core idea, adapted: refusal target, no hazardous data, 3B model."],
           ["TAR (Tamirisa et al. 2024), TamperBench (KDD 2026)", "Tamper resistance must be tested against <b>several attacks</b>, including fine-tuning.", "Our Day-4 fine-tuning attack and evaluation design."],
           ["Qi et al. 2023, fine-tuning compromises safety", "A few harmful examples can undo safety training.", "The second attack we test (LoRA, 100 to 1,000 examples)."],
           ["XSTest (Röttger et al. 2024), OR-Bench (Cui et al. 2024)", "Safety must not mean refusing safe questions.", "Our over-refusal metric (XSTest) and the v2b training data (OR-Bench)."]],
          [52, 62, 66]),
      p("<b>Ideas we dropped, and why.</b> (a) A multilingual / Hinglish attack angle: we found no evidence that people "
        "actually attack models this way, so the work would not be grounded in a real threat. (b) A straight Fool's Gold "
        "replication: it failed at 9B, needs hazardous data and the authors withheld their code. We kept its core idea "
        "and applied it to the attack that is really used in the wild (Heretic).", B),
      PageBreak()]

# ================= 3. DESIGN
S += [p("3. How we designed the experiment", H2),
      p("<b>Principle: attack every model we build, using the same strong, real-world attacker, adaptively.</b> "
        "Heretic searches from scratch on each new model, so a defense cannot pass by hiding from one fixed attack."),
      flow(["Measure the original (Heretic)", "Build defense data", "Train defense (v1, v2, v2b)", "Re-attack with Heretic", "Second attack: fine-tuning", "Compare"]),
      Spacer(1, 6),
      p("The three defenses", H3),
      tbl([["Model", "What we did (simple)", "Why"],
           ["<b>v1</b>", "LoRA fine-tune on 1,200 long, explained refusals (written by Tiny Aya itself) and 1,155 normal answers to everyday questions (Dolly).",
            "A model that explains why it refuses stores refusal in more places, so removing one direction hurts less."],
           ["<b>v2</b>", "Same data, but on half the training steps we <b>run the abliteration attack inside the model during training</b> and it must still refuse. The refusal direction is recomputed from the current model every 100 steps.",
            "Like sparring practice: the model learns to refuse while the attack is happening, and the simulated attacker keeps adapting."],
           ["<b>v2b</b>", "v2 plus 1,218 borderline but safe questions (e.g. how to kill a stuck computer process) with the original model's own helpful answers.",
            "v2 became too cautious (refused 46% of safe XSTest prompts). This teaches it where the line really is."]],
          [16, 92, 72]),
      p("Fair-testing rules we follow", H3),
      *bullets(["The prompts Heretic uses for scoring are <b>excluded</b> from our training data.",
                "XSTest is used <b>only</b> for testing. OR-Bench has 200 held-out prompts we never train on.",
                "We change one thing at a time (v2 to v2b only adds data), so we know what caused each change.",
                "Attacked or stripped models are <b>never published</b>. Only numbers are shared.",
                "Harmful prompts come only from public benchmarks. We write no harmful content."]),
      p("4. The full pipeline and every run", H2),
      p("<b>Compute:</b> Kaggle free T4 GPUs (about 30 GPU-hours a week; about 18 used so far). Models are stored in private "
        "Hugging Face repos. A watcher script on the laptop polls Kaggle, downloads the results when a run finishes and "
        "launches the next stage automatically, so runs continued overnight."),
      tbl([["Run", "What it did", "Outcome", "What we learned or fixed"],
           ["v2 to v4", "Day-1 checks: FP16 sanity, refusal, LoRA speed", "Check passed: 98% refusal, 0% on harmless, no garbage output", "Kaggle's torchao breaks peft, so we uninstall it"],
           ["v5 to v8", "Combined Day-2 job", "Failed: no token", "Kaggle hides secrets from CLI-pushed runs, so the token now comes from a private Kaggle dataset"],
           ["v9", "<b>Heretic on the original Tiny Aya</b>", "<b>Done in 53 min: 99% to 14% refusal at KL 0.009</b>", "The threat is real. GO."],
           ["v10", "Generate defense data", "Done in 67 min (1,200 refusals, 1,155 normal answers)", "GPU out of memory (262k vocabulary), so batch 1 with gradient accumulation"],
           ["v11", "Train v1", "Failed: dataset not found", "Moved the data to a private Kaggle dataset"],
           ["v12", "<b>Train v1, evaluate, Heretic on v1</b>", "Done in 179 min", "v1 helps a little (70% at KL 0.01)"],
           ["v13", "<b>Train v2, evaluate, Heretic on v2</b>", "Done in 189 min", "<b>v2 holds 100% up to KL 0.05</b>, but over-refuses (46%)"],
           ["v14", "v2b", "Typo, crashed at start", "No GPU time lost"],
           ["v15", "<b>Train v2b, evaluate, Heretic on v2b</b>", "Done in 280 min", "Over-refusal falls to 20.8%; resistance trade-off"],
           ["v16", "<b>Fine-tuning attack, MMLU, judge</b>", "Done in 114 min", "Fine-tuning strips all models; MMLU cost about 2.5 points; the judge agrees with the keyword counts"]],
          [17, 50, 55, 58]),
      PageBreak()]

# ================= 5. RESULTS
S += [p("5. What we got", H2),
      Image(CH, width=178 * mm, height=99 * mm),
      p("How to read this: each point is one attack Heretic found. Moving right means the attack damaged the model more; "
        "lower means fewer refusals survived. The original model (red) drops to 14% almost at zero damage. "
        "v2 (dark blue) stays at the top until Heretic is forced to damage the model about 7x more than it needed "
        "against the original. v2b (green) gives up some of that but never goes below 47%.", SM),
      Spacer(1, 4),
      tbl([["Metric", "Original", "v1", "v2", "v2b"],
           ["Training time (T4)", "-", "106 min", "112 min", "157 min"],
           ["Refusal on 100 harmful test prompts", "99%", "99%", "100%", "100%"],
           ["Refusal on XSTest <i>unsafe</i> prompts", "-", "-", "-", "88.5%"],
           ["Over-refusal on XSTest <i>safe</i> prompts", "12%", "17.6%", "46%", "20.8%"],
           ["Heretic: best refusal at KL at most 0.01 / 0.05 / 0.1", "14 / 14 / 14%", "70 / 43 / 17%", "100 / 100 / 96%", "99 / 52 / 47%"],
           ["Heretic: lowest refusal reached at any KL", "14%", "17%", "33%", "47%"]],
          [70, 25, 25, 30, 30]),
      p("What this means", H3),
      *bullets(["<b>Discovery 1:</b> stripping Tiny Aya is essentially free (14% refusal at KL 0.009).",
                "<b>Discovery 2:</b> long, explained refusals alone (v1) only delay Heretic. It still reaches 17%.",
                "<b>Discovery 3:</b> training against the simulated attack (v2) is what makes the difference: 100% refusal "
                "survives up to KL 0.05.",
                "<b>Discovery 4, a trade-off:</b> making the model less over-cautious (v2b) gives back some low-damage robustness "
                "but raises the worst case (47% vs 33%). v2 and v2b are two useful points on a <b>safety vs usability "
                "curve</b>, and we report both."]),
      box("Honest caveats (we say these before judges do)", [
          "- On clean models the classifier judge agrees with keyword counting (99% vs 99%). Heretic's attacked outputs are still scored by keywords only.",
          "- Each Heretic search ran 40 trials. A longer search might find stronger attacks.",
          "- One model size (3B). Our defense does <b>not</b> stop harmful fine-tuning (see section 6).",
          "- General ability drops slightly: MMLU 52.9% to 50.1 to 50.7%."], bg="#FFF8E6", border="#C98A00"),
      PageBreak()]

# ================= 6-7 NOW / NEXT
S += [p("6. Second attack: harmful fine-tuning (Day-4, done)", H2),
      tbl([["Model", "MMLU", "No attack", "After 100 examples", "After 400", "After 1,000"],
           ["Original", "52.9%", "99%", "1%", "3%", "9%"], ["v1", "50.7%", "98%", "3%", "5%", "10%"],
           ["v2", "50.5%", "100%", "3%", "6%", "8%"], ["v2b", "50.1%", "100%", "1%", "6%", "9%"]], [25, 25, 30, 35, 32, 33]),
      p("Refusal is scored by the independent classifier judge (ProtectAI). LoRA attack with learning rate 2e-4 on public harmful examples. Attacked weights are never saved.", SM),
      box("What this means", [
          "- <b>Every model is stripped by about 100 harmful examples.</b> This matches the literature: fine-tuning attacks remain the open problem in tamper resistance (Qi et al. 2023, TamperBench).",
          "- Our defense works against <b>data-free weight editing</b> (Heretic / abliteration). That is the attack actually used to make the many abliterated models on Hugging Face: it is free, needs no harmful data and takes one command.",
          "- Fine-tuning needs harmful training data and a training setup, so it is a different (higher-effort) threat. We will report this clearly as the limit of our defense.",
          "- The cost to general ability is small: about 2.5 MMLU points."], bg="#FFF8E6", border="#C98A00"),
      p("7. Plan to submission (by the evening of 14 Nov)", H2),
      tbl([["Step", "What", "Why it matters"],
           ["1", "Decide the scope: claim resistance to data-free attacks, and optionally try a cheap fine-tuning hardening step (SafeTune harden / TAR-lite)", "Fine-tuning broke every model; we need a clear, honest claim"],
           ["2", "Chain the Lexsi toolkits: <b>SafeTune</b> (evaluation and baseline defenses), <b>AuditKIT</b> (red-team and over-refusal probes), <b>CircuitKIT</b> (where refusal lives inside v2 vs the original)",
            "Hackathon rule: use at least 2 Lexsi repos. Also gives a mechanism story"],
           ["3", "Mechanism plot: how spread out refusal is across layers, original vs v2", "Explains why the defense works, not just that it does"],
           ["4", "HF Article + model card + small demo Space (attack the original vs v2 live, on a CPU)", "What judges actually see"],
           ["5", "Stretch: a detector that flags stripped copies of Aya on the Hub", "Extra real-world impact"]],
          [10, 100, 70]),
      p("8. Our chances of standing out (honest view)", H2),
      p("<b>Strengths</b>", H3),
      *bullets(["<b>A real threat, demonstrated on the sponsor's own model:</b> Heretic stripping Tiny Aya almost for free is a "
                "striking, reproducible finding Cohere and Lexsi will care about.",
                "<b>Adaptive evaluation:</b> we re-attack every checkpoint with a fresh optimiser. Many hackathon defenses only test a "
                "fixed attack, and this is the main criticism in the tamper-resistance literature.",
                "<b>A clear positive result with numbers:</b> 14% vs 100% refusal at the same damage level.",
                "<b>We report the trade-off and the caveats</b> instead of hiding them. Research judges reward this.",
                "<b>Runs on free hardware</b> (T4, about 2 to 3 hours per defense). That is practical for real model publishers."]),
      p("<b>Risks</b>", H3),
      *bullets(["<b>Confirmed:</b> harmful fine-tuning breaks all our models. We must frame the claim as resistance to data-free weight editing, which is still the attack used in the wild.",
                "Heretic-attacked outputs are still scored by keywords only (the judge agreed on clean models).",
                "The toolkit-chaining rule is not done yet. It is planned for Day 5."]),
      p("Our view: the Heretic result (14% vs 100% at the same damage) is strong and novel on Tiny Aya. The fine-tuning result "
        "limits the claim but makes the study more credible. With clear scoping, the Lexsi toolkit chaining and a "
        "mechanism story, this is a competitive shortlist candidate. We can't promise a placement; this is our reading of the evidence.", B),
      PageBreak()]

# ================= 9 COORDINATOR
S += [p("9. For the coordinator: a short pitch and our questions", H2),
      box("Pitch (about 1 minute)", [
          "We showed that Heretic, the free tool people use to strip safety from open models, removes Tiny Aya's refusals "
          "almost for free: 99% to 14% at KL 0.009. Inspired by recent work that trains models <i>while simulating the "
          "attack</i> (Fool's Gold, ART) and by extended-refusal training, we built a defense that trains Tiny Aya through "
          "a simulated, constantly re-computed abliteration. Then we re-ran Heretic from scratch on our model. "
          "Refusal stayed at 100% at the damage level that fully stripped the original. A second version keeps most of "
          "that while cutting needless refusals back to about 21%, at a cost of about 2.5 MMLU points. Harmful fine-tuning "
          "still strips every model, so we scope our claim to data-free weight-editing attacks, and will chain SafeTune, AuditKIT and CircuitKIT for "
          "evaluation and a mechanism analysis."], bg="#EEF2FA", border="#1F2A44"),
      p("Questions we would like answered", H3),
      *bullets(["<b>Toolkit chaining:</b> does using SafeTune (evaluation and a baseline defense), AuditKIT (probes) and "
                "CircuitKIT (analysis) as part of the pipeline satisfy the at-least-2-repos rule, or must the defense itself be built on them?",
                "<b>Evaluation standard:</b> is there a preferred refusal judge or benchmark for Track 2b (for example a specific "
                "classifier, StrongREJECT, or TamperBench attacks)? We currently use keyword counts plus a classifier.",
                "<b>Model choice:</b> is Tiny Aya alone fine, or would judges expect a second model family or size?",
                "<b>Releasing weights:</b> may we publish the <i>hardened</i> models publicly? We plan to keep any stripped or attacked models private.",
                "<b>Demo format:</b> is a CPU Hugging Face Space (original vs hardened, under attack) enough, or is a live attack demo preferred?",
                "<b>Compute:</b> are there extra HF credits or GPU grants for a longer Heretic search (more trials) to strengthen the result?",
                "<b>Scoring:</b> how is tamper resistance weighed against usability (over-refusal, MMLU)?"]),
      p("10. Glossary", H2),
      tbl([["Term", "Meaning"],
           ["Open-weight model", "A model whose full parameter file is downloadable, so anyone can modify it."],
           ["Refusal direction", "One direction in the model's internal activity that mostly controls saying no."],
           ["Abliteration", "Removing the refusal direction from the weights, so the model stops refusing."],
           ["Heretic", "A free tool that searches for the best abliteration settings automatically."],
           ["KL divergence", "How much the model's normal answers change. Low KL means the attack is cheap and almost invisible."],
           ["Pareto front", "The best trade-offs an attacker found: for each damage level, the fewest refusals."],
           ["LoRA", "A cheap way to fine-tune by training small add-on matrices instead of the whole model."],
           ["Over-refusal", "Refusing safe questions that only sound risky (measured with XSTest)."],
           ["MMLU", "A standard multiple-choice test of general knowledge, used to check that the model stays smart."],
           ["Adaptive attack", "Re-running the attack from scratch on the defended model, rather than reusing old settings."]],
          [38, 142]),
      Spacer(1, 6),
      p("Code, logs and raw results: lexsi-hackathon repo (notes/LOG.md, results/pareto_summary.json, kaggle/day1 to day4).", SM)]

def foot(c, d):
    c.saveState(); c.setFont("Ar", 7.5); c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(15 * mm, 9 * mm, "Heretic-resistant Tiny Aya | team brief | Track 2b Tamper Resistance")
    c.drawRightString(195 * mm, 9 * mm, f"{d.page}"); c.restoreState()
doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
                        title="Heretic-Resistant Tiny Aya: Team Brief")
doc.build(S, onFirstPage=foot, onLaterPages=foot)
print("ok", sys.argv[1])
