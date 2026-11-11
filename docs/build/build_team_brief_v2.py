"""Final team brief. Usage: python docs/build/build_team_brief_v2.py docs/Team_Brief.pdf
Reuses the style helpers from build_tamper_pdf.py and the Pareto chart from build_team_brief.py."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = open(os.path.join(os.path.dirname(__file__), "build_tamper_pdf.py"), encoding="utf-8").read()
exec(src[:src.index("S = []")])            # p, bullets, table, box, flow, styles
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Image
F = "C:/Windows/Fonts/"
for n, f in [("Ar", "arial.ttf"), ("ArB", "arialbd.ttf"), ("ArI", "ariali.ttf"), ("ArBI", "arialbi.ttf")]:
    pdfmetrics.registerFont(TTFont(n, F + f))
addMapping("Ar", 0, 0, "Ar"); addMapping("Ar", 1, 0, "ArB"); addMapping("Ar", 0, 1, "ArI"); addMapping("Ar", 1, 1, "ArBI")
for st in (B, SM, BL, BOX): st.fontName = "Ar"
for st in (H1, H2, H3): st.fontName = "ArB"
CH = os.path.join(ROOT, "docs", "build", "pareto_chart.png")
GREEN, BLUEBG, AMBER = ("#EAF6EE", "#2E8B57"), ("#EEF2FA", "#1F2A44"), ("#FFF8E6", "#C98A00")

S = []
# ============ COVER
S += [p("Heretic-Resistant Tiny Aya", H1),
      p("<b>Team brief, final results.</b> Everything we researched, built, tested and found, in plain words. "
        "Lexsi Labs x Cohere x Hugging Face AI Safety &amp; Alignment Hackathon, Track 2b (Tamper Resistance). "
        "Submission deadline: 14 Nov (end of day AoE).", B), Spacer(1, 4),
      box("The story in 30 seconds", [
          "1. Anyone can download an open model and <b>remove its safety in one command</b> with a free tool called <b>Heretic</b>.",
          "2. On Cohere's <b>Tiny Aya Global</b>, Heretic cut refusal of harmful requests from <b>99% to 1%</b>, and general "
          "knowledge (MMLU) did not drop at all. The threat is real and free.",
          "3. We trained Tiny Aya <b>while simulating that attack inside the model</b>, so it learned to refuse even with its "
          "refusal direction removed.",
          "4. Our <b>v2</b> refuses <b>100%</b> at the attack strength that fully broke the original. Heretic's strongest attack "
          "only gets it to <b>23%</b>, and needs about <b>16x more damage</b> to the model's normal behaviour.",
          "5. Lexsi's own <b>SafeTune</b> confirms it independently (original 28%, v2 100%). <b>Limits:</b> harmful fine-tuning "
          "and prefilling still break it. We say so openly."]),
      Spacer(1, 6), p("Key numbers (refusal on 100 harmful prompts, scored by the classifier judge; higher is safer)", H3),
      table([["", "Original", "v1", "v2", "v2b"],
             ["No attack", "99%", "98%", "100%", "100%"],
             ["Heretic, weak attack (KL up to 0.01, keyword score)", "14%", "70%", "<b>100%</b>", "99%"],
             ["Heretic, strongest attack found (judge)", "<b>1%</b>", "~17% (kw)", "<b>23%</b>", "24%"],
             ["SafeTune refusal-direction ablation (judge)", "28%", "-", "<b>100%</b>", "<b>100%</b>"],
             ["Harmful LoRA fine-tune, 100 examples (judge)", "1%", "3%", "3%", "1%"],
             ["Over-refusal of safe questions (XSTest)", "12%", "17.6%", "46%", "<b>20.8%</b>"],
             ["MMLU, general knowledge", "52.9%", "50.7%", "50.5%", "50.1%"]], [70, 26, 26, 27, 27]),
      p("v1's strongest-attack figure is Heretic's keyword score; it was not re-judged. \"-\" means not measured.", SM),
      PageBreak()]

# ============ 1 PROBLEM + TERMS
S += [p("1. The problem, in plain words", H2),
      p("Companies publish <b>open-weight</b> models: the whole model file is downloadable. Safety training teaches the model "
        "to refuse harmful requests. Research (Arditi et al., 2024) found that this refusal behaviour lives mostly in "
        "<b>one direction</b> inside the model's internal activity. Delete that direction and the model stops refusing. "
        "This is <b>abliteration</b>. <b>Heretic</b> automates it: it tries many settings and keeps the ones that remove the "
        "most refusals while changing normal behaviour the least. It needs no data and no skill, and takes about 1 hour on "
        "a free GPU. That is why there are so many \"abliterated\" copies of open models on Hugging Face."),
      p("<b>Our question:</b> can we make Tiny Aya keep its safety against this exact, real-world attack, without making "
        "it much less useful?"),
      box("Who is the \"classifier judge\"? (important)", [
          "- <b>What it is:</b> a small, separate AI model, "
          "<b>ProtectAI/distilroberta-base-rejection-v1</b> (Apache-2.0). It reads a model's answer and labels it "
          "<b>REJECTION</b> (a refusal) or <b>NORMAL</b> (it answered). It is not one of our models.",
          "- <b>Why we need it:</b> the simple method, a <b>keyword score</b>, counts an answer as a refusal if it contains "
          "words like \"sorry\", \"I can't\" or \"illegal\". That is fooled easily: a model that complies but adds \"this is "
          "illegal\" gets counted as refusing.",
          "- <b>Proof it matters:</b> after Heretic's attack, the keyword score said the original Tiny Aya still refused "
          "14%. The judge showed the real figure is <b>1%</b>.",
          "- <b>Where we use each:</b> Heretic scores its own search with keywords, which we can't change. All our final "
          "numbers use the judge, and we report keyword scores beside them.",
          "- <b>Next check:</b> a teammate is re-scoring the saved responses with stronger safety judges "
          "(Llama Guard / Granite Guardian)."], *BLUEBG),
      Spacer(1, 4), p("Other key terms", H3),
      table([["Term", "Meaning"],
             ["Refusal direction", "One direction in the model's internal activity that mostly controls \"saying no\"."],
             ["Abliteration / Heretic", "Deleting that direction from the weights / the free tool that does it automatically."],
             ["KL divergence", "How much an attack changes the model's normal answers on harmless prompts. Low = a cheap, invisible attack."],
             ["Pareto front", "For each damage level (KL), the lowest refusal the attacker managed. Our attack-curve chart."],
             ["LoRA", "Cheap fine-tuning: train small add-on matrices instead of the whole model."],
             ["MMLU", "A standard multiple-choice test of general knowledge (1,000-question sample)."],
             ["XSTest / OR-Bench", "Safe questions that sound risky (e.g. \"how do I kill a Python process\"), used to measure over-refusal."],
             ["Prefilling", "Forcing the answer to start with \"Sure, here is...\" (possible with open weights)."],
             ["CAA steering", "Pushing the model's internal activity away from refusing while it runs."],
             ["Lexsi toolkits", "SafeTune (attacks and directions), AuditKIT (red-team probes), CircuitKIT (circuits). The rules require at least 2."]],
            [38, 142]),
      PageBreak()]

# ============ 2 RESEARCH + 3 DESIGN
S += [p("2. What we researched, and what we took from it", H2),
      table([["Source", "Idea", "Used for"],
             ["Arditi et al. 2024", "Refusal is one direction", "Why Heretic works; the direction we attack and train against"],
             ["Heretic (p-e-w)", "Automatic abliteration with an optimiser", "Our main attacker, re-run fresh on every model"],
             ["Extended refusal (arXiv 2505.19056)", "Long, explained refusals spread refusal across directions", "v1"],
             ["Fool's Gold (arXiv 2608.17202), ART (arXiv 2605.26526)", "Train while simulating the attack", "v2 (refusal target, no hazardous data)"],
             ["TAR, TamperBench, Qi et al. 2023", "Test several attacks; fine-tuning is the hard one", "Our fine-tuning attack and honest limits"],
             ["XSTest, OR-Bench", "Safety must not mean refusing safe questions", "Over-refusal metric (XSTest), v2b data (OR-Bench)"]],
            [52, 62, 66]),
      p("<b>Dropped:</b> (a) a Hinglish/multilingual attack angle, because we found no evidence of real-world use; "
        "(b) a straight Fool's Gold replication, because it failed at 9B, needs hazardous data and its code is withheld.", B),
      p("3. How we designed the experiment", H2),
      flow(["Attack the original (Heretic)", "Build training data", "Train v1 / v2 / v2b", "Re-attack each one fresh",
            "Other attacks + toolkits", "Judge + MMLU"]),
      Spacer(1, 5),
      table([["Model", "What we did", "Why"],
             ["<b>v1</b>", "LoRA fine-tune on 1,200 long, explained refusals plus 1,155 everyday answers, all written by Tiny Aya itself.",
              "Explaining refusals spreads them across the model."],
             ["<b>v2</b>", "Same data, but on half the training steps we <b>remove the refusal direction from every layer during training</b>, "
              "and the model must still refuse. The direction is recomputed from the current model every 100 steps.",
              "Like sparring: the model learns to refuse while being attacked, and the simulated attacker keeps adapting."],
             ["<b>v2b</b>", "v2 plus 1,218 borderline-but-safe questions with helpful answers.", "v2 was over-cautious (46%); this teaches it where the line is."]],
            [14, 100, 66]),
      p("<b>Fair-testing rules:</b>", B),
      *bullets(["Heretic's scoring prompts are excluded from training.",
                "XSTest is used for testing only, and 200 OR-Bench prompts are held out.",
                "One change at a time between versions.",
                "Stripped models are never saved or published, and their outputs are never shown.",
                "All harmful prompts come from public benchmarks."]),
      PageBreak()]

# ============ 4 RUNS
S += [p("4. The full pipeline and every run", H2),
      p("<b>Compute:</b> free Kaggle T4 GPUs (about 26.5 of the week's ~30 GPU-hours used). Scripts are pushed from the laptop. "
        "A watcher script polls Kaggle, downloads the results and launches the next stage automatically, so runs continued overnight. "
        "Models are on Hugging Face. Every run is logged in notes/LOG.md."),
      table([["Run", "What it did", "Result / fix"],
             ["v2-v8", "Setup checks; the token problem", "Kaggle hides secrets from CLI pushes, so the token now comes from a private Kaggle dataset"],
             ["v9", "<b>Heretic on the original</b>", "99% to 14% refusal (keyword) at KL 0.009 in 53 min. The threat is confirmed."],
             ["v10-v11", "Generate training data; storage", "1,200 refusals + 1,155 answers; out-of-memory fix (batch 1 x 8)"],
             ["v12", "<b>Train v1, attack it</b>", "Helps a little: 70% at KL 0.01"],
             ["v13", "<b>Train v2, attack it</b>", "100% up to KL 0.05, but over-refuses (46%)"],
             ["v15", "<b>Train v2b, attack it</b>", "Over-refusal 20.8%; robustness trade-off"],
             ["v16", "<b>Fine-tuning attack, MMLU, judge</b>", "Fine-tuning breaks all models; MMLU -2.5 points"],
             ["v17", "Toolkit smoke test (CPU, free)", "All 4 tools install on Kaggle"],
             ["v18", "<b>Lexsi toolkits + other attacks</b>", "SafeTune ablation 28% vs 100%; sampling, 4-bit, prefill, steering, AuditKIT"],
             ["v19", "<b>Exact Heretic attacks re-applied, then judged and MMLU'd</b>", "Original 1% (MMLU 52.7%); v2 23% (MMLU 51.0%)"],
             ["v20-21", "CircuitKIT circuit analysis", "Model does not fit a 16 GB T4 for discovery. Stopped."]],
            [17, 63, 100]),
      p("5. Results", H2),
      Image(CH, width=178 * mm, height=99 * mm),
      p("Each point is one attack Heretic found (keyword-scored). Moving right means more damage to the model; higher means more "
        "refusals survived. The original (red) collapses almost at zero damage. v2 (dark blue) stays at the top until Heretic "
        "is forced to damage the model about 7-16x more.", SM),
      PageBreak()]

# ============ 5 contd: attack matrix + Heretic judged
S += [p("All attacks, scored by the judge (refusal on 100 harmful prompts; higher is safer)", H3),
      table([["Attack", "Original", "v2", "v2b", "Tool"],
             ["Heretic, strongest attack found", "1% (MMLU 52.7%)", "23% (MMLU 51.0%)", "24%", "Heretic"],
             ["Refusal-direction ablation at run time", "28%", "<b>100%</b>", "<b>100%</b>", "SafeTune"],
             ["CAA steering, strength -1", "4%", "<b>42%</b>", "30%", "SafeTune"],
             ["CAA steering, strength -4", "0%", "0%", "0%", "SafeTune"],
             ["Random sampling, best of 3 (share that got through)", "9%", "<b>0%</b>", "2%", "ours"],
             ["Model compressed to 4-bit", "99%", "<b>100%</b>", "<b>100%</b>", "bitsandbytes"],
             ["Prefill \"Sure, here is\"", "31%", "38%", "41%", "ours"],
             ["Harmful LoRA fine-tune (100 / 1,000 examples)", "1% / 9%", "3% / 8%", "1% / 9%", "ours"],
             ["Prompt-injection probes (pass rate)", "62.5%", "<b>87.5%</b>", "75%", "AuditKIT"]],
            [62, 30, 30, 22, 26]),
      box("What the results mean", [
          "- <b>Works against:</b> attacks that remove or steer the refusal direction (Heretic, SafeTune ablation, mild steering), "
          "random sampling, and 4-bit compression.",
          "- <b>Raises the price, not a wall:</b> Heretic's strongest attack still gets v2 to 23% without hurting MMLU. "
          "But it needs about 16x more behaviour change, and v2 still refuses 62% of clearly unsafe XSTest prompts "
          "(the attacked original: 17%).",
          "- <b>Doesn't work against:</b> harmful fine-tuning, prefilling, strong steering. These are known open problems.",
          "- <b>Costs:</b> about 2.5 MMLU points; v2 over-refuses (46%), and v2b fixes most of that (20.8%)."], *GREEN),
      Spacer(1, 4),
      p("Inside the model (mechanism)", H3),
      p("SafeTune's per-layer refusal directions show v2's direction is <b>10-20% weaker in the middle and late layers</b> than the "
        "original's (layer 24: 40.2 vs 46.4), with the same peak layer (18). Our reading is that v2 no longer relies on one "
        "dominant direction, so deleting it no longer deletes refusal."),
      p("6. How we used the Lexsi toolkits", H2),
      table([["Toolkit", "How we used it", "Outcome"],
             ["<b>SafeTune</b>", "Extracted refusal directions; attacked every model with direction ablation and CAA steering",
              "Independent confirmation (28% vs 100%); mechanism evidence"],
             ["<b>AuditKIT</b>", "Red-team probes (jailbreak, encoding, prompt injection, refusal) on every model",
              "Independent probes: v2 passes prompt injection 87.5% vs 62.5% for the original"],
             ["<b>CircuitKIT</b>", "Attempted refusal-circuit comparison, original vs v2",
              "Its dataset helpers build unusable pairs (worked around); discovery does not fit a 16 GB T4 for this model"]],
            [24, 70, 86]),
      p("The rule needs at least 2 connected repos: SafeTune and AuditKIT are both in the evaluation chain. CircuitKIT is reported "
        "as attempted.", SM),
      PageBreak()]

# ============ 7 PUBLIC + STATUS + CHANCES
S += [p("7. Everything that is public", H2),
      table([["What", "Where (Hugging Face user StarWarrior24x7)"],
             ["Models (v1, v2, v2b) with full cards", "tiny-aya-global-xref-v1, tiny-aya-global-xref-v2 (v2b in the v2b/ subfolder)"],
             ["Training data", "datasets/tiny-aya-extended-refusal"],
             ["Results, Heretic journals, run logs", "datasets/tiny-aya-tamper-resistance-results"],
             ["Collections (required)", "\"Heretic-Resistant Tiny Aya: Models\" and \"...: Data &amp; Results\""],
             ["Demo Space (results explorer)", "spaces/heretic-resistant-tiny-aya (ready; being published)"],
             ["Code and full log", "github.com/ShivenduShivu/Tamper_Resistance_OSModels"],
             ["HF Article (the submission)", "Draft ready (docs/HF_ARTICLE.md); posting before the deadline"]],
            [62, 118]),
      p("Never public: stripped or attacked models, their raw outputs, refusal-direction files, and tokens.", SM),
      p("8. Where we stand and who does what", H2),
      table([["Item", "Owner", "Status"],
             ["Open-weight track (this work), article, Space, submission", "Shivendu", "Results final; article in review; Space ready to publish"],
             ["Stronger judge pass (Llama Guard / Granite Guardian) on saved responses", "Teammate", "Responses can be shared privately on request"],
             ["Fine-tuning defense (TAR-lite: simulate LoRA attacks during training)", "Teammates, ft-defense/* branches", "Exploring; merge only through a reviewed PR"]],
            [78, 40, 62]),
      p("9. Our chances, honestly", H2),
      *bullets(["<b>Strengths:</b>",
                "&nbsp;&nbsp;- A real, free attack demonstrated on the sponsor's own model (99% to 1%).",
                "&nbsp;&nbsp;- A defense that clearly works against it.",
                "&nbsp;&nbsp;- Adaptive re-attacks on every model.",
                "&nbsp;&nbsp;- Independent confirmation with Lexsi's own toolkit.",
                "&nbsp;&nbsp;- Honest limits.",
                "&nbsp;&nbsp;- Everything public and reproducible on free GPUs.",
                                "<b>Risks:</b>",
                "&nbsp;&nbsp;- Fine-tuning and prefilling still break the model.",
                "&nbsp;&nbsp;- Heretic's strongest attack reaches 23%.",
                "&nbsp;&nbsp;- One model size only.",
                "&nbsp;&nbsp;- CircuitKIT could not run on free hardware.",
                "<b>Our view:</b> a strong shortlist candidate if the article tells this story clearly. We can't promise a placement."]),
      p("10. One-minute pitch (for the coordinator or Demo Day)", H2),
      box("Pitch", [
          "Heretic, the free tool people use to strip safety from open models, removes Tiny Aya's refusals almost completely: "
          "99% to 1%, with no loss in general knowledge. Inspired by Fool's Gold and ART, we trained Tiny Aya while simulating "
          "that attack inside the model. At the same attack strength our model still refuses 100%, and Lexsi's SafeTune confirms "
          "it independently. The attacker now has to distort the model about 16 times more, and still only gets to 23%. "
          "It also survives 4-bit compression and sampling attacks. We are honest about what it doesn't stop yet: "
          "fine-tuning and prefilling, which is our next step."], *BLUEBG)]


def foot(c, d):
    c.saveState(); c.setFont("Ar", 7.5); c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(15 * mm, 9 * mm, "Heretic-Resistant Tiny Aya | team brief (final results) | Track 2b Tamper Resistance")
    c.drawRightString(195 * mm, 9 * mm, f"{d.page}"); c.restoreState()


doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
                        title="Heretic-Resistant Tiny Aya: Team Brief (final)")
doc.build(S, onFirstPage=foot, onLaterPages=foot)
print("ok", sys.argv[1])
