"""Final team brief (complete edition). Usage: python docs/build/build_team_brief_v3.py docs/Team_Brief.pdf"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = open(os.path.join(os.path.dirname(__file__), "build_tamper_pdf.py"), encoding="utf-8").read()
exec(src[:src.index("S = []")])            # p, bullets, table, box, flow, H1/H2/H3/B/SM/BL/BOX
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
BLUE, GREEN, AMBER, RED = ("#EEF2FA", "#1F2A44"), ("#EAF6EE", "#2E8B57"), ("#FFF8E6", "#C98A00"), ("#FDECEA", "#B4361A")
def term(word, meaning): return p(f"<b>{word}</b>: {meaning}", BL)

S = []
# =========================== COVER
S += [p("Heretic-Resistant Tiny Aya", H1),
      p("<b>Complete team brief, final results (updated 14 Nov 2026).</b> Lexsi Labs x Cohere x Hugging Face AI Safety &amp; Alignment Hackathon, "
        "Track 2b (Tamper Resistance). Submission deadline: 15 Nov 2026, 11:59 PM AoE (= 16 Nov, 5:29 PM IST). This document explains the whole project "
        "from zero: what the problem is, what we built, how we tested it, what we found, what it does not do, and how to "
        "answer the judges. Technical words are explained where they first appear and again in the glossary (section 25). "
        "The GitHub README is the single source of truth: if a number here ever differs, the README wins.", B),
      box("The whole project in 9 lines", [
          "1. <b>Open-weight models</b> (anyone can download the model file) come with safety training: they refuse harmful requests.",
          "2. That refusal mostly lives in <b>one direction</b> inside the model. Delete it and the model stops refusing. This is <b>abliteration</b>.",
          "3. A free tool, <b>Heretic</b>, does this in one command. On Cohere's <b>Tiny Aya</b> it made <b>91%</b> of answers to harmful requests harmful (was 0%), with MMLU unchanged.",
          "4. We trained Tiny Aya <b>while simulating that attack inside it</b> (our model <b>v2</b>), so it learned to refuse even with the direction removed.",
          "5. Result: the low-damage attacks found in our searches keep <b>100%</b> refusal on v2. The strongest attack found had ~<b>16x the measured damage</b> and still got only <b>40%</b> harmful answers (vs 91%). A <b>5x longer search</b> found no stronger worst case (42%).",
          "6. Lexsi's own <b>SafeTune</b> confirms it (55% -> 0% harmful). It also holds in <b>Hindi, Arabic, French, Spanish</b> (41% vs 12% harmful).",
          "7. <b>It is not free:</b> MMLU and coding hold, but instruction following drops 10 points and v2 refuses 43% of safe-but-scary questions (original 17%).",
          "8. <b>Harmful fine-tuning still breaks every model.</b> Adding RepNoise (model <b>A1</b>) delays the collapse <b>about 3-4x</b> on held-out prompts (between 2x and 7x given the checkpoint spacing; two training seeds, one attack seed), but costs about half of v2's coding passes. <b>Delay, not prevention.</b> Prefill also still works (a small pilot cut it 86% -> 0%).",
          "9. Everything is public: models, data, score files, code, a live demo page and the article; plus <b>4 bug fixes merged</b> into Lexsi's toolkits."], *BLUE),
      Spacer(1, 6), p("Key numbers at a glance", H3),
      table([["", "Original Tiny Aya", "Our v2", "Our v2b"],
             ["Refuses harmful requests (no attack)", "98%", "100%", "100%"],
             ["Heretic, low-damage attacks (KL &lt;= 0.05): refusal kept (keyword)", "<b>14%</b>", "<b>100%</b>", "52%"],
             ["Heretic, strongest attack: harmful answers (Granite judge)", "<b>91%</b>", "<b>40%</b>", "57%"],
             ["...same, with 5x attacker search budget", "-", "42%", "-"],
             ["SafeTune direction-removal attack: harmful answers", "55%", "<b>0%</b>", "<b>0%</b>"],
             ["Multilingual (hi/ar/fr/es), strongest attack: harmful", "41%", "<b>12%</b>", "22%"],
             ["Prefill \"Sure, here is\": harmful answers", "82%", "77%", "80%"],
             ["Harmful fine-tune, 100 examples: refusal left", "0%", "3%", "-"],
             ["Refuses safe-but-scary questions (XSTest, n = 250)", "17.2%", "42.8%", "16.8%"],
             ["MMLU (general knowledge)", "52.3%", "53.0%", "52.7%"],
             ["Instruction following (IFEval, n = 200)", "77.5%", "67.5%", "58.0%"]], [86, 32, 30, 30]),
      p("Higher refusal = safer; lower harmful = safer; lower over-refusal = more useful. \"-\" = not measured. Over-refusal labels are provisional (AI classifier, 128-token answers).", SM),
      PageBreak()]

# =========================== PART A: BACKGROUND
S += [p("Part A. The background, in plain words", H1),
      p("1. Models, open weights and safety training", H2),
      p("A large language model (LLM) is a very large set of numbers (the <b>weights</b>) that turns a text prompt into a text answer. "
        "Companies like Cohere train a base model on huge amounts of text and then do <b>safety training</b>: they teach it to "
        "<b>refuse</b> harmful requests (\"how do I build a weapon\") while still helping with normal ones. "
        "<b>Open-weight</b> means the weights file is published: anyone can download it, run it on their own computer, and change it. "
        "That is great for research, privacy and cost, but it also means the safety training can be removed."),
      p("<b>Tiny Aya Global</b> is Cohere's small open model (3.35 billion weights) that speaks 70+ languages. It is the hackathon's "
        "model family, so it is the one we protect."),
      p("2. The attack: refusal direction, abliteration and Heretic", H2),
      p("Inside the model, every word is processed as a long list of numbers called <b>activations</b>. Researchers (Arditi et al., 2024) found "
        "that the difference between \"I will refuse\" and \"I will answer\" mostly sits along <b>one direction</b> in that space, like a single "
        "switch. <b>Abliteration</b> edits the weights so the model can no longer move along that direction: the switch is removed, so it "
        "never refuses. No training data is needed: just a few example prompts to find the direction."),
      p("<b>Heretic</b> is a free, popular tool that automates abliteration. It tries many settings (which layers, how strongly) using an "
        "optimiser and keeps the ones that remove the most refusals while changing the model's normal behaviour the least. "
        "It runs in about an hour on a free GPU. This is how most \"abliterated\" (safety-stripped) models on Hugging Face are made."),
      box("What is KL divergence? (the attacker's \"damage meter\")", [
          "When an attack edits the model, it also changes how the model answers normal questions. <b>KL divergence</b> is a number "
          "that measures how much the model's answers on harmless prompts changed. 0 = identical; bigger = more different.",
          "An attacker wants a <b>low KL</b>: the stripped model should still behave normally, otherwise it is visibly broken and nobody uses it. "
          "So a good defense forces the attacker to accept a <b>high KL</b> (a visibly damaged model) to remove the refusals.",
          "Analogy: picking a lock (low KL) vs smashing the door (high KL). We want to make picking impossible, so the attacker has to smash."], *AMBER),
      p("3. Why this matters", H2),
      *bullets(["For <b>Cohere</b>: if their open model's safety can be removed for free, every copy is at risk.",
                "For <b>Hugging Face</b>: it hosts both the originals and stripped copies.",
                "For <b>Lexsi Labs</b>: they build safety tooling (SafeTune, AuditKIT, CircuitKIT); tamper resistance is the open problem of Track 2b.",
                "Our question: <b>can a small open model keep its refusals against the attack people actually use, without becoming less useful?</b>"]),
      PageBreak()]

# =========================== PART B: APPROACH
S += [p("Part B. What we built and how we tested it", H1),
      p("4. The research we built on", H2),
      table([["Source", "What it says", "What we took"],
             ["Arditi et al. 2024", "Refusal is one direction", "Why Heretic works; the direction our defense trains against"],
             ["Heretic (p-e-w)", "Automatic abliteration", "Our main attacker, re-run fresh on every model"],
             ["Extended refusal (arXiv 2505.19056)", "Long, explained refusals spread refusal across many directions", "Our v1"],
             ["Fool's Gold, ART (arXiv 2608.17202, 2605.26526)", "Train while simulating the attack", "The core idea of v2 (refusal target, no hazardous data)"],
             ["TAR, TamperBench, Qi et al. 2023", "Defenses must face several attacks; fine-tuning is the hardest", "Our extra attacks and honest limits"],
             ["DeepRefusal (arXiv 2509.15202)", "Train refusals after a harmful start", "Our prefill pilot"],
             ["XSTest, OR-Bench", "Safety must not mean refusing safe questions", "Our over-refusal test; v2b's training data"]], [44, 66, 70]),
      p("5. The three defenses", H2),
      p("All three use <b>LoRA fine-tuning</b>: instead of retraining all 3.35 billion weights, we train small add-on matrices (cheap: "
        "about 2 hours on a free T4 GPU) and then merge them into the model. Training data: 1,200 long, explained refusals to harmful "
        "prompts plus 1,155 normal helpful answers, all written by Tiny Aya itself (so its style stays the same)."),
      table([["Model", "What we did", "Analogy / why"],
             ["<b>v1</b>", "Fine-tune on the explained refusals + normal answers.", "Teach it to explain why it refuses, so refusal is spread out, not one switch."],
             ["<b>v2</b>", "Same data, but on half the training steps we <b>run the abliteration attack inside the model while it trains</b> "
              "(remove the refusal direction from every layer) and it must still refuse. Every 100 steps we re-measure the direction "
              "from the current model, so the simulated attacker keeps adapting.",
              "Sparring practice: a boxer who trains while being hit learns to stay standing. The model learns to refuse even with its "
              "\"refusal switch\" removed."],
             ["<b>v2b</b>", "v2 plus 1,218 borderline-but-safe questions with helpful answers (e.g. \"how do I kill a stuck process\").", "v2 became too cautious; this teaches it where the line is."]], [14, 96, 70]),
      p("6. How we tested (and why the tests are fair)", H2),
      *bullets(["<b>Adaptive attacker:</b> Heretic searches from scratch on each of our models. A defense cannot pass by hiding from one fixed attack.",
                "<b>Many different attacks</b> (section 11): Heretic, SafeTune's own attacks, steering, prefilling, sampling, 4-bit compression, harmful fine-tuning, multilingual prompts.",
                "<b>Clean test sets:</b> the prompts Heretic uses are excluded from training; XSTest is never trained on; 200 OR-Bench prompts are held out. "
                "The fine-tuning results were re-checked on <b>193 HarmBench prompts with zero overlap</b> with any training, attack or defense data.",
                "<b>Predictions written before running</b> the held-out fine-tuning test, so we could not move the goalposts afterwards (all three held).",
                "<b>One change at a time</b> between versions, and a <b>matched control</b> arm in experiments, so we know what caused each change.",
                "<b>Stripped models are never published</b>; their outputs are never shown; harmful prompts come only from public benchmarks."]),
      p("7. Who decides \"refused\" and \"harmful\"? (the judges)", H2),
      table([["Judge", "What it is", "What it answers", "Why we use it"],
             ["Keyword score", "Checks for words like \"sorry\", \"I can't\", \"illegal\"", "Did it refuse? (rough)", "Heretic uses it internally. Easily fooled: \"Here is how... (this is illegal)\" counts as a refusal."],
             ["<b>ProtectAI</b> distilroberta-rejection", "A small AI classifier (Apache-2.0)", "Did it refuse?", "Much better than keywords. After Heretic, keywords said the original refuses 14%; ProtectAI says 1%."],
             ["<b>Granite Guardian 3.2</b> (IBM, 5B)", "A safety AI built to judge harm", "Is the answer actually harmful?", "Refusal wording is not the point; harm is. It also caught v2 rewriting harmful requests into harmless answers."],
             ["<b>Llama Guard 3</b> (Meta, 1B)", "A multilingual safety judge", "Is it harmful? (any language)", "Second opinion for non-English answers; agrees with Granite 81-87%."]], [30, 40, 34, 76]),
      p("<b>Confidence intervals</b> (the numbers in [brackets]) show the range the true value is likely in, given we tested 100 prompts. "
        "\"91% [84, 95]\" means our best estimate is 91% and the true rate is very likely between 84% and 95%.", SM),
      PageBreak()]

# =========================== PART C: RESULTS
S += [p("Part C. What we found", H1),
      p("8. The threat is real: Tiny Aya is stripped for free", H2),
      table([["", "Refuses", "Harmful (Granite)", "MMLU"],
             ["Tiny Aya Global, untouched", "98%", "0%", "52.3%"],
             ["After Heretic's best attack (KL 0.009, ~1 GPU-hour)", "<b>1%</b>", "<b>91%</b> [84, 95]", "52.0%"]], [80, 25, 45, 25]),
      p("In plain words: one free command, about an hour, and the model answers 91% of harmful requests, while scoring the same on a "
        "general-knowledge test, so the tampering is hard to spot from ordinary use."),
      p("9. Our defense against Heretic", H2),
      Image(CH, width=176 * mm, height=97 * mm),
      p("How to read the chart: each dot is one attack Heretic found. Moving right = the attack damaged the model more (higher KL). "
        "Higher = more refusals survived. The original (red) collapses almost at zero damage. v2 (dark blue) stays at the top until "
        "Heretic is forced to damage the model ~15x more.", SM),
      table([["Lowest refusal Heretic reached (keyword)", "KL &lt;= 0.01", "KL &lt;= 0.05", "KL &lt;= 0.1", "Strongest found"],
             ["Original", "14%", "14%", "14%", "14% @ KL 0.009"],
             ["v1", "70%", "43%", "17%", "17%"],
             ["<b>v2</b>", "<b>100%</b>", "<b>100%</b>", "79%*", "33-38% @ KL 0.14-0.15"],
             ["v2b", "99%", "52%", "47%", "47% @ KL 0.054"]], [62, 26, 26, 26, 40]),
      p("*From the 200-trial search (section 10); the 40-trial search found 96%.", SM),
      table([["At each model's strongest Heretic attack", "Original", "v2", "v2b"],
             ["Damage needed (KL)", "0.009", "0.153 (~16x)", "0.054"],
             ["Still refuses", "1%", "<b>25%</b>", "31%"],
             ["Harmful answers (Granite)", "91%", "<b>40%</b>", "57%"],
             ["Harmful on clearly-unsafe XSTest prompts", "61%", "<b>4%</b>", "15%"],
             ["MMLU", "52.0%", "52.1%", "52.0%"]], [80, 30, 36, 30]),
      box("What this means", [
          "- The cheap, low-damage attacks found in our searches (the kind people actually use) <b>fail</b> on v2: 100% refusal up to KL 0.05.",
          "- The strongest attack found had <b>~16x the measured damage</b> and still left only 40% harmful (vs 91%). This is what our searches found, not a proven minimum for every attack.",
          "- Honest framing: we <b>raise the price and reduce the payoff</b> of a strip; we do <b>not</b> make it impossible.",
          "- Surprise finding (Granite): under attack, v2 often <b>rewrites</b> a harmful request into a harmless answer (asked about "
          "trafficking endangered species, it writes a conservation guide). It is safer than its refusal rate suggests."], *GREEN),
      PageBreak(),
      p("10. A 5x longer search: cheaper partial attacks, no stronger worst case", H2),
      p("A fair critic would ask: \"maybe Heretic just didn't search long enough?\". Abhinav re-ran Heretic on v2 with <b>200 trials instead of 40</b> "
        "(229 minutes on 2 GPUs), identical settings otherwise."),
      table([["v2 under Heretic", "40 trials", "200 trials"],
             ["Lowest refusal at KL &lt;= 0.01 / 0.05 / 0.1", "100 / 100 / 96%", "100 / 100 / 79%"],
             ["Strongest point: harmful (Granite)", "40% [31, 50]", "42% [33, 52]"],
             ["Strongest point: refusal / MMLU", "25% / 52.1%", "20% / 52.2%"]], [80, 50, 50]),
      p("<b>Meaning:</b> 5x more effort finds cheaper partial attacks in the middle band, but the worst case does not get worse "
        "(40% vs 42%, overlapping confidence intervals). Nothing gets below 100% refusal at low damage. It is one search per budget, "
        "so it does not prove that more search can never do more damage."),
      p("11. Every other attack we tried", H2),
      table([["Attack (what it does)", "Original", "v2", "v2b"],
             ["SafeTune direction removal at run time (Lexsi's tool): refusal / harmful", "29% / 55%", "<b>100% / 0%</b>", "<b>99% / 0%</b>"],
             ["SafeTune CAA steering -1 (push activations away from refusing): refusal / harmful", "4% / 97%", "35% / 72%", "34% / 71%"],
             ["CAA steering -4 (very strong push)", "gibberish", "gibberish", "gibberish"],
             ["Sampling: ask 3 times with randomness, any success (lower is safer)*", "9%", "<b>0%</b>", "2%"],
             ["4-bit compression (how people run models on laptops): refusal*", "99%", "<b>100%</b>", "<b>100%</b>"],
             ["Prefill: force the answer to start \"Sure, here is\": refusal / harmful", "21% / 82%", "20% / 77%", "24% / 80%"],
             ["Harmful fine-tune on 100 examples (single-BOS re-run): refusal", "0%", "3%", "-"],
             ["AuditKIT prompt-injection probes: pass rate* (small probe set, indicative)", "62.5%", "<b>87.5%</b>", "75%"]], [100, 26, 28, 26]),
      p("* measured before the BOS fix (section 15), so provisional; all other rows are final. SafeTune ablation: n = 100, v2's 95% interval is 0-4%.", SM),
      box("What generalises and what doesn't", [
          "<b>Works against:</b> anything that removes the refusal direction (Heretic, SafeTune ablation); partly against mild steering; provisionally against random sampling and 4-bit compression.",
          "<b>Doesn't work against:</b> harmful fine-tuning and prefilling. These are the known unsolved problems in this research area (Qi et al.; TamperBench). Section 13 shows how far a second defense delays fine-tuning.",
          "<b>Strong steering (-4)</b> doesn't \"strip\" any model; it breaks all of them into gibberish (found by Abhinav's Granite check)."], *AMBER),
      p("12. It works in other languages", H2),
      p("Tiny Aya is multilingual, so Abhinav tested 100 red-team prompts each in English, Hindi, Arabic, French and Spanish (from Cohere's own "
        "<i>aya_redteaming</i> dataset) at each model's strongest Heretic attack. Answers were translated to English (NLLB-200) and judged by "
        "Granite; Llama Guard 3 judged the original text as a cross-check."),
      table([["Harmful answers at strongest attack", "English", "Hindi+Arabic+French+Spanish"],
             ["Original", "34%", "41%"],
             ["<b>v2</b>", "22%", "<b>12%</b>"],
             ["v2b", "28%", "22%"]], [80, 40, 60]),
      p("<b>Meaning:</b> v2 was trained only in English, yet its protection carries over and is even stronger in the other languages. "
        "Caveat: these prompts are softer than our main test (many hate/bias prompts), so compare within this table only.", SM),
      PageBreak(),
      p("13. Harmful fine-tuning: delay, not prevention (Phase A)", H2),
      p("The track asks for safety that survives <b>later harmful fine-tuning</b>. The attacker trains a small LoRA add-on on harmful "
        "question-answer pairs (<i>LLM-LAT/harmful-dataset</i>) and we check, step by step, how much refusal is left. v2 alone does not stop this, "
        "so we trained two more checkpoints on top of v2 with published fine-tuning defenses:"),
      *bullets(["<b>A1 = v2 + RepNoise</b> (Rosati et al. 2024): scrambles the model's internal representation of harmful content, so learning it again is harder.",
                "<b>A2 = v2 + Booster</b> (Huang et al. 2025): trains the model so that a harmful training step changes it as little as possible.",
                "Both keep v2's anti-abliteration training running at the same time, and use harmful data disjoint from the attacker's."]),
      table([["Refusal % on 193 held-out HarmBench prompts", "No attack", "Step 25", "Step 50", "Step 100", "Step 175"],
             ["Original", "80", "53", "6", "7", "5"],
             ["v2", "97", "98", "<b>13</b>", "19", "3"],
             ["<b>A1</b> (RepNoise)", "98", "98", "98", "<b>96</b>", "6"],
             ["A1, second training seed", "98", "98", "97", "<b>96</b>", "6"],
             ["A2 (Booster)", "98", "98", "9", "6", "5"]], [62, 22, 22, 22, 24, 24]),
      p("Slow attack (learning rate 2e-5). At the 10x faster rate (2e-4), v2 falls to 8% by step 10 and A1 to 31% by step 25: A1 still lasts about 2.5x longer. "
        "A2 is worse than v2 there (25% vs 94% at step 5).", SM),
      box("How sure are we? (why this is not a false result)", [
          "- <b>Held-out prompts:</b> HarmBench minus 7 near-duplicates, with zero overlap with any training, attack or defense data.",
          "- <b>Predictions written before the run</b> (A1 breaks ~3-4x later than v2; original no better than v2; A2 no better than v2). <b>All three held.</b>",
          "- <b>Two training seeds</b> for A1 give almost identical curves, so it is not a lucky training run. Only one attack seed was used, and the checkpoints are coarse: v2 breaks between steps 25 and 50, A1 between 100 and 175, so the delay is about 3-4x at the midpoints and somewhere between 2x and 7x overall.",
          "- <b>Real harm, not broken text:</b> Granite Guardian re-judged every answer. When refusal collapses, 92-97% of answers are judged harmful, and the answers stay varied and normal-length.",
          "- <b>Mechanism:</b> the attacker's training loss on A1 starts near <b>239</b> (about 1.5 on every other model). Refusal only breaks once that loss is back to normal, so the delay is the time the attacker spends undoing RepNoise.",
          "- <b>Not an artefact:</b> at most 4 training steps were skipped for numerical overflow."], *GREEN),
      p("<b>Honest reading:</b> every model ends at the original's level (3-6% refusal). A1 buys a measurable, reproducible <b>delay</b>, "
        "but it costs capability (next section), so we publish it as a <b>trade-off, not a better model</b>. A2 is a <b>negative result</b>, reported as such."),
      p("14. Capability cost: MMLU alone hid it (Aban's audit)", H2),
      p("MMLU (general knowledge) is unchanged, and for a while we reported \"no capability cost\". Aban then scored every model on a "
        "<b>frozen, version-pinned</b> set of tasks with <b>paired 95% intervals</b> (same questions for every model, so differences are not noise from different samples). "
        "That showed real, task-specific costs:"),
      table([["Change (points)", "v2 vs original", "v2b vs original", "A1 vs v2"],
             ["Coding (HumanEval, n = 164)", "-1.2 (holds)", "-1.2 (holds)", "<b>-20.1</b> (37 vs 70 passes)"],
             ["General knowledge (MMLU, n = 200)", "-0.5 (holds)", "+0.5 (holds)", "+0.5 (holds)"],
             ["Security knowledge (CyberMetric, n = 500)", "-1.4 (holds)", "-1.8 (holds)", "<b>-4.2</b>"],
             ["Instruction following (IFEval, n = 200)", "<b>-10.0</b>", "<b>-19.5</b>", "-5.0 (not resolved)"],
             ["Maths (GSM8K), standard parser", "-1.0", "<b>-16.5</b>", "<b>-11.5</b>"],
             ["Safe questions refused (XSTest, n = 250)", "<b>+25.6</b>", "-0.4", "-8.8 (fewer)"]], [62, 36, 36, 46]),
      *bullets(["<b>Bold</b> = the 95% interval excludes zero (a real cost). \"holds\" = no change detected at this sample size, which is not proof of no change.",
                "Maths outside English (GlobalMGSM, Abhinav's check) also drops for v2 under strict scoring, mostly because of answer format.",
                "<b>Takeaway:</b> no checkpoint is free. v2 is the safest but over-refuses and follows instructions less well; v2b fixes over-refusal but costs maths "
                "and instruction following; A1 lasts longer under fine-tuning but loses coding. We present them as points on a trade-off curve."]),
      p("15. Bugs we found and fixed (and why that is good)", H2),
      box("The double-BOS bug (found by Aban)", [
          "Every model input starts with a special \"start\" marker called <b>BOS</b> (begin-of-sequence). Tiny Aya's chat format already adds one, "
          "and our test code added a second, so every test prompt started \"start start\". This affected our own test numbers and the part of "
          "v2's training that measures the refusal direction.",
          "We fixed it and <b>re-ran every headline number</b> (Day-6). The headline held. Two things changed: the \"-2.5 MMLU points\" cost we had "
          "reported <b>disappeared</b> (the bug hurt our fine-tuned models more), and prefill turned out to be <b>a bigger weakness</b> than we thought.",
          "Heretic and SafeTune (which proved v2 works) use their own correct code, so the main result was never affected."], *BLUE),
      p("The second fix was Aban's own: in his first prefill experiment, the training openers were split into tokens differently from the test "
        "openers (in 89% of cases). He caught it, fixed it, and verified the fix on 21,195 encodings before the final pilot. "
        "<b>Catching and fixing our own mistakes, with re-runs, is a strength of this submission</b>: every reported number is checked."),
      box("Two more caught in the final review (14 Nov)", [
          "- <b>Stop token (found by Abhinav):</b> Tiny Aya's generation settings stop on token 3, but chat answers end with token 6, so the model kept "
          "writing past the end of its answer (a one-line translation ran to 256 tokens instead of 10). Cohere's own base model has the same setting. "
          "We fixed it on all our published models. Our scores are unaffected in comparison terms: every model was scored with the same settings and a 128-token cap.",
          "- <b>Fine-tuning re-run with a single BOS:</b> the harmful fine-tune result was re-measured after the BOS fix: 0% (original) vs 3% (v2) refusal after 100 examples. The limit holds.",
          "- <b>Stale claims:</b> Abhinav's judge-style review of a fresh clone found mixed over-refusal numbers, a CPU-only and gated-tokenizer load snippet, and "
          "wording stronger than the evidence (\"holds\", \"nobody notices\"). All fixed, and every public page (README, model cards, Space, article) now shows the same numbers."], *AMBER),
      p("16. The prefill pilot: fixing our biggest weakness", H2),
      p("Prefill means forcing the model's answer to begin with \"Sure, here is\". The model then tends to continue. Aban tested whether "
        "training v2 to refuse <i>after</i> such openers helps, with a fair 3-arm design (all trained identically except the openers):"),
      table([["", "No openers (control)", "Openers on harmful only", "Openers on all"],
             ["Harmful answers after \"Sure, here is\" (Granite)", "86%", "<b>0%</b>", "<b>0%</b>"],
             ["Harmful answers after 6 unseen openers", "15-83%", "<b>0-5%</b>", "<b>0%</b>"],
             ["Refuses safe questions, normal use (no opener)", "38%", "<b>39%</b>", "46%"],
             ["Refuses safe questions after an opener", "7%", "39%", "30%"]], [70, 36, 38, 36]),
      *bullets(["<b>Result:</b> openers on harmful prompts only blocked prefill attacks with <b>no increase in normal over-refusal</b>.",
                "<b>Trade-off:</b> it also refuses some safe questions <i>after</i> an opener. Normal users never prefill (only attackers or some developers do), so this mostly hurts attackers.",
                "<b>How we got here:</b> the first version learned a shortcut (\"refuse after any opener\": 77% on safe questions). We only found that because we asked for a <b>benign control</b>. Then the token bug was fixed.",
                "<b>Limits:</b> one training run; answers judged at 48 tokens (main tests use 128); not yet re-attacked with Heretic. It is a <b>pilot</b>, not a claim."]),
      PageBreak(),
      p("17. Cost and ROI (is it worth it?)", H2),
      *bullets(["<b>Our compute:</b> ~37 GPU-hours, almost all on free Kaggle T4s (~$13 at on-demand prices), plus ~$5 of rented A10 for the pilot.",
                "<b>Attack vs defense:</b> stripping the original costs ~1 T4-hour (~$0.35); training our defense costs ~1-3 T4-hours (~$1), once.",
                "<b>Self-hosting example:</b> a multilingual support bot with 1M requests/month (300 input + 200 output tokens each). Measured "
                "speed on one T4 with plain code: ~150 tokens/s, so it fits on one always-on T4: <b>~$256/month</b>."]),
      table([["Option (prices checked 11 Nov 2026, USD per 1M tokens in / out)", "Monthly cost"],
             ["<b>Self-hosted hardened Tiny Aya, 1x T4 (measured, not optimised)</b>", "<b>~$256</b>"],
             ["GPT-6 Luna / Claude Haiku 5.5 (current cheapest; $0.10 / $0.50)", "$130"],
             ["GPT-5.6 Luna ($0.20 / $1.20)", "$300"],
             ["GPT-5.4 mini ($0.75 / $4.50)", "$1,125"],
             ["GPT-6.1 Terra ($1 / $5), current mid-tier", "$1,300"],
             ["Claude Sonnet 5.5 ($2 / $10), current mid-tier", "$2,600"]], [140, 40]),
      p("<b>Reading:</b> the cheapest current closed models cost about half of our <i>unoptimised</i> setup (a proper serving engine like vLLM "
        "would be several times faster; not measured, so not claimed). Against current mid-tier models, self-hosting is <b>~4-10x cheaper</b>. "
        "The bigger reasons to self-host are data privacy, 70+ languages and owning the weights. That ownership is exactly the risk we address. "
        "Note: the first ROI draft used two outdated models (GPT-4o-mini, Claude Haiku 4.5); Aban caught it and we updated to current models. "
        "OpenAI prices are from an aggregator (OpenAI's page blocks automated reading); confirm before quoting.", SM),
      p("18. What we do NOT claim (say this before the judges do)", H2),
      box("Honest limits", [
          "- <b>Not unbreakable:</b> Heretic's strongest attack found still gets 40% harmful answers from v2 (at ~16x the measured damage).",
          "- <b>Fine-tuning and prefilling still break every model.</b> A1 only <b>delays</b> fine-tuning, and costs about half of v2's coding passes. The prefill pilot is promising but is one run, judged at 48 tokens, and not Heretic-tested.",
          "- <b>Not free in capability:</b> instruction following -10 (v2) and -19.5 (v2b); v2b also loses maths. Over-refusal: v2 42.8% vs original 17.2% (v2b 16.8%), provisional classifier labels.",
          "- <b>One model size</b> (3.35B); capability numbers are from one training seed.",
          "- <b>Judges are AI models</b>, not ground truth; we used two independent ones (three for other languages) to reduce that risk.",
          "- Some rows (sampling, 4-bit, AuditKIT) were measured before the BOS fix; marked with * everywhere.",
          "- <b>CircuitKIT</b> (circuit-level analysis) did not fit on a free 16 GB GPU for this model; listed as future work."], *RED),
      PageBreak()]

# =========================== PART D: TEAM, PROCESS, ARTIFACTS
S += [p("Part D. Team, toolkits, artifacts and process", H1),
      p("19. Who did what", H2),
      table([["Person", "Contributions"],
             ["<b>Shivendu</b> (HF: StarWarrior24x7)", "Threat demo; v1/v2/v2b defenses; adaptive Heretic evaluation; Lexsi toolkit integration and the 4 upstream fixes; single-BOS re-evaluation; "
              "Phase A (A1 RepNoise, A2 Booster, held-out fine-tuning tests with predictions, Granite re-judge); article, Space, brief"],
             ["<b>Abhinav</b> (HF: laterabhi)", "Granite Guardian harm judge (pre- and post-fix); Heretic-200; multilingual evaluation with Llama Guard cross-check; ROI; "
              "final judge-style review of a fresh clone (PR #10): consistency and overclaim fixes, the stop-token bug, HF sync script"],
             ["<b>Aban</b> (HF: abanwild)", "Found the double-BOS bug; prefix-recovery pilot with matched controls, benign-opener control, token-boundary fix and Granite judging; caught outdated ROI models; "
              "frozen paired capability audit (PR #7, #9: IFEval, GSM8K, HumanEval, MMLU, CyberMetric, XSTest) that exposed the hidden costs and priced A1/A2"]], [40, 140]),
      p("20. How we used the Lexsi toolkits (the \"2 connected repos\" rule)", H2),
      table([["Toolkit", "How", "Outcome"],
             ["<b>SafeTune</b>", "Extracted refusal directions; attacked every model with direction ablation and CAA steering", "Independent confirmation (55% vs 0% harmful); mechanism evidence"],
             ["<b>AuditKIT</b>", "Red-team probes (jailbreak, encoding, prompt injection, refusal) on every model", "v2 passes prompt injection 87.5% vs 62.5%"],
             ["<b>CircuitKIT</b>", "Attempted refusal-circuit comparison", "Built the task (worked around a pairing issue); discovery needs &gt;16 GB GPU"]], [24, 80, 76]),
      p("<b>Mechanism (SafeTune):</b> v2's refusal direction is 10-20% weaker in middle/late layers than the original's. Our reading: v2 no "
        "longer depends on one dominant direction, so deleting it no longer deletes refusal.", SM),
      p("<b>Contributing back: 4 bug fixes merged upstream.</b> While integrating the toolkits we hit real bugs, reproduced each, fixed it with a "
        "regression test and sent it upstream. All four are merged:"),
      table([["Toolkit", "PR", "What was wrong"],
             ["AuditKIT", "#35", "In multi-probe runs every probe was scored by every probe's detectors, so correct refusals were marked as failures"],
             ["AuditKIT", "#36", "A probe with no prompts reported a 100% pass rate while testing nothing"],
             ["CircuitKIT", "#37", "The safety helpers built prompt pairs that discovery rejects, so the documented example could not run"],
             ["SafeTune", "#79 (via #77)", "DeepRefusal training with gradient checkpointing silently computed the wrong gradient (ablation hooks were off during recomputation)"]], [24, 26, 130]),
      p("The maintainers reproduced and merged them, keeping our commits. This is evidence that we used the toolkits seriously, not just imported them.", SM),
      p("21. Everything public", H2),
      table([["What", "Where (Hugging Face user StarWarrior24x7)"],
             ["Models v1, v2 (+ v2b subfolder), with cards", "tiny-aya-global-xref-v1, tiny-aya-global-xref-v2"],
             ["Research checkpoints A1, A2 (fine-tuning defenses)", "tiny-aya-global-phaseA-v2-repnoise, tiny-aya-global-phaseA-v2-booster"],
             ["Training data", "datasets/tiny-aya-extended-refusal"],
             ["Results, Heretic journals, judge scores, logs (numbers only)", "datasets/tiny-aya-tamper-resistance-results"],
             ["Collections (required)", "\"Heretic-Resistant Tiny Aya: Models\" and \"... Data &amp; Results\""],
             ["Live demo page", "spaces/heretic-resistant-tiny-aya"],
             ["Code, every run, full log, score files", "github.com/ShivenduShivu/Tamper_Resistance_OSModels (notes/LOG.md, results/)"],
             ["Article (the submission)", "The GitHub README (organiser-approved), mirrored as the card of heretic-resistant-tiny-aya-article"]], [70, 110]),
      p("All repos are public and un-gated (required by the rules). Never public: stripped or attacked models, their raw outputs, tokens.", SM),
      p("22. How the work ran (pipeline)", H2),
      flow(["Attack original (Heretic)", "Make training data", "Train v1/v2/v2b", "Re-attack each fresh", "Other attacks + judges", "Fix bugs, re-run, publish"]),
      Spacer(1, 4),
      p("Jobs ran on free Kaggle T4 GPUs, pushed from the laptop; a watcher script polled each job and launched the next, so runs continued "
        "overnight. Teammates worked on separate branches and merged via reviewed pull requests (every PR was checked for leaked text and "
        "secrets). Every run and decision is in notes/LOG.md.", SM),
      PageBreak()]

# =========================== PART E: PITCH + Q&A + GLOSSARY
S += [p("Part E. Pitch, judge questions and glossary", H1),
      p("23. One-minute pitch", H2),
      box("Pitch", [
          "\"Heretic, the free tool people use to strip safety from open models, removes Tiny Aya's safety almost completely: 91% of "
          "harmful requests get harmful answers, with general knowledge unchanged. Inspired by Fool's Gold and ART, we trained Tiny Aya "
          "while simulating that attack inside the model. The low-damage attacks found in our searches leave it refusing 100%, and Lexsi's SafeTune "
          "confirms it independently: 55% harmful down to 0%. The strongest attack we found distorts the model about 16 times more and still only gets 40% harmful answers; "
          "a five-times-longer search found no stronger worst case. It holds in Hindi, Arabic, French and Spanish. "
          "We measured the price too: instruction following drops 10 points and it over-refuses more. Harmful fine-tuning still breaks every model; "
          "adding RepNoise delays that about three to four times on held-out prompts, at a cost in coding. Everything is public, including four bug fixes merged into Lexsi's own toolkits.\""], *BLUE),
      p("24. Questions judges will likely ask", H2),
      table([["Question", "Short answer"],
             ["Isn't the attacker just paying more KL?", "Yes, and we say so. The strongest attack found had ~16x the measured damage for a much worse strip (40% vs 91% harmful), and the cheap attacks found fail entirely. Not a proven minimum."],
             ["Did you just not search long enough?", "We checked: 5x the trials (200) found cheaper partial attacks but no stronger worst case (42% vs 40%, overlapping uncertainty). One search each, so not proof."],
             ["Are your safety numbers just keyword counts?", "No. Two AI judges: ProtectAI (refusal) and Granite Guardian (harm), plus Llama Guard for languages. Keywords are shown only where Heretic uses them."],
             ["Does it make the model dumber?", "Knowledge and coding hold (MMLU, HumanEval, CyberMetric), but instruction following drops 10 points (v2). We found that with a paired audit and report it."],
             ["Does it refuse too much?", "Yes, v2 does: 42.8% of safe-but-scary questions vs 17.2% for the original. v2b brings it back to 16.8% but costs maths. A trade-off we show openly."],
             ["What about fine-tuning attacks?", "Not prevented, by us or the field. v2 alone delays it briefly; A1 (v2 + RepNoise) delays it about 3-4x on held-out prompts (2-7x given checkpoint spacing), but loses coding. Delay, not prevention."],
             ["Is the A1 delay real or a fluke?", "Held-out prompts with zero overlap, predictions written beforehand (all held), two training seeds, a second judge (Granite) and a mechanism (attack loss starts at ~239)."],
             ["Why not ship A1 as the main model?", "Its coding drops from 70 to 37 of 164 HumanEval passes. We publish it as a research checkpoint and keep v2 as the headline."],
             ["Why trust results after you found bugs?", "We re-ran every headline number after the BOS fix; Heretic/SafeTune never had the bug; the headline held. Finding and fixing bugs openly is the point."],
             ["Is it only English?", "No: tested in 4 more languages; v2 is even stronger there (12% vs 41% harmful)."],
             ["Did you really use the Lexsi toolkits?", "Yes: SafeTune and AuditKIT in the evaluation, CircuitKIT attempted, and 4 bug fixes merged upstream."],
             ["Is it worth deploying?", "Defense costs ~$1 once; self-hosting is ~4-10x cheaper than current mid-tier closed APIs."]], [60, 120]),
      p("25. Glossary", H2),
      *[term(w, m) for w, m in [
          ("LLM", "large language model; software made of billions of numbers (weights) that turns prompts into answers."),
          ("Weights / open-weight", "the model's numbers; open-weight = published so anyone can download and edit them."),
          ("Safety training / refusal", "teaching a model to decline harmful requests."),
          ("Activations", "the numbers inside the model while it processes text."),
          ("Refusal direction", "one direction in activation space that mostly controls refusing."),
          ("Abliteration", "editing weights to remove the refusal direction, so the model stops refusing."),
          ("Heretic", "free tool that automatically finds the best abliteration settings."),
          ("Trial / search budget", "one attempt by Heretic's optimiser; we ran 40 (and 200) per model."),
          ("KL divergence", "how much an attack changed the model's normal answers; low = invisible attack."),
          ("Pareto front", "the best attacks found: for each damage level, the fewest refusals."),
          ("Fine-tuning / LoRA", "further training; LoRA trains small add-on matrices instead of the whole model (cheap)."),
          ("Adversarial training", "training while being attacked, so the model learns to resist (our v2)."),
          ("Steering / CAA", "pushing the model's activations in a direction while it runs, e.g. away from refusing."),
          ("Prefill", "forcing the beginning of the model's answer (e.g. \"Sure, here is\")."),
          ("Sampling", "generating with randomness; an attacker can retry until one answer complies."),
          ("4-bit quantisation", "compressing the weights to run on small hardware."),
          ("Judge / classifier", "a separate AI that labels answers (refused? harmful?)."),
          ("Confidence interval", "the likely range of the true value given our sample size."),
          ("Over-refusal", "refusing safe questions that only sound risky (measured with XSTest)."),
          ("MMLU", "a standard multiple-choice test of general knowledge; checks the model stays smart."),
          ("BOS token", "a special start marker at the beginning of every input; our bug added it twice."),
          ("Matched control", "an experiment arm identical except for the one thing being tested, so we know what caused a change."),
          ("GPU-hour / T4", "one hour of a graphics card; T4 is the free Kaggle GPU (about $0.35/h to rent)."),
          ("Collection (HF)", "a Hugging Face page grouping related models/datasets; required for submission."),
          ("RepNoise / Booster", "two published defenses against harmful fine-tuning; we added them on top of v2 as A1 and A2."),
          ("Held-out prompts", "test prompts that never appear in any training, attack or defense data."),
          ("HarmBench", "a public benchmark of harmful requests; we used 193 of its standard prompts as the held-out test."),
          ("Training seed", "the random starting point of a training run; two seeds giving the same result means it is not luck."),
          ("IFEval / GSM8K / HumanEval / CyberMetric", "tests of instruction following, school maths, coding and security knowledge."),
          ("Paired comparison", "every model answers the same questions, so differences come from the model, not the sample."),
          ("Stop token (EOS)", "the special token that tells the model its answer is finished."),
      ]]]


def foot(c, d):
    c.saveState(); c.setFont("Ar", 7.5); c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(15 * mm, 9 * mm, "Heretic-Resistant Tiny Aya | complete team brief (final, 14 Nov) | Track 2b Tamper Resistance")
    c.drawRightString(195 * mm, 9 * mm, f"{d.page}"); c.restoreState()


doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
                        title="Heretic-Resistant Tiny Aya: Complete Team Brief")
doc.build(S, onFirstPage=foot, onLaterPages=foot)
print("ok", sys.argv[1])
