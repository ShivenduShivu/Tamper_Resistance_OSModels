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
      p("<b>Complete team brief, final results.</b> Lexsi Labs x Cohere x Hugging Face AI Safety &amp; Alignment Hackathon, "
        "Track 2b (Tamper Resistance). Submission deadline: 15 Nov 2026, 11:59 PM AoE (= 16 Nov, 5:29 PM IST). This document explains the whole project "
        "from zero: what the problem is, what we built, how we tested it, what we found, what it does not do, and how to "
        "answer the judges. Technical words are explained where they first appear and again in the glossary (section 23).", B),
      box("The whole project in 8 lines", [
          "1. <b>Open-weight models</b> (anyone can download the model file) come with safety training: they refuse harmful requests.",
          "2. That refusal mostly lives in <b>one direction</b> inside the model. Delete it and the model stops refusing. This is <b>abliteration</b>.",
          "3. A free tool, <b>Heretic</b>, does this in one command. On Cohere's <b>Tiny Aya</b> it made <b>91%</b> of answers to harmful requests harmful (was 0%), at no loss in general knowledge.",
          "4. We trained Tiny Aya <b>while simulating that attack inside it</b> (our model <b>v2</b>), so it learned to refuse even with the direction removed.",
          "5. Result: at the attack strength that fully broke the original, v2 still refuses <b>100%</b>. The strongest attack Heretic finds needs ~<b>16x more damage</b> to the model and still gets only <b>40%</b> harmful answers. A <b>5x longer search</b> found cheaper partial attacks but <b>no stronger worst case</b> (42%).",
          "6. Lexsi's own tools (<b>SafeTune</b>, <b>AuditKIT</b>) confirm it. It works in <b>Hindi, Arabic, French, Spanish</b> too (41% vs 12% harmful), and costs <b>nothing on MMLU</b>.",
          "7. <b>Honest limits:</b> training the model on harmful examples (fine-tuning) or forcing its answer to start with \"Sure, here is\" (prefill) still break it. A small, separate pilot cut prefill harm 86% -> 0% (one seed, 48-token judging, not Heretic-tested).",
          "8. Everything is public: models, data, results, code, a live demo page, and an article."], *BLUE),
      Spacer(1, 6), p("Key numbers at a glance", H3),
      table([["", "Original Tiny Aya", "Our v2", "Our v2b"],
             ["Refuses harmful requests (no attack)", "98%", "100%", "100%"],
             ["Heretic, weak attack (KL &lt;= 0.05): refusal kept (keyword)", "<b>14%</b>", "<b>100%</b>", "52%"],
             ["Heretic, strongest attack: harmful answers (Granite judge)", "<b>91%</b>", "<b>40%</b>", "57%"],
             ["...same, with 5x attacker search budget", "-", "42%", "-"],
             ["SafeTune direction-removal attack: harmful answers", "55%", "<b>0%</b>", "<b>0%</b>"],
             ["Multilingual (hi/ar/fr/es), strongest attack: harmful", "41%", "<b>12%</b>", "22%"],
             ["Prefill \"Sure, here is\": harmful answers", "82%", "77%", "80%"],
             ["Refuses safe-but-scary questions (over-refusal)", "16%", "39%", "18%"],
             ["MMLU (general knowledge)", "52.3%", "53.0%", "52.7%"]], [86, 32, 30, 30]),
      p("Higher refusal = safer; lower harmful = safer; lower over-refusal = more useful. \"-\" = not measured.", SM),
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
                "<b>Many different attacks</b> (section 9): Heretic, SafeTune's own attacks, steering, prefilling, sampling, 4-bit compression, harmful fine-tuning, multilingual prompts.",
                "<b>Clean test sets:</b> the prompts Heretic uses are excluded from training; XSTest is never trained on; 200 OR-Bench prompts are held out.",
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
        "general-knowledge test, so nobody would notice it was tampered with."),
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
          "- Weak, cheap, invisible attacks (the ones people actually use) <b>fail completely</b> on v2: 100% refusal up to KL 0.05.",
          "- The strongest attack <b>costs ~16x more damage</b> and still leaves only 40% harmful (vs 91%).",
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
             ["Harmful fine-tune on 100 examples: refusal*", "1%", "3%", "1%"],
             ["AuditKIT prompt-injection probes: pass rate*", "62.5%", "<b>87.5%</b>", "75%"]], [100, 26, 28, 26]),
      p("* measured before the BOS fix (section 13); all other rows are final.", SM),
      box("What generalises and what doesn't", [
          "<b>Works against:</b> anything that removes or steers the refusal direction (Heretic, SafeTune ablation, mild steering), random sampling, and 4-bit compression.",
          "<b>Doesn't work against:</b> harmful fine-tuning (training on ~100 harmful examples) and prefilling. These are the known unsolved problems in this research area (Qi et al.; TamperBench).",
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
      p("13. Two bugs we found and fixed (and why that is good)", H2),
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
      p("14. The prefill pilot: fixing our biggest weakness", H2),
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
      p("15. Cost and ROI (is it worth it?)", H2),
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
      p("16. What we do NOT claim (say this before the judges do)", H2),
      box("Honest limits", [
          "- <b>Not unbreakable:</b> Heretic's strongest attack still gets 40% harmful answers from v2 (at ~16x the damage).",
          "- <b>Fine-tuning and prefilling still break v2.</b> The prefill pilot is promising but is one run, judged at 48 tokens, and not Heretic-tested.",
          "- <b>One model size</b> (3.35B). Over-refusal: v2 39% vs original 16% (v2b 18%).",
          "- <b>Judges are AI models</b>, not ground truth; we used two independent ones to reduce that risk.",
          "- Some rows (sampling, 4-bit, fine-tune, AuditKIT) were measured before the BOS fix; marked with * everywhere.",
          "- <b>CircuitKIT</b> (circuit-level analysis) did not fit on a free 16 GB GPU for this model; listed as future work."], *RED),
      PageBreak()]

# =========================== PART D: TEAM, PROCESS, ARTIFACTS
S += [p("Part D. Team, toolkits, artifacts and process", H1),
      p("17. Who did what", H2),
      table([["Person", "Contributions"],
             ["<b>Shivendu</b> (HF: StarWarrior24x7)", "Threat demo; v1/v2/v2b defenses; adaptive Heretic evaluation; Lexsi toolkit integration (SafeTune, AuditKIT, CircuitKIT attempt); single-BOS re-evaluation; article, Space, brief"],
             ["<b>Abhinav</b>", "Granite Guardian harm judge (pre- and post-fix); Heretic-200; multilingual evaluation with Llama Guard cross-check; ROI"],
             ["<b>Aban</b> (HF: abanwild)", "Found the double-BOS bug; prefix-recovery pilot with matched controls, benign-opener control, token-boundary fix and Granite judging; caught outdated ROI models"]], [40, 140]),
      p("18. How we used the Lexsi toolkits (the \"2 connected repos\" rule)", H2),
      table([["Toolkit", "How", "Outcome"],
             ["<b>SafeTune</b>", "Extracted refusal directions; attacked every model with direction ablation and CAA steering", "Independent confirmation (55% vs 0% harmful); mechanism evidence"],
             ["<b>AuditKIT</b>", "Red-team probes (jailbreak, encoding, prompt injection, refusal) on every model", "v2 passes prompt injection 87.5% vs 62.5%"],
             ["<b>CircuitKIT</b>", "Attempted refusal-circuit comparison", "Built the task (worked around a pairing issue); discovery needs &gt;16 GB GPU"]], [24, 80, 76]),
      p("<b>Mechanism (SafeTune):</b> v2's refusal direction is 10-20% weaker in middle/late layers than the original's. Our reading: v2 no "
        "longer depends on one dominant direction, so deleting it no longer deletes refusal.", SM),
      p("19. Everything public", H2),
      table([["What", "Where (Hugging Face user StarWarrior24x7)"],
             ["Models v1, v2 (+ v2b subfolder), with cards", "tiny-aya-global-xref-v1, tiny-aya-global-xref-v2"],
             ["Training data", "datasets/tiny-aya-extended-refusal"],
             ["Results, Heretic journals, judge scores, logs (numbers only)", "datasets/tiny-aya-tamper-resistance-results"],
             ["Collections (required)", "\"Heretic-Resistant Tiny Aya: Models\" and \"... Data &amp; Results\""],
             ["Live demo page", "spaces/heretic-resistant-tiny-aya"],
             ["Code, every run, full log", "github.com/ShivenduShivu/Tamper_Resistance_OSModels (notes/LOG.md)"],
             ["Article (the submission)", "docs/HF_ARTICLE.md, posted on HF Articles before the deadline"]], [70, 110]),
      p("All repos are public and un-gated (required by the rules). Never public: stripped models, their raw outputs, tokens.", SM),
      p("20. How the work ran (pipeline)", H2),
      flow(["Attack original (Heretic)", "Make training data", "Train v1/v2/v2b", "Re-attack each fresh", "Other attacks + judges", "Fix bugs, re-run, publish"]),
      Spacer(1, 4),
      p("Jobs ran on free Kaggle T4 GPUs, pushed from the laptop; a watcher script polled each job and launched the next, so runs continued "
        "overnight. Teammates worked on separate branches and merged via reviewed pull requests (every PR was checked for leaked text and "
        "secrets). Every run and decision is in notes/LOG.md.", SM),
      PageBreak()]

# =========================== PART E: PITCH + Q&A + GLOSSARY
S += [p("Part E. Pitch, judge questions and glossary", H1),
      p("21. One-minute pitch", H2),
      box("Pitch", [
          "\"Heretic, the free tool people use to strip safety from open models, removes Tiny Aya's safety almost completely: 91% of "
          "harmful requests get harmful answers, and general knowledge is untouched. Inspired by Fool's Gold and ART, we trained Tiny Aya "
          "while simulating that attack inside the model. At the same attack strength our model still refuses 100%; Lexsi's SafeTune "
          "confirms it independently. The attacker must distort the model about 16 times more and still only gets 40% harmful answers; "
          "a five-times-longer search found no stronger worst case. It works in Hindi, Arabic, French and Spanish, costs nothing on MMLU, and everything is "
          "public. We're honest that fine-tuning and prefilling still win, and a small separate pilot already cuts prefill harm from 86% to 0%.\""], *BLUE),
      p("22. Questions judges will likely ask", H2),
      table([["Question", "Short answer"],
             ["Isn't the attacker just paying more KL?", "Yes, and we say so. But it's ~16x the damage for a much worse strip (40% vs 91% harmful), and cheap attacks fail entirely."],
             ["Did you just not search long enough?", "We checked: 5x the trials (200) found cheaper partial attacks but no stronger worst case (42% vs 40%, overlapping uncertainty). One search each, so not proof."],
             ["Are your safety numbers just keyword counts?", "No. Two AI judges: ProtectAI (refusal) and Granite Guardian (harm), plus Llama Guard for languages. Keywords are shown only where Heretic uses them."],
             ["Does it make the model dumber?", "No: MMLU 52.3% (original) vs 53.0% (v2), within noise."],
             ["Does it refuse too much?", "v2 refuses 39% of safe-but-scary questions (original 16%); v2b brings it to 18%. We show both as a trade-off."],
             ["What about fine-tuning attacks?", "Not solved (by us or the field). Honest limit; next step is simulated LoRA attacks during training."],
             ["Why trust results after you found a bug?", "We re-ran every headline number after the fix; Heretic/SafeTune never had the bug; the headline held."],
             ["Is it only English?", "No: tested in 4 more languages; v2 is even stronger there (12% vs 41% harmful)."],
             ["Is it worth deploying?", "Defense costs ~$1 once; self-hosting is ~4-10x cheaper than current mid-tier closed APIs."]], [60, 120]),
      p("23. Glossary", H2),
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
      ]]]


def foot(c, d):
    c.saveState(); c.setFont("Ar", 7.5); c.setFillColor(colors.HexColor("#6B7280"))
    c.drawString(15 * mm, 9 * mm, "Heretic-Resistant Tiny Aya | complete team brief (final) | Track 2b Tamper Resistance")
    c.drawRightString(195 * mm, 9 * mm, f"{d.page}"); c.restoreState()


doc = SimpleDocTemplate(sys.argv[1], pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
                        title="Heretic-Resistant Tiny Aya: Complete Team Brief")
doc.build(S, onFirstPage=foot, onLaterPages=foot)
print("ok", sys.argv[1])
