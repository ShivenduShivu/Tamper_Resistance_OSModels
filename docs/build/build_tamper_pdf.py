import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

OUT = sys.argv[1]
ss = getSampleStyleSheet()
ACC = colors.HexColor("#B4361A"); NAVY = colors.HexColor("#1F2A44")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=19, leading=23, spaceAfter=4, textColor=ACC)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=NAVY)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=2, textColor=NAVY)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=10.6, spaceAfter=0)
BL = ParagraphStyle("BL", parent=B, leftIndent=11, spaceAfter=2)
BOX = ParagraphStyle("BOX", parent=B, fontSize=9.4, leading=12.8)

def p(t, s=B): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BL, bulletText="•") for i in items]
def table(rows, widths):
    data = [[Paragraph(f"<font color='white'><b>{c}</b></font>", SM) for c in rows[0]]]
    data += [[p(c, SM) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9BEC9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t
def box(title, lines, bg="#FFF4EC", border="#B4361A"):
    content = [Paragraph(f"<b>{title}</b>", BOX)] + [Paragraph(l, BOX) for l in lines]
    t = Table([[content]], colWidths=[180 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t
def flow(steps):
    cells = []
    for i, s in enumerate(steps):
        cells.append(Paragraph(f"<b>{s}</b>", ParagraphStyle("f", parent=SM, alignment=1)))
        if i < len(steps) - 1:
            cells.append(Paragraph("<b>→</b>", ParagraphStyle("a", parent=B, alignment=1, fontSize=13)))
    n = len(steps); w = (180 - 7 * (n - 1)) / n
    widths = []
    for i in range(len(cells)):
        widths.append((w if i % 2 == 0 else 7) * mm)
    t = Table([cells], colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i in range(0, len(cells), 2):
        st += [("BOX", (i, 0), (i, 0), 0.8, NAVY), ("BACKGROUND", (i, 0), (i, 0), colors.HexColor("#E8ECF5"))]
    t.setStyle(TableStyle(st))
    return t

S = []
# ---------------- Cover / TL;DR
S += [p("Tamper Resistance for Open-Weight LLMs", H1),
      p("A beginner's guide to the problem, the attacks, the defenses, the 2026 state of research, and what we can uniquely build "
        "for the Lexsi Labs hackathon. Prepared 9 Nov 2026. Words in <b>bold</b> on first use are explained in the Glossary (Section 11).", SM),
      Spacer(1, 6),
      box("The whole thing in 6 lines", [
          "1. When a company releases a model's <b>weights</b> openly, anyone can download and retrain it. A few minutes of retraining can strip out its safety.",
          "2. <b>Tamper resistance</b> means building safety that <i>survives</i> that retraining, which is called a <b>harmful fine-tuning attack</b>.",
          "3. Attackers have three levels of access: the prompt, the model's internal signals (<b>activations</b>), and the weights. Cheap tricks exist at every level.",
          "4. The best-known defense (TAR, 2025) and a dozen others have been broken or weakened by 2026 studies (TamperBench, Kuo et al.). No defense is solved.",
          "5. That is an opportunity. Honest, worst-case measurement plus a cheap new defense angle is publishable-quality work.",
          "6. Our angle (revised 9 Nov after checking real-world data): defend Tiny Aya against <b>Heretic</b>, the one-click abliteration tool people actually use "
          "(over 9,000 'abliterated' models on Hugging Face), and make tampering <b>detectable</b>.",
      ]),
      ]

# ---------------- 1. Problem
S += [p("1. The problem in plain words", H2),
      p("<b>Analogy.</b> Think of an AI model as a very knowledgeable employee who has been trained to say no to dangerous requests. "
        "With a closed model (like a chatbot website), the company controls the employee, so you can only talk to them. With an <b>open-weight</b> model, "
        "you get a full copy of the employee's brain. Anyone can retrain that copy with a handful of bad examples, and the "
        "\"say no\" habit disappears. Tamper resistance asks: <i>can we build the safety habit so deep that retraining the copy fails to remove it, "
        "while the model stays useful for honest retraining?</i>"),
      p("<b>Why it matters, with numbers.</b>"),
      ]
S += bullets([
    "Fine-tuning GPT-3.5 Turbo on just <b>10 harmful examples (cost under $0.20)</b> removed most of its safety (Qi et al., 2023).",
    "<b>LoRA</b> fine-tuning undid the safety of Llama-2-Chat 70B on one GPU for <b>under $200</b> (Lermen et al., 2023).",
    "About <b>100 examples and 1 GPU-hour</b> were enough to \"shadow-align\" many open models (Yang et al., 2023).",
    "Even <b>harmless</b> fine-tuning data erodes safety, and a few unusual (\"outlier\") benign samples can break it badly (Qi et al., 2023; arXiv 2505.06843).",
])
S += [p("The hackathon statement: <i>\"Make safety survive later harmful fine-tuning by someone else, and demonstrate it with attack fine-tunes.\"</i> "
        "The words <b>\"by someone else\"</b> matter. The attacker controls training, not us. This decides which defenses even qualify (next section).")]

# ---------------- 2. Threat models
S += [p("2. Two different threat models (the most common confusion)", H2),
      p("A <b>threat model</b> is a precise statement of what the attacker can and cannot do. Papers in this area use two, and mixing them up is the most common beginner mistake."),
      table([
          ["", "Fine-tuning-as-a-service", "Open-weight (our problem statement)"],
          ["Who runs training?", "The provider (e.g. OpenAI's fine-tuning API). The user only uploads data", "The attacker, on their own GPUs"],
          ["What can the defender do?", "Filter data, change the training recipe, check the result before serving", "Only shape the weights <i>before</i> release. After that, nothing"],
          ["Example defenses", "Lisa, SafeGrad, freezing safety layers, Antidote, ReSta re-patching", "TAR, RepNoise, Vaccine/Booster, SEAM, SDD, extended refusal, ART, Deep Ignorance"],
          ["Difficulty", "Manageable", "Very hard. Considered unsolved in 2026"],
      ], [34, 70, 76]),
      Spacer(1, 4),
      box("Important for us", [
          "SafeTune's <b>harden</b> methods (SafeGrad, LISA) and <b>recover</b> methods (ReSta etc.) are fine-tuning-as-a-service tools: the defender controls training. "
          "They do not by themselves answer the tamper-resistance statement. We can use them as <b>baselines</b> (\"what a normal safety method does under attack\"), "
          "but our defense must act before release, through the weights alone.",
      ], bg="#EEF2FA", border="#1F2A44")]

# ---------------- 3. Why easy to remove
S += [p("3. Why safety is so easy to remove: four scientific findings", H2)]
S += bullets([
    "<b>Safety is shallow.</b> Alignment mainly changes the <i>first few words</i> of an answer (\"I can't help with that\"). If those first words are forced or retrained, the rest flows as if no safety existed. "
    "(Qi et al., \"Safety Alignment Should Be Made More Than Just a Few Tokens Deep\", ICLR 2025.)",
    "<b>Refusal lives on one \"wire\".</b> Inside the model, refusing is controlled mostly by a single <b>refusal direction</b> in the activations. Remove that one direction and the model stops refusing, "
    "with no training at all (Arditi et al., 2024). This is the basis of <b>abliteration</b>.",
    "<b>Safety sits in a tiny part of the weights.</b> Roughly 3% of neurons (or a very low-rank slice of the weights) carry most safety behaviour. Pruning just those breaks safety while the model stays smart (Wei et al., 2024).",
    "<b>The harmful knowledge is still there.</b> Safety training mostly teaches the model to <i>refuse</i>, not to <i>forget</i>. The knowledge from pretraining remains, so attackers only need to unlock it, not teach it (Kuo, Yadav, Smith, 2026).",
])
S += [p("<b>Lesson for defenders:</b> a good defense must (a) make safety <i>deep</i> (many tokens, many directions, many layers), and/or (b) make relearning harmful behaviour "
        "<i>expensive or self-destructive</i>, and/or (c) actually remove the knowledge.")]

# ---------------- 4. Attacker playbook
S += [PageBreak(), p("4. The attacker's playbook: study it to counter it", H2),
      p("Attacks are grouped by how deep the attacker reaches. A 2026 ICML paper (\"Beyond Harmful Fine-Tuning\") argues every defense must be tested at "
        "<b>all three tiers</b>, because many defenses only consider Tier 3. Descriptions here are conceptual. The goal is to know what to defend against."),
      table([
          ["Attack", "How it works (plain words)", "Cost", "How defenders counter"],
          ["<b>Tier 1: Prompt (no weight access needed)</b>", "", "", ""],
          ["Jailbreak prompts", "Clever wording, role-play or encodings that trick the model into answering", "Very low", "Robust refusal training; input filters"],
          ["Prefilling", "With open weights you can write the first words of the answer yourself (\"Sure, here is...\"). This exploits shallow safety", "Near zero", "Deep safety alignment: train recovery from a bad start (Qi 2025)"],
          ["Low-resource language", "Ask in a language where safety training was thin. Translation alone raised attack success sharply on GPT-4 (Yong et al., 2023)", "Near zero", "Multilingual safety data"],
          ["<b>Tier 2: Activations / representations (no training)</b>", "", "", ""],
          ["Abliteration", "Find the refusal direction (average activation on harmful prompts minus average on harmless ones), then subtract it from the weights permanently", "Minutes, no GPU training", "Spread refusal across many directions: extended refusal (2025), ART (2026)"],
          ["Activation steering", "Push activations away from refusal while the model is generating", "Minutes", "Same as above; circuit breakers"],
          ["Embedding attack", "Optimise the input vectors directly instead of words", "Low", "Latent adversarial training (LAT)"],
          ["<b>Tier 3: Weights (training)</b>", "", "", ""],
          ["Harmful fine-tune (full or LoRA)", "Retrain on harmful Q&amp;A pairs. LoRA makes it cheap on one GPU", "Low", "TAR, RepNoise, Vaccine/Booster, SEAM, SDD"],
          ["Benign fine-tune", "Retrain on harmless data. Safety erodes as a side-effect, worst with outlier samples", "Low", "Booster, CRL; deep alignment"],
          ["Jailbreak-tuning", "Train in a trigger, style or \"competing objective\" so the model complies. TamperBench's strongest attack on 14 of 21 models", "Low", "Largely open problem"],
          ["Multilingual fine-tune", "Do the harmful retraining in another language to slip past English-trained defenses", "Low", "<b>Little work. Our opportunity</b>"],
          ["Model merging / task vectors", "Add an \"uncensored\" weight difference, or subtract the safety difference", "Very low", "Hard; detection is easier"],
          ["Quantization trick", "A model that looks safe in full precision becomes harmful once compressed (Egashira et al., 2024)", "Medium", "Test safety after quantization"],
          ["Pruning safety neurons", "Delete the small weight region that carries safety (Wei et al., 2024)", "Low", "Spread safety; redundancy"],
          ["<b>Adaptive attacker (the real enemy)</b>", "Sweep learning rate, steps, LoRA rank and data; combine tiers (abliterate, then fine-tune). Defenses tuned against one fixed attack usually fail here", "Low to medium", "Report <b>worst case</b> over sweeps, not one attack"],
      ], [33, 75, 22, 50]),
      Spacer(1, 3),
      p("<b>Pattern across attacks:</b> each one either removes the <i>refusal trigger</i> (prefill, abliteration, pruning) or <i>relearns compliance</i> (fine-tuning). "
        "Knowledge removal (Deep Ignorance) is the only defense that blocks both. Refusal-spreading is the cheapest counter to the first group.", SM)]

# ---------------- 5. Defenses
S += [PageBreak(), p("5. The defense map: what researchers have tried", H2),
      table([
          ["Family and idea", "Methods (year)", "Evidence", "Weakness", "Cost"],
          ["<b>A. Make harmful relearning hard.</b> Train against simulated attackers so their training gets stuck",
           "TAR (2024/ICLR'25); MLAC / self-destructing models (2023); AntiDote (2025)",
           "TAR resisted many fine-tuning attacks in its own tests", "Breaks under attack-setting sweeps (Qi 2024, TamperBench); large utility loss; results vary with random seed", "Very high (inner attack loops)"],
          ["<b>B. Scramble harmful representations.</b> Push the internal signals of harmful content toward noise",
           "RepNoise (2024); Vaccine, T-Vaccine, Booster (2024); CTRL, RSN-Tune", "Good against fixed-setting attacks", "TamperBench: mostly fail under sweeps; some destroy capability", "Medium"],
          ["<b>C. Make tampering self-destructive.</b> Harmful fine-tuning wrecks the model's general ability, so the stolen model is useless",
           "SEAM (2025); SDD (2025)", "Attacked models become low quality", "Abliteration bypasses SEAM: harmful output above 70% (Kuo 2026)", "Medium"],
          ["<b>D. Make refusal hard to cut out.</b> Spread refusal across many directions and tokens",
           "Extended refusal (2025); ART (2026); deep safety alignment (2025); ReFAT, LAT, circuit breakers (2024)",
           "Extended refusal kept refusal above 90% after abliteration; ART cut abliteration success from 93% to 49% on Gemma-1B", "Tested on small models, few attack types; fine-tuning attacks still break most", "<b>Low</b> (one supervised fine-tune)"],
          ["<b>E. Remove the knowledge itself.</b> Filter harmful data out of pretraining, or unlearn and distill",
           "Deep Ignorance (2025); RMU unlearning; distillation", "Deep Ignorance resisted about 10,000 steps / 300M tokens of adversarial fine-tuning, far beyond post-training methods", "Needs pretraining from scratch; does not stop help when harmful info is pasted into the prompt", "Very high"],
      ], [40, 34, 38, 44, 24]),
      Spacer(1, 4),
      p("<b>Reading the table:</b> family E is the most durable but needs pretraining, which is impossible in 5 days. Families A to C are expensive and were broken in 2026 audits. "
        "Family D is cheap, recent and matches the cheapest attack (abliteration). That is where a small team can contribute something new.")]

# ---------------- 6. State of the field
S += [p("6. State of the field (November 2026): the honest scoreboard", H2)]
S += bullets([
    "<b>TamperBench (KDD 2026)</b> tested 21 open models (up to 8B) against 9 attacks with 40 hyperparameter trials each. Every model had at least one attack pushing harmfulness above 0.74 on StrongREJECT (0 = safe, 1 = fully harmful) while utility was kept. "
    "Of 7 re-implemented defenses, most failed under sweeps. TAR lowered worst-case harm by only about 0.12, at a large utility cost. No defense held against full-strength LoRA. Open-source toolkit: github.com/criticalml-uw/TamperBench.",
    "<b>Kuo, Yadav, Smith (CMU, 2026):</b> TAR and SEAM are bypassed by abliteration and prefilling, with attack success of 16% to 96%. Their fix, ART, helps but does not reach the unattacked baseline.",
    "<b>\"Beyond Harmful Fine-Tuning\" (ICML 2026):</b> defenses that only consider fine-tuning attacks are tested too narrowly. They recommend tiered evaluation: prompt, representation and weight attacks.",
    "<b>Qi et al. (2024), \"On Evaluating the Durability of Safeguards\":</b> small changes to the attack (learning rate, data shuffling) flip published conclusions. Lesson: report sweeps and worst case.",
    "<b>Consensus:</b> no post-training defense gives durable tamper resistance against an adaptive attacker. The honest research goal is to <i>raise the attacker's cost</i> (steps, data, compute) and measure that cost carefully.",
])

# ---------------- 6b. Fool's Gold
S += [PageBreak(), p("6b. Case study: \"Fool's Gold\" (Russinovich, Microsoft, 2026), a different way to think about defense", H2),
      p("<b>Paper:</b> arXiv 2608.17202. <b>Code:</b> github.com/markrussinovich/fools-gold (MIT licence). <b>Project page:</b> markrussinovich.github.io/fools-gold"),
      p("<b>Big idea in one line:</b> stop trying to prevent refusal removal, because it will happen. Instead, make what the attacker gets afterwards <i>untrustworthy</i>. "
        "This is <b>defensive deception</b>, the same idea as a <b>honeypot</b> in cybersecurity: a trap that looks valuable but is fake."),
      p("<b>Analogy:</b> a thief cracks the safe (removes refusal) and finds gold bars, but the gold is painted lead. The thief can't tell by looking, and the stolen goods are worthless."),
      p("Core methodology, step by step", H3)]
S += bullets([
    "<b>Step 1: Attack your own model.</b> The defender abliterates their own model and records the real harmful answers it now gives. This shows exactly what an attacker would get.",
    "<b>Step 2: Write decoys.</b> Each real answer is rewritten to keep the same style, detail level and confidence, but with the <i>critical</i> details made false (a \"fatally wrong\" answer). "
    "Decoys are then checked for <b>tells</b> (giveaway patterns) that an attacker's filter could use to spot fakes, and fixed if found.",
    "<b>Step 3: Bind decoys to the attacked state.</b> The model is fine-tuned <i>inside a differentiable simulation of the abliteration attack</i>, meaning the training runs through a mathematically removed refusal direction. "
    "So the decoy behaviour appears only once refusal has been stripped. Two safeguards keep the normal model unchanged: a <b>refusal pin</b> (the normal model must still refuse) and a <b>benign KL term</b> "
    "(harmless answers must stay close to the original model's).",
    "<b>Step 4: Tighten on-policy.</b> Supervised fine-tuning first teaches the decoy behaviour. Then <b>preference optimization</b> on the model's own outputs in the attacked state reduces the share of answers that are still truthful. "
    "Training stops at pre-registered <b>gates</b>, fixed in advance, for benign behaviour and capability. This avoids cherry-picking.",
    "<b>Step 5: Adaptive evaluation.</b> For every defended checkpoint, the abliteration attack is <i>re-computed from scratch</i> on that checkpoint before any metric is read. "
    "This ensures the defense isn't only beating an attack built for the old model.",
])
S += [p("Results (from the paper)", H3)]
S += bullets([
    "Tested on 7 models from 5 families, 9B to 122B, dense and mixture-of-experts (Qwen3.5, Qwen3, gpt-oss-20b, gemma-4-31B, GLM-4.5-Air).",
    "On the 6 models that passed its quality gates, 51% to 90% of attacked-state answers to unseen hazardous prompts were fatally wrong. "
    "That is +0.27 to +0.84 more than the same attack on an undefended model.",
    "The normal released model is unchanged: refusal stays near 1.0, and MMLU, GSM8K, WMDP-bio and IFEval stay within noise. The knowledge is not erased; only attacked-state behaviour changes.",
    "Attackers can't filter decoys by quality, and majority voting over 64 samples still yields mostly falsified procedures.",
    "Training on one hazard area also spread to related procedural categories (+0.63 on a 200-prompt probe).",
])
S += [p("Limitations (stated by the author)", H3)]
S += bullets([
    "Raises cost, doesn't make the attack impossible. An attacker with another trusted source can check answers, though they then didn't need the model.",
    "Works only for a model's <b>first release</b>. Once a clean copy is public, it can be used as an <b>oracle</b> to check answers.",
    "Does not address prompt jailbreaks (by design). Evaluated mainly against abliteration-style attacks.",
    "The <b>9B model failed</b> the efficacy gate, so it is a boundary case. Decoy data, attacked and defended checkpoints are <b>withheld</b> for safety.",
])
S += [box("Verdict for our hackathon: borrow, don't replicate", [
    "<b>Not a fit as the main project:</b> (1) it failed at 9B and Tiny Aya is 3.35B; (2) its core data is real hazardous answers rewritten as decoys, which can't safely be produced or published; "
    "(3) the hackathon requires all artefacts on HF, but the paper itself withholds them; (4) a smaller re-run of a full published recipe has low originality; (5) the pipeline is too heavy for 5 days.",
    "<b>Borrow two things:</b> (a) <b>adaptive evaluation</b>: re-derive the attack on every defended checkpoint; (b) <b>training in the simulated attacked state</b>, used for a harmless goal: "
    "make the model keep giving a full, reasoned refusal even after abliteration. This is close to ART (Kuo et al., 2026).",
], bg="#EEF2FA", border="#1F2A44")]

# ---------------- 6c. Real-world evidence
S += [p("6c. Reality check: what attackers actually do (data pulled 9 Nov 2026)", H2),
      table([
          ["Evidence", "Number", "What it tells us"],
          ["Hugging Face models with \"abliterated\" in the name", "9,111", "Abliteration is mainstream, not academic"],
          ["Models with \"heretic\" in the name", "7,456", "Most recent stripping uses one automated tool"],
          ["Models with \"uncensored\" in the name", "8,372", "Strong demand for safety-stripped copies"],
          ["Abliterated copies of Aya Expanse (8B/32B), incl. quantized", "13 or more repos", "Cohere's own models are already being stripped"],
          ["Downloads of two Aya Expanse 8B abliterated repos (all time)", "about 25,000", "Compare 1.14M for the original: a real, measurable share"],
          ["Abliterated Tiny Aya copies", "0 found", "New model: we can defend it before stripped copies spread"],
          ["Hinglish jailbreak evidence", "99% ASR in one study (Aswal &amp; Jaiswal, 2025)", "Shows a <i>prompt-level</i> vulnerability, but no evidence of in-the-wild use. With open weights, attackers don't need language tricks: they run Heretic"],
      ], [62, 36, 82]),
      Spacer(1, 3),
      p("<b>Conclusion:</b> for tamper resistance, the realistic attacker is someone running <b>Heretic</b> (or a LoRA fine-tune) on downloaded weights. "
        "The multilingual angle is dropped from this project. It belongs to prompt-level safety of deployed apps, not to open-weight tampering.", B),
      p("How Heretic works (simple)", H3)]
S += bullets([
    "Computes a refusal direction for each layer from harmful vs harmless prompts, the same idea as abliteration.",
    "Removes that direction from the model's weights, using a tunable strength pattern across layers and components.",
    "Uses <b>Optuna</b> (a <b>TPE</b> optimizer) to search automatically for the settings that minimise two things at once: the number of refusals on harmful prompts, and the <b>KL divergence</b> from the original model on harmless prompts (damage).",
    "Keeps the <b>Pareto front</b>: the best trade-offs between \"fewest refusals\" and \"least damage\". It is an <i>adaptive attacker in a box</i>, which makes it ideal for honest evaluation.",
    "Reported example (author's numbers): Gemma-3-12B with 3 of 100 refusals at KL 0.16. Licence: AGPL-3.0, about 21.7k GitHub stars.",
])

# ---------------- 7. How to measure
S += [p("7. How tamper resistance is measured", H2),
      flow(["Base model", "Apply our defense", "Release weights", "Attacker tampers (sweep of attacks)", "Measure harm, usefulness, over-refusal"]),
      Spacer(1, 5)]
S += bullets([
    "<b>Harm after attack:</b> attack success rate (ASR) or StrongREJECT score on HarmBench / AdvBench prompts, judged by a classifier model. Report the <b>worst case</b> across the attack sweep.",
    "<b>Attack-cost curve (the key plot):</b> harmfulness on the y-axis against attacker effort on the x-axis (fine-tuning steps or number of harmful examples). A good defense pushes the curve to the right. This is more honest than a single number.",
    "<b>Utility:</b> general ability (MMLU, MMLU-Pro or Global-MMLU subset). TamperBench caps acceptable loss at a 10% relative drop.",
    "<b>Over-refusal:</b> refusing harmless-but-scary-sounding prompts (XSTest). A model that refuses everything is \"safe\" but useless.",
    "<b>Trainability:</b> can honest users still fine-tune it for a normal task? Fine-tune on a benign task and check task accuracy.",
    "<b>Four immunization conditions</b> (Rosati et al., 2024): <i>resistance</i> (attack fails or is costly), <i>stability</i> (usefulness kept), <i>generalization</i> (works against attack data not seen during defense), <i>trainability</i> (benign fine-tuning still works). Judges will recognise this checklist.",
])

# ---------------- 8. Unique ideas
S += [PageBreak(), p("8. What we can do that is unique", H2),
      p("Ranked by novelty multiplied by feasibility in 5 days on Tiny Aya (3.35B) with free or low-cost GPUs.")]
S += [p("Idea A (headline): Heretic-resistant Tiny Aya", H3)]
S += bullets([
    "<b>Real problem:</b> over 9,000 abliterated and over 7,000 Heretic-stripped models on Hugging Face, including 13 or more stripped Aya Expanse copies. Tiny Aya has none yet, so we can defend it first.",
    "<b>Use the attacker's own scoreboard:</b> run Heretic's optimizer on the original and the defended model and compare their <b>Pareto fronts</b> (refusals vs KL damage). "
    "Success means Heretic can only remove refusals by also badly damaging the model (high KL, MMLU drop), so the attacker gets a broken model.",
])
S += [p("Idea B (the defense): spread the refusal and train in the attacked state", H3)]
S += bullets([
    "Extended refusal (2025): teach full, reasoned refusals instead of one-line templates, so refusal is spread across many tokens and directions.",
    "Plus the Fool's Gold and ART trick: also train <i>through a simulated abliteration</i> (refusal direction projected out), so the model learns to keep refusing even when that direction is removed. "
    "The target is a harmless refusal, not a decoy, so no hazardous data is needed.",
    "Refusal text is harmless to create, so CuratorKIT can generate and quality-gate it at no risk.",
])
S += [p("Idea C (the mechanism, Lexsi's interpretability strength)", H3)]
S += bullets([
    "Measure <b>how concentrated</b> refusal is before and after the defense: the share of refusal explained by the top direction, the number of layers involved, and the cosine similarity of refusal directions across languages (SafeTune refusal direction, CircuitKIT components).",
    "This turns \"it worked\" into \"it worked because refusal is no longer on one wire\", which is exactly the explanation Lexsi's team values.",
])
S += [p("Idea D (the honest scorecard)", H3)]
S += bullets([
    "Tiered stress test: Tier 1 prefilling, Tier 2 Heretic (adaptive abliteration, re-run on every defended checkpoint), Tier 3 LoRA harmful fine-tune at 3 budgets (for example 50, 200 and 1000 steps).",
    "Report worst case, attack-cost curves, utility, over-refusal and trainability.",
    "Include honest negatives. We expect a large fine-tuning budget to still break the model, and saying so builds trust with judges.",
])
S += [p("Idea E (second pillar): \"Is this a stripped Aya?\" tamper detector", H3)]
S += bullets([
    "If safety can't be made unbreakable, make tampering <b>visible</b>. Fingerprint the model's refusal behaviour and weights along the refusal direction at release, then build a check that flags uploads that were abliterated or safety-stripped.",
    "<b>Real test data already exists:</b> the public abliterated Aya Expanse 8B copies on Hugging Face (positives) vs honest Aya fine-tunes (negatives), plus our own Heretic outputs for Tiny Aya.",
    "Measured by detection rate and false-alarm rate. Useful to model publishers (Cohere) and hubs. Inference only, so cheap.",
])
S += [box("Combined thesis (one sentence for judges)", [
    "\"Over 9,000 safety-stripped models on Hugging Face show that abliteration is the real-world threat to open weights. We make Tiny Aya resist Heretic, "
    "so the attacker's best result is a damaged model, and we make stripping detectable. Both are measured with adaptive, worst-case evaluation.\"",
    "Repo chain: <b>CuratorKIT</b> (extended-refusal data) -> <b>SafeTune</b> (refusal direction, baselines, HarmBench evaluation, SpectralEntropyMonitor) -> <b>CircuitKIT</b> (where refusal lives, before and after) -> <b>AuditKIT</b> (over-refusal, utility, throughput). That is four repos.",
])]

# ---------------- 9. Feasibility
S += [p("9. Feasibility, risks, and what not to claim", H2),
      table([
          ["Component", "Compute on Tiny Aya 3.35B", "Risk"],
          ["Extended-refusal fine-tune (LoRA, a few thousand examples)", "About 1-2 GPU-hours on an L4/A10G; slower on a T4", "Low"],
          ["Abliteration sweep (layers by strength)", "Minutes per setting; no training", "Low"],
          ["LoRA harmful fine-tune attacks (3 budgets by 2 languages, before and after defense)", "About 6-12 short runs; 15-40 min each", "Medium (credit budget)"],
          ["Evaluation (HarmBench subset, XSTest, Global-MMLU subset)", "The judge model needs about 14.5 GB; batch it on one GPU session", "Medium"],
          ["Full TAR / RepNoise baselines", "TAR needs inner attack loops; too costly for 5 days", "High: skip, cite instead"],
      ], [70, 70, 40]),
      Spacer(1, 4)]
S += bullets([
    "<b>Do not claim</b> \"tamper-proof\" or \"unbreakable\". Claim \"raises attack cost by X times\" with curves. The 2026 literature will make judges sceptical of anything stronger.",
    "<b>Expected outcome:</b> a clear gain against abliteration and prefilling, a partial gain against small LoRA attacks, and none against large budgets. That is still a strong, honest result.",
    "<b>Go/no-go check on Day 1:</b> (1) abliteration of base Tiny Aya clearly works (harm jumps), and (2) one LoRA fine-tune finishes in under an hour on the GPU we have. If either fails, return to the multilingual safety + steering plan.",
    "<b>Compute warning for Heretic:</b> it runs many trials, each scoring about 100 prompts. Use a reduced trial count and an L4/A10G-class GPU; a free T4 may be too slow for repeated runs.",
    "<b>Related claim to check:</b> Abliterix (a Heretic derivative) says it released an abliteration-resistant alignment method that Heretic is weak against. This is unverified; read it before claiming novelty.",
])
S += [p("Suggested 5-day schedule (if we switch)", H3),
      table([
          ["Day", "Work"],
          ["1", "Setup; go/no-go: (1) Heretic strips base Tiny Aya (reduced trial count), (2) one LoRA run finishes in under 1 hour. Start CuratorKIT extended-refusal data plus a benign retain set"],
          ["2", "Train defense variants: (i) extended refusal, (ii) extended refusal plus training in the simulated attacked state. Baseline: base model. Start the detector: collect public abliterated Aya copies and honest fine-tunes"],
          ["3", "Attack all variants: Heretic re-run on each checkpoint (compare Pareto fronts), prefill, LoRA attacks at 3 budgets (public harmful datasets only)"],
          ["4", "Evaluate harm, utility, over-refusal, trainability. Mechanism: refusal concentration across layers and directions. Detector accuracy"],
          ["5", "HF artefacts (defended model, data, logs; attacked checkpoints stay private), Space demo, HF Article, ROI, submit"],
      ], [12, 168])]

# ---------------- 10. Assist zones
S += [p("10. How I can help on this topic without stopping midway", H2),
      table([
          ["Zone", "What"],
          ["Green (full help)", "All defense code; extended-refusal data generation (refusals are harmless text); evaluation harness; abliteration and attack code used for measurement (published, standard methods); mechanism analysis; plots; HF Article, Space, ROI"],
          ["Amber (with care)", "Running harmful-fine-tune attacks using <b>public</b> harmful datasets (e.g. those used in TamperBench and HarmBench). Attacked checkpoints stay private and only metrics are published. Harmful outputs are scored by a judge model, not read out or rewritten"],
          ["Red (I will not)", "Writing new harmful training data or answers with real-world uplift; publishing safety-stripped models"],
      ], [32, 148])]

# ---------------- 11. Glossary
S += [PageBreak(), p("11. Glossary (jargon explained simply)", H2),
      table([
          ["Term", "Simple meaning"],
          ["Weights / parameters", "The billions of numbers that make up the model. Changing them changes behaviour."],
          ["Open-weight model", "A model whose weights anyone can download, run and modify (e.g. Tiny Aya, Llama)."],
          ["Fine-tuning", "Further training a model on new examples to change its behaviour."],
          ["LoRA / QLoRA", "Cheap fine-tuning that trains a small add-on instead of all weights. QLoRA also compresses the model so it fits on a small GPU."],
          ["Alignment / safety training", "Training that teaches a model to be helpful and to refuse harmful requests."],
          ["Refusal", "The model declining a request (\"I can't help with that\")."],
          ["Jailbreak", "A trick (usually a prompt) that makes a model ignore its safety training."],
          ["Tampering", "Modifying the model itself (weights or internal signals) to remove safety."],
          ["Tamper resistance", "Safety that keeps working even after someone tampers with the model."],
          ["Threat model", "A precise statement of what the attacker can and cannot do."],
          ["Harmful fine-tuning attack", "Fine-tuning on harmful examples to make the model comply."],
          ["Fine-tuning-as-a-service", "A provider fine-tunes the model for you on its servers, so the provider can add safeguards."],
          ["Activations / hidden states", "The model's internal signals while it reads and writes text, like brain activity."],
          ["Residual stream", "The main internal channel of a transformer, which each layer reads from and adds to."],
          ["Representation", "How the model internally encodes a concept, as a pattern of activations."],
          ["Direction (in activation space)", "A specific pattern of activations. Concepts like \"refuse\" often line up with one direction."],
          ["Refusal direction", "The single activation direction that mostly controls refusing (Arditi et al., 2024)."],
          ["Abliteration", "Removing the refusal direction from the weights (projecting it out), so the model can no longer refuse. Needs no training."],
          ["Projection / orthogonalization", "The maths of removing one direction from vectors or weight matrices."],
          ["Activation steering", "Adding or removing a direction during generation to change behaviour."],
          ["Prefilling", "Writing the start of the model's answer yourself to force a compliant beginning."],
          ["Shallow alignment", "Safety that mostly affects only the first few words of an answer."],
          ["Gradient / gradient descent", "The signal that tells training which way to change each weight to reduce the error (loss)."],
          ["Gradient ascent", "Moving weights to <i>increase</i> the error on some data, used to make a model worse at something on purpose."],
          ["Loss", "A number measuring how wrong the model is on training examples. Training lowers it."],
          ["Learning rate", "How big each training step is. Attackers sweep it to find settings that break defenses."],
          ["Adversarial training", "Training a model while simulating attacks against it, so it learns to resist them."],
          ["Meta-learning (in TAR)", "Training on the outcome of simulated attacker training runs, so future real attacks make little progress."],
          ["Unlearning", "Removing specific knowledge from a model (forget set) while keeping the rest (retain set)."],
          ["Model merging / task vector", "Adding or subtracting weight differences between models to add or remove skills."],
          ["Quantization", "Storing weights with fewer bits (e.g. 4-bit) to make the model smaller and faster."],
          ["Pruning", "Deleting weights or neurons that seem unimportant."],
          ["Circuit", "A small set of model components that together carry out a specific behaviour."],
          ["Attack success rate (ASR)", "The percentage of harmful prompts the model answers harmfully. Lower is safer."],
          ["StrongREJECT / HarmBench / AdvBench", "Standard sets of harmful prompts plus scoring methods for measuring harm."],
          ["XSTest / over-refusal", "Harmless prompts that sound scary (\"how do I kill a Python process\"). Refusing them is over-refusal."],
          ["MMLU / Global-MMLU", "Multiple-choice knowledge tests measuring general ability. Global-MMLU covers many languages."],
          ["Judge model", "A classifier model that scores whether an answer is harmful."],
          ["Hyperparameter sweep", "Trying many attack settings (learning rate, steps, rank) and keeping the worst case."],
          ["Worst-case vs average-case", "Security cares about the strongest attack, not the typical one."],
          ["Null result", "An experiment that finds no effect. It is honest and still useful when reported clearly."],
          ["Heretic", "Open-source tool that abliterates any model automatically, searching for the settings that remove the most refusals with the least damage."],
          ["KL divergence", "A number measuring how different two models' output probabilities are. Here it measures how much an attack damaged the model."],
          ["Pareto front", "The set of best trade-offs where you can't improve one goal (fewer refusals) without worsening the other (more damage)."],
          ["Optuna / TPE", "A library and search method that tries settings intelligently, learning from earlier trials which ones look promising."],
          ["Defensive deception / honeypot", "A defense that fools the attacker with convincing fakes instead of blocking them."],
          ["Decoy", "A fake answer that looks real but has its critical details made wrong (Fool's Gold)."],
          ["Preference optimization (e.g. DPO)", "Training by showing the model pairs of answers and teaching it to prefer the better one."],
          ["On-policy", "Training on the model's own current outputs rather than on fixed, pre-written examples."],
          ["Oracle", "A trusted reference an attacker can use to check answers, such as a clean copy of the model."],
          ["GGUF / quantized copy", "A compressed model file format for running on laptops. Many stripped models are shared this way."],
      ], [48, 132])]

# ---------------- 12. References
S += [p("12. Key references", H2)]
S += bullets([
    "Qi et al. 2023, Fine-tuning Aligned Language Models Compromises Safety. arXiv 2310.03693",
    "Lermen et al. 2023, LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B. arXiv 2310.20624",
    "Yang et al. 2023, Shadow Alignment. arXiv 2310.02949 ; Yong et al. 2023, Low-Resource Languages Jailbreak GPT-4. arXiv 2310.02446",
    "Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction. arXiv 2406.11717",
    "Wei et al. 2024, Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications. arXiv 2402.05162",
    "Qi et al. 2025 (ICLR), Safety Alignment Should Be Made More Than Just a Few Tokens Deep. arXiv 2406.05946",
    "Tamirisa et al. 2025 (ICLR), Tamper-Resistant Safeguards for Open-Weight LLMs (TAR). arXiv 2408.00761",
    "Rosati et al. 2024, Representation Noising (RepNoise). arXiv 2405.14577 ; Huang et al. 2024, Vaccine. arXiv 2402.01109 ; Booster. arXiv 2409.01586 ; Lisa. arXiv 2405.18641",
    "Zou et al. 2024, Circuit Breakers. arXiv 2406.04313 ; Egashira et al. 2024, Exploiting LLM Quantization. arXiv 2405.18137",
    "Huang et al. 2024, Harmful Fine-tuning Attacks and Defenses for LLMs: A Survey. arXiv 2409.18169",
    "Qi et al. 2024, On Evaluating the Durability of Safeguards for Open-Weight LLMs. arXiv 2412.07097",
    "Abu Shairah et al. 2025, An Embarrassingly Simple Defense Against LLM Abliteration Attacks. arXiv 2505.19056",
    "O'Brien et al. 2025, Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards. arXiv 2508.06601",
    "SDD: Self-Degraded Defense against Malicious Fine-tuning, 2025. arXiv 2507.21182 ; Benign outlier samples break safety, 2025. arXiv 2505.06843",
    "Hossain et al. 2026 (KDD), TamperBench. arXiv 2602.06911, github.com/criticalml-uw/TamperBench",
    "Kuo, Yadav, Smith 2026, Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks (ART). arXiv 2605.26526",
    "ICML 2026, Beyond Harmful Fine-Tuning: Evaluating Open-Weight Safeguards Under Training-Free Tampering. icml.cc/virtual/2026/78555",
    "Russinovich 2026, Fool's Gold: Defensive Deception for Open-Weight Models. arXiv 2608.17202, markrussinovich.github.io/fools-gold",
    "Heretic (p-e-w), automatic abliteration with Optuna. github.com/p-e-w/heretic",
    "Aswal &amp; Jaiswal 2025, Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs. arXiv 2505.14226",
    "Hugging Face Hub API model counts, queried 9 Nov 2026 (search terms: abliterated, heretic, uncensored, aya).",
])

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(15 * mm, 8 * mm, "Tamper Resistance: beginner guide + hackathon plan")
    c.drawRightString(195 * mm, 8 * mm, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                        title="Tamper Resistance Guide", author="ShivenduShivu")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("ok", OUT)
