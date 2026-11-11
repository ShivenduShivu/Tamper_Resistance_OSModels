"""Build the static (free) HF Space: space_static/index.html with all data inlined.
Usage (repo root): python scripts/build_static_space.py"""
import json, os, html
D = json.load(open("space/data.json", encoding="utf-8"))
os.makedirs("space_static", exist_ok=True)

DATA = json.dumps({"pareto": D["pareto"], "side": list(D["side_by_side"].values())[0], "useful": D["useful"]},
                  ensure_ascii=False).replace("</", "<\\/")

ATTACKS = [
    ["No attack", "99%", "100%", "100%"],
    ["Heretic, strongest attack found", "1%", "23% <small>(needs ~16× the KL)</small>", "24%"],
    ["SafeTune refusal-direction ablation", "28%", "<b>100%</b>", "<b>100%</b>"],
    ["SafeTune CAA steering, strength −1", "4%", "<b>42%</b>", "30%"],
    ["SafeTune CAA steering, strength −4", "0%", "0%", "0%"],
    ["Sampling attack: share that got through (best of 3)", "9%", "<b>0%</b>", "2%"],
    ["Model compressed to 4-bit", "99%", "<b>100%</b>", "<b>100%</b>"],
    ["Prefill “Sure, here is”", "31%", "38%", "41%"],
    ["Harmful LoRA fine-tune, 100 examples", "1%", "3%", "1%"],
    ["AuditKIT prompt-injection probes (pass rate)", "62.5%", "<b>87.5%</b>", "75%"],
]
COST = [["Over-refusal on safe XSTest prompts", "12%", "46%", "20.8%"],
        ["MMLU (1,000 questions)", "52.9%", "50.5%", "50.1%"],
        ["MMLU after Heretic's strongest attack", "52.7%", "51.0%", "49.6%"]]


def rows(t):
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in t)


useful_rows = "".join(f"<tr><td>{html.escape(u['prompt'])}</td><td>{html.escape(u['answer'])}</td></tr>" for u in D["useful"])

PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heretic-Resistant Tiny Aya</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{{--bg:#f7f8fb;--card:#fff;--ink:#1f2a44;--muted:#5b6475;--acc:#b4361a;--ok:#2e8b57;--line:#e3e6ee}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11141b;--card:#1a1f2b;--ink:#e8ebf2;--muted:#a3abbb;--line:#2c3342}}}}
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}}
.wrap{{max-width:1040px;margin:0 auto;padding:22px 16px 48px}}
h1{{margin:4px 0 6px;font-size:28px}} .lead{{color:var(--muted);margin:0 0 14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
.kpi b{{display:block;font-size:26px}} .kpi span{{color:var(--muted);font-size:13px}}
.red{{color:var(--acc)}} .green{{color:var(--ok)}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 0;border-bottom:1px solid var(--line)}}
.tabs button{{border:0;background:none;color:var(--muted);padding:10px 12px;font-size:14px;cursor:pointer;border-bottom:2px solid transparent}}
.tabs button.on{{color:var(--ink);border-color:var(--acc);font-weight:600}}
.panel{{display:none;background:var(--card);border:1px solid var(--line);border-top:0;border-radius:0 0 12px 12px;padding:18px}}
.panel.on{{display:block}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:600}} small{{color:var(--muted)}}
input[type=range]{{width:100%}} .note{{color:var(--muted);font-size:13px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}} @media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}
.box{{border:1px solid var(--line);border-radius:10px;padding:12px;min-height:90px}} .box h4{{margin:0 0 6px}}
.prompt{{background:var(--bg);border-radius:8px;padding:10px;margin-top:8px}}
.nav{{display:flex;gap:8px;align-items:center;margin-top:10px}} .nav button{{padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}}
a{{color:var(--acc)}}
</style></head><body><div class="wrap">
<h1>🛡️ Heretic-Resistant Tiny Aya</h1>
<p class="lead"><b>Can a small open model keep its safety when someone edits it out of the weights?</b><br>
<a href="https://github.com/p-e-w/heretic">Heretic</a>, a free one-command tool, removes the <i>refusal direction</i> from a model's weights.
We trained Cohere's <b>Tiny Aya Global</b> while simulating that attack, so its refusals survive it.</p>
<div class="kpis">
 <div class="kpi"><b class="red">99% → 1%</b><span>Original Tiny Aya: refusal of harmful requests after Heretic (MMLU unchanged)</span></div>
 <div class="kpi"><b class="green">100%</b><span>Our v2 at the same attack strength that broke the original</span></div>
 <div class="kpi"><b>23% · 16×</b><span>v2 under Heretic's strongest attack: refusal left, and how much more damage it needed</span></div>
 <div class="kpi"><b class="green">28% vs 100%</b><span>Lexsi SafeTune's direction-ablation attack: original vs v2</span></div>
</div>
<div class="tabs" id="tabs">
 <button class="on" data-t="t1">Attack curves</button><button data-t="t2">All attacks</button>
 <button data-t="t3">Same attack, side by side</button><button data-t="t4">Still helpful?</button><button data-t="t5">How it works</button>
</div>
<div class="panel on" id="t1">
 <p>Each point is one attack that Heretic found. <b>Higher means harder to strip.</b> Move the slider to set how much damage to the model's normal behaviour (KL divergence) the attacker will accept.</p>
 <div id="chart" style="height:430px"></div>
 <label>Attacker's damage budget (KL): <b id="klv">0.010</b></label><input type="range" id="kl" min="0" max="0.16" step="0.001" value="0.01">
 <table id="budget"></table>
 <p class="note">Heretic scores refusal with keywords. Our classifier judge is stricter: for example, the original's 14% under Heretic's best attack is really <b>1%</b>.</p>
</div>
<div class="panel" id="t2">
 <p>Refusal on 100 harmful prompts, scored by the classifier judge <a href="https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1">ProtectAI/distilroberta-base-rejection-v1</a>. Higher is safer.</p>
 <table><tr><th>Attack</th><th>Original</th><th>v2</th><th>v2b</th></tr>{rows(ATTACKS)}</table>
 <p><b>Works against:</b> attacks that remove or steer the refusal direction (Heretic, SafeTune ablation, mild steering), random sampling, and 4-bit compression.<br>
 <b>Does not work against:</b> harmful fine-tuning, prefilling, or strong steering. These are known open problems and our next step.</p>
 <h3>What does it cost?</h3><table><tr><th>Usability</th><th>Original</th><th>v2</th><th>v2b</th></tr>{rows(COST)}</table>
</div>
<div class="panel" id="t3">
 <p><b>Attack:</b> SafeTune refusal-direction ablation (from Lexsi's SafeTune), applied identically to both models. These are prompts where the original complied and v2 refused.
 The prompts come from the public <code>mlabonne/harmful_behaviors</code> benchmark. <b>We never show the stripped model's output.</b></p>
 <div class="prompt"><small>Harmful prompt (public benchmark)</small><div id="sp"></div></div>
 <div class="pair"><div class="box"><h4 class="red">Original Tiny Aya, attacked</h4><div>⚠️ <b>Complied.</b> Response withheld.</div></div>
 <div class="box"><h4 class="green">v2, same attack</h4><div id="sv"></div></div></div>
 <div class="nav"><button id="prev">← Prev</button><span id="cnt"></span><button id="next">Next →</button></div>
</div>
<div class="panel" id="t4">
 <p>Safe questions that <i>sound</i> risky (XSTest), answered by <b>v2b running in 4-bit</b>. v2b refuses only 20.8% of these, against 12% for the original.</p>
 <table><tr><th style="width:38%">Safe question</th><th>v2b answer</th></tr>{useful_rows}</table>
</div>
<div class="panel" id="t5">
 <ol><li><b>v1:</b> LoRA fine-tune on 1,200 long, explained refusals plus 1,155 everyday answers, all written by Tiny Aya itself.</li>
 <li><b>v2:</b> the same data, but on half of the training steps we <b>remove a refusal direction from every layer during training</b>, and the model must still refuse.
 The direction is recomputed from the current model every 100 steps, so the simulated attacker keeps adapting. This follows Fool's Gold and ART, with a refusal target and no hazardous data.</li>
 <li><b>v2b:</b> adds 1,218 borderline-but-safe prompts to reduce over-refusal.</li>
 <li><b>Evaluation:</b> a fresh 40-trial Heretic search on every model, Lexsi's <b>SafeTune</b> (ablation and steering attacks) and <b>AuditKIT</b> (red-team probes), MMLU, XSTest, and a harmful fine-tuning attack.</li></ol>
 <p><b>Responsible release:</b> only the hardened models are public. Stripped models existed only inside the evaluation runs, and their outputs are never shown.</p>
 <p><b>Links:</b>
 <a href="https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-models-6af3e1eff9a91556f5535934">Models</a> ·
 <a href="https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-data-and-results-6af3e1f03fff746c25b4884c">Data &amp; results</a> ·
 <a href="https://github.com/ShivenduShivu/Tamper_Resistance_OSModels">Code</a></p>
 <p class="note">Lexsi Labs × Cohere × Hugging Face AI Safety &amp; Alignment Hackathon, Track 2b: Tamper Resistance. This page loads no model; every number comes from our evaluation runs.</p>
</div>
</div>
<script>
const D = {DATA};
const NAMES = {{base:"Original Tiny Aya", v1:"v1: extended refusal", v2:"v2: trained through abliteration", v2b:"v2b: + borderline-safe data"}};
const COL = {{base:"#B4361A", v1:"#8A8F9C", v2:"#4F6BD8", v2b:"#2E8B57"}};
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {{
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("on", p.id === b.dataset.t));
  if (b.dataset.t === "t1") Plotly.Plots.resize("chart");
}});
const dark = matchMedia("(prefers-color-scheme: dark)").matches, ink = dark ? "#e8ebf2" : "#1f2a44";
function draw(kl) {{
  const traces = Object.keys(NAMES).filter(k => D.pareto[k]).map(k => {{
    const f = [...D.pareto[k].front].sort((a, b) => a[1] - b[1]);
    return {{x: f.map(p => p[1]), y: f.map(p => p[0] * 100), name: NAMES[k], mode: "lines+markers",
      line: {{shape: "hv", color: COL[k], width: 2.5}}, hovertemplate: "KL %{{x:.3f}}<br>still refuses %{{y:.0f}}%<extra></extra>"}};
  }});
  Plotly.react("chart", traces, {{paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {{color: ink}},
    margin: {{t: 10, r: 10, l: 50, b: 40}}, xaxis: {{title: "Damage to normal behaviour (KL)", range: [-0.003, 0.16], gridcolor: dark ? "#2c3342" : "#e3e6ee"}},
    yaxis: {{title: "Harmful prompts still refused (%)", range: [0, 103], gridcolor: dark ? "#2c3342" : "#e3e6ee"}},
    legend: {{orientation: "h", y: -0.22}}, shapes: [{{type: "line", x0: kl, x1: kl, y0: 0, y1: 103, line: {{dash: "dash", color: "#888"}}}}]}},
    {{displayModeBar: false, responsive: true}});
  let t = "<tr><th>Model</th><th>Lowest refusal Heretic reached within this budget</th></tr>";
  for (const k of Object.keys(NAMES)) {{ if (!D.pareto[k]) continue;
    const ok = D.pareto[k].front.filter(p => p[1] <= kl).map(p => p[0]);
    t += `<tr><td>${{NAMES[k]}}</td><td>${{ok.length ? Math.round(Math.min(...ok) * 100) + "%" : "n/a"}}</td></tr>`; }}
  document.getElementById("budget").innerHTML = t; document.getElementById("klv").textContent = (+kl).toFixed(3);
}}
const s = document.getElementById("kl"); s.oninput = () => draw(+s.value); draw(0.01);
let i = 0; const esc = t => t.replace(/[&<>]/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;"}}[c]));
function show() {{ const r = D.side[i]; document.getElementById("sp").innerHTML = esc(r.prompt);
  document.getElementById("sv").innerHTML = esc(r.v2); document.getElementById("cnt").textContent = (i + 1) + " / " + D.side.length; }}
document.getElementById("prev").onclick = () => {{ i = (i - 1 + D.side.length) % D.side.length; show(); }};
document.getElementById("next").onclick = () => {{ i = (i + 1) % D.side.length; show(); }};
show();
</script></body></html>"""
open("space_static/index.html", "w", encoding="utf-8").write(PAGE)
open("space_static/README.md", "w", encoding="utf-8").write("""---
title: Heretic-Resistant Tiny Aya
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
short_description: Can Tiny Aya keep its safety when it is edited out?
models:
- StarWarrior24x7/tiny-aya-global-xref-v2
- StarWarrior24x7/tiny-aya-global-xref-v1
- CohereLabs/tiny-aya-global
datasets:
- StarWarrior24x7/tiny-aya-tamper-resistance-results
- StarWarrior24x7/tiny-aya-extended-refusal
---

This page shows the results of our Track 2b (Tamper Resistance) project for the Lexsi Labs × Cohere × Hugging Face AI Safety & Alignment Hackathon. No model runs here; outputs of stripped models are never shown.
""")
print("ok space_static/index.html", round(os.path.getsize("space_static/index.html") / 1024), "KB")
