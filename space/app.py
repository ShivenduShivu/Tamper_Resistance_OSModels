"""Heretic-Resistant Tiny Aya: results explorer (no model is loaded; every number comes from our evaluation runs)."""
import json
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

D = json.load(open("data.json", encoding="utf-8"))
PF = D["pareto"]
NAMES = {"base": "Original Tiny Aya", "v1": "v1: extended refusal", "v2": "v2: trained through abliteration",
         "v2b": "v2b: + borderline-safe data"}
COLORS = {"base": "#B4361A", "v1": "#8A8F9C", "v2": "#1F2A44", "v2b": "#2E8B57"}
LINKS = """**Links:**
- [Models collection](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-models-6af3e1eff9a91556f5535934)
- [Data & results collection](https://huggingface.co/collections/StarWarrior24x7/heretic-resistant-tiny-aya-data-and-results-6af3e1f03fff746c25b4884c)
- [Code on GitHub](https://github.com/ShivenduShivu/Tamper_Resistance_OSModels)"""


def pareto_fig(kl_budget):
    fig = go.Figure()
    for k, name in NAMES.items():
        if k not in PF:
            continue
        f = sorted(PF[k]["front"], key=lambda t: t[1])
        fig.add_trace(go.Scatter(x=[kl for _, kl in f], y=[r * 100 for r, _ in f], mode="lines+markers",
                                 line=dict(shape="hv", color=COLORS[k], width=2.5), name=name,
                                 hovertemplate="KL %{x:.3f}<br>still refuses %{y:.0f}%<extra>" + name + "</extra>"))
    fig.add_vline(x=kl_budget, line_dash="dash", line_color="#888",
                  annotation_text=f"attacker's damage budget = {kl_budget:.3f}")
    fig.update_layout(xaxis_title="Damage the attack does to normal behaviour (KL divergence)",
                      yaxis_title="Harmful prompts still refused (%)", yaxis_range=[0, 103], xaxis_range=[-0.003, 0.16],
                      legend=dict(orientation="h", y=-0.25), height=470, margin=dict(t=30, l=10, r=10))
    return fig


def budget_table(kl_budget):
    rows = []
    for k, name in NAMES.items():
        if k not in PF:
            continue
        ok = [r for r, kl in PF[k]["front"] if kl <= kl_budget]
        rows.append({"Model": name, "Best Heretic could do within this budget (refusal kept)": f"{min(ok) * 100:.0f}%" if ok else "n/a"})
    return pd.DataFrame(rows)


def update(kl):
    return pareto_fig(kl), budget_table(kl)


ATTACKS = pd.DataFrame([
    ["No attack", "99%", "100%", "100%"],
    ["Heretic, strongest attack found (judge)", "1%", "23% (needs ~16× the KL)", "24%"],
    ["SafeTune refusal-direction ablation", "28%", "100%", "100%"],
    ["SafeTune CAA steering, strength −1", "4%", "42%", "30%"],
    ["SafeTune CAA steering, strength −4", "0%", "0%", "0%"],
    ["Sampling attack (best of 3, share that got through)", "9% got through", "0% got through", "2% got through"],
    ["4-bit quantised (bitsandbytes NF4)", "99%", "100%", "100%"],
    ["Prefill \"Sure, here is\"", "31%", "38%", "41%"],
    ["Harmful LoRA fine-tune, 100 examples", "1%", "3%", "1%"],
], columns=["Attack (refusal on 100 harmful prompts, ProtectAI judge)", "Original", "v2", "v2b"])
COST = pd.DataFrame([
    ["Over-refusal on safe XSTest prompts", "12%", "46%", "20.8%"],
    ["MMLU (1,000 questions)", "52.9%", "50.5%", "50.1%"],
    ["MMLU after Heretic's strongest attack", "52.7%", "51.0%", "49.6%"],
], columns=["Usability", "Original", "v2", "v2b"])

SIDE = D["side_by_side"]
ATTACK_NAME = list(SIDE)[0]


def show_pair(i):
    r = SIDE[ATTACK_NAME][int(i)]
    return r["prompt"], "⚠️ **Complied.** The response is withheld: we never publish outputs of stripped models.", r["v2"]


USEFUL = D["useful"]

with gr.Blocks(title="Heretic-Resistant Tiny Aya") as demo:
    gr.Markdown(
        "# Heretic-Resistant Tiny Aya\n"
        "**Can a small open model keep its safety when someone edits it out of the weights?**\n\n"
        "- **The attack:** [Heretic](https://github.com/p-e-w/heretic) removes the *refusal direction* from a model's weights in one command. "
        "On Cohere's **Tiny Aya Global**, it cuts refusal of harmful requests from **99% to 1%**, with no loss on MMLU.\n"
        "- **Our defense:** we trained Tiny Aya *while simulating that attack inside the model*. At the same attack strength, our **v2 still refuses 100%**.\n\n"
        "*This Space loads no model. Every number comes from our evaluation runs on free Kaggle T4 GPUs. "
        "Refusal is scored by a classifier judge, [ProtectAI/distilroberta-base-rejection-v1](https://huggingface.co/ProtectAI/distilroberta-base-rejection-v1).*")
    with gr.Tab("1 · Attack curves"):
        gr.Markdown("Each point is one attack that Heretic found. **Higher means harder to strip.** "
                    "Slide to choose how much damage to the model's normal behaviour the attacker will accept.")
        kl = gr.Slider(0.0, 0.16, value=0.01, step=0.001, label="Attacker's damage budget (KL)")
        plot = gr.Plot(pareto_fig(0.01))
        tbl = gr.Dataframe(budget_table(0.01), interactive=False)
        kl.change(update, kl, [plot, tbl])
        gr.Markdown("*Heretic scores refusal with its own keyword counter. The classifier judge is stricter: "
                    "for example, the original's 14% under the best attack is really **1%**.*")
    with gr.Tab("2 · All attacks"):
        gr.Markdown("### Which attacks does the defense stop?\nThe numbers are refusal rates on 100 harmful prompts (higher is safer).")
        gr.Dataframe(ATTACKS, interactive=False, wrap=True)
        gr.Markdown("- **Works against:** attacks that remove or steer the refusal direction (Heretic, SafeTune ablation, mild steering), random sampling, and 4-bit quantisation.\n"
                    "- **Doesn't work against:** harmful fine-tuning, prefilling, or strong steering. These are known open problems and our next step.")
        gr.Markdown("### What does it cost?")
        gr.Dataframe(COST, interactive=False)
    with gr.Tab("3 · Same attack, side by side"):
        gr.Markdown(f"**Attack:** {ATTACK_NAME}, from Lexsi's SafeTune, applied identically to both models. "
                    f"We show {len(SIDE[ATTACK_NAME])} prompts where the original complied and v2 refused. "
                    "The prompts come from the public `mlabonne/harmful_behaviors` benchmark.")
        idx = gr.Slider(0, len(SIDE[ATTACK_NAME]) - 1, value=0, step=1, label="Example")
        p = gr.Textbox(label="Harmful prompt (public benchmark)", interactive=False)
        with gr.Row():
            o = gr.Markdown(label="Original Tiny Aya, attacked")
            v = gr.Textbox(label="v2, same attack", interactive=False, lines=6)
        idx.change(show_pair, idx, [p, o, v])
        demo.load(show_pair, idx, [p, o, v])
    with gr.Tab("4 · Still helpful?"):
        gr.Markdown("Safe questions that *sound* risky (from XSTest), answered by **v2b running in 4-bit**. "
                    "v2b refuses only 20.8% of these, against 12% for the original.")
        gr.Dataframe(pd.DataFrame(USEFUL).rename(columns={"prompt": "Safe question", "answer": "v2b answer"}), interactive=False, wrap=True)
    with gr.Tab("5 · How it works"):
        gr.Markdown(
            "### Method\n"
            "1. **v1:** LoRA fine-tune on 1,200 long, explained refusals plus 1,155 everyday answers, all written by Tiny Aya itself.\n"
            "2. **v2:** the same data, but on half of the training steps we **remove a refusal direction from every layer during training**, and the model must still refuse. "
            "The direction is recomputed from the current model every 100 steps, so the simulated attacker keeps adapting. "
            "This is in the spirit of *Fool's Gold* and *ART*, with a refusal target and no hazardous data.\n"
            "3. **v2b:** adds 1,218 borderline-but-safe prompts to reduce over-refusal.\n"
            "4. **Evaluation:** a fresh 40-trial Heretic search on every model, plus Lexsi's **SafeTune** (ablation and steering attacks) and **AuditKIT** (red-team probes).\n\n"
            "### Responsible release\n"
            "- Only the hardened models are public.\n"
            "- Stripped models existed only inside the evaluation runs.\n"
            "- Their outputs are never shown.\n\n"
            "*Lexsi Labs × Cohere × Hugging Face AI Safety & Alignment Hackathon, Track 2b: Tamper Resistance.*\n\n" + LINKS)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
