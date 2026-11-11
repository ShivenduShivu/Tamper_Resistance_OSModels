---
title: Heretic-Resistant Tiny Aya
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
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

This Space explores the results of our Track 2b (Tamper Resistance) project for the Lexsi Labs × Cohere × Hugging Face AI Safety & Alignment Hackathon. It covers the attack curves, every attack we tried, a side-by-side comparison under the same attack, and the usability cost.

No model runs in this Space; all numbers come from our evaluation runs. Outputs of stripped models are never shown.
