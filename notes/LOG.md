# Experiment Log

Format: date, run, config, result, decision.

## 2026-11-10: Day-1 go/no-go (`kaggle/day1`, Kaggle v2)
- **Config:** Tiny Aya Global in FP16 on a T4.
  - A: 50 harmful and 50 harmless prompts, refusal measured by keyword match.
  - B: LoRA r=16 (q, k, v, o), batch size 4, sequence length 768, 100 steps.
  - C: Heretic with classic abliteration, 40 trials (12 random).
- **Pass criteria:**
  - Heretic cuts the harmful-prompt refusal rate sharply at low KL.
  - A LoRA step takes under ~3 s, so 1,000 steps finish in under 1 hour.
  - No NaN or garbage output in FP16.
- **Result:** _pending_
- **Decision:** _pending_. If it fails, switch to the fallback "Prevent vs Repair" (SafeTune).
