# Heretic-Resistant Tiny Aya (Lexsi Labs AI Safety Hackathon, Track 2b: Tamper Resistance)

**Goal:** make Tiny Aya Global keep refusing harmful requests after automated abliteration (Heretic) and small LoRA attacks, without hurting usefulness or over-refusal. Stretch goal: a detector for safety-stripped Aya copies.

**Deadline:** submit by the evening of 14 Nov 2026.

## Folder map
| Path | What lives here |
|---|---|
| `docs/` | Strategy and beginner PDFs; `docs/build/` holds the scripts that generate them |
| `kaggle/<job>/` | One folder per Kaggle job (`kernel-metadata.json` + script). Push with `kaggle kernels push -p kaggle/<job>` |
| `src/` | Shared code: data building, defense training, evaluation |
| `data/` | Local data (git-ignored; real copies live on HF) |
| `results/<job>/` | Outputs pulled from Kaggle (summaries, small JSON/CSV only) |
| `notes/LOG.md` | Experiment log: every run, its config, its result, and the decision taken |
| `scripts/` | Helper scripts (e.g. `fetch.ps1` to pull Kaggle outputs) |

## Compute
- **Kaggle** (main): GPU jobs run in the background. Secret `HF_TOKEN` must be attached to each new notebook once, in the UI.
- **Colab** (secondary): short interactive jobs. Push checkpoints to HF.
- **HF credits ($20):** final evaluation reruns and the Space.
- **Laptop (RTX 3050 6 GB):** code and 4-bit smoke tests only.

## Rules to remember
- Use 2 or more Lexsi repos chained together (plan: CuratorKIT, SafeTune, CircuitKIT, AuditKIT).
- All artefacts go on HF. **Stripped or attacked checkpoints stay private**; publish only metrics.
- Show before/after on more than one metric, plus an HF Article, a Space and (optionally) ROI.
