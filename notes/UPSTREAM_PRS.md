# Upstream contribution candidates (Lexsi Labs repos)

We log issues found while integrating SafeTune, CircuitKIT and AuditKIT.
A PR is opened only when a fix is genuine, small and tested.

| # | Repo | What we found | Evidence (file:line / run) | Fix size | Status |
|---|---|---|---|---|---|
| 1 | AuditKIT | When no detectors are given, `RedTeamRunner` merges every probe's detectors and applies them to all probes. A correct refusal on a jailbreak probe then trips `RefusalDetector` and is marked failed. `total_probes` counts prompt×detector rows, not prompts. | `auditkit/redteam/runner.py:147, 200-206`; no multi-probe test in `tests/test_redteam.py` | small | **REPRODUCED** (Kaggle v17): a model that always refuses scores 16/16 on jailbreak alone, but in a jailbreak+refusal run gets 39 "probes" (not 21) and 13 failures, e.g. `[jailbreak/refusal]`. Fix + test committed (a622052, ShivenduShivu) on branch `fix/per-probe-detectors`; the new test fails on 1.1.7 and passes with the fix; full-suite failures identical. **PR opened: https://github.com/Lexsi-Labs/AuditKIT/pull/35** |
| 2 | AuditKIT | The docstring example `Probe("jailbreak_dan")` creates a probe with no prompts, so it returns 0 results with `pass_rate` = 1.0 (a misleading pass). | `redteam/__init__.py:9`, `runner.py:59-60` | small | **PR opened: https://github.com/Lexsi-Labs/AuditKIT/pull/36** (raises a clear error; docstring fixed; test added) |
| 3 | AuditKIT | No CI workflow runs pytest (the step is commented out). | `patch-release.yml:244` | small/med | candidate |
| 4 | CircuitKIT | `register_advbench` and the other `register_*` functions don't expose `refusal_token`, `compliance_token` or `pairing_mode`, although the internals support them. Tokenizers like Cohere's need them. | `tasks/safety_datasets.py:86-114`, `safety_prompt.py:169` | small | **HIT IN OUR WORK (Day-5b):** every public `register_*` helper builds `answer_contrastive` pairs, whose corrupt prompt equals the clean prompt, so `ck.discover()` rejects them ("32/32 records have a corrupt prompt identical to their clean prompt"). The public API cannot produce a discoverable safety task. Fix idea: pass `pairing_mode`, `refusal_token`, `compliance_token` and `benign_controls` through `register_*`, and/or document the needed mode. Needs: CircuitKIT fork |
| 5 | SafeTune | The `SpectralEntropyMonitor.scan` docstring says it returns 3-tuples; it actually returns `(idx, layer, entropy, z)`. | `instrumentation/evaluate/suite/spectral.py:167-170` | tiny (docs) | prepared together with #6 (see below) |
| 6 | SafeTune | The progress-bar labels "harmless" and "harmful" are swapped when called from `extract_refusal_direction`. | `core/runtime/inference/vector_extraction.py:198-199` | tiny | **prepared** (commit b20944c on local branch `fix/extraction-labels-and-scan-doc`; adds a keyword-only `labels` arg, callers pass correct names). Needs: SafeTune fork + a CPU test run |
| 7 | all | CI doesn't test Python 3.13, which Kaggle runs. | CI configs | small | candidate: only if we hit a 3.13 issue |

Notes:
- All three use the LSAL-1.2 licence (academic and personal use free; contributions grant Lexsi relicensing rights).
- Maintainers fold outside PRs into combined PRs (SafeTune #64/#68, CircuitKIT #35).

## Pacing (user preference)
- At most 1–2 PRs per repo at a time. Space new ones out over the week so they don't look like spam.
- Raise next: SafeTune #5+#6 (one PR), after the fork and a test run.
- Then CircuitKIT #4, only if our CircuitKIT run needs custom refusal or compliance tokens.
- Hold: #3 and #7 (CI), unless the maintainers welcome them.

- Note (Day-5c): CircuitKIT EAP discovery on tiny-aya (3.35B, 262k vocab) runs out of memory on a 16 GB T4 even with plain EAP and 8 examples. This is worth mentioning in the #4 PR description as context, not as a separate PR.
