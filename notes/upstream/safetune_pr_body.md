Hi! Two small things I noticed while using `extract_refusal_direction` and `SpectralEntropyMonitor` on Tiny Aya for the Lexsi hackathon.

**1. Swapped progress-bar labels**

`extract_refusal_direction` passes the harmful prompts as `safe_prompts` to `extract()`, on purpose, to flip the sign of the direction. A side effect is that the progress bar says "harmless" while it's reading the harmful prompts, and the other way round. CAA has the same thing with its positive and negative sets.

`extract()` now takes an optional keyword-only `labels` argument, and both callers pass the right names. The default labels are unchanged, so nothing else is affected.

**2. `scan()` docstring**

The `SpectralEntropyMonitor` docstring says `scan()` returns `(prompt_idx, layer_idx, z_score)`. It actually returns `(prompt_idx, layer_idx, entropy, z_score)`, so I updated the docstring.

**Testing**

I ran `tests/diagnostic/test_vector_extraction.py`, `test_spectral_monitor.py`, `tests/audit/test_S02_caa.py` and `test_E07_spectral.py` before and after the change: 25 passed in both cases (Python 3.13, Linux). This only changes labels and a docstring.
