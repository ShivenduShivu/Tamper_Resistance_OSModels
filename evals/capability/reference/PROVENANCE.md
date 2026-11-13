# Vendored reference: openai/human-eval execution module

- File: reference/human_eval_execution.py
- Source: https://raw.githubusercontent.com/openai/human-eval/37c4dd63798c3c9ba32fa69a2fb49c5e2c43a181/human_eval/execution.py
- Repo: https://github.com/openai/human-eval (archived; commit 37c4dd63798c3c9ba32fa69a2fb49c5e2c43a181,
  master HEAD at retrieval, 2025-01-17 "fix broken eval", verified via the GitHub commits API)
- SHA256: 79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c
  (matches the independent reviewer's retrieved copy, hash stated in review)
- License: MIT (reference/HUMAN_EVAL_LICENSE, same commit)
- Retrieved: 2026-11-13
- Enforcement: coding_exec.py re-hashes this file at import and the per-item
  driver re-hashes its in-sandbox copy before importing it; any mismatch
  refuses to run. test_reference_module_pinned locks the pin.

The scoring path calls this module's own check_correctness() unchanged. The
only harness-level deviations are OUTSIDE the module and frozen in
coding_exec.py: timeout 5s (upstream default 3.0; hardware-dependent,
recalibrated on the actual image at the first authorized run), the frozen
raw-generation extraction rule, and the Modal Sandbox host-isolation wrapper.
