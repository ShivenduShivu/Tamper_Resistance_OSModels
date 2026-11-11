Hi! I ran into this while trying to find refusal circuits in Tiny Aya with CircuitKIT for the Lexsi hackathon.

**What happens**

The `register_*` helpers in `tasks/safety_datasets.py` always build tasks with `pairing_mode="answer_contrastive"`. In that mode, the corrupt prompt is the same as the clean prompt; only the answer token differs. `discover()` then rejects the task:

```
ValidationError: 32/32 (100%) records have a corrupt prompt identical to their clean prompt ...
EAP discovery has no contrastive signal for these examples.
```

So the example at the top of the module (`eap-ig` on `advbench`) can't run as written. There's also no way to choose another mode through the public helpers. `_register` supports `pairing_mode`, but the helpers don't pass it on.

**The fix**

- `_register` now defaults to `"harmful_vs_benign"`, which matches the module docstring ("pairs each harmful prompt with a benign control").
- The `register_*` helpers and `register_all_safety` accept `**kwargs` and pass them on to `SafetyPromptAdapter.adapt()`. That covers `pairing_mode`, `refusal_token`, `compliance_token` and `benign_controls`. The tokens matter for models whose first reply token isn't `" I"` or `" Sure"` (Tiny Aya's chat template, for example).
- Added `tests/data/test_safety_register_pairing.py`. It checks that the default pairs are contrastive and that the options are passed through.

**Testing**

- The new test fails on current `main` and passes with this change.
- The existing tests in `tests/data` give the same results before and after.

Happy to keep `"answer_contrastive"` as the default and only expose the options, if you'd rather not change the default.
