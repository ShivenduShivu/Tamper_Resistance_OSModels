Hi! I ran into this while using AuditKIT's red-team runner to evaluate a model for the Lexsi hackathon.

**What happens**

If you run more than one probe without passing detectors, the runner gathers the default detectors of all the probes and applies every one of them to every probe. So in a run like `probes=["jailbreak", "refusal"]`, the jailbreak prompts also get checked by `RefusalDetector`. A model that correctly refuses a jailbreak is then counted as a failure.

A small example with a model that always refuses:

```python
from auditkit.redteam import RedTeamRunner

refuse = lambda prompts: ["I'm sorry, but I can't help with that." for _ in prompts]

RedTeamRunner(refuse).run(probes=["jailbreak"])              # 16 checks, 16 passed
RedTeamRunner(refuse).run(probes=["refusal"])                # 5 checks, 0 passed (expected)
RedTeamRunner(refuse).run(probes=["jailbreak", "refusal"])   # 39 checks, 13 failed, e.g. [jailbreak/refusal]
```

The jailbreak part should still pass when it runs together with other probes, and the total should be 21, not 39.

**The fix**

- When no detectors are given, each probe is now checked only with its own default detectors. If you pass detectors explicitly, they still apply to every probe, as before.
- `detectors_used` keeps a stable order. It came from a `set` before, so the order could change between runs.
- Added a test that a multi-probe run gives the same totals as running each probe on its own.

**Testing**

- `tests/test_redteam.py`: the new test fails on the current `main` and passes with this change. The other tests still pass.
- Full suite: the failures are the same before and after. The only remaining error on my machine is `test_vllm_offline_engine`, which needs vLLM.
