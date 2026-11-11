
import sys, json, torch, circuitkit as ck
from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.tasks.registry import register_task
from datasets import load_dataset
name, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
harm = list(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"))[:32]
benign = list(load_dataset("mlabonne/harmless_alpaca", split="test")["text"])[:32]
ds = SafetyPromptAdapter().adapt(harm, name="jbb_hvb", source="JBB", max_records=32, refusal_token="I",
                                 compliance_token="Sure", benign_controls=benign, pairing_mode="harmful_vs_benign")
register_task(NormalizedTaskSpec(ds, name="jbb_hvb"))
r = {}
m = ck.load_model(path, dtype="float16")
r["mem_after_load_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
c = ck.discover(m, "jbb_hvb", algorithm="eap", n_examples=8, batch_size=1)
r["mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
r["top30"] = [str(n) for n in c.top_nodes(30)]
for attr in ["faithfulness", "n_edges", "n_nodes"]:
    if hasattr(c, attr):
        v = getattr(c, attr); r[attr] = str(v() if callable(v) else v)[:400]
try:
    c.save(out + ".circuit")
except Exception as e:
    r["save_err"] = str(e)[:200]
json.dump(r, open(out, "w"))
