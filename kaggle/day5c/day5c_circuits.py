# Day-5c (v2): each model in its own process; EAP (not IG), 8 examples, expandable segments.
# Day-5c: CircuitKIT refusal-circuit comparison, original vs v2 (harmful_vs_benign pairing; the public register_* helpers only build degenerate pairs).
import os, sys, json, time, glob, re, subprocess, random, urllib.request, shutil
OUT = "/kaggle/working"; os.makedirs(OUT, exist_ok=True)
DTYPE = "float16"
def log(k, v):
    print(k, v, flush=True)
    with open(f"{OUT}/summary.jsonl", "a") as f: f.write(json.dumps({k: v}) + "\n")
def find_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e: print("secret unavailable:", e, flush=True)
    for fp in glob.glob("/kaggle/input/**/*", recursive=True):
        if os.path.isfile(fp) and os.path.getsize(fp) < 2048:
            m = re.search(r"hf_[A-Za-z0-9]{20,}", open(fp, errors="ignore").read())
            if m: return m.group(0)
tk = find_token()
if not tk: log("FATAL", "no HF_TOKEN"); sys.exit(1)
os.environ["HF_TOKEN"] = tk
def step(name, fn, *a):
    t0 = time.time()
    try:
        r = fn(*a); log(f"{name}_sec", round(time.time() - t0)); return r
    except Exception as e:
        import traceback; log(f"{name}_ERR", f"{type(e).__name__}: {e}"[:600] + " | " + traceback.format_exc()[-900:])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "circuitkit"], check=True)
import circuitkit as ck, gc, torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
from circuitkit.data.normalized_task import NormalizedTaskSpec
from circuitkit.tasks.registry import register_task
USER = HfApi().whoami()["name"]
def build_task():
    harm = list(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"))[:32]
    benign = list(load_dataset("mlabonne/harmless_alpaca", split="test")["text"])[:32]
    ds = SafetyPromptAdapter().adapt(harm, name="jbb_hvb", source="JailbreakBench/JBB-Behaviors", max_records=32,
                                     refusal_token="I", compliance_token="Sure", benign_controls=benign, pairing_mode="harmful_vs_benign")
    log("task_paired", f"{ds.n_paired}/{len(ds)}"); register_task(NormalizedTaskSpec(ds, name="jbb_hvb"))
def path_of(name):
    if name == "base": return "CohereLabs/tiny-aya-global"
    p = "/kaggle/working/v2"; snapshot_download(f"{USER}/tiny-aya-global-xref-v2", local_dir=p, ignore_patterns=["v2b/*"]); return p
WORKER = r"""
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
"""
open("/kaggle/working/ck_worker.py", "w").write(WORKER)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
res = {}
for name in ["base", "v2"]:
    def run(name=name):
        out = f"{OUT}/circuit_{name}.json"
        p = subprocess.run([sys.executable, "/kaggle/working/ck_worker.py", name, path_of(name), out], env=env, capture_output=True, text=True, timeout=3600)
        if p.returncode != 0: raise RuntimeError(p.stderr[-1500:])
        res[name] = json.load(open(out)); log(f"circuit_{name}", {k: v for k, v in res[name].items() if k != "top30"})
        log(f"circuit_{name}_top30", res[name]["top30"])
    step(f"circuit_{name}", run)
if len(res) == 2:
    a, b = set(res["base"]["top30"]), set(res["v2"]["top30"])
    log("circuit_top30_overlap", {"shared": len(a & b), "only_base": sorted(a - b)[:15], "only_v2": sorted(b - a)[:15]})
shutil.rmtree("/kaggle/working/v2", ignore_errors=True)
log("DONE", True)
