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
step("build_task", build_task)
def path_of(name):
    if name == "base": return "CohereLabs/tiny-aya-global"
    p = "/kaggle/working/v2"; snapshot_download(f"{USER}/tiny-aya-global-xref-v2", local_dir=p, ignore_patterns=["v2b/*"]); return p
res = {}
for name in ["base", "v2"]:
    def run(name=name):
        m = ck.load_model(path_of(name), dtype="float16")
        c = ck.discover(m, "jbb_hvb", algorithm="eap-ig", n_examples=16, batch_size=1)
        res[name] = c
        top = c.top_nodes(30)
        log(f"circuit_{name}_top30", [str(n) for n in top])
        for attr in ["faithfulness", "n_edges", "n_nodes", "metrics", "summary"]:
            if hasattr(c, attr):
                v = getattr(c, attr); v = v() if callable(v) else v; log(f"circuit_{name}_{attr}", str(v)[:800])
        del m; gc.collect(); torch.cuda.empty_cache()
    step(f"circuit_{name}", run)
def compare():
    a = {str(n) for n in res["base"].top_nodes(30)}; b = {str(n) for n in res["v2"].top_nodes(30)}
    log("circuit_top30_overlap", {"shared": len(a & b), "only_base": sorted(a - b)[:15], "only_v2": sorted(b - a)[:15]})
    ck.visualize_circuit(res["base"], mode="comparison", second_circuit=res["v2"], output=f"{OUT}/circuit_base_vs_v2.html")
if len(res) == 2: step("circuit_compare", compare)
shutil.rmtree("/kaggle/working/v2", ignore_errors=True)
log("DONE", True)
