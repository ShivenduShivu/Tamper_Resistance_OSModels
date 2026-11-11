import sys, json, subprocess, os
arm = sys.argv[1]; M = "/tmp/mv2/bin/modal"
subprocess.run([M, "volume", "get", "lexsi-v2-ckpt", f"/pfx/{arm}_dynamic_eval.json", f"/tmp/{arm}.json", "--force"], check=True, capture_output=True)
d = json.load(open(f"/tmp/{arm}.json")); bad = []
if arm in ("B", "C"):
    for k in ("grad_norm_all_finite", "loss_all_finite", "hooks_inactive_after_train", "trained_full_epoch"):
        if d.get(k) is not True: bad.append((k, d.get(k)))
    if d.get("n_steps") != 295: bad.append(("n_steps", d.get("n_steps")))
    ts = d.get("train_stats", {})
    if ts.get("harm") != 1200 or ts.get("benign") != 1155 or ts.get("dropped_no_label") != 0: bad.append(("train_stats", ts))
    if arm == "C" and ts.get("with_opener") != 1200: bad.append(("opener", ts))
    if arm == "B" and ts.get("with_opener") != 0: bad.append(("opener", ts))
c = d["counts"]
if c["harm_test"] != 100 or c["xstest_safe"] != 250 or c["mmlu"] != 1000: bad.append(("counts", c))
if d["v2_rev"] != "377d402a45f92c5c787f3509084afca75eda3e8f": bad.append(("rev", d["v2_rev"]))
print("VALID" if not bad else f"INVALID {bad}"); sys.exit(1 if bad else 0)
