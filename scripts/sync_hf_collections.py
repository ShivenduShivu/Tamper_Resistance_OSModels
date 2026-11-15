"""Bring the HF collections in line with the submission rules. Dry run by default; add --apply to change anything.

- Models collection: add A1 and A2, and drop the gated Cohere base model (the rules ask for public, un-gated repos;
  the base model is linked from every model card and from the README instead).
- Data & results collection: upload the score-only result folders that are newer than the results dataset
  (Phase A fine-tuning, A1/A2 capability, the paired clean audit, the single-BOS fine-tuning check).

Run from the repo root after `hf auth login`:
    python scripts/sync_hf_collections.py           # shows what would change
    python scripts/sync_hf_collections.py --apply
"""
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

APPLY = "--apply" in sys.argv
api = HfApi()
U = "StarWarrior24x7"
MODELS = f"{U}/heretic-resistant-tiny-aya-models-6af3e1eff9a91556f5535934"
RESULTS_REPO = f"{U}/tiny-aya-tamper-resistance-results"
ADD = [(f"{U}/tiny-aya-global-phaseA-v2-repnoise", "A1: v2 + RepNoise, delays harmful fine-tuning about 3-4x; research checkpoint"),
       (f"{U}/tiny-aya-global-phaseA-v2-booster", "A2: v2 + Booster, negative result; kept for reproducibility")]
DROP = ["CohereLabs/tiny-aya-global"]
FOLDERS = ["results/phaseA_finetune", "results/phaseA_capability", "results/general_audit", "results/phaseA_A0"]
SKIP = {"responses"}   # never upload attacked outputs

col = api.get_collection(MODELS)
have = {i.item_id: i for i in col.items}
for repo, note in ADD:
    if repo in have:
        print(f"already in models collection: {repo}")
        continue
    print(f"{'ADD' if APPLY else 'would add'} {repo}")
    if APPLY:
        api.add_collection_item(MODELS, item_id=repo, item_type="model", note=note, exists_ok=True)
for repo in DROP:
    if repo in have:
        print(f"{'REMOVE' if APPLY else 'would remove'} gated {repo}")
        if APPLY:
            api.delete_collection_item(MODELS, have[repo].item_object_id)

for folder in FOLDERS:
    # only files already public in the GitHub repo: untracked local files (e.g. attacked responses) are never sent
    tracked = subprocess.run(["git", "ls-files", folder], capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in tracked if not (SKIP & set(Path(f).parts))]
    rel = [str(Path(f).relative_to(folder)).replace("\\", "/") for f in files]
    print(f"{'UPLOAD' if APPLY else 'would upload'} {folder}/ ({len(files)} tracked files) -> datasets/{RESULTS_REPO}/{Path(folder).name}/")
    for f in rel: print("    ", f)
    if APPLY and rel:
        api.upload_folder(folder_path=folder, path_in_repo=Path(folder).name, repo_id=RESULTS_REPO, repo_type="dataset",
                          allow_patterns=rel, commit_message=f"Add score-only {Path(folder).name}")
if not APPLY:
    print("\nDry run only. Re-run with --apply to change the collections.")
