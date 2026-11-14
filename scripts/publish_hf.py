"""Sync the Hugging Face model cards, Space and generation configs with this repo.

Dry run by default: prints what would change. Add --apply to upload.
Needs write access to StarWarrior24x7: log in with `hf auth login` (never paste the token anywhere).

    pip install huggingface_hub
    python scripts/publish_hf.py            # dry run
    python scripts/publish_hf.py --apply    # upload

Cards that still contain a TODO marker are skipped until the TODO is resolved.
"""
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

APPLY = "--apply" in sys.argv
api = HfApi()
U = "StarWarrior24x7"

CARDS = [  # (repo file, repo id, repo type)
    ("docs/hf_cards/v2.md", f"{U}/tiny-aya-global-xref-v2", "model"),
    ("docs/hf_cards/v1.md", f"{U}/tiny-aya-global-xref-v1", "model"),
    ("docs/hf_cards/a1.md", f"{U}/tiny-aya-global-phaseA-v2-repnoise", "model"),
    ("docs/hf_cards/a2.md", f"{U}/tiny-aya-global-phaseA-v2-booster", "model"),
    ("docs/hf_cards/ds.md", f"{U}/tiny-aya-extended-refusal", "dataset"),
]
SPACE = (f"{U}/heretic-resistant-tiny-aya", ["space_static/index.html", "space_static/README.md"])

# Chat turns end with <|END_OF_TURN_TOKEN|> (id 6), the tokenizer's eos. The shipped generation_config.json
# only lists <EOS_TOKEN> (id 3), so generate() never stops at the end of an answer.
END_OF_TURN = 6
GEN_CONFIGS = [  # (repo id, subfolder)
    (f"{U}/tiny-aya-global-xref-v2", None),
    (f"{U}/tiny-aya-global-xref-v2", "v2b"),
    (f"{U}/tiny-aya-global-xref-v1", None),
    (f"{U}/tiny-aya-global-phaseA-v2-repnoise", None),
    (f"{U}/tiny-aya-global-phaseA-v2-booster", None),
]


def upload(local, path_in_repo, repo_id, repo_type, msg):
    print(f"{'UPLOAD' if APPLY else 'would upload'} {local} -> {repo_type}:{repo_id}/{path_in_repo}")
    if APPLY:
        api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type, commit_message=msg)


def remote_text(repo_id, filename, repo_type):
    try:
        return Path(hf_hub_download(repo_id, filename, repo_type=repo_type)).read_text(encoding="utf-8")
    except Exception:
        return None


for local, repo_id, repo_type in CARDS:
    text = Path(local).read_text(encoding="utf-8")
    if "TODO" in text:
        print(f"SKIP {local}: resolve the TODO first")
        continue
    if remote_text(repo_id, "README.md", repo_type) == text:
        print(f"up to date: {repo_id}")
        continue
    upload(local, "README.md", repo_id, repo_type, "Sync model card with the project repo")

space_id, files = SPACE
for local in files:
    name = Path(local).name
    if remote_text(space_id, name, "space") == Path(local).read_text(encoding="utf-8"):
        print(f"up to date: {space_id}/{name}")
        continue
    upload(local, name, space_id, "space", "Sync results explorer with the project repo")

for repo_id, sub in GEN_CONFIGS:
    name = f"{sub}/generation_config.json" if sub else "generation_config.json"
    cfg = json.loads(remote_text(repo_id, name, "model"))
    eos = cfg.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if END_OF_TURN in eos:
        print(f"up to date: {repo_id}/{name} eos={eos}")
        continue
    cfg["eos_token_id"] = [END_OF_TURN] + [e for e in eos if e is not None and e != END_OF_TURN]
    print(f"{repo_id}/{name}: eos_token_id {eos} -> {cfg['eos_token_id']}")
    tmp = Path(tempfile.mkdtemp()) / "generation_config.json"
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    upload(str(tmp), name, repo_id, "model", "Stop generation at <|END_OF_TURN_TOKEN|>")

if not APPLY:
    print("\nDry run only. Re-run with --apply to upload.")
