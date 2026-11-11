"""Publish the results-explorer Space. Run from the repo root: python scripts/publish_space.py"""
from huggingface_hub import HfApi
a = HfApi(); S = "StarWarrior24x7/heretic-resistant-tiny-aya"
a.create_repo(S, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
a.upload_folder(folder_path="space", repo_id=S, repo_type="space",
                allow_patterns=["app.py", "data.json", "requirements.txt", "README.md"], commit_message="Results explorer")
for slug in [c.slug for c in a.list_collections(owner="StarWarrior24x7")]:
    a.add_collection_item(slug, item_id=S, item_type="space", exists_ok=True)
print("https://huggingface.co/spaces/" + S)
