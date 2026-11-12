"""Publish docs/HF_ARTICLE.md as the model card of a public HF repo (allowed by the organisers when HF Articles need PRO).
Run from the repo root: python scripts/publish_article.py"""
from huggingface_hub import HfApi
FRONT = """---
license: cc-by-nc-4.0
language: [en, hi, ar, fr, es]
tags: [safety, tamper-resistance, abliteration, heretic, lexsi-hackathon]
base_model: CohereLabs/tiny-aya-global
---

"""
a = HfApi(); R = "StarWarrior24x7/heretic-resistant-tiny-aya-article"
a.create_repo(R, repo_type="model", private=False, exist_ok=True)
card = FRONT + open("docs/HF_ARTICLE.md", encoding="utf-8").read()
a.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=R, commit_message="Hackathon article: Heretic-Resistant Tiny Aya")
for c in a.list_collections(owner="StarWarrior24x7"):
    a.add_collection_item(c.slug, item_id=R, item_type="model", exists_ok=True, note="Project article")
print("Article live at: https://huggingface.co/" + R)
