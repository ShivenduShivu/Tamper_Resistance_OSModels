"""Publish the repository README as the model card of the public HF article repo (organisers accept a README when
HF Articles need PRO). The README is the single source of truth; relative links are rewritten to absolute GitHub URLs.

Run from the repo root:
    python scripts/publish_article.py           # dry run: writes build/article_preview.md, uploads nothing
    python scripts/publish_article.py --apply   # upload
"""
import re
import sys
from pathlib import Path

GH = "https://github.com/ShivenduShivu/Tamper_Resistance_OSModels"
RAW = "https://raw.githubusercontent.com/ShivenduShivu/Tamper_Resistance_OSModels/main/"
R = "StarWarrior24x7/heretic-resistant-tiny-aya-article"
FRONT = """---
license: cc-by-nc-4.0
language: [en, hi, ar, fr, es]
tags: [safety, tamper-resistance, abliteration, heretic, lexsi-hackathon]
base_model: CohereLabs/tiny-aya-global
---

> This article mirrors the project README: [""" + GH + """](""" + GH + """). Code, score files and figure sources live there.

"""


def absolute(m):
    bang, text, target = m.group(1), m.group(2), m.group(3)
    if re.match(r"^(https?:|mailto:|#)", target):
        return m.group(0)
    path = target.lstrip("./")
    if bang or re.search(r"\.(png|jpg|jpeg|gif|svg)$", path.split("#")[0], re.I):
        return f"{bang}[{text}]({RAW}{path})"
    kind = "tree" if path.endswith("/") else "blob"
    return f"[{text}]({GH}/{kind}/main/{path})"


text = Path("README.md").read_text(encoding="utf-8")
text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", absolute, text)
text = re.sub(r'src="(?!https?:)([^"]+)"', lambda m: f'src="{RAW}{m.group(1).lstrip("./")}"', text)
card = FRONT + text

if "--apply" not in sys.argv:
    out = Path("build/article_preview.md"); out.parent.mkdir(exist_ok=True)
    out.write_text(card, encoding="utf-8")
    left = re.findall(r"\]\((?!https?:|mailto:|#)[^)]+\)", card)
    print(f"Dry run: wrote {out} ({len(card)} chars); relative links left: {len(left)}")
    print("Re-run with --apply to upload to https://huggingface.co/" + R)
    sys.exit(0)

from huggingface_hub import HfApi
a = HfApi()
a.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=R,
              commit_message="Sync article with the project README")
print("Article live at: https://huggingface.co/" + R)
