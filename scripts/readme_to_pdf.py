"""README.md -> docs/Heretic_Resistant_Tiny_Aya.pdf (Mermaid rendered, details expanded). Run from repo root."""
import re, subprocess, os, pathlib, markdown
ROOT = pathlib.Path(__file__).resolve().parent.parent
md = (ROOT / "README.md").read_text(encoding="utf-8")
blocks = []
def keep(m):
    blocks.append(m.group(1)); return f"\n\nMERMAIDBLOCK{len(blocks) - 1}\n\n"
md = re.sub(r"```mermaid\n(.*?)```", keep, md, flags=re.S)
md = md.replace("<details>", "<details open>").replace('<div align="center">', '<div align="center" markdown="1">')
blocks = [b.replace("flowchart LR", "flowchart TD") for b in blocks]   # tall diagrams read better on A4
html = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html", "toc", "sane_lists"])
for i, b in enumerate(blocks):
    html = html.replace(f"<p>MERMAIDBLOCK{i}</p>", f'<pre class="mermaid">{b}</pre>')
css = """
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1F2A44; font-size: 10.4pt; line-height: 1.5; }
h1 { font-size: 26pt; margin: 4px 0; color: #0B1426; } h2 { font-size: 15pt; border-bottom: 2px solid #36D399; padding-bottom: 3px; margin-top: 22px; color: #0B1426; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #0B1426; } a { color: #2563EB; text-decoration: none; }
img { max-width: 100%; page-break-inside: avoid; } p img[alt*="badge"], a img { max-width: none; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.2pt; page-break-inside: avoid; }
th { background: #0B1426; color: #fff; text-align: left; padding: 5px 6px; } td { border-bottom: 1px solid #E3E6EE; padding: 4px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #F6F8FB; }
blockquote { border-left: 4px solid #36D399; background: #EEF8F3; margin: 10px 0; padding: 6px 12px; }
code { background: #EEF1F6; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre.mermaid { background: #fff; text-align: center; page-break-inside: avoid; }
details summary { font-weight: 600; margin: 8px 0; } sub { color: #5B6475; }
"""
doc = f"""<!doctype html><html><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/"><style>{css}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'base', themeVariables:{{fontFamily:'Segoe UI', primaryColor:'#EEF2FA', primaryBorderColor:'#1F2A44', lineColor:'#5B6475'}}}});</script>
</head><body>{html}</body></html>"""
tmp = ROOT / "docs" / "_readme_print.html"; tmp.write_text(doc, encoding="utf-8")
out = ROOT / "docs" / "Heretic_Resistant_Tiny_Aya.pdf"
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
import tempfile, time
before = out.stat().st_mtime if out.exists() else 0
with tempfile.TemporaryDirectory() as prof:   # own profile: a running Edge window otherwise swallows the job silently
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--user-data-dir={prof}", f"--print-to-pdf={out}", tmp.as_uri()], check=True, capture_output=True, timeout=180)
tmp.unlink()
if not out.exists() or out.stat().st_mtime <= before:
    raise SystemExit("Edge did not write a new PDF")
print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
