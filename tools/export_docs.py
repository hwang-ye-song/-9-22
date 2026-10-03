# -*- coding: utf-8 -*-
"""노트북 → 정리본 Markdown → HTML → PDF"""
import json, re, sys
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

proj = Path(sys.argv[1]).resolve()
nb = json.loads((proj / "Vision_AI_특강_정리.ipynb").read_text(encoding="utf-8"))

parts = []
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        parts.append(src)
        continue
    m = re.search(r'Image\("images/([^"]+)"\)', src)
    if m:
        cap = src.splitlines()[0].lstrip("# ").strip()
        parts.append(f"![{cap}](images/{m.group(1)})\n\n*{cap}*")
        continue
    out = "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
    block = f"```python\n{src}\n```"
    if out.strip():
        block += f"\n\n**실행 결과**\n\n```text\n{out.rstrip()}\n```"
    parts.append(block)

doc = "\n\n".join(parts)
doc = doc.replace("> 💡 이 노트북의 **코드 셀은", "> 💡 원본 노트북의 **코드 셀은")
(proj / "특강_정리본.md").write_text(doc + "\n", encoding="utf-8")

# ---------- HTML for PDF ----------
def prep(text):
    # python-markdown 은 목록/표 앞에 빈 줄이 필요하다
    lines, out = text.split("\n"), []
    in_code = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and i > 0:
            prev = lines[i - 1]
            starts_block = re.match(r"^(\s*[-*] |\s*\d+\. |\|)", line)
            prev_same = re.match(r"^(\s*[-*] |\s*\d+\. |\||\s{2,}\S)", prev)
            if starts_block and prev.strip() and not prev_same and not prev.strip().startswith("```"):
                out.append("")
        line = line.replace("- [ ]", "- ☐")
        out.append(line)
    t = "\n".join(out)
    # 목록 항목 안의 들여쓴 코드블록(3칸) → 4칸
    t = re.sub(r"\n   (```|source|python)", lambda m: "\n    " + m.group(1), t)
    return t

body = markdown.markdown(prep(doc), extensions=["tables", "fenced_code", "sane_lists"])
css = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Malgun Gothic', 'Segoe UI Emoji', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1f2328; }
h1 { font-size: 22pt; border-bottom: 3px solid #1a73e8; padding-bottom: 6px; }
h2 { font-size: 16pt; color: #1a73e8; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 28px; page-break-after: avoid; }
h3 { font-size: 13pt; margin-top: 18px; page-break-after: avoid; }
h4 { font-size: 11pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; vertical-align: top; }
th { background: #f0f5ff; }
code { font-family: Consolas, 'Malgun Gothic', monospace; background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #0d1117; color: #e6edf3; padding: 10px 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-all; font-size: 8.6pt; line-height: 1.45; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; font-family: Consolas, 'Malgun Gothic', monospace; }
blockquote { border-left: 4px solid #1a73e8; background: #f6f9ff; margin: 8px 0; padding: 6px 12px; color: #333; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #d0d7de; }
"""
html = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Vision AI 특강 정리</title><style>{css}</style></head><body>{body}</body></html>"
tmp = proj / "_print.html"
tmp.write_text(html, encoding="utf-8")
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    page.goto(tmp.as_uri())
    page.wait_for_load_state("networkidle")
    page.pdf(path=str(proj / "Vision_AI_특강_정리.pdf"), format="A4", print_background=True,
             display_header_footer=True, header_template="<span></span>",
             footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span> / <span class='totalPages'></span></div>",
             margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
    if len(sys.argv) > 2:  # 미리보기 스크린샷
        page.set_viewport_size({"width": 900, "height": 1200})
        page.screenshot(path=sys.argv[2], full_page=True)
    b.close()
tmp.unlink()
print("ok")
