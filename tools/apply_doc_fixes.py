#!/usr/bin/env python
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO/"docs/_build/doc_drift_report.json"

REPLACE_RULES = [
    # Master parquet naming
    (re.compile(r"<prefix>_flightline_master\.parquet|flight\s*line[_\s-]*master\.parquet", re.I),
     "<prefix>_merged_pixel_extraction.parquet"),
    # Stage list ensure Merge + QA present
    (re.compile(r"(^|\n)5\.\s*\*\*Parquet Export\*\*.*", re.I),
     r"\g<1>5. **Parquet Export** for all ENVI products"),
    (re.compile(r"(^|\n)6\.\s*\*\*(?:QA|Quality).*", re.I),
     r"\g<1>6. **Merge Stage** → `<prefix>_merged_pixel_extraction.parquet`\n7. **QA Panel** → `<prefix>_qa.png`"),
    # Common outputs table names
    (re.compile(r"_qa\.png", re.I), "_qa.png"),
    (re.compile(r"_brdfandtopo_corrected_envi\.(img|hdr|json|parquet)", re.I), r"_brdfandtopo_corrected_envi.\1"),
]

def apply_replacements(md_path: Path) -> bool:
    txt = md_path.read_text(encoding="utf-8")
    orig = txt
    for pat, repl in REPLACE_RULES:
        txt = pat.sub(repl, txt)
    if txt != orig:
        md_path.write_text(txt, encoding="utf-8")
        return True
    return False

def main():
    if not REPORT.exists():
        raise SystemExit("Run tools/doc_drift_audit.py first.")
    changed: list[str] = []
    docs_root = REPO/"docs"
    doc_paths = [REPO/"README.md"]
    if docs_root.exists():
        doc_paths.extend(sorted(docs_root.rglob("*.md")))
    for mf in doc_paths:
        if mf.exists() and apply_replacements(mf):
            changed.append(str(mf))
    # Append missing outputs/sensors if not mentioned
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    missing_outputs = rep.get("missing_in_docs", {}).get("outputs", [])
    if missing_outputs and (REPO/"README.md").exists():
        readme = REPO/"README.md"
        txt = readme.read_text(encoding="utf-8")
        marker = "**Missing outputs (auto-added)**"
        if marker not in txt and "Pipeline Outputs" in txt:
            # Append missing outputs list under that section
            txt = re.sub(r"(##\s*Pipeline Outputs[^\n]*\n(?:.*\n){0,50})",
                         r"\1\n**Missing outputs (auto-added)**: " + ", ".join(sorted(set(missing_outputs))) + "\n",
                         txt, flags=re.I)
            readme.write_text(txt, encoding="utf-8")
            changed.append(str(readme))

    # de-duplicate and sort for stable reporting
    changed = sorted(set(changed))

    # Write a short summary
    out = REPO/"docs/_build/doc_fix_summary.txt"
    out.write_text("Changed files:\n" + "\n".join(changed), encoding="utf-8")
    print(f"✅ Applied replacements. Changed: {len(changed)} files")
    print(f"   See {out}")

if __name__ == "__main__":
    main()
