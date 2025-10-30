#!/usr/bin/env python
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]

# ---------- Helpers ----------
def read(p: Path) -> str:
    try: return p.read_text(encoding="utf-8")
    except: return ""

def py_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.py") if "venv" not in p.parts and ".eggs" not in p.parts and "site-packages" not in p.parts]

def uniq(xs): return sorted(set(xs))

# ---------- Extract “truth” from code ----------
def extract_cli_entrypoints(pyproject: Path) -> Dict[str,str]:
    d = {}
    txt = read(pyproject)
    # toml-lite parse for entrypoints
    m = re.search(r"\[project\.scripts\](.+?)\n\s*\[", txt, re.S)
    if not m:
        m = re.search(r"\[project\.scripts\](.+)", txt, re.S)
    if m:
        block = m.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not line or "=" not in line: continue
            k,v = [s.strip() for s in line.split("=",1)]
            v = v.strip('"\'')
            d[k] = v
    return d

def extract_argparse_calls(py: Path) -> Dict[str, List[str]]:
    """Return CLI flags per script by static scan of argparse add_argument calls."""
    flags = []
    code = read(py)
    for m in re.finditer(r"add_argument\(([^)]*)\)", code):
        args = m.group(1)
        # crude capture of flag names
        for f in re.findall(r"--[a-zA-Z0-9\-_]+", args):
            flags.append(f)
    return {py.name: uniq(flags)} if flags else {}

def extract_log_messages() -> List[str]:
    msgs = []
    for p in py_files(REPO / "src"):
        txt = read(p)
        for m in re.finditer(rf'logger\.(info|warning|error)\(\s*f?["\'](.+?)["\']', txt):
            msgs.append(m.group(2))
        for m in re.finditer(rf'print\(\s*f?["\'](.+?)["\']', txt):
            msgs.append(m.group(1))
        for m in re.finditer(r'UserWarning:\s*(.+)', txt):
            msgs.append(m.group(1))
    return uniq(msgs)

def extract_constants_and_patterns() -> Dict[str, List[str]]:
    truths = {"outputs":[], "sensors":[], "suffixes":[], "globs":[], "qa_flags":[], "stage_markers":[]}
    patt_outputs = [
        r"(_envi\.img)", r"(_envi\.hdr)", r"(_brdfandtopo_corrected_envi\.(?:img|hdr|json|parquet))",
        r"(_landsat_[a-z0-9\+]+_envi\.(?:img|hdr|parquet))",
        r"(_micasense(?:_to_match_[a-z_+\d]+)?_envi\.(?:img|hdr|parquet))",
        r"(_merged_pixel_extraction\.parquet)", r"(_qa\.png)", r"(_qa\.json)"
    ]
    patt_sensors = [r"landsat_tm", r"landsat_etm\+", r"landsat_oli2?", r"micasense(?:_to_match_[a-z_+]+)?"]
    patt_suffix = [r"_brdfandtopo_corrected_envi", r"_merged_pixel_extraction", r"_envi", r"_qa"]
    patt_globs = [r"\*\*/\*original\*\.parquet", r"\*\*/\*corrected\*\.parquet", r"\*\*/\*resampl\*\.parquet"]
    patt_qa = [r"scene_ratio_out_of_range", r"scene_negatives_excess", r"scene_over_one_excess"]

    code = "\n".join(read(p) for p in py_files(REPO / "src"))
    for pat in patt_outputs:
        truths["outputs"] += re.findall(pat, code)
    for pat in patt_sensors:
        truths["sensors"] += re.findall(pat, code)
    for pat in patt_suffix:
        truths["suffixes"] += re.findall(pat, code)
    for pat in patt_globs:
        truths["globs"] += re.findall(pat, code)
    for pat in patt_qa:
        truths["qa_flags"] += re.findall(pat, code)

    # stage markers: look for emojis/phrases used in logs
    for msg in extract_log_messages():
        if any(s in msg for s in ["Download", "ENVI export", "BRDF+topo", "Convolving", "Parquet", "Merge", "QA panel", "Finished pipeline"]):
            truths["stage_markers"].append(msg)
    for k in truths: truths[k] = uniq(truths[k])
    return truths

def derive_prefix_rule() -> str:
    # Verify new naming rule exists in code
    code = "\n".join(read(p) for p in py_files(REPO / "src"))
    if "_merged_pixel_extraction.parquet" in code:
        return "<prefix>_merged_pixel_extraction.parquet"
    return "<prefix>_flightline_master.parquet?"

# ---------- Scan docs ----------
def scan_docs(root: Path) -> Dict[str, List[Tuple[Path,str]]]:
    targets = {}
    pats = {
        "outputs": r"(?:_envi\.img|_envi\.hdr|_brdfandtopo_corrected_envi\.(?:img|hdr|json|parquet)|_landsat_[a-z0-9\+]+_envi\.(?:img|hdr|parquet)|_micasense[^\s]*_envi\.(?:img|hdr|parquet)|_merged_pixel_extraction\.parquet|_qa\.png|_qa\.json)",
        "stages": r"(Download|Export|Topographic|BRDF|Convolution|Parquet|Merge|QA panel)",
        "cli": r"(python\s+-m\s+bin\.[a-zA-Z0-9_]+|csc-[a-z0-9\-]+)",
        "flags": r"(--[a-z0-9\-]+)",
        "sensors": r"(landsat_tm|landsat_etm\+|landsat_oli2?|micasense[^\s]*)",
    }
    docs_root = REPO/"docs"
    doc_paths = [REPO/"README.md"]
    if docs_root.exists():
        doc_paths.extend(sorted(docs_root.rglob("*.md")))
    for mf in doc_paths:
        txt = read(mf)
        hits = []
        for key, pat in pats.items():
            for m in re.finditer(pat, txt, re.I):
                hits.append((key, m.group(0)))
        targets[str(mf)] = hits
    return targets

# ---------- Compare ----------
def compare(truth: Dict[str, List[str]], docs_hits: Dict[str, List[Tuple[str,str]]]) -> Dict:
    report = {"missing_in_docs": {}, "stale_in_docs": {}, "notes": []}
    # Build doc aggregates
    doc_values = {"outputs": set(), "sensors": set(), "flags": set()}
    for f, pairs in docs_hits.items():
        for key, val in pairs:
            if key == "sensors":
                doc_values["sensors"].add(val)
            if key == "outputs":
                doc_values["outputs"].add(val)
            if key == "flags":
                doc_values["flags"].add(val)
    # Missing: in code truth but not in docs
    for k in ["outputs","sensors"]:
        report["missing_in_docs"][k] = sorted(set(truth.get(k, [])) - set(doc_values.get(k, [])))
    # Stale: in docs but not in code (e.g., old names)
    for k in ["outputs","sensors"]:
        report["stale_in_docs"][k] = sorted(set(doc_values.get(k, [])) - set(truth.get(k, [])))

    # Naming rule
    report["naming_rule"] = derive_prefix_rule()
    # Stage presence (coarse)
    report["stages_found"] = truth.get("stage_markers", [])
    return report

def main():
    pyproject = REPO/"pyproject.toml"
    entry = extract_cli_entrypoints(pyproject)
    flags = {}
    for p in py_files(REPO/"bin"):
        flags.update(extract_argparse_calls(p))
    for p in py_files(REPO/"src"):
        flags.update(extract_argparse_calls(p))

    truth = extract_constants_and_patterns()
    docs_hits = scan_docs(REPO)
    rep = compare(truth, docs_hits)
    rep["entry_points"] = entry
    rep["cli_flags"] = flags

    out_dir = REPO/"docs/_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"doc_drift_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # Pretty markdown
    md = ["# Documentation Drift Report",
          "",
          "## Naming rule",
          f"- Merged parquet naming in code: **{rep['naming_rule']}**",
          "",
          "## Missing items in docs",
          f"- Outputs: {', '.join(rep['missing_in_docs'].get('outputs', [])) or '—'}",
          f"- Sensors: {', '.join(rep['missing_in_docs'].get('sensors', [])) or '—'}",
          "",
          "## Stale items in docs",
          f"- Outputs: {', '.join(rep['stale_in_docs'].get('outputs', [])) or '—'}",
          f"- Sensors: {', '.join(rep['stale_in_docs'].get('sensors', [])) or '—'}",
          "",
          "## Entry points (pyproject)",
          json.dumps(rep.get("entry_points", {}), indent=2),
          "",
          "## CLI flags discovered",
          json.dumps(rep.get("cli_flags", {}), indent=2),
          "",
          "## Stage markers found in logs",
          *[f"- {s}" for s in rep.get("stages_found", [])],
          "",
         ]
    if rep.get("stale_in_docs", {}).get("sensors"):
        md.append("TODO: Review sensor labels in docs for formatting issues.")
        md.append("")
    (out_dir/"doc_drift_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Wrote {out_dir/'doc_drift_report.md'} and doc_drift_report.json")

if __name__ == "__main__":
    main()
