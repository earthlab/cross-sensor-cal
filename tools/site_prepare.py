from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "_build"
DEV_PAGE = ROOT / "docs" / "dev" / "doc_drift_report.md"


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    # Run the audit (tolerate failures)
    try:
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "doc_drift_audit.py")])
    except Exception:
        pass
    md = BUILD / "doc_drift_report.md"
    DEV_PAGE.parent.mkdir(parents=True, exist_ok=True)
    if md.exists():
        shutil.copyfile(md, DEV_PAGE)
    else:
        DEV_PAGE.write_text(
            "# Documentation Drift Report\n\n(No drift report generated.)\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
