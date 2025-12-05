import subprocess
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "ty_tuff_contributions_last_year.md"

AUTHOR_FILTERS = ["ttuff", "Ty Tuff"]
SINCE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

TAG_KEYWORDS = {
    "brdf": "BRDF",
    "topographic": "BRDF",
    "terrain": "BRDF",
    "parquet": "Parquet",
    "pixel": "Parquet",
    "extract": "Parquet",
    "neon": "NEON pipeline",
    "envi": "NEON pipeline",
    "config": "NEON pipeline",
    "convolution": "Sensor convolution",
    "regression": "Sensor convolution",
    "landsat": "Sensor convolution",
    "micasense": "Sensor convolution",
    "docs": "Docs",
    "doc": "Docs",
    "mkdocs": "Docs",
    "ci": "CI",
    "gocmd": "gocmd",
    "workflow": "CI",
    "github": "CI",
}

LEADING_MAP = {
    "src": "code",
    "gocmd": "gocmd",
    "docs": "docs",
    "tests": "tests",
    "scripts": "scripts",
    "notebooks": "notebooks",
    "constraints": "config",
    "requirements": "config",
    "pyproject.toml": "config",
    "mkdocs.yml": "config",
    "Makefile": "config",
}

def run_git_log():
    commits = {}
    for author in AUTHOR_FILTERS:
        cmd = [
            "git",
            "-C",
            str(REPO_ROOT),
            "log",
            f"--since={SINCE}",
            f"--author={author}",
            "--pretty=format:%H\t%ad\t%s",
            "--date=short",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            commit_hash, date_str, subject = line.split("\t", 2)
            commits[commit_hash] = {
                "hash": commit_hash,
                "date": date_str,
                "subject": subject,
            }
    return list(commits.values())


def detect_tags(subject, files):
    lower_subject = subject.lower()
    tags = set()
    for keyword, tag in TAG_KEYWORDS.items():
        if keyword in lower_subject:
            tags.add(tag)
        else:
            for path in files:
                if keyword in path.lower():
                    tags.add(tag)
                    break
    return tags


def summarize_files(files):
    categories = Counter()
    leading = set()
    for path in files:
        parts = Path(path).parts
        if not parts:
            continue
        lead = parts[0]
        leading.add(lead)
        categories[LEADING_MAP.get(lead, "other")] += 1
    primary = None
    if categories:
        primary = categories.most_common(1)[0][0]
    return leading, primary


def git_show_stat(commit_hash):
    cmd = ["git", "-C", str(REPO_ROOT), "show", "--stat", "--oneline", commit_hash]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    files = []
    for line in lines[1:]:
        if " | " in line:
            files.append(line.split(" | ")[0].strip())
    return files


def group_by_month(commits):
    grouped = defaultdict(list)
    for entry in commits:
        month = entry["date"][:7]
        grouped[month].append(entry)
    for month_entries in grouped.values():
        month_entries.sort(key=lambda c: c["date"], reverse=True)
    return dict(sorted(grouped.items(), reverse=True))


def format_key_changes(leading, primary):
    leading_parts = sorted(leading)
    main = primary if primary else "misc"
    return f"Touched {'/'.join(leading_parts)}; primary focus: {main}."


def build_thematic_summary(commits):
    theme_map = {
        "BRDF": "BRDF + topographic correction",
        "Parquet": "Pixel extraction & Parquet streaming",
        "Sensor convolution": "Sensor convolution and regression",
        "NEON pipeline": "NEON pipeline",
        "Docs": "Data movement, docs, gocmd, and CI",
        "CI": "Data movement, docs, gocmd, and CI",
        "gocmd": "Data movement, docs, gocmd, and CI",
    }
    theme_entries = defaultdict(list)
    for commit in commits:
        if commit["tags"]:
            for tag in commit["tags"]:
                theme = theme_map.get(tag)
                if theme:
                    theme_entries[theme].append(commit)
        else:
            theme_entries["Data movement, docs, gocmd, and CI"].append(commit)
    return theme_entries


def write_markdown(commits):
    if not commits:
        raise SystemExit("No commits found for Ty Tuff in the last year.")

    commits.sort(key=lambda c: c["date"], reverse=True)
    grouped = group_by_month(commits)
    total = len(commits)
    start_date = commits[-1]["date"]
    end_date = commits[0]["date"]

    lines = []
    lines.append("# Ty Tuff contributions to cross-sensor-cal (last 12 months)\n")
    lines.append("_Generated from git history using author filters for \"ttuff\" / \"Ty Tuff\"._\n")
    lines.append("## Summary")
    lines.append(f"- Total commits: {total}")
    lines.append(f"- Time window analyzed: {start_date} to {end_date}")
    lines.append("- High-level themes:")
    lines.append("  - BRDF/topographic correction improvements")
    lines.append("  - NEON ENVI export + config plumbing")
    lines.append("  - Streaming Parquet export / pixel extraction")
    lines.append("  - Sensor convolution & regression tables (Landsat/MicaSense/etc.)")
    lines.append("  - Documentation, examples, gocmd scripts, CI robustness\n")

    lines.append("## Monthly breakdown\n")
    for month, month_commits in grouped.items():
        lines.append(f"### {month}")
        for entry in month_commits:
            tags_str = "".join([f"[{tag}]" for tag in sorted(entry["tags"])]).strip()
            tag_display = f" {tags_str}" if tags_str else ""
            lines.append(f"- `{entry['short']}` ({entry['date']}){tag_display} – {entry['subject']}")
            lines.append(f"  - Key changes: {entry['key_changes']}")
        lines.append("")

    lines.append("## Thematic summary (for NSF reporting)\n")
    themes = build_thematic_summary(commits)
    for theme, items in themes.items():
        hashes = ", ".join([f"{c['short']} ({c['date']})" for c in items])
        lines.append(f"- **{theme}**  ")
        lines.append(f"  - {len(items)} commits ({hashes})")
        lines.append("  - Emphasized robustness, scalability, and reproducibility across processing steps.\n")

    OUTPUT_FILE.write_text("\n".join(lines))


def main():
    commits = run_git_log()
    processed = []
    for entry in commits:
        files = git_show_stat(entry["hash"])
        leading, primary = summarize_files(files)
        tags = detect_tags(entry["subject"], files)
        processed.append(
            {
                **entry,
                "short": entry["hash"][:7],
                "files": files,
                "leading": leading,
                "primary": primary,
                "tags": tags,
                "key_changes": format_key_changes(leading, primary),
            }
        )
    write_markdown(processed)


if __name__ == "__main__":
    main()
