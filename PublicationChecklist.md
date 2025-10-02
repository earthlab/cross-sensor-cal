# Publication checklist (JOSS readiness)

This checklist tracks the tasks required to prepare **cross-sensor-cal** for a
Journal of Open Source Software (JOSS) submission. Items marked as ✅ have been
completed in this pass. Items with **TODO** require human follow-up.

## Project metadata
- ✅ LICENSE present (GPL-3.0-or-later); confirm suitability for target venues.
- ✅ Clear repository description in `README.md`.
- ✅ `CITATION.cff` populated with placeholders (**TODO: replace author details and DOI once available**).

## Packaging & installation
- ✅ `pyproject.toml` defines metadata and dependencies; `pip install .[dev]` works.
- ✅ Minimum Python version stated (>=3.9).
- ✅ `environment.yml` provides reproducible conda environment (**TODO: validate heavy dependencies on clean machines**).

## Documentation
- ✅ README includes badges, quickstart, example, citation, support, and license sections.
- ✅ MkDocs configuration with Material theme; docs build via `mkdocs build`.
- ✅ API reference wired through mkdocstrings (**TODO: expand narrative around each public function**).

## Testing & CI
- ✅ Pytest suite includes import and example smoke tests.
- ✅ GitHub Actions CI runs linting (ruff/black), type checks, and pytest with coverage across Python 3.9–3.12.
- **TODO:** Improve automated test coverage beyond smoke tests to target ≥60%.

## Examples & data
- ✅ `examples/basic_calibration_workflow.py` demonstrates a minimal workflow.
- ✅ `examples/data/README.md` explains how to obtain real NEON data (**TODO: add direct links to recommended sample datasets**).

## Quality & style
- ✅ Ruff, Black, and mypy configurations documented (`docs/dev-guide.md`).
- ✅ CONTRIBUTING.md and CODE_OF_CONDUCT.md available (**TODO: insert maintainer contact email**).

## Community
- ✅ Issue and PR templates added under `.github/`.
- ✅ Support policy outlined in README (**TODO: confirm support cadence with maintainers**).

## Reuse & archival
- ✅ Version placeholder `0.1.0` defined in metadata and package.
- **TODO:** Create initial tagged release (`v0.1.0`) and publish artifacts.
- **TODO:** Register Zenodo concept DOI and add badge/link.
- **TODO:** Consider Software Heritage archival.

## JOSS paper
- ✅ `paper/paper.md` skeleton created with all required sections.
- ✅ `paper/bibliography.bib` seeded with placeholder references (**TODO: insert real DOIs and expand references**).
- ✅ `paper/README.md` documents submission steps.
- **TODO:** Fill in summary, statement of need, validation evidence, and acknowledgements.

## Submission
- **TODO:** Perform final review against the [JOSS author guidelines](https://joss.theoj.org/about#author_guidelines).
- **TODO:** Submit the paper through the JOSS portal and respond to reviewer feedback.

Keep this checklist up to date as milestones are completed.
