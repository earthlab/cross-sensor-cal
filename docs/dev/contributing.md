# Contributing & Development Workflow

We welcome contributions from the community. This page outlines expectations for maintaining scientific guarantees, the checks that enforce them, and where to change code safely.

---

## Design philosophy and invariants

- Preserve reproducibility: stages are idempotent and restart-safe; do not add side effects that bypass existing skip logic.
- Respect ordering: `process_one_flightline` and `go_forth_and_multiply` must continue to run stages in the documented sequence.
- Treat filenames as contracts: `FlightlinePaths` and naming utilities define how users and CI locate outputs.
- Outputs over returns: ENVI/Parquet/QA artifacts are the primary interface and must remain stable.

---

## Continuous integration and guarantees

CI workflows enforce the invariants above:

- **Lint + smoke (`ci.yml`, lite job):** `ruff check src tests` and `pytest -q -m "lite"` with `CSCAL_TEST_MODE=lite` on Python 3.11.
- **Full tests (`ci.yml`, unit job):** `pytest -q` with `CSCAL_TEST_MODE=unit` after lite completes.
- **QA quick check (`qa-ci.yml`):** runs `pytest tests/test_qa -q` and renders a QA fixture image/JSON to ensure `_qa.*` outputs remain consistent.
- **Docs build (`docs.yml`):** `python tools/site_prepare.py` then `mkdocs build --strict` plus a best-effort link check.
- **Docs drift (`docs-drift.yml`):** `python tools/doc_drift_audit.py` flags missing mentions of critical artifacts; `_merged_pixel_extraction.parquet` and `_qa.png` are treated as required outputs in examples.

Run the key checks locally before opening a PR:

```bash
ruff check src tests
python -m pytest -m lite
python -m pytest
python tools/site_prepare.py && mkdocs build --strict
```

---

## How to extend the system safely

### Adding or modifying a target sensor
- Update spectral parameters in `spectralbridge/data/landsat_band_parameters.json` (or add an analogous entry) and ensure `standard_resample.py` can consume them.
- Confirm `get_flightline_products`/`FlightlinePaths` emit filenames for the new sensor and that merged Parquet and QA outputs remain unchanged.
- Add tests that validate resampled bands and naming; do not change stage ordering or skip logic.

### Updating brightness or calibration coefficients
- Coefficients live under `spectralbridge/data/brightness/` and are loaded via `brightness_config`. Keep keys stable so regression lookups continue to work.
- Re-run QA-focused tests and inspect QA outputs against Landsat-referenced expectations after changes.

### Modifying QA outputs
- QA artifacts (`<flight_id>_qa.png`, `<flight_id>_qa.json`) are consumed by docs, drift checks, and downstream users. Preserve these filenames even if adding metrics or plots.
- Update `tests/test_qa` and any quick-mode fixtures if output contents change; ensure CI still passes without downloading large datasets.

### Changing extraction or merge logic
- Merged Parquet files (`<flight_id>_merged_pixel_extraction.parquet`) are the contract for downstream analysis. Maintain column consistency and schema ordering expected by users and tests.
- If adding columns, keep existing ones intact and document the change in the outputs contract page.

---

## Relationship to scientific reproducibility

- The repository implements the workflow described in the RSE manuscript, so code changes can affect published analyses.
- Explicit artifacts (ENVI, Parquet, QA) plus drift and QA checks provide an audit trail; contributors are stewards of that record.
- When altering calibration tables, QA logic, or stage ordering, consider how prior runs would be reproduced and update documentation accordingly.

---

## Development environment

- Install in editable mode: `pip install -e ".[dev]"`
- Prefer pure functions and clear logging; avoid needless in-memory copies.
- Update documentation alongside code changes to keep drift checks green.
- Open an issue, work on a feature branch, add tests, and submit a pull request once checks pass.
