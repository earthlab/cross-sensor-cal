# Cross-Sensor Cal Publication Checklist

> Living document for preparing the project for packaging and public release. Update
> the status boxes and notes as tasks are completed.

## 1. Package Structure & Metadata
- [x] Confirm the canonical package name (`cross-sensor-cal` vs `cross_sensor_cal`) and ensure the code lives under a single importable package directory (e.g., `src/cross_sensor_cal`). *(2024-03-22: moved modules into `src/cross_sensor_cal/` and updated imports project-wide.)*
- [x] Replace the minimal `setup.py` with a `pyproject.toml` using PEP 621 metadata (name, version, description, authors, URLs, keywords, classifiers) and optional `setup.cfg` for configuration. Align metadata with README and docs. *(Added `pyproject.toml` with setuptools build-backend and metadata; removed `setup.py`.)*
- [x] Add `__init__.py` exports and package-level documentation so users can discover public APIs easily. *(Created `cross_sensor_cal/__init__.py` exposing key pipeline functions.)*
- [x] Decide on versioning scheme (CalVer or SemVer) and document it in CONTRIBUTING along with release tagging conventions. *(Documented SemVer workflow and release tagging steps in CONTRIBUTING.)*
- [ ] Audit repository for large data or notebooks that should be excluded from source distributions. Use `.gitignore`/`MANIFEST.in` to prevent shipping bulky artifacts.

## 2. Dependencies & Environment
- [ ] Inventory runtime vs. optional dependencies by reviewing modules (e.g., GDAL, HyTools, Ray) and refactor imports so optional features are guarded or extra-requirements are defined.
- [ ] Translate `environment.yaml` into concise dependency groups (`install_requires`, `extras_require`, dev/test/doc extras). Remove pinned Windows-specific or conda-only packages that do not belong on PyPI.
- [ ] Provide a lightweight sample dataset or clearly document external data requirements so users can run example pipelines after installing from PyPI.
- [x] Add a `requirements-dev.txt` or equivalent to unify tooling for contributors (formatters, linters, docs builders). *(Created `requirements-dev.txt` aligned with the `[project.optional-dependencies]` section.)*

## 3. Code Quality, Testing & Tooling
- [ ] Expand automated test coverage beyond `tests/test_file_sort.py` to cover each major pipeline stage (`neon_to_envi`, resampling, masking, polygon extraction, MESMA). Include integration tests with fixture data.
- [ ] Configure continuous integration (e.g., GitHub Actions) to run unit tests, linting (ruff/flake8), type checking (mypy/pyright), and documentation builds on each push/PR.
- [ ] Adopt consistent code style (Black, Ruff, isort) and document formatting commands in `CONTRIBUTING.md`.
- [ ] Evaluate adding type hints and optional static typing checks to critical modules for maintainability.
- [ ] Ensure `pytest` configuration (`pyproject.toml`/`pytest.ini`) ignores heavyweight data paths and sets up necessary environment variables for tests.

## 4. Documentation & Community Files
- [ ] Finish filling placeholders in `README.md`, `CITATION.cff`, and MkDocs pages. Include feature overview, supported sensors, and end-to-end workflow diagrams.
- [ ] Verify that documentation builds cleanly with `mkdocs build` and publish instructions (`mkdocs gh-deploy` or Read the Docs) as part of release workflow.
- [ ] Add usage examples demonstrating both library API calls and CLI entry points, ideally with runnable Jupyter notebooks linked from docs.
- [ ] Update `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (create if missing), and issue/PR templates to guide external contributors once the package is public.
- [ ] Provide citation and acknowledgement guidance consistent across README, docs, and metadata.

## 5. Distribution Artifacts & QA
- [ ] Run `python -m build` to generate sdist/wheel and inspect contents (ensure no unnecessary files, confirm console scripts are installed).
- [ ] Execute `twine check dist/*` to validate metadata and `pip install dist/*.whl` in a clean virtual environment for smoke tests.
- [ ] Document hardware/software prerequisites (GDAL, PROJ) and include troubleshooting tips for installation on Linux/macOS/Windows.
- [ ] Automate changelog generation (`CHANGELOG.md`) per release with notable features and breaking changes.
- [ ] Establish release checklist (tagging, GitHub release notes, PyPI upload) and capture in this document or `RELEASING.md`.

## 6. Licensing & Governance
- [x] Confirm license (GPLv3) is acceptable for target distribution venues; include `LICENSE` file in repository and package manifest. *(Added the GPLv3 text at `LICENSE` and referenced it from `pyproject.toml`.)*
- [ ] Ensure all third-party code, data, and documentation comply with the chosen license and attribution requirements.
- [ ] Identify maintainers and add contact information/support policy in README and docs.

## 7. Post-Release Follow-up
- [ ] Monitor initial PyPI release install stats/issues and iterate on documentation gaps.
- [ ] Announce release channels (EarthLab blog, mailing lists) and track feedback for roadmap planning.
- [ ] Schedule periodic dependency and security audits (Dependabot, `pip-audit`) and plan for long-term maintenance.

---
_Last updated: 2025-08-15_
