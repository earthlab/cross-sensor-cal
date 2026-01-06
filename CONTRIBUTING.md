# Contributing

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
Thank you for your interest in improving Cross-Sensor Calibration! This guide
summarizes the project conventions so that contributions are easy to review and
safe to deploy in the production cloud environment.

## Ways to contribute
- Report issues and pipeline regressions through
  [GitHub Issues](https://github.com/earthlab/spectralbridge/issues).
- Improve documentation, tutorials, and example notebooks.
- Add tests that increase coverage without expanding production data size.
- Triage dependency compatibility problems or OS-specific installation bugs.

## Development workflow
1. **Fork and branch**: create a feature branch for every change.
2. **Environment**: use Python 3.10 with the dependencies listed in
   `environment.yaml`. Avoid pinning additional packages unless strictly
   necessary.
3. **Editable install**: `pip install -e .[dev]` (see `requirements-dev.txt` once
   available) to install tooling such as pytest, Ruff, and MyPy.
4. **Pre-commit checks**: run formatting (`ruff format`, `ruff check`, `isort`)
   and tests (`pytest`) before opening a pull request.
5. **Documentation**: update MkDocs pages or docstrings when behaviour changes.
   New user-facing features must include a docs update or changelog entry.

## Versioning and releases
- We follow **Semantic Versioning (SemVer)**: `MAJOR.MINOR.PATCH`.
- Tag releases in Git with `vMAJOR.MINOR.PATCH` (e.g., `v0.2.0`).
- Update the version in `pyproject.toml`/`setup.cfg` (temporary home: `setup.py`)
  and in `CITATION.cff` as part of release preparation.
- Document notable changes in `CHANGELOG.md` once it is introduced.

## Testing guidelines
- Tests must run via `pytest` using only fixture data stored under
  `tests/data/`.
- Integration tests should rely on the minimal public sample flight line; do not
  reference private or cloud-only paths.
- When working on modules that interact with external systems (iRODS, CyVerse,
  cloud buckets), isolate the integration points behind feature flags so that
  local unit tests remain deterministic.

## Coding standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) and ensure Ruff passes with
  the repository configuration.
- Use type hints for new or refactored functions. Run `mypy` before submitting.
- Keep file path manipulations centralized in the existing helper modules. When
  in doubt, open an issue before changing any production path conventions.
- Avoid committing large data artifacts; use `.gitignore` and document download
  steps instead.

## Communication
- Use GitHub Issues or Discussions for asynchronous questions.
- Flag urgent operational issues with the `priority:high` label so maintainers
  can triage quickly.
- Pull requests require at least one review from a maintainer before merging.

Thanks for helping make Cross-Sensor Calibration reliable and reproducible!
<!-- FILLME:END -->
