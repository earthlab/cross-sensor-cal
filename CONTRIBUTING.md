# Contributing

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->

Thank you for your interest in improving **cross-sensor-cal**! This document
describes how to set up a development environment, the project's release
process, and the expectations for opening pull requests.

## Getting started

1. Create and activate a Python 3.10+ virtual environment.
2. Install the project in editable mode with development tools:

   ```bash
   pip install -e .[dev]
   ```

   The `[dev]` extra installs the formatting, linting, and testing utilities
   listed in `requirements-dev.txt`.

## Coding standards

- Format code with **Black** and sort imports via **Ruff**:

  ```bash
  black .
  ruff check --fix .
  ```

- Run the unit tests before submitting a pull request:

  ```bash
  pytest
  ```

- Optional: run **mypy** for a light-weight static type check:

  ```bash
  mypy src/cross_sensor_cal
  ```

## Versioning and releases

The project follows **Semantic Versioning (SemVer)**: MAJOR.MINOR.PATCH.

- Increment the **MAJOR** version for incompatible API changes.
- Increment the **MINOR** version for new functionality that remains
  backward-compatible.
- Increment the **PATCH** version for bug fixes and documentation-only
  improvements.

When preparing a release:

1. Update the `version` field in `pyproject.toml`.
2. Add a new entry to the changelog (to be created) summarizing notable
   changes.
3. Tag the release commit as `vMAJOR.MINOR.PATCH` and push the tag to GitHub.
4. Publish a GitHub release and upload artifacts to PyPI.

## Opening a pull request

1. Create a new branch from `main` describing your change, e.g.
   `git checkout -b feature/resampling-cli`.
2. Commit logically grouped changes with descriptive messages.
3. Ensure `pytest` passes locally and include output in the PR description.
4. Submit the PR with a summary of the changes and any follow-up TODO items.

<!-- FILLME:END -->
