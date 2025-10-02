# Developer guide

This page summarizes the development workflow, coding standards, and release
process for contributors working on cross-sensor-cal.

## Local setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

This installs the project in editable mode plus the tooling stack used in CI:
`ruff`, `black`, `pytest`, `pytest-cov`, `mypy`, and MkDocs dependencies.

## Quality checks

Run the automated checks locally before pushing commits:

```bash
ruff check .
black --check .
mypy src/cross_sensor_cal
pytest --cov
```

The GitHub Actions workflow enforces the same checks across Python 3.9–3.12.

## Documentation

Documentation is authored with MkDocs Material. To preview locally run:

```bash
mkdocs serve
```

New guides live in the `docs/` directory; keep navigation entries synced via
`mkdocs.yml`. Use Google-style docstrings in the source code so mkdocstrings can
render API documentation automatically.

## Releases

1. Update `pyproject.toml` and `CITATION.cff` with the new version number.
2. Run the full test suite and build artifacts:

   ```bash
   python -m build
   twine check dist/*
   ```

3. Tag the release (`git tag vX.Y.Z && git push --tags`).
4. Publish the release on GitHub and upload to PyPI.
5. Archive the release on Zenodo to mint a DOI (see `PublicationChecklist.md`).
6. Announce the release and update any downstream integrations.

## Additional resources

- [CONTRIBUTING.md](https://github.com/earthlab/cross-sensor-cal/blob/main/CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](https://github.com/earthlab/cross-sensor-cal/blob/main/CODE_OF_CONDUCT.md)
- [RELEASING.md](https://github.com/earthlab/cross-sensor-cal/blob/main/RELEASING.md)
