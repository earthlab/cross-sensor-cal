# Releasing cross-sensor-cal

Follow this checklist when cutting a new release.

1. Ensure `main` is green in CI and documentation builds without warnings.
2. Update version numbers:
   - `pyproject.toml`
   - `CITATION.cff`
   - Any badges or documentation references
3. Update the changelog (create `CHANGELOG.md` if it does not exist) with user-facing changes.
4. Run the quality gates locally:

   ```bash
   ruff check .
   black --check .
   mypy .
   pytest --cov
   ```

5. Build distributions:

   ```bash
   python -m build
   twine check dist/*
   ```

6. Test installation from the wheel in a clean environment.
7. Tag the release (`git tag vX.Y.Z && git push --tags`).
8. Publish a GitHub release with release notes and attach the artifacts if needed.
9. Upload the distributions to PyPI with `twine upload dist/*` (requires credentials).
10. Archive the release on Zenodo to mint/update the DOI (**TODO: add Zenodo concept DOI**).
11. Update `PublicationChecklist.md` to reflect the completed release tasks.
12. Announce the release via the appropriate communication channels.
