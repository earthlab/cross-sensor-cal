# Contributing & Development Workflow

We welcome contributions from the community. This page outlines the workflow for proposing changes, running tests, and maintaining documentation quality.

---

## Basic principles

- Keep all processing steps **transparent and auditable**  
- Do not introduce new scientific claims without citation or review  
- Maintain consistent naming and directory conventions  
- Ensure any new feature is accompanied by documentation updates  

---

## Development environment

Install the package in editable mode:

```bash
pip install -e ".[dev]"
Run tests:
pytest
Build documentation:
mkdocs build
Making code changes
Open an issue describing the proposed change
Create a feature branch
Add or update unit tests
Update relevant documentation pages
Submit a pull request
Modifying documentation
Update Markdown files directly under docs/
Ensure navigation in mkdocs.yml remains valid
Keep sections coherent and avoid duplication
Run mkdocs serve locally to preview
Adding datasets or SRFs
Place files under cross_sensor_cal/data/
Update relevant stages to load the new assets
Document usage in tutorials
Coding standards
Prefer pure functions and isolated dependencies
Avoid large in-memory operations unless necessary
Log processing decisions clearly
Next steps
Package architecture
Codex edit guidelines

---
