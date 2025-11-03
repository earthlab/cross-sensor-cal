# Contributing

- Run tests
- Run docs drift audit before committing doc changes:
  ```bash
  python tools/doc_drift_audit.py
  python tools/apply_doc_fixes.py
  ```
- Preview docs locally:
  ```bash
  pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-macros-plugin
  mkdocs serve
  ```

## Docs PR checklist
- [ ] Use **Purpose • Inputs • Outputs • Run it • Pitfalls** on how-to pages
- [ ] Include a CLI copy block and (when relevant) a short Python snippet
- [ ] Link Stages ↔ Schemas ↔ Outputs ↔ Troubleshooting
- [ ] `mkdocs build` passes locally

## Continuous integration checks

Pull requests to `main` run four main checks:

1. **CI / lite** – fast linting and smoke tests.
2. **CI / unit** – full Python test suite.
3. **Docs Drift Check / audit** – verifies docs and code stay in sync.
4. **QA quick check / qa** – runs a minimal QA pipeline on a small fixture and uploads the PNG/JSON/PDF artifacts.

This layout keeps feedback fast while ensuring the QA pipeline and docs stay
healthy. The QA quick check now runs once per PR (and optionally on pushes to
`main`) instead of duplicating work on every branch push.
