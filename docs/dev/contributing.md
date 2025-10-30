# Contributing

- Run tests
- Run docs drift audit before committing doc changes:
  ```bash
  python tools/doc_drift_audit.py
  python tools/apply_doc_fixes.py
  ```
- Preview docs locally:
  ```bash
  pip install mkdocs mkdocs-material mkdocstrings[python]
  mkdocs serve
  ```
