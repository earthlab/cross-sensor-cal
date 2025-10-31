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
