.PHONY: check-docs
check-docs:
	python scripts/check_docs_links.py

docs-audit:
	python tools/doc_drift_audit.py
	echo "Report at docs/_build/doc_drift_report.md"

docs-fix:
	python tools/apply_doc_fixes.py
	git status --porcelain
