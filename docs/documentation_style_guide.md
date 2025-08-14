# Cross-Sensor Calibration Documentation Style Guide

This guide establishes conventions for writing documentation in the Cross-Sensor Calibration project. Its goal is to create a linear, pedagogical narrative that makes the package easy to understand and adopt.

## Philosophy
- **Clarity first.** Explain concepts in plain language before introducing technical jargon.
- **Narrative flow.** Documentation should guide the reader from inputs through processing to outputs in a logical order.
- **Pragmatic examples.** Every section should include code snippets or workflows that users can run directly.
- **Minimal prerequisites.** Link to background materials rather than assuming extensive prior knowledge.

## Structure
1. **Overview** – Briefly describe the purpose of the component and how it fits into the larger workflow.
2. **Prerequisites** – List required data, dependencies, and setup steps.
3. **Step-by-step tutorial** – Present instructions in chronological order.
4. **Reference** – Provide detailed API descriptions, parameters, and links to source code.
5. **Next steps** – Suggest follow-on tasks or sections.

## Style
- Use Markdown headings (`#`, `##`, `###`) to organize content.
- Write in the second person (“you”) and active voice.
- Keep sentences concise; aim for one idea per sentence.
- Use numbered lists for sequences and bullet lists for options.
- Highlight file names, parameters, and code using backticks (`like_this`).
- Wrap code examples in fenced blocks with the appropriate language tag.
- Include diagrams or figures when they clarify complex processes.
- Cross-link related documents with relative paths.

## Formatting
- Line length: soft wrap at 100 characters.
- Use American English spelling.
- Date format: YYYY-MM-DD.
- Reference issues or pull requests with full links.

## Maintenance
- Each documentation page must include a `Last updated: YYYY-MM-DD` line at the end.
- When updating docs, ensure examples are tested against the current codebase.
- Run `pytest` before committing changes that affect code examples.

Following this guide will keep the documentation consistent and approachable for new contributors and users.
