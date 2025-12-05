# Guidelines for AI / Codex Edits

This page defines explicit rules for using AI tools (such as GitHub Copilot, ChatGPT, or Codex) when editing this repository. The goal is to maintain scientific accuracy, structural integrity, and documentation consistency.

---

## Core rules

### 1. Do NOT invent scientific claims
All scientific statements must come from:
- NEON documentation  
- sensor metadata  
- existing package logic  
- peer-reviewed sources  

### 2. Do NOT alter conceptual meaning without review
Edits must preserve:
- scientific correctness  
- processing assumptions  
- pipeline logic  

### 3. Do NOT change navigation or folder structure unless instructed
Documentation layout is intentional and should not be reorganized automatically.

### 4. Do NOT remove placeholders, schemas, or metadata
These are required for reproducibility.

---

## Code-editing rules

- Do not introduce new APIs without discussion  
- Do not refactor internal architecture without review  
- Avoid automatic renaming of variables or functions  
- Confirm that unit tests pass after any edit  

---

## Documentation-editing rules

- Only modify the files explicitly referenced in a prompt  
- Never rewrite entire sections unless provided with exact text  
- Preserve all fenced code blocks  
- Maintain consistency with mkdocs navigation  

---

## When NOT to use AI tools

Avoid automatic editing when:

- adding new scientific explanations  
- modifying BRDF/topographic formulas  
- updating spectral response files  
- changing reflectance scaling logic  

These require domain expertise.

---

## Safe tasks for AI tools

- inserting text provided by the user  
- reorganizing text when explicitly instructed  
- generating placeholder scaffolding  
- performing mechanical refactors with tests  

---

## Summary

AI tools in this repository should behave like careful editors—not authors.  
Always prioritize scientific correctness, reproducibility, and transparency.

---

## Next steps

- [Contributing & development workflow](contributing.md)  
- [Package architecture](architecture.md)
