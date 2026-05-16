---
description: always check toolsets
# applyTo: 'Describe when these instructions should be loaded by the agent based on task context' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

# Project Instructions: locpick

## Scope
These instructions apply to the entire repository.

## Environment
- **Always** run Python commands inside the `locpick` conda environment: `conda run -n locpick python ...`
- **Always** run pytest with pytest-xdist: `pytest tests/ -n auto`

## Tool Preferences
- **Prefer** MCP/language-server tools (pylance, serena) for reading and manipulating source files when they are available.
- Shell tools like `sed`, `awk`, or regex-based bulk replacement are acceptable when faster or more practical, but prefer structured edits when possible.
- Use `multi_replace_string_in_file` for batch edits when appropriate.

## Code Conventions
- Private subpackages and modules must use the underscore prefix (e.g., `_jax`, `_kernels`, `_solvers`, `_sampling`).
- The public API surface is minimal: `data`, `models`, `dgp`, `results`, and `spec` are the only top-level public subpackages.

## Workflow
- For non-trivial changes, **draft a plan first** and present it for approval before implementation.
- After any edit that touches code (not documentation), run the full test suite to verify no regressions.
