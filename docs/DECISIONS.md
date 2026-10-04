# Decisions Log

This file records assumptions and design decisions made during implementation
when the specification was ambiguous.

## Phase 0

- **2024-01-01 — Build system**: Using `setuptools` with `src` layout. Chose over
  `hatchling`/`flit` for maximum compatibility.
- **2024-01-01 — Makefile vs scripts**: Using a `Makefile` with `test` and `lint`
  targets. On Windows, users can use `make` via Git Bash, WSL, or run the
  underlying commands directly.
- **2024-01-01 — Config library**: Using `pydantic` v2 with `pydantic-settings`
  for YAML config loading. Provides validation, type safety, and serialization.
