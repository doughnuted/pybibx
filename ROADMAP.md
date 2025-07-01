# Project Roadmap

This document outlines the plan for modernizing the `pybibx` library.

## Phase 1: Project Setup & Analysis (Completed)
- [x] Create `ROADMAP.md` and `ISSUES.md`
- [x] Analyze `setup.py` and existing `pyproject.toml`
- [x] Identify all project dependencies from imports
- [x] Formalize the project structure

## Phase 2: Dependency Management (In Progress)
- [x] Migrate dependency management to `pyproject.toml`
- [ ] Update all dependencies to latest compatible versions
- [ ] Remove legacy packaging files (`setup.py`, `MANIFEST.in`)

## Phase 3: Automated Testing
- [ ] Set up `pytest` framework
- [ ] Create a `tests/` directory
- [ ] Implement initial smoke tests for core functionality

## Phase 4: Continuous Integration (CI)
- [ ] Create a `.github/workflows/` directory
- [ ] Implement a GitHub Actions workflow for CI
- [ ] The workflow will run linters (`ruff`) and `pytest`

## Phase 5: Automated Documentation
- [ ] Set up `MkDocs` for documentation generation
- [ ] Create initial documentation structure in a `docs/` directory
- [ ] Configure `mkdocs.yml`
- [ ] Generate initial version of the documentation website
