# CI verification

This file documents the CI workflow that was added to the repository and the steps used to verify it locally.

Workflow
- Path: `.github/workflows/ci.yml`
- Purpose: Install the package in editable mode (`pip install -e .`) and run the test-suite with `PYTHONPATH=.` so CI imports the repository code (not an installed package named `mlx`).

How to verify locally (match CI)

1. Create / activate a virtualenv (or use an existing `.venv_test`):

```bash
python -m venv .venv_test
source .venv_test/bin/activate
```

2. Install editable package and test deps:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e . --no-deps
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
```

3. Run tests with PYTHONPATH set to the repository root:

```bash
PYTHONPATH=. pytest -q
```

Expected result (local verification performed):
- `70 passed` (test count and runtime may vary on different machines).

CI trigger
- The workflow runs automatically on pushes and on pull requests targeting `main`.
- To run CI for verification, open a PR from your branch (for example `chore/fix-tests-shim`) or push to the branch.

Notes
- If CI shows failures on a runner (ubuntu/macOS), include the CI logs in the PR and I can iterate on dependency or runner configuration.
- This workflow mirrors the local setup used by developers and helps avoid import collisions with third-party packages named `mlx`.
