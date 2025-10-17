Here’s a **CONTRIBUTING.md** for TinyRecursiveModels-MLX, with best practices, code review, and the test-driven development workflow:

***

# Contributing to TinyRecursiveModels-MLX

Thank you for your interest in contributing!  
We actively welcome improvements, bugfixes, feature additions, and documentation updates as we build the MLX-native Mac version of TinyRecursiveModels.

## Project Philosophy

- **Test-driven development:** Every new feature or bugfix must be accompanied by one or more tests, written before code changes.
- **Atomic, incremental changes:** Prefer small, understandable PRs that only do one thing and are fully covered by tests.
- **Clarity and transparency:** All architectural changes and major decisions are documented in comments and `docs/migration_guide.md`.
- **Community and compatibility:** All contributors are credited—please preserve original authorship in adapted files and cite third-party code as described in the LICENSE.

***

## How to Contribute

### 1. Prepare Your Environment

- Use **Mac Apple Silicon** (M1, M2, M3, M4) and **Python ≥3.9 ARM**.
- Install all dependencies from `requirements.txt` or use the provided `environment.yml`.

### 2. Fork & Branch

- [Fork](https://github.com/gajsanders/TinyRecursiveModels-MLX) the repo and clone it locally.
- Create a new branch for your feature/bugfix:
  ```bash
  git checkout -b feature/my-feature
  ```

### 3. Develop with Tests First

- Write a failing or incomplete test that demonstrates the desired change in `tests/`.
- Develop your feature/code to make the test pass.
- Add or update documentation for all public methods and any complex logic.
- **Don't leave dead or orphaned code:** If you add a new module or utility, ensure it gets wired into an integration test.

### 4. Code Style & Quality

- Run style checks:
  ```bash
  black .
  isort .
  flake8 .
  mypy mlx/
  ```
- Ensure all code is type-annotated and follows PEP8.
- Use clear variable/class/function names.
- Use meaningful docstrings and comments for complex logic.

### 5. Run the Test Suite

- Run all tests before submitting your PR:
  ```bash
  pytest tests/
  ```
- Make sure no test is skipped or failing except those explicitly marked as "experimental" or "deferred".

### 6. Submit a Pull Request

- Push your branch and open a PR against `main`.
- In your PR description, reference the related issue (if any) and summarize your change, affected files, and new tests.
- **If you adapted code from another project or third-party MLX example,** note this in your PR and in comments in the code.
- Tag your PR as “critical”, “experimental”, or “deferred” if appropriate.
- For new contributors, add yourself (and coauthors) to `CONTRIBUTORS.md`.

### 7. Review and Merge

- PRs require at least one code review and all tests passing in CI.
- Major refactors/doc changes should also update `docs/migration_guide.md` with rationale and affected files.
- Prefers “squash and merge” for atomic history.

***

## Contribution Best Practices

- **One test per feature**: Every new function/method should have a direct test and be included in integration tests if relevant.
- **No big jumps**: Break large features into multiple small PRs for easier review.
- **Integration first**: Wire new modules/utilities into the pipeline/testing as soon as possible—no dead code.
- **Document as you go**: Short architecture notes or migration guides are greatly valued, and help future contributors.

***

## Attribution and Licensing

- **Preserve original authorship in all adapted files.**
- Clearly annotate any **third-party code or MLX examples** in docstrings and code comments.
- All contributions are MIT licensed and explicitly credited.

***

## Need Help?

- Open an [issue](https://github.com/gajsanders/TinyRecursiveModels-MLX/issues) for questions, design decisions, or feature requests.
- Check `README.md`, `docs/migration_guide.md`, and example scripts for guidance.
- Tag project maintainers in PRs if a review lag occurs.

***

**Happy coding & testing!**  
— TinyRecursiveModels-MLX Community

---