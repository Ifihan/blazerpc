# Contributing to BlazeRPC

Thank you for your interest in contributing to BlazeRPC. This guide walks you through the process of setting up a development environment, making changes, and submitting a pull request.

## Getting started

### Prerequisites

- **Python 3.10 or later**
- **[uv](https://docs.astral.sh/uv/)** for dependency management

### Setting up the development environment

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/Ifihan/blazerpc.git
cd blazerpc
uv sync --extra dev
```

This installs BlazeRPC in editable mode along with all testing and linting tools.

### Verifying the setup

Run the test suite to confirm everything is working:

```bash
uv run pytest tests/ -v
```

## Making changes

### Branch naming

Create a descriptive branch from `main`:

```bash
git checkout -b feat/adaptive-batching-timeout
git checkout -b fix/health-check-status
git checkout -b docs/streaming-example
```

Use prefixes like `feat/`, `fix/`, `docs/`, or `refactor/` to signal the nature of the change.

### Code style

BlazeRPC uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before committing, run:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

A few conventions to follow:

- **Imports at the top.** All imports belong at the top of the file, before any function or class definitions. The only exception is `import uvloop` in the CLI, which is wrapped in a `try/except` because it is an optional platform-specific dependency.
- **Type annotations.** Every public function should have type annotations on its parameters and return value.
- **Docstrings.** Use triple-quoted docstrings for modules, classes, and public functions. Follow the NumPy docstring style for parameter documentation.

### Type checking

BlazeRPC ships a `py.typed` marker and aims for strict type coverage:

```bash
uv run mypy src/blazerpc/
```

### Writing tests

Tests live in the `tests/` directory and use [pytest](https://docs.pytest.org/). Follow these guidelines:

- **One test file per module.** If you modify `src/blazerpc/runtime/batcher.py`, the corresponding tests belong in `tests/test_batcher.py`.
- **Integration tests** go in `tests/integration/`. These tests start a real gRPC server and exercise the full request lifecycle.
- **Use `pytest.mark.asyncio`** for async test functions. The project is configured with `asyncio_mode = "auto"`, so async tests are detected automatically.
- **Conditional imports.** If a test depends on an optional library (e.g., PyTorch), use `pytest.importorskip("torch")` at the top of the test file so it is skipped gracefully when the library is not installed.

Run the full suite with:

```bash
uv run pytest tests/ -v
```

Run a specific test file or test:

```bash
uv run pytest tests/test_batcher.py -v
uv run pytest tests/test_batcher.py::test_single_item_batch -v
```

### Commit messages

Write clear, concise commit messages that explain **why** the change was made, not just what changed:

```
feat: add partial failure handling to adaptive batcher
```

Use conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Refer to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for more details.

## Submitting a pull request

1. **Push your branch** to your fork or the upstream repository.
2. **Open a pull request** against the `main` branch.
3. **Fill in the description.** Explain what the PR does, why it is needed, and how you tested it.
4. **Ensure CI passes.** The GitHub Actions workflow runs tests across Python 3.10, 3.11, and 3.12, plus linting with Ruff.
5. **Respond to review feedback.** Maintainers may request changes -- push follow-up commits to the same branch.

### What makes a good pull request

- **Focused scope.** Each PR should address a single concern. If you find an unrelated issue while working, open a separate PR for it.
- **Tests included.** Every new feature or bug fix should come with tests that demonstrate the expected behavior.
- **Documentation updated.** If your change affects the public API, update the README or add an example.

## Reporting issues

If you find a bug or have a feature request, [open an issue](https://github.com/Ifihan/blazerpc/issues) on GitHub. Include:

- A clear title and description.
- Steps to reproduce (for bugs).
- The Python version and operating system you are using.
- Any relevant error messages or stack traces.

## Code of conduct

Be respectful and constructive in all interactions. We are building something together, and every contribution -- whether code, documentation, a bug report, or a question -- is valued.
