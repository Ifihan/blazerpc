# Contributing

For the full contributing guide -- including environment setup, code style conventions, testing instructions, and pull request guidelines -- see [CONTRIBUTING.md](https://github.com/Ifihan/blazerpc/blob/main/CONTRIBUTING.md) in the repository root.

## Quick reference

```bash
# Clone and install
git clone https://github.com/Ifihan/blazerpc.git
cd blazerpc
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/blazerpc/
```
