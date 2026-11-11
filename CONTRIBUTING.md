# Contributing

Bug reports, documentation improvements, and code contributions are welcome. Open an issue for defects or proposed changes that would benefit from discussion, then submit a pull request with a focused change and its verification steps.

## Development setup

Python contributions use Python 3.11 or later and `uv`:

```bash
uv sync --locked --extra dev --extra data
uv run ruff check src tests
uv run pytest
```

Frontend changes use Node.js 22 or later and `pnpm`:

```bash
cd web
pnpm install
pnpm test
pnpm run build
```

The MLX training and inference paths require macOS on Apple Silicon and the optional `mlx` dependency group. Tests should not require downloading model weights. BANKING77 and all examples must retain their source and license attribution. Synthetic examples must be identified as synthetic; do not submit customer messages, credentials, or other sensitive data.

## Pull requests

Describe the user-visible change, link related issues, and list the checks that were run. Include tests for behavior changes and update documentation when command behavior, configuration, or limitations change. Keep changes focused and follow the existing Ruff and TypeScript formatting conventions.
