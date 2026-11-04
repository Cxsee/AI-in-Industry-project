# AI-in-Industry-project

Movie recommendations beyond accuracy: we compare classic recommenders, an LLM used as a ranker and
a hybrid with a constrained selection step (OR-Tools CP-SAT), on accuracy, novelty, diversity,
exposure and cost.

Andrea Caselli and Francesco Forlani · AI in Industry exam project + Project Work

## Setup

You need [uv](https://docs.astral.sh/uv/); it installs Python 3.12 by itself.

```bash
uv sync
uv run pytest
```

The heavier parts of the stack are optional, so install only what you are working on:

| Extra | Used for | Install with |
| --- | --- | --- |
| `mf` | Matrix factorization (`implicit`) | `uv sync --extra mf` |
| `content` | Content-based embeddings (`sentence-transformers`) | `uv sync --extra content` |
| `llm` | LLM ranker through Ollama | `uv sync --extra llm` |

`uv sync` removes whatever you don't ask for, so pass every extra you need each time.

## Layout

```text
src/movierec/   project code
tests/          tests
notebooks/      exploration and result analysis
data/           datasets (not committed)
results/        experiment outputs (not committed)
```

Copy `.env.example` to `.env` to change the data folders or the Ollama server.

## Before committing

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

## Docker

```bash
docker build -t movierec .
docker run --rm -v "$PWD/data:/data" -v "$PWD/results:/results" movierec
```
