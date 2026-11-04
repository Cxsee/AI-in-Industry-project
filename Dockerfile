FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first so that code changes do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra mf --extra llm

COPY src ./src
RUN uv sync --locked --no-dev --extra mf --extra llm

# Mount datasets and outputs here at run time.
ENV MOVIEREC_DATA_DIR=/data MOVIEREC_RESULTS_DIR=/results
VOLUME ["/data", "/results"]

CMD ["uv", "run", "--no-sync", "python", "-c", "import movierec; print(movierec.__version__)"]
