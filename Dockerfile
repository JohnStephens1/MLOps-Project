FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md src/ ./

RUN uv sync --frozen

# entry point necessary, otherwise uv will be used as entrypoint -> janky error messages
ENTRYPOINT []


FROM base AS notebook

RUN uv sync --frozen --group notebook


FROM notebook AS dev

RUN uv tool install rust-just
