FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY . .

RUN uv sync

CMD ["uv", "run", "python", "main.py"]
