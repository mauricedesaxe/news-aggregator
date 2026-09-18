# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-default-groups

EXPOSE 8080
CMD ["sh", "-c", "COOKIE_SECURE=true uvicorn romanian_news.reader.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
