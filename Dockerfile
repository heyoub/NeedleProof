# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS web
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/package.json
COPY packages/contracts/package.json packages/contracts/package.json
RUN pnpm install --frozen-lockfile
COPY apps/web apps/web
COPY packages/contracts packages/contracts
RUN pnpm --filter @needleproof/web build

FROM ghcr.io/astral-sh/uv:0.10.3 AS uv

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NEEDLEPROOF_DATA_DIR=/app/data \
    NEEDLEPROOF_WEB_DIST=/app/apps/web/dist
WORKDIR /app
COPY --from=uv /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock README.md ./
COPY apps/api apps/api
RUN uv sync --frozen --no-dev
COPY data data
COPY --from=web /app/apps/web/dist apps/web/dist
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "needleproof_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
