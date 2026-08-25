# ──────────────────────────────────────────────────────────────
# Stage 1: Dependency resolution with uv
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN uv pip install --no-deps .

# ──────────────────────────────────────────────────────────────
# Stage 2: Slim runtime
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 venom && \
    useradd --uid 1000 --gid venom --create-home venom

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/lancedb && \
    chown -R venom:venom /app

USER venom

ENTRYPOINT ["venomsearch"]
CMD ["--help"]
