# ─────────────────────────────────────────────────────────────────────────────
# RAGSmith – Dockerfile
# Multi-stage build: keeps the final image lean
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build / install dependencies ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for psycopg2, PyMuPDF, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY . .

# Create data directories (will be overridden by EBS volume mount on EC2)
RUN mkdir -p data/indexes data/chunks data/uploads exports

# Non-root user for security
RUN adduser --disabled-password --gecos "" ragsmith && \
    chown -R ragsmith:ragsmith /app
USER ragsmith

EXPOSE 8000

# Health check — hits /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Gunicorn with uvicorn workers for production
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
