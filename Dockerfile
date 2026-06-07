# =============================================================================
# Thunders AI - Multi-stage Docker Build
# =============================================================================
# Build:  docker build -t thunders-ai -f deployment/docker/Dockerfile .
# Run:    docker run -p 8000:8000 thunders-ai
# GPU:    docker run --gpus all -p 8000:8000 thunders-ai
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements/base.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r base.txt

# Copy source and build the package
COPY . .
RUN python -m build

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="Thunders AI Team <team@thunders-ai.dev>"
LABEL org.opencontainers.image.title="Thunders AI"
LABEL org.opencontainers.image.description="Unified AI Platform — LLM, Vision, Speech, Robotics & Security"
LABEL org.opencontainers.image.url="https://github.com/thunders-ai/thunders-ai"
LABEL org.opencontainers.image.source="https://github.com/thunders-ai/thunders-ai"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built wheel from builder and install
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy application source (for configs, templates, etc.)
COPY . .

# Create non-root user for security
RUN groupadd -r thunders && useradd -r -g thunders -d /home/thunders -s /sbin/nologin thunders && \
    mkdir -p /home/thunders/.thunders_ai && chown -R thunders:thunders /home/thunders
USER thunders

# Environment defaults
ENV THUNDERS_AI_HOST=0.0.0.0
ENV THUNDERS_AI_PORT=8000
ENV THUNDERS_AI_DEVICE=cpu
ENV THUNDERS_AI_LOG_LEVEL=INFO
ENV THUNDERS_AI_CACHE_DIR=/home/thunders/.thunders_ai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["thunders-ai", "serve", "--host", "0.0.0.0", "--port", "8000"]
