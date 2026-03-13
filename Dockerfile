# ─── Stage 1: builder ────────────────────────────────────────────────────────
# Use a slim Python image; users who need CUDA should swap this base image for
# e.g. pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime and remove the torch line.
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed by OpenCV / Pillow / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first so Docker can cache the pip layer
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package (and all its dependencies) into a dedicated prefix
RUN pip install --no-cache-dir --prefix=/install ".[all]" 2>/dev/null || \
    pip install --no-cache-dir --prefix=/install .

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Volumes: mount your dataset and receive outputs here
VOLUME ["/workspace/data", "/workspace/runs", "/workspace/mlruns"]

# Default entrypoint — override at `docker run` time
ENTRYPOINT ["yolo-boost"]
CMD ["--help"]
