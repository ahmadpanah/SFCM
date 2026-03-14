# docker/Dockerfile
# ==================
# Exact environment for reproducing all SFCM experiments.
# Matches the compute environment described in §4.4:
#   Python 3.10, PyTorch, GPyTorch, NVIDIA A100 (CUDA 12.1)
#
# Build (CPU-only, for CI / reviewing without GPU):
#   docker build -f docker/Dockerfile --target cpu -t sfcm:cpu .
#
# Build (GPU, full experiments):
#   docker build -f docker/Dockerfile --target gpu -t sfcm:gpu .
#
# Run unit tests (CPU):
#   docker run --rm sfcm:cpu pytest tests/ -v
#
# Run full experiment suite (GPU):
#   docker run --rm --gpus all sfcm:gpu \
#     python experiments/run_experiments.py --all --n_runs 30

# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.10.12-slim AS base

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /workspace/

# ── CPU target (CI / review) ──────────────────────────────────────────────
FROM base AS cpu

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    gpytorch==1.11 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    matplotlib==3.8.2 \
    pyyaml==6.0.1 \
    pytest==8.0.0 \
    pytest-cov==4.1.0

# Install optional BO baselines (CPU mode)
RUN pip install --no-cache-dir \
    smac==2.0.2 \
    cma==3.3.0 \
    || true   # non-fatal: baselines fall back gracefully if absent

# Create results directory
RUN mkdir -p /workspace/results/tables /workspace/results/figures

# Smoke test
RUN python -c "import sfcm; print('SFCM import OK')" && \
    pytest tests/test_sfcm.py::TestHyperparameterConsistency -v

CMD ["pytest", "tests/", "-v", "--tb=short"]

# ── GPU target (full experiments) ─────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git curl build-essential \
    && ln -sf python3.10 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/

RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    gpytorch==1.11 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    matplotlib==3.8.2 \
    pyyaml==6.0.1 \
    pytest==8.0.0 \
    smac==2.0.2 \
    cma==3.3.0

RUN mkdir -p /workspace/results/tables /workspace/results/figures

CMD ["python", "experiments/run_experiments.py", "--all", "--n_runs", "30"]
