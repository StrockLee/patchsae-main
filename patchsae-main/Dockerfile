FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

# Install CUDA-enabled PyTorch first, then project dependencies.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchvision==0.17.2 && \
    python -m pip install -r /tmp/requirements.txt

COPY . /workspace

RUN mkdir -p /workspace/data /workspace/out /workspace/logs

ENV PYTHONPATH=/workspace \
    HF_HOME=/workspace/data/hf_cache \
    HF_DATASETS_CACHE=/workspace/data/hf_cache/datasets \
    TRANSFORMERS_CACHE=/workspace/data/hf_cache/transformers \
    WANDB_DIR=/workspace/out/wandb \
    MPLCONFIGDIR=/workspace/out/matplotlib

CMD ["bash"]
