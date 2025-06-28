# ───────────────────────────────────────────────────────────────
# Lightweight CUDA runtime + system Python 3.10
# Only needs rebuilding when requirements.txt changes
# ───────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- system packages ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git build-essential libsndfile1 ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- python packages ----------
COPY requirements.txt /tmp/requirements.txt
RUN mkdir /app && cd /app && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install flash-attn --no-build-isolation

# ---------- runtime setup ----------
WORKDIR /app
EXPOSE 8003
