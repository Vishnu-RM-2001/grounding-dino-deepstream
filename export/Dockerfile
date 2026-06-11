# Export environment for the GroundingDINO SwinT-OGC ONNX export.
# CPU-only: tracing on CPU forces the pure-PyTorch deformable-attention path
# (multi_scale_deformable_attn_pytorch), so we never need the CUDA `_C` op.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.hf_cache
