#!/usr/bin/env bash
# Build gdino-app (our own app — no NVIDIA sample source needed).
# Run 01_build_libs.sh first.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}

[ -f build/libgdino_common.so ] || {
  echo "ERROR: build/libgdino_common.so missing — run 01_build_libs.sh first."
  exit 1
}

docker run --rm --gpus all -v "$PWD":/work -w /work "$IMAGE" bash -lc '
  set -e
  if [ ! -f /usr/local/cuda-13.1/include/cuda_runtime_api.h ]; then
    echo "installing CUDA dev headers..."
    apt-get update -qq && apt-get install -y -qq cuda-nvcc-13-1
  fi
  make build/gdino-app \
    CUDA=/usr/local/cuda-13.1 \
    EXTRA_INC=-I/usr/local/cuda-13.1/targets/x86_64-linux/include
  echo "built: build/gdino-app"
'
