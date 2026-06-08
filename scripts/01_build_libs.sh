#!/usr/bin/env bash
# Build the three plugin libraries. The samples image has no nvcc, so install it
# (needed for normalize.cu) on first use.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
docker run --rm --gpus all -v "$PWD":/work -w /work "$IMAGE" bash -lc '
  set -e
  if [ ! -x /usr/local/cuda-13.1/bin/nvcc ]; then
    echo "installing nvcc..."
    apt-get update -qq && apt-get install -y -qq cuda-nvcc-13-1
  fi
  make all \
    CUDA=/usr/local/cuda-13.1 \
    EXTRA_INC=-I/usr/local/cuda-13.1/targets/x86_64-linux/include
  echo "built libs:"; ls -la build/
'
