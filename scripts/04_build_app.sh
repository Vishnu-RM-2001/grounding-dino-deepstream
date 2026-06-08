#!/usr/bin/env bash
# Build the app from NVIDIA's stock deepstream-preprocess-test plus our edits.
# Run 01_build_libs.sh first. (Installs CUDA dev headers if the image lacks them.)
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
if [ ! -f build/libgdino_common.so ]; then
  echo "ERROR: build/libgdino_common.so missing — run 01_build_libs.sh first."
  exit 1
fi
docker run --rm --gpus all -v "$PWD":/work -w /work "$IMAGE" bash -lc '
  set -e
  if [ ! -f /usr/local/cuda-13.1/include/cuda_runtime_api.h ]; then
    echo "installing CUDA dev headers..."
    apt-get update -qq && apt-get install -y -qq cuda-nvcc-13-1
  fi
  python3 app/integrate_sample_app.py --cuda-ver 13.1
'
echo "app -> build/gdino-preprocess-test"
