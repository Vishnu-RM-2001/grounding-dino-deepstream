#!/usr/bin/env bash
# Build the FP16 TensorRT engine. The dynamic batch profile is required for
# Gst-nvinfer to load it.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
if [ ! -f onnx/gdino_single_input.onnx ]; then
  echo "ERROR: onnx/gdino_single_input.onnx missing — run 02_make_onnx.sh first."
  exit 1
fi
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD":/work -w /work "$IMAGE" bash -lc '
  set -e
  /usr/src/tensorrt/bin/trtexec --onnx=/work/onnx/gdino_single_input.onnx --fp16 \
    --minShapes=packed:1x1633280x1x1 \
    --optShapes=packed:1x1633280x1x1 \
    --maxShapes=packed:2x1633280x1x1 \
    --memPoolSize=workspace:4096 \
    --saveEngine=/work/onnx/gdino_single_input_fp16.engine
  ls -lh /work/onnx/gdino_single_input_fp16.engine
'
