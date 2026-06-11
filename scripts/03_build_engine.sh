#!/usr/bin/env bash
# Build the dynamic-batch TensorRT engine for the selected MODEL.
#   MODEL=tao|gdino_b   (default gdino_b)   reads onnx/<MODEL>_packed.onnx
#   PRECISION=fp32|fp16|bf16  (default fp32 = max accuracy)
#
# Precision notes (verified on this model family):
#   - fp32: matches the reference; recommended for accuracy.
#   - tao + fp16: ~1.8x faster but DROPS most "person" detections (scores collapse
#     below threshold); fine for strong/large classes. Not max accuracy.
#   - gdino_b + fp16: BROKEN (scores collapse, boxes degenerate). Use fp32 or bf16.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
# --model tao|gdino_b ; --precision fp32|fp16|bf16  (env MODEL/PRECISION are fallbacks)
MODEL=${MODEL:-gdino_b}
PRECISION=${PRECISION:-fp32}
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --model=*) MODEL="${1#*=}"; shift;;
  --precision) PRECISION="$2"; shift 2;;
  --precision=*) PRECISION="${1#*=}"; shift;;
  -h|--help) echo "usage: $0 [--model tao|gdino_b] [--precision fp32|fp16|bf16]"; exit 0;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
PACKED=onnx/${MODEL}_packed.onnx
ENGINE=onnx/${MODEL}.engine
[ -f "$PACKED" ] || { echo "ERROR: $PACKED missing — run MODEL=$MODEL ./scripts/02_make_onnx.sh"; exit 1; }

case "$PRECISION" in
  fp32) FLAG="" ;;
  fp16) FLAG="--fp16"; [ "$MODEL" = gdino_b ] && echo "WARNING: gdino_b+fp16 is broken; use fp32 or bf16" ;;
  bf16) FLAG="--bf16" ;;
  *) echo "unknown PRECISION='$PRECISION' (fp32|fp16|bf16)"; exit 1 ;;
esac
echo "building $MODEL engine ($PRECISION) -> $ENGINE"

docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD":/workspace -w /workspace "$IMAGE" bash -lc "
  set -e
  /usr/src/tensorrt/bin/trtexec --onnx=/workspace/$PACKED $FLAG \
    --minShapes=packed:1x1633280x1x1 \
    --optShapes=packed:1x1633280x1x1 \
    --maxShapes=packed:2x1633280x1x1 \
    --memPoolSize=workspace:4096 \
    --saveEngine=/workspace/$ENGINE
  ls -lh /workspace/$ENGINE
"
