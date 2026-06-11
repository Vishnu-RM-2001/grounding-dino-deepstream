#!/usr/bin/env bash
# Fetch the model assets for the selected variant.
#   MODEL=tao       (default) NVIDIA TAO Grounding-DINO Swin-Tiny commercial-deployable
#                   ONNX from NGC (anonymous download).
#   MODEL=gdino_b  IDEA-Research GroundingDINO: clone the repo, download the
#                   SwinT-OGC .pth, and build the CPU export image (gdino-export).
set -e
cd "$(dirname "$0")/.."
# --model tao|gdino_b  (env MODEL is the fallback default)
MODEL=${MODEL:-tao}
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --model=*) MODEL="${1#*=}"; shift;;
  -h|--help) echo "usage: $0 [--model tao|gdino_b]"; exit 0;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
echo "model: $MODEL"

if [ "$MODEL" = "tao" ]; then
  mkdir -p model
  if [ ! -f model/grounding_dino_swin_tiny_commercial_deployable.onnx ]; then
    echo "downloading TAO Grounding-DINO from NGC..."
    curl -L -o model/gdino_tao.zip \
      "https://api.ngc.nvidia.com/v2/models/nvidia/tao/grounding_dino/versions/grounding_dino_swin_tiny_commercial_deployable_v1.0/zip"
    (cd model && unzip -o gdino_tao.zip && rm -f gdino_tao.zip)
  fi
  ls -lh model/grounding_dino_swin_tiny_commercial_deployable.onnx

elif [ "$MODEL" = "gdino_b" ]; then
  [ -d GroundingDINO ] || git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git
  mkdir -p weights
  if [ ! -f weights/groundingdino_swint_ogc.pth ]; then
    curl -L -o weights/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  fi
  ls -lh weights/groundingdino_swint_ogc.pth
  echo "building CPU export image (gdino-export)..."
  docker build -t gdino-export export/

else
  echo "unknown MODEL='$MODEL' (use tao|gdino_b)"; exit 1
fi
echo "done. next: ./scripts/02_make_onnx.sh --model $MODEL"
