#!/usr/bin/env bash
# Produce the packed single-input ONNX for the selected MODEL.
#   MODEL=tao       pack the TAO ONNX (model/...onnx) -> onnx/tao_packed.onnx
#   MODEL=gdino_b  export IDEA-Research GroundingDINO from .pth, then pack
#                   -> onnx/gdino_b_packed.onnx   (needs the gdino-export image)
# Usage: [MODEL=gdino_b] ./scripts/02_make_onnx.sh [path/to/tao.onnx]
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
# --model tao|gdino_b ; --onnx PATH (tao: source ONNX, default model/...onnx)
MODEL=${MODEL:-gdino_b}
IN=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --model=*) MODEL="${1#*=}"; shift;;
  --onnx) IN="$2"; shift 2;;
  --onnx=*) IN="${1#*=}"; shift;;
  -h|--help) echo "usage: $0 [--model tao|gdino_b] [--onnx path/to/tao.onnx]"; exit 0;;
  *) [ -z "$IN" ] && { IN="$1"; shift; } || { echo "unknown arg: $1"; exit 1; };;
esac; done
PACKED=onnx/${MODEL}_packed.onnx
LAYOUT=assets/${MODEL}_pack_layout.json
mkdir -p onnx assets

if [ "$MODEL" = "tao" ]; then
  IN=${IN:-model/grounding_dino_swin_tiny_commercial_deployable.onnx}
  [ -f "$IN" ] || { echo "ERROR: TAO ONNX not found: $IN — run MODEL=tao ./scripts/00_get_model.sh"; exit 1; }
  docker run --rm -v "$PWD":/workspace -w /workspace "$IMAGE" bash -lc '
    set -e
    python3 -c "import onnx, onnx_graphsurgeon" 2>/dev/null || {
      command -v pip3 >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-pip; }
      python3 -m pip install --quiet --break-system-packages onnx onnx-graphsurgeon; }
    python3 onnx/build_single_input_onnx.py --variant tao \
      --in "'"$IN"'" --out "'"$PACKED"'" --layout "'"$LAYOUT"'"
  '

elif [ "$MODEL" = "gdino_b" ]; then
  [ -f weights/groundingdino_swint_ogc.pth ] || { echo "ERROR: weights missing — run MODEL=gdino_b ./scripts/00_get_model.sh"; exit 1; }
  docker run --rm -v "$PWD":/workspace -w /workspace -e HF_HOME=/workspace/.hf_cache gdino-export bash -lc "
    set -e
    python3 export/export_onnx.py --out onnx/gdino_b_raw.onnx --verify
    python3 onnx/build_single_input_onnx.py --variant gdino_b \
      --in onnx/gdino_b_raw.onnx --out '$PACKED' --layout '$LAYOUT'
  "
else
  echo "unknown MODEL='$MODEL' (use tao|gdino_b)"; exit 1
fi
ls -lh "$PACKED" "$LAYOUT"
