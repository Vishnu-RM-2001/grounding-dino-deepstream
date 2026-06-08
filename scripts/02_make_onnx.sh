#!/usr/bin/env bash
# Pack the model's 6 inputs into one and split them back inside the graph.
# Usage: ./scripts/02_make_onnx.sh [path/to/original.onnx]
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}
IN=${1:-model/grounding_dino_swin_tiny_commercial_deployable.onnx}
if [ ! -f "$IN" ]; then
  echo "ERROR: original ONNX not found at: $IN"
  echo "Download it first (see README, 'Get the model') and pass its path, e.g.:"
  echo "  ./scripts/02_make_onnx.sh model/grounding_dino_swin_tiny_commercial_deployable.onnx"
  exit 1
fi
# install onnx tooling if the image doesn't have it
docker run --rm -v "$PWD":/work -w /work "$IMAGE" bash -lc '
  set -e
  python3 -c "import onnx, onnx_graphsurgeon" 2>/dev/null || {
    command -v pip3 >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-pip; }
    python3 -m pip install --quiet --break-system-packages onnx onnx-graphsurgeon
  }
  python3 onnx/build_single_input_onnx.py \
    --in "'"$IN"'" \
    --out onnx/gdino_single_input.onnx \
    --layout assets/pack_layout.json
  ls -lh onnx/gdino_single_input.onnx
'
