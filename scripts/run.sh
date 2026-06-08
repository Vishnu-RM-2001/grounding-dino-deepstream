#!/usr/bin/env bash
# Run the pipeline. By default it saves an annotated MP4; with --live it shows
# the result in a window instead (EGL sink, needs an X11 desktop session).
#   ./scripts/run.sh        "<prompt>" [video] [out.mp4] ["<switch-to-prompt>"]
#   ./scripts/run.sh --live "<prompt>" [video]           ["<switch-to-prompt>"]
# Prompt: phrases separated by commas or periods (e.g. "car, man"). Use "man" for
# people. Threshold = GDINO_THR (default 0.25). A final arg switches the prompt
# live ~7s in. Press Ctrl+C to stop.
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}

LIVE=0
case "$1" in --live|-l) LIVE=1; shift;; esac

PROMPT=${1:-car . man .}
VIDEO=${2:-file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4}
if [ "$LIVE" = 1 ]; then SWITCH=${3:-}; else OUT=${3:-out/gdino_out.mp4}; SWITCH=${4:-}; fi
THR=${GDINO_THR:-0.25}

[ -x build/gdino-preprocess-test ]       || { echo "ERROR: app missing — run 04_build_app.sh first.";   exit 1; }
[ -f onnx/gdino_single_input_fp16.engine ] || { echo "ERROR: engine missing — run 03_build_engine.sh first."; exit 1; }

# Share the control FIFO between host and container (a container's /tmp is its
# own), so you can change the prompt from the host while it runs:
#   echo "person, backpack" > /tmp/gdino_prompt
FIFO=${GDINO_FIFO:-/tmp/gdino_prompt}
[ -p "$FIFO" ] || { rm -f "$FIFO"; mkfifo "$FIFO"; }

# mode-specific docker arguments
EXTRA=()
if [ "$LIVE" = 1 ]; then
  : "${DISPLAY:?No DISPLAY — --live needs an X11 desktop session. Omit --live to save an MP4.}"
  xhost +local:root >/dev/null 2>&1 || xhost +local: >/dev/null 2>&1 || true   # let the container reach your X server
  [ -d /dev/dri ] && EXTRA+=(--device /dev/dri)                                # nveglglessink needs the render node
  EXTRA+=(-e GDINO_SINK=nveglglessink -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
  echo "prompt: '$PROMPT'${SWITCH:+   (live switch -> '$SWITCH')}   (window)"
else
  mkdir -p out/frames && rm -f out/frames/*.jpg
  EXTRA+=(-e GDINO_OUT=/work/out/frames/ds_%05d.jpg)
  echo "prompt: '$PROMPT'${SWITCH:+   (live switch -> '$SWITCH')}   thr=$THR  -> $OUT"
fi

# Prompt/video/switch are passed as env vars so the in-container script can stay
# fully single-quoted.
INNER='
  set -e
  export USE_NEW_NVSTREAMMUX=no
  export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/work/build:/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
  export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
  rm -rf ~/.cache/gstreamer-1.0
  sed "s#@@ROOT@@#/work#g" configs/config_preprocess_gdino.txt | \
    sed "s#^prompt=.*#prompt=$GDINO_PROMPT#" > /tmp/pre.txt
  sed "s#@@ROOT@@#/work#g" configs/config_infer_gdino.txt > /tmp/inf.txt
  if [ -n "$GDINO_SWITCH" ]; then
    ( sleep 7; echo "$GDINO_SWITCH" > /tmp/gdino_prompt; echo "### switched -> $GDINO_SWITCH" ) &
  fi
  exec ./build/gdino-preprocess-test /tmp/pre.txt /tmp/inf.txt "$GDINO_VIDEO"
'

set +e
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e GDINO_PROMPT="$PROMPT" -e GDINO_SWITCH="$SWITCH" -e GDINO_VIDEO="$VIDEO" -e GDINO_THR="$THR" \
  ${GDINO_LOG:+-e GDINO_LOG="$GDINO_LOG"} \
  -v "$FIFO":/tmp/gdino_prompt \
  "${EXTRA[@]}" -v "$PWD":/work -w /work "$IMAGE" bash -lc "$INNER" 2>&1 \
  | { if [ "$LIVE" = 1 ]; then cat; \
      else grep -E "Number of objects|switched|End of stream|\[gdino\]|ERROR" | sed -n "1,8p;\$p"; fi; }
set -e

if [ "$LIVE" != 1 ]; then
  N=$(ls out/frames/*.jpg 2>/dev/null | wc -l)
  echo "frames written: $N"
  if [ "$N" -gt 0 ] && command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -y -framerate 25 -start_number 0 -i out/frames/ds_%05d.jpg \
      -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT" 2>/dev/null || \
    ffmpeg -y -framerate 25 -start_number 0 -i out/frames/ds_%05d.jpg -c:v mpeg4 -q:v 3 "$OUT" 2>/dev/null
    echo "saved: $OUT"; ls -lh "$OUT"
  elif [ "$N" -gt 0 ]; then
    echo "frames are in out/frames/ (install ffmpeg to encode an MP4)"
  fi
fi
