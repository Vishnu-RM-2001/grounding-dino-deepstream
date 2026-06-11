#!/usr/bin/env bash
# Run the pipeline. Saves an annotated MP4 by default; --live shows a window.
#   ./scripts/run.sh [flags] "<prompt>"
# Flags:
#   --model tao|gdino_b   model (default tao)
#   --live                 EGL/X11 window instead of saving an MP4 (needs a desktop)
#   --video URI            input (default: data/sample.mp4 from get_test_videos.sh, else
#                          the in-container clip; accepts file:// or rtsp://)
#   --out FILE             output MP4 (default out/gdino_<model>.mp4; ignored with --live)
#   --switch "<prompt>"    demo a live prompt change once frames are flowing
#   --thr F                detection score threshold (default 0.3)
#   --nms-iou F            class-agnostic NMS IoU (default 0.5; <=0 disables)
#   --max-area F           drop boxes bigger than this frac of the frame (default 0.92; >=1 off)
#   --out-w N / --out-h N  output resolution (default 1280x720)
#   --bitrate BPS          H.264 bitrate for --out
#   --sink ELEM            custom sink element (e.g. fakesink for headless throughput)
#   --log                  log every detection to stdout
# Prompt: phrases separated by '.' or ',' (e.g. "car . person ."). Change it live at
# runtime from any shell:   echo "bus . backpack ." > /tmp/gdino_prompt
set -e
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nvcr.io/nvidia/deepstream:9.0-samples-multiarch}

MODEL=tao; LIVE=0; LOG=0
VIDEO=""
OUT=""; SWITCH=""; THR=0.3; NMS_IOU=""; MAX_AREA=""
OUT_W=""; OUT_H=""; BITRATE=""; SINK=""; PROMPT=""
while [ $# -gt 0 ]; do case "$1" in
  --model)    MODEL="$2"; shift 2;;
  --live|-l)  LIVE=1; shift;;
  --video)    VIDEO="$2"; shift 2;;
  --out)      OUT="$2"; shift 2;;
  --switch)   SWITCH="$2"; shift 2;;
  --thr)      THR="$2"; shift 2;;
  --nms-iou)  NMS_IOU="$2"; shift 2;;
  --max-area) MAX_AREA="$2"; shift 2;;
  --out-w)    OUT_W="$2"; shift 2;;
  --out-h)    OUT_H="$2"; shift 2;;
  --bitrate)  BITRATE="$2"; shift 2;;
  --sink)     SINK="$2"; shift 2;;
  --log)      LOG=1; shift;;
  -h|--help)  sed -n '2,20p' "$0"; exit 0;;
  -*)         echo "unknown flag: $1 (see --help)"; exit 1;;
  *)          PROMPT="$1"; shift;;
esac; done
PROMPT=${PROMPT:-person . backpack . handbag .}
OUT=${OUT:-out/gdino_${MODEL}.mp4}
ENGINE=onnx/${MODEL}.engine
# default input: the downloaded GDINO sample if present, else the in-container clip
if [ -z "$VIDEO" ]; then
  if [ -f data/sample.mp4 ]; then VIDEO=file:///work/data/sample.mp4
  else VIDEO=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4; fi
fi

[ -x build/gdino-app ] || { echo "ERROR: app missing — run ./scripts/04_build_app.sh"; exit 1; }
[ -f "$ENGINE" ]       || { echo "ERROR: $ENGINE missing — run ./scripts/03_build_engine.sh --model $MODEL"; exit 1; }
echo "model: $MODEL   engine: $ENGINE"

# Bind-mount the control FIFO so a host `echo` reaches the running pipeline.
FIFO=${GDINO_FIFO:-/tmp/gdino_prompt}
[ -p "$FIFO" ] || { rm -f "$FIFO"; mkfifo "$FIFO"; }

# Assemble the app's flags (sink branch + output knobs) and docker extras.
APP_ARGS=""; EXTRA=()
if [ "$LIVE" = 1 ]; then
  : "${DISPLAY:?No DISPLAY — --live needs an X11 desktop session. Omit --live to save an MP4.}"
  xhost +local:root >/dev/null 2>&1 || xhost +local: >/dev/null 2>&1 || true
  [ -d /dev/dri ] && EXTRA+=(--device /dev/dri)
  EXTRA+=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
  APP_ARGS="--sink ${SINK:-nveglglessink}"
  echo "prompt: '$PROMPT'${SWITCH:+   (live switch -> '$SWITCH')}   (window)"
elif [ -n "$SINK" ]; then
  APP_ARGS="--sink $SINK"
  echo "prompt: '$PROMPT'${SWITCH:+   (live switch -> '$SWITCH')}   sink=$SINK"
else
  mkdir -p "$(dirname "$OUT")"
  APP_ARGS="--out /work/$OUT"
  echo "prompt: '$PROMPT'${SWITCH:+   (live switch -> '$SWITCH')}   thr=$THR  -> $OUT"
fi
[ -n "$OUT_W" ]   && APP_ARGS="$APP_ARGS --out-w $OUT_W"
[ -n "$OUT_H" ]   && APP_ARGS="$APP_ARGS --out-h $OUT_H"
[ -n "$BITRATE" ] && APP_ARGS="$APP_ARGS --bitrate $BITRATE"
[ "$LOG" = 1 ]    && APP_ARGS="$APP_ARGS --log"

# Configs are templated in-container (@@ROOT@@, @@VARIANT@@, @@ENGINE@@, prompt).
INNER='
  set -e
  export USE_NEW_NVSTREAMMUX=no
  export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/work/build:/usr/local/cuda-13.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
  export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
  rm -rf ~/.cache/gstreamer-1.0
  sed -e "s#@@ROOT@@#/work#g" -e "s#@@VARIANT@@#$GDINO_MODEL#g" configs/config_preprocess_gdino.txt | \
    sed "s#^prompt=.*#prompt=$GDINO_PROMPT#" > /tmp/pre.txt
  sed -e "s#@@ROOT@@#/work#g" -e "s#@@ENGINE@@#/work/onnx/$GDINO_MODEL.engine#g" configs/config_infer_gdino.txt > /tmp/inf.txt
  exec ./build/gdino-app $GDINO_APP_ARGS /tmp/pre.txt /tmp/inf.txt "$GDINO_VIDEO"
'

# Live-switch demo: write the new prompt to the FIFO from the HOST once frames flow.
LOGF=$(mktemp)
if [ -n "$SWITCH" ]; then
  ( until grep -q "frame=20 " "$LOGF" 2>/dev/null; do sleep 1; done
    echo "$SWITCH" > "$FIFO"; echo "### live switch -> '$SWITCH'" ) &
fi

# Detection knobs go to the bbox parser, an nvinfer plugin with no argv → via env.
set +e
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e GDINO_PROMPT="$PROMPT" -e GDINO_VIDEO="$VIDEO" -e GDINO_MODEL="$MODEL" \
  -e GDINO_APP_ARGS="$APP_ARGS" -e GDINO_THR="$THR" \
  ${NMS_IOU:+-e GDINO_NMS_IOU="$NMS_IOU"} ${MAX_AREA:+-e GDINO_MAX_AREA="$MAX_AREA"} \
  -v "$FIFO":/tmp/gdino_prompt \
  "${EXTRA[@]}" -v "$PWD":/work -w /work "$IMAGE" bash -lc "$INNER" 2>&1 \
  | tee "$LOGF" \
  | { if [ "$LIVE" = 1 ]; then cat; \
      else grep --line-buffered -E "PERF:|live switch|End of stream|\[gdino\]|ERROR|frame=0 "; fi; }
set -e
rm -f "$LOGF"

if [ "$LIVE" != 1 ] && [ -f "$OUT" ]; then
  echo "saved: $OUT"; ls -lh "$OUT"
fi
