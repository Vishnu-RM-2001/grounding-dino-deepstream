# Live-Text Grounding-DINO on NVIDIA DeepStream 9.0

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DeepStream](https://img.shields.io/badge/DeepStream-9.0-76b900.svg)
![Grounding-DINO](https://img.shields.io/badge/TAO-Grounding--DINO-blue.svg)
![Open-Vocabulary](https://img.shields.io/badge/detection-open--vocabulary-orange.svg)

Run NVIDIA's **TAO Grounding-DINO Swin-Tiny** open-vocabulary detector on NVIDIA
**DeepStream 9.0**, on the real `Gst-nvinfer` element — and **change what it detects
while the stream is running**, just by typing words. No restart.

```bash
echo "dog, bicycle" > /tmp/gdino_prompt        # boxes switch on the next frame
```

![demo](assets/demo_gdino.jpg)

> `car, man` on a live stream — cars and people boxed and labelled with their phrase.
> Type `bus, bicycle, backpack` and the classes change mid-stream. It's open-vocabulary,
> so the classes are whatever words you give it, not a fixed list.

---

## How it works

### Pipeline

```
┌─────────────────────────────────────────┐
│            Input (file / RTSP)          │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│              nvstreammux                │
│        Batches decoded frames           │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│         nvdspreprocess (our lib)        │
│  • resize + normalize the frame         │
│  • tokenize the CURRENT prompt          │
│  • pack image + 5 text tensors → 1      │
└──────────────────┬──────────────────────┘
                   ▼          ▲  echo "dog, bicycle" > /tmp/gdino_prompt
┌─────────────────────────────────────────┐   (live, no restart)
│                nvinfer                  │
│   Grounding-DINO ONNX → TensorRT FP16   │
│  NvDsInferParseCustomGDINO (our parser):│
│   • decode pred_logits / pred_boxes     │
│   • threshold + class-agnostic NMS      │
│   • emit boxes, class_id = phrase idx   │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│           nvtiler → nvdsosd             │
│  probe stamps the live phrase label     │
│  on each box; OSD draws boxes + text    │
│  FPS printed every 300 frames           │
└──────────────────┬──────────────────────┘
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│   run.sh         │  │   run.sh --live  │
│  nvv4l2h264enc   │  │   live X11 window│
│  → h264parse     │  │   (nveglglessink)│
│  → mp4mux        │  │                  │
│  → out/...mp4    │  │                  │
└──────────────────┘  └──────────────────┘
```

### The trick: one tensor, six inputs, live text

Grounding-DINO is open-vocabulary — it takes an **image plus a text prompt** and detects
whatever the prompt names. So the network has **6 inputs**: the image, plus 5 tensors that
describe the tokenized prompt. DeepStream's per-frame tensor path only carries one input,
so this project does three things:

1. **Pack and split.** `onnx/build_single_input_onnx.py` packs the image + 5 text tensors
   into one `packed` tensor and adds a small in-graph preamble that splits it back into the
   6 real inputs. The model's output is unchanged.
2. **Per-frame text.** A custom `nvdspreprocess` plugin normalizes the frame and writes the
   *current* prompt's tokens into that packed tensor every batch — this is what makes the
   text live. A control FIFO (`/tmp/gdino_prompt`) swaps the prompt atomically at runtime.
3. **Decode + label.** `Gst-nvinfer` runs the engine and a custom bbox parser turns the raw
   outputs into boxes; a probe in the app labels each box with the live phrase.

### About the app

This repo ships its own DeepStream application at [`app/gdino_app.cpp`](app/gdino_app.cpp).
It is a self-contained GStreamer app (~250 lines) that:

- Builds the full `nvdspreprocess → nvinfer → nvdsosd` pipeline
- Selects the output sink from an env var (`GDINO_OUT` for MP4, `GDINO_SINK` for a custom
  element, or the default EGL/X11 display sink)
- Encodes directly to H.264 MP4 via `nvv4l2h264enc → h264parse → mp4mux → filesink`
  (GPU-accelerated, no intermediate JPEG frames, no ffmpeg dependency)
- Stamps live phrase labels on each detection via a pad probe on `nvinfer`'s src pad
- Prints FPS every 300 frames

---

## Prerequisites

**DeepStream 9.0 requirements** ([source](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html))
- Ubuntu 24.04
- NVIDIA driver ≥ 590.48.01
- CUDA 13.1
- TensorRT 10.14.1.48
- GStreamer 1.24.2

> This project uses the **Docker image** (`nvcr.io/nvidia/deepstream:9.0-samples-multiarch`),
> which bundles all of the above — only the host driver needs to meet the minimum version.

**Host requirements**
- NVIDIA GPU
- NVIDIA driver ≥ 590.48.01
- Docker + **NVIDIA Container Toolkit** (`--gpus all` must work):
  ```bash
  docker run --rm --gpus all nvcr.io/nvidia/deepstream:9.0-samples-multiarch nvidia-smi
  ```
- A free [NGC account](https://ngc.nvidia.com) to download the model.

---

## Get the model

Download the **Grounding-DINO Swin-Tiny (commercial deployable)** ONNX from NGC:

> https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino?version=grounding_dino_swin_tiny_commercial_deployable_v1.0

Using the [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli):
```bash
mkdir -p model

ngc registry model download-version \
  "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0" \
  --dest model/
```

## Setup

Make the scripts executable
```bash
chmod +x scripts/*.sh
```

---

## Quick start

From the repo root. Steps 1–4 are a one-time setup; after that, `run.sh` is all you need.

```bash
# 1) build the three plugin libraries  (installs nvcc in-container on first run)
./scripts/01_build_libs.sh

# 2) pack the model's 6 inputs into one  -> onnx/gdino_single_input.onnx
./scripts/02_make_onnx.sh model/grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0/grounding_dino_swin_tiny_commercial_deployable.onnx

# 3) build the FP16 TensorRT engine  -> onnx/gdino_single_input_fp16.engine
./scripts/03_build_engine.sh

# 4) build the app  -> build/gdino-app
./scripts/04_build_app.sh

# run: detect cars and people, save an annotated MP4 -> out/gdino_out.mp4
./scripts/run.sh "car, man"
```
Open `out/gdino_out.mp4`.

To run on your own footage, pass a file URI:
```bash
./scripts/run.sh "dog, bicycle" file:///path/to/your_video.mp4 out.mp4
```

---

## Step by step

### 1. Build the plugin libraries
```bash
./scripts/01_build_libs.sh
```
Builds `libgdino_common.so`, `libnvds_gdino_preprocess.so`, and `libnvds_gdino_parser.so`
into `build/`. The DeepStream samples image has no `nvcc`, so the script installs it into
the container the first time (needs internet that once).

### 2. Pack the ONNX
```bash
./scripts/02_make_onnx.sh model/.../grounding_dino_swin_tiny_commercial_deployable.onnx
```
Produces `onnx/gdino_single_input.onnx` — the same model with one `packed` input and an
in-graph preamble that splits it back into the original six. Keeps a dynamic batch dim.

### 3. Build the TensorRT engine
```bash
./scripts/03_build_engine.sh
```
Builds `onnx/gdino_single_input_fp16.engine` (FP16, dynamic batch). Takes ~2 minutes;
cached afterwards.

### 4. Build the app
```bash
./scripts/04_build_app.sh
```
Compiles `app/gdino_app.cpp` (our own GStreamer app) to `build/gdino-app`. No NVIDIA sample
source is required — the app is self-contained in this repo.

### 5. Run
```bash
./scripts/run.sh "car, man"                                   # bundled sample video → MP4
./scripts/run.sh "dog, bicycle" file:///path/to/video.mp4 out.mp4
```
With no video argument it uses the H.264 sample stream shipped in the DeepStream container
(decoded by NVDEC). Press **Ctrl+C** to stop cleanly.

---

## Changing what it detects, live

This is the point of the project. While a run is going, write new words to the control file
**from the host** in another terminal — the next frame is detected against them, no restart:

```bash
echo "person, backpack, traffic light" > /tmp/gdino_prompt
```

(`run.sh` shares `/tmp/gdino_prompt` between the host and the container as a named pipe, so a
plain `echo` from your host reaches the running pipeline.)

`run.sh`'s 4th argument demonstrates this automatically by switching partway through:
```bash
./scripts/run.sh "car, man" "" demo.mp4 "bus, bicycle"        # switches ~7s in
```

---

## Configuration reference

Key settings in [`configs/config_preprocess_gdino.txt`](configs/config_preprocess_gdino.txt):

| Property | Value | Why |
|---|---|---|
| `network-input-order` | `2` | CUSTOM — our lib owns the packed-tensor layout |
| `network-input-shape` | `1;1633280;1;1` | the flat `packed` tensor (image + 5 text tensors) |
| `processing-width/height` | `960` / `544` | the model's fixed input resolution |
| `maintain-aspect-ratio` | `0` | stretch to fill, matching the model's export |
| `draw-roi` | `0` | off — otherwise nvdspreprocess draws a green frame border |
| `[user-configs] prompt` | `car . man .` | initial prompt (overridden per run by the scripts) |
| `[user-configs] mean / std` | ImageNet | per-channel normalization the model expects |
| `[user-configs] fifo-path` | `/tmp/gdino_prompt` | the live-prompt control file |

Key settings in [`configs/config_infer_gdino.txt`](configs/config_infer_gdino.txt):

| Property | Value | Why |
|---|---|---|
| `network-type` | `0` | detector path — nvinfer calls our bbox parser |
| `cluster-mode` | `4` | no DeepStream clustering — our parser does its own NMS |
| `parse-bbox-func-name` | `NvDsInferParseCustomGDINO` | our custom parser |
| `model-engine-file` | `…/gdino_single_input_fp16.engine` | the dynamic-batch FP16 engine |
| `gie-unique-id` | `1` | must equal the preprocess `target-unique-ids` |

Detection threshold is the `GDINO_THR` environment variable (default `0.25`), read by the parser.

---

## Use your own input video

```bash
./scripts/run.sh "car, man" file:///path/to/your_video.mp4 out.mp4
```
Put the file anywhere under the repo (it's mounted at `/work` in the container, so a file in
`data/` is `file:///work/data/your_video.mp4`). For an RTSP camera, pass
`rtsp://user:pass@host:554/stream` as the URI.

### Sample video

`run.sh` already defaults to a street-scene sample (`sample_720p.mp4`) that ships **inside**
the DeepStream image, so `./scripts/run.sh "car, man"` just works with no input file.

To pull that sample out to a local `data/` file, use the following command:

```bash
./scripts/get_test_videos.sh
./scripts/run.sh "car, person, bus, traffic light" file:///work/data/sample_720p.mp4 out/sample.mp4
```

---

## Live display instead of a file (optional)

`run.sh` saves an MP4. To watch it render live in a window via the EGL sink, add `--live`
— it passes your X display and the GPU render node into the container:

```bash
./scripts/run.sh --live "car, man"
```

Requires an **X11 desktop session** (`echo $DISPLAY` is set). On Wayland, log into an
"Xorg"/X11 session, or use `run.sh` to save an MP4 instead. The script grants the container
X access via `xhost +local:root` and adds `--device /dev/dri`.

---

## Interactive shell in the container (for debugging)

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/work -w /work \
  -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/work/build \
  nvcr.io/nvidia/deepstream:9.0-samples-multiarch bash
```

---

## License

The files in this repository (source, configs, scripts, documentation) are released under the
**MIT License** — see [LICENSE](LICENSE).

This project depends on third-party components with separate licenses — see [NOTICE](NOTICE):

| Component | License | Notes |
|---|---|---|
| NVIDIA DeepStream SDK 9.0 | NVIDIA Proprietary | Runs inside the Docker container; not distributed here |
| TAO Grounding-DINO Swin-Tiny | NVIDIA model license | Downloaded from NGC; ONNX/engine are gitignored |
| BERT vocabulary (`assets/vocab.txt`) | Apache-2.0 | `bert-base-uncased` WordPiece vocab |

---

## References
- DeepStream SDK docs — https://docs.nvidia.com/metropolis/deepstream/dev-guide/
- TAO Grounding-DINO (NGC) — https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino
- Grounding-DINO paper — https://arxiv.org/abs/2303.05499
