# Live-Text Grounding-DINO on NVIDIA DeepStream 9.0

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DeepStream](https://img.shields.io/badge/DeepStream-9.0-76b900.svg)
![Grounding-DINO](https://img.shields.io/badge/Grounding--DINO-TAO%20%7C%20IDEA--Research-blue.svg)
![Open-Vocabulary](https://img.shields.io/badge/detection-open--vocabulary-orange.svg)

Run Grounding-DINO Swin-Tiny open-vocabulary detection on NVIDIA DeepStream 9.0, using the
`Gst-nvinfer` element. The prompt is read from a control file, so you can change what the
model detects while the stream is running, without a restart.

Two interchangeable models, selected with `--model gdino_b` (default) or `--model tao`:

| `MODEL` | what it is | source |
|---|---|---|
| `tao` | NVIDIA TAO Grounding-DINO Swin-Tiny (commercial deployable) | ONNX from NGC |
| `gdino_b` | IDEA-Research GroundingDINO SwinT-OGC | `.pth`, exported to ONNX here |

Both run through the same pipeline, plugins and app. They differ only in which engine is
loaded and the order of the packed text tensors (the `model-variant` config key).

```bash
echo "cat . bicycle ." > /tmp/gdino_prompt      # classes switch on the next frame
```

![demo](assets/demo_gdino.jpg)

> `tomato . hand .` on `data/tomatoes.mp4`. The classes are just the words in the prompt;
> change them at runtime and the next frame is detected against the new ones.

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
│        nvdspreprocess (custom lib)      │
│  • resize + normalize the frame         │
│  • tokenize the current prompt          │
│  • pack image + 5 text tensors → 1      │
└──────────────────┬──────────────────────┘
                   ▼          ▲  echo "cat . bicycle ." > /tmp/gdino_prompt
┌─────────────────────────────────────────┐   (live, no restart)
│                nvinfer                  │
│   Grounding-DINO ONNX → TensorRT engine │
│  NvDsInferParseCustomGDINO (custom):    │
│   • decode pred_logits / pred_boxes     │
│   • threshold + class-agnostic NMS      │
│   • emit boxes, class_id = phrase idx   │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│           nvtiler → nvdsosd             │
│  probe stamps the live phrase label     │
│  on each box; OSD draws boxes + text    │
│  periodic perf line on stdout           │
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

### Packing six inputs into one tensor

Grounding-DINO takes an image plus a text prompt and detects whatever the prompt names, so
the network has six inputs: the image and five tensors describing the tokenized prompt.
DeepStream's per-frame tensor path carries only one input, so the project does three things:

1. Pack and split. `onnx/build_single_input_onnx.py` packs the image and five text tensors
   into one `packed` tensor and adds an in-graph preamble that splits it back into the six
   real inputs. The model's output is unchanged.
2. Per-frame text. A custom `nvdspreprocess` plugin normalizes the frame and writes the
   current prompt's tokens into the packed tensor every batch. A control FIFO
   (`/tmp/gdino_prompt`) swaps the prompt atomically at runtime, which is what makes the
   text live.
3. Decode and label. `Gst-nvinfer` runs the engine and a custom bbox parser turns the raw
   outputs into boxes; a probe in the app labels each box with the current phrase.

The only per-model difference is the order of the five text tensors in the packed buffer,
and which mask is the `[256,256]` block-diagonal: `tao` and `gdino_b` swap the roles of
`attention_mask` and `text_token_mask`. The `model-variant` config key (set by `run.sh`
from `--model`) tells the preprocess plugin which order to write. Everything else —
tokenizer, mask recipe, decoder, app — is shared. See `src/gdino_layout.h`.

### About the app

The DeepStream application is [`app/gdino_app.cpp`](app/gdino_app.cpp). It is a single
GStreamer app that:

- Builds the full `nvdspreprocess → nvinfer → nvdsosd` pipeline
- Selects the output sink from flags (`--out` for MP4, `--sink` for a custom element,
  or the default EGL/X11 display sink)
- Encodes directly to H.264 MP4 via `nvv4l2h264enc → h264parse → mp4mux → filesink`
  (GPU-accelerated, no intermediate JPEG frames, no ffmpeg dependency)
- Stamps live phrase labels on each detection via a pad probe on `nvinfer`'s src pad
- Prints a periodic DeepStream-style `**PERF:` status line (per source)

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
- Internet on first setup: `--model tao` pulls the ONNX from NGC (anonymous, no account
  needed); `--model gdino_b` clones the GroundingDINO repo + weights and fetches
  `bert-base-uncased`. `00_get_model.sh` handles both.

---

## Quick start

Pick a model with `--model gdino_b` (default) or `--model tao`. Steps 0–4 are a
one-time setup per model; after that, `run.sh` is all you need. The plugin libs and app
(steps 1, 4) are **model-agnostic** — build them once and they work for both.

```bash
chmod +x scripts/*.sh
M=gdino_b                   # or: M=tao  (just a shell var for the examples)

# 0) fetch the model assets
#    tao      -> downloads the ONNX from NGC
#    gdino_b -> clones IDEA-Research/GroundingDINO, downloads the .pth, builds the
#                CPU export image (gdino-export)
./scripts/00_get_model.sh   --model $M

# 1) build the three plugin libraries (model-agnostic; installs nvcc in-container once)
./scripts/01_build_libs.sh

# 2) produce the packed single-input ONNX  -> onnx/<model>_packed.onnx
#    tao = pack the NGC ONNX; gdino_b = export from .pth, then pack
./scripts/02_make_onnx.sh   --model $M

# 3) build the TensorRT engine             -> onnx/<model>.engine   (FP32 default)
./scripts/03_build_engine.sh --model $M

# 4) build the app (model-agnostic)        -> build/gdino-app
./scripts/04_build_app.sh

# 5) download the two demo clips          -> data/dog_park.mp4, data/tomatoes.mp4
./scripts/get_test_videos.sh

# run: detect the dog and its owner, save an annotated MP4 -> out/gdino_<model>.mp4
./scripts/run.sh --model $M --video file:///workspace/data/dog_park.mp4 "dog . person . ball"
```

Switch model and clip with the same flags:
```bash
./scripts/run.sh --model gdino_b --video file:///workspace/data/dog_park.mp4 "dog . person . ball"
./scripts/run.sh --model tao     --video file:///workspace/data/tomatoes.mp4 "tomato . hand ."
```

> The `gdino_b` model also needs the **gdino-export** image (built by `00_get_model.sh`)
> for the PyTorch→ONNX export. `tao` needs no extra image.
>
> Every script also accepts `MODEL`/`PRECISION` env vars as a fallback, but **flags are
> the documented interface** and take precedence.

---

## Choosing precision

`--precision` flag for `03_build_engine.sh` (default `fp32`). Both models can only run in
TensorRT — for `tao` the ONNX carries a custom deformable-attn plugin; the `gdino_b`
export uses pure-PyTorch ops — so the **FP32 engine is the accuracy reference**.

| precision | tao | gdino_b |
|---|---|---|
| **fp32** (default) | max accuracy | max accuracy |
| **fp16** | faster, accuracy drops | broken — scores collapse, boxes degenerate |
| **bf16** | degraded (worse than fp16 here) | correct boxes, scores ~0.15 lower |

```bash
./scripts/03_build_engine.sh --model tao --precision fp16    # faster, lower recall
```
---

## Changing what it detects, live

While a run is going, write new words to the control file from the host in another terminal.
The next frame is detected against them, with no restart:

```bash
echo "dog . tree ." > /tmp/gdino_prompt
```

Prompts use phrases separated by `.` or `,` (e.g. `"dog . person ."` or `"dog, person"`).

---

## Configuration reference

Key settings in [`configs/config_preprocess_gdino.txt`](configs/config_preprocess_gdino.txt):

| Property | Value | Why |
|---|---|---|
| `network-input-order` | `2` | CUSTOM — the preprocess lib owns the packed-tensor layout |
| `network-input-shape` | `1;1633280;1;1` | the flat `packed` tensor (image + 5 text tensors) |
| `processing-width/height` | `960` / `544` | the model's fixed input resolution |
| `maintain-aspect-ratio` | `0` | stretch to fill, matching the model's export |
| `[user-configs] model-variant` | `@@VARIANT@@` | `tao` or `gdino_b` — packed text-tensor order (set by `run.sh` from `--model`) |
| `[user-configs] prompt` | `car . person .` | initial prompt (overridden per run by the scripts) |
| `[user-configs] mean / std` | ImageNet | per-channel normalization the model expects |
| `[user-configs] fifo-path` | `/tmp/gdino_prompt` | the live-prompt control file |

Key settings in [`configs/config_infer_gdino.txt`](configs/config_infer_gdino.txt):

| Property | Value | Why |
|---|---|---|
| `network-type` | `0` | detector path — nvinfer calls the bbox parser |
| `cluster-mode` | `4` | no DeepStream clustering — the custom parser does its own NMS |
| `parse-bbox-func-name` | `NvDsInferParseCustomGDINO` | the custom parser |
| `model-engine-file` | `@@ENGINE@@` → `onnx/<MODEL>.engine` | the dynamic-batch engine (filled by `run.sh`) |
| `gie-unique-id` | `1` | must equal the preprocess `target-unique-ids` |

Detection threshold is the `--thr` flag of `run.sh` (default `0.3`).

---

## Customization reference


**Model & engine** (setup scripts)
| Want to… | Flag |
|---|---|
| Choose the model | `--model tao` \| `--model gdino_b` (on `00`/`02`/`03`/`run.sh`) |
| Choose precision | `./scripts/03_build_engine.sh --model M --precision fp32\|fp16\|bf16` |
| Use your own GDINO ONNX | `./scripts/02_make_onnx.sh --onnx path.onnx --model M` (input names auto-detect the variant) |

**What/how it detects (no rebuild — flags to `run.sh`)**
| Knob | Flag | Default | Effect |
|---|---|---|---|
| Prompt (start) | positional arg | `dog . person .` | initial classes |
| Prompt (live) | `echo "…" > /tmp/gdino_prompt` | — | change classes mid-stream, no restart |
| Score threshold | `--thr` | `0.30` | lower = more (weaker) boxes |
| NMS IoU | `--nms-iou` | `0.50` | overlap to suppress; `<=0` disables NMS |
| Whole-image filter | `--max-area` | `0.92` | drop boxes bigger than this frac; `>=1` disables |

**Output video (no rebuild — flags to `run.sh`)**
| Knob | Flag | Default |
|---|---|---|
| Output WxH | `--out-w` / `--out-h` | 1280×720 |
| H.264 bitrate | `--bitrate` | encoder default |
| Sink / output | `--sink ELEM` (e.g. `fakesink`) · `--out FILE` (MP4) · `--live` (window) | MP4 |
| Log detections | `--log` | off |

---

## Use your own input video

```bash
./scripts/run.sh --video file:///workspace/data/your_video.mp4 --out out.mp4 "car . person ."
```
Put the file anywhere under the repo. It is mounted at `/workspace` in the container, so a file in
`data/` is `file:///workspace/data/your_video.mp4`. For an RTSP camera, pass
`--video rtsp://user:pass@host:554/stream`.

### Demo clips

`./scripts/get_test_videos.sh` downloads two short clips into `data/`:

| file | prompt | scene |
|---|---|---|
| `data/dog_park.mp4` | `dog . person . ball` | a dog and its owner in a park |
| `data/tomatoes.mp4` | `tomato . hand .` | hands washing tomatoes in a bowl |

```bash
./scripts/get_test_videos.sh
./scripts/run.sh --video file:///workspace/data/dog_park.mp4 "dog . person . ball"
./scripts/run.sh --video file:///workspace/data/tomatoes.mp4 "tomato . hand ."
```

Both files are gitignored; re-fetch them any time with `get_test_videos.sh`. When `--video`
is omitted, `run.sh` uses `data/dog_park.mp4`. The clips are from
[Pexels](https://www.pexels.com/license/) (free to use, no attribution required).

---

## Live display instead of a file (optional)

`run.sh` saves an MP4. To watch it render live in a window via the EGL sink, add `--live`
— it passes your X display and the GPU render node into the container:

```bash
./scripts/run.sh --live --video file:///workspace/data/dog_park.mp4 "dog . person . ball"
```

Requires an **X11 desktop session** (`echo $DISPLAY` is set). On Wayland, log into an
"Xorg"/X11 session, or use `run.sh` to save an MP4 instead. The script grants the container
X access via `xhost +local:root` and adds `--device /dev/dri`.

---

## Interactive shell in the container (for debugging)

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/workspace/build \
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
