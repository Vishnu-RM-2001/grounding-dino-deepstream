# Live-Text Grounding-DINO on NVIDIA DeepStream 9.0

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DeepStream](https://img.shields.io/badge/DeepStream-9.0-76b900.svg)
![Grounding-DINO](https://img.shields.io/badge/Grounding--DINO-TAO%20%7C%20IDEA--Research-blue.svg)
![Open-Vocabulary](https://img.shields.io/badge/detection-open--vocabulary-orange.svg)

Run **Grounding-DINO Swin-Tiny** open-vocabulary detection on NVIDIA **DeepStream 9.0**,
on the real `Gst-nvinfer` element — and **change what it detects while the stream is
running**, just by typing words. No restart.

**Two interchangeable models** — pick with `--model tao` (default) or `--model gdino_b`:

| `MODEL` | what it is | source |
|---|---|---|
| **`tao`** | NVIDIA TAO Grounding-DINO Swin-Tiny *commercial deployable* | ONNX from NGC |
| **`gdino_b`** | IDEA-Research **GroundingDINO** SwinT-OGC | `.pth` → exported to ONNX here |

Both run through the **same DeepStream pipeline, plugins and app**; they differ only in
which engine is loaded and the packed text-tensor order (a `model-variant` config key).

```bash
echo "dog, bicycle" > /tmp/gdino_prompt        # boxes switch on the next frame
```

![demo](assets/demo_gdino.jpg)

> `person . backpack . handbag .` on the bundled sample clip — a busy concourse where every
> pedestrian and their bag is boxed and labelled with its phrase. Type `suitcase . umbrella .`
> and the classes change mid-stream. It's open-vocabulary, so the classes are whatever words
> you give it, not a fixed list.

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
│   Grounding-DINO ONNX → TensorRT engine │
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
│  **PERF: FPS printed every 5s           │
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

**The only per-model difference** is the order of the 5 text tensors in the packed buffer
(and which mask is the `[256,256]` block-diagonal): `tao` and `gdino_b` swap the roles of
`attention_mask` and `text_token_mask`. The `model-variant` config key (set automatically
by `run.sh` from `--model`) tells the preprocess plugin which order to write; everything else —
tokenizer, mask recipe, decoder, app — is shared. See `src/gdino_layout.h`.

### About the app

This repo ships its own DeepStream application at [`app/gdino_app.cpp`](app/gdino_app.cpp).
It is a self-contained GStreamer app that:

- Builds the full `nvdspreprocess → nvinfer → nvdsosd` pipeline
- Selects the output sink from flags (`--out` for MP4, `--sink` for a custom element,
  or the default EGL/X11 display sink)
- Encodes directly to H.264 MP4 via `nvv4l2h264enc → h264parse → mp4mux → filesink`
  (GPU-accelerated, no intermediate JPEG frames, no ffmpeg dependency)
- Stamps live phrase labels on each detection via a pad probe on `nvinfer`'s src pad
- Prints a DeepStream-style `**PERF: FPS<src> <fps>` line every 5 s (per source)

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

Pick a model with `--model tao` (default) or `--model gdino_b`. Steps 0–4 are a
one-time setup per model; after that, `run.sh` is all you need. The plugin libs and app
(steps 1, 4) are **model-agnostic** — build them once and they work for both.

```bash
chmod +x scripts/*.sh
M=tao                       # or: M=gdino_b  (just a shell var for the examples)

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

# 5) download the demo clip               -> data/sample.mp4
./scripts/get_test_videos.sh

# run: detect people + their bags, save an annotated MP4 -> out/gdino_<model>.mp4
./scripts/run.sh --model $M "person . backpack . handbag ."
```

Switch model at run time with the same flag:
```bash
./scripts/run.sh --model gdino_b "person . backpack . handbag ."                # onnx/gdino_b.engine
./scripts/run.sh --model tao --video file:///path/to/video.mp4 --out out.mp4 "dog . bicycle ."
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
| **fp32** (default) | max accuracy (~17 FPS) | max accuracy (~10 FPS) |
| **fp16** | ~32 FPS but accuracy drops | **broken** — scores collapse, boxes degenerate |
| **bf16** | degraded (worse than fp16 here) | correct boxes, scores ~0.15 lower (~15 FPS) |

```bash
./scripts/03_build_engine.sh --model tao --precision fp16    # faster, lower recall
```
Verified: FP32 reproduces the model exactly through the whole pipeline. FP16/BF16 trade
accuracy for speed differently per model — see `verify/verify_tao.py`.

---

## Changing what it detects, live

This is the point of the project. While a run is going, write new words to the control file
**from the host** in another terminal — the next frame is detected against them, no restart:

```bash
echo "suitcase . umbrella ." > /tmp/gdino_prompt
```

`run.sh` shares `/tmp/gdino_prompt` between the host and the container as a named pipe, so a
plain `echo` from your host reaches the running pipeline. (This works identically for both
models.) The `--switch` flag demonstrates it automatically, switching once real frames
are flowing:
```bash
./scripts/run.sh --out demo.mp4 --switch "suitcase . umbrella ." "person . backpack . handbag ."   # switches mid-stream
```

Prompts use phrases separated by `.` or `,` (e.g. `"car . person ."` or `"car, person"`).

---

## Configuration reference

Key settings in [`configs/config_preprocess_gdino.txt`](configs/config_preprocess_gdino.txt):

| Property | Value | Why |
|---|---|---|
| `network-input-order` | `2` | CUSTOM — our lib owns the packed-tensor layout |
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
| `network-type` | `0` | detector path — nvinfer calls our bbox parser |
| `cluster-mode` | `4` | no DeepStream clustering — our parser does its own NMS |
| `parse-bbox-func-name` | `NvDsInferParseCustomGDINO` | our custom parser |
| `model-engine-file` | `@@ENGINE@@` → `onnx/<MODEL>.engine` | the dynamic-batch engine (filled by `run.sh`) |
| `gie-unique-id` | `1` | must equal the preprocess `target-unique-ids` |

Detection threshold is the `--thr` flag of `run.sh` (default `0.3`).

---

## Customization reference

`run.sh` exposes everything as **flags** (`./scripts/run.sh --thr 0.2 --out-w 640 …`);
config keys live in `configs/`. (`run.sh` translates the detection flags into the env the
bbox parser reads, since an `nvinfer` plugin has no `argv` — but you only ever type flags.)

**Model & engine** (setup scripts)
| Want to… | Flag |
|---|---|
| Choose the model | `--model tao` \| `--model gdino_b` (on `00`/`02`/`03`/`run.sh`) |
| Choose precision | `./scripts/03_build_engine.sh --model M --precision fp32\|fp16\|bf16` |
| Use your own GDINO ONNX | `./scripts/02_make_onnx.sh --onnx path.onnx --model M` (input names auto-detect the variant) |

**What/how it detects (no rebuild — flags to `run.sh`)**
| Knob | Flag | Default | Effect |
|---|---|---|---|
| Prompt (start) | positional arg | `car . person .` | initial classes |
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

**Preprocessing** (`configs/config_preprocess_gdino.txt`, `[user-configs]`): `mean`, `std`,
`pixel-scale` (normalization), `model-variant`, `fifo-path`. `[property]`:
`processing-width/height`, `maintain-aspect-ratio`.

**Deeper (needs rebuild):**
- **Input resolution** — change `IMG_H`/`IMG_W` in [`src/gdino_layout.h`](src/gdino_layout.h)
  (multiples of 32), re-export the ONNX at that size, update the preprocess
  `processing-*` + `network-input-shape` and the engine `--*Shapes`. The header
  documents the exact steps.
- **Tokenizer / vocab** — `assets/vocab.txt` (bert-base-uncased) + `src/bert_tokenizer.*`.
- **Decode/labelling logic** — [`src/gdino_decode.cpp`](src/gdino_decode.cpp) (scoring, NMS,
  box math) and the label probe in [`app/gdino_app.cpp`](app/gdino_app.cpp).

> Tip for low-precision recall: on `tao`, FP16/BF16 drop weak classes like `person` —
> use a stronger word (`man`) or lower `--thr` to ~0.2. See the precision section.

---

## Use your own input video

```bash
./scripts/run.sh --video file:///path/to/your_video.mp4 --out out.mp4 "car . person ."
```
Put the file anywhere under the repo (it's mounted at `/work` in the container, so a file in
`data/` is `file:///work/data/your_video.mp4`). For an RTSP camera, pass
`--video rtsp://user:pass@host:554/stream`.

### Sample video

`./scripts/get_test_videos.sh` downloads a short (~14 s) Grounding-DINO demo clip to
`data/sample.mp4` — the overhead "people-walking" concourse clip used throughout
GroundingDINO / `supervision` demos: a crowd of pedestrians with bags, which shows off
open-vocabulary prompts far better than a generic street scene. Once it's present,
`run.sh` uses it automatically:

```bash
./scripts/get_test_videos.sh
./scripts/run.sh "person . backpack . handbag . suitcase ."
```

(`data/sample.mp4` is gitignored — re-fetch it any time with `get_test_videos.sh`. With no
clip downloaded, `run.sh` falls back to the sample that ships inside the DeepStream image.)

---

## Live display instead of a file (optional)

`run.sh` saves an MP4. To watch it render live in a window via the EGL sink, add `--live`
— it passes your X display and the GPU render node into the container:

```bash
./scripts/run.sh --live "car . person ."
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
