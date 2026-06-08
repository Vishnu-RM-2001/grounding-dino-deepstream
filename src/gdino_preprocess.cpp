// nvdspreprocess custom lib. Each batch, CustomTensorPreparation normalizes the
// frame into the image region of the packed tensor and copies the current
// prompt's text tensors into the text region (the prompt comes from PromptStore
// and can change live via the FIFO).
//
// [user-configs] keys: vocab-path (required), prompt, fifo-path, mean, std,
// pixel-scale.
//
// nvdspreprocess_interface.h uses std:: containers without including them, so
// pull these in first.
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "nvdspreprocess_interface.h"
#include "gdino_layout.h"
#include "gdino_text.h"
#include "gdino_prompt_store.h"
#include "normalize.h"

struct CustomCtx {
  cudaStream_t stream = nullptr;
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float stdv[3] = {0.229f, 0.224f, 0.225f};
  float inv_scale = 1.0f / 255.0f;
  float* text_host = nullptr;          // pinned, TEXT_COUNT floats
};

static std::string cfg(const std::unordered_map<std::string, std::string>& m,
                       const char* k, const std::string& def) {
  auto it = m.find(k);
  return it == m.end() ? def : it->second;
}
static void parse3(const std::string& s, float out[3]) {
  std::stringstream ss(s); std::string t; int i = 0;
  while (std::getline(ss, t, ',') && i < 3) out[i++] = std::stof(t);
}

extern "C" CustomCtx* initLib(CustomInitParams initparams) {
  auto* ctx = new CustomCtx();
  auto& uc = initparams.user_configs;

  std::string vocab = cfg(uc, "vocab-path", "");
  std::string prompt = cfg(uc, "prompt", "person .");
  std::string fifo = cfg(uc, "fifo-path", "/tmp/gdino_prompt");
  parse3(cfg(uc, "mean", "0.485,0.456,0.406"), ctx->mean);
  parse3(cfg(uc, "std", "0.229,0.224,0.225"), ctx->stdv);
  ctx->inv_scale = 1.0f / std::stof(cfg(uc, "pixel-scale", "255.0"));

  if (vocab.empty()) { fprintf(stderr, "[gdino] vocab-path is required\n"); delete ctx; return nullptr; }
  if (!gdino::PromptStore::instance().init(vocab, prompt, fifo)) { delete ctx; return nullptr; }

  // sanity-check the configured tensor shape against the engine's packed input
  long prod = 1;
  for (int d : initparams.tensor_params.network_input_shape) prod *= d;
  if (prod != gdino::PACKED_TOTAL)
    fprintf(stderr, "[gdino] WARN: network-input-shape product=%ld != PACKED_TOTAL=%lld; "
            "set network-input-shape: 1;%lld\n", prod,
            (long long)gdino::PACKED_TOTAL, (long long)gdino::PACKED_TOTAL);

  cudaStreamCreate(&ctx->stream);
  cudaMallocHost((void**)&ctx->text_host, gdino::TEXT_COUNT * sizeof(float));
  fprintf(stderr, "[gdino] initLib OK (mean %.3f,%.3f,%.3f std %.3f,%.3f,%.3f)\n",
          ctx->mean[0], ctx->mean[1], ctx->mean[2], ctx->stdv[0], ctx->stdv[1], ctx->stdv[2]);
  return ctx;
}

extern "C" void deInitLib(CustomCtx* ctx) {
  if (!ctx) return;
  gdino::PromptStore::instance().shutdown();
  if (ctx->text_host) cudaFreeHost(ctx->text_host);
  if (ctx->stream) cudaStreamDestroy(ctx->stream);
  delete ctx;
}

extern "C" NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx* ctx, NvDsPreProcessBatch* batch,
                        NvDsPreProcessCustomBuf*& buf,
                        CustomTensorParams& tensorParam,
                        NvDsPreProcessAcquirer* acquirer) {
  static int calls = 0;
  buf = acquirer->acquire();
  if (!buf || !buf->memory_ptr) return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  float* dst = (float*)buf->memory_ptr;
  if (calls < 3)
    fprintf(stderr, "[gdino] CustomTensorPreparation call#%d units=%zu fmt=%d pitch=%u\n",
            calls, batch->units.size(), (int)batch->scaling_pool_format, batch->pitch);
  ++calls;

  // CURRENT prompt -> contiguous text floats (same for all frames in the batch)
  auto snap = gdino::PromptStore::instance().current();
  if (!snap) { acquirer->release(buf); return NVDSPREPROCESS_TENSOR_NOT_READY; }
  gdino::writeTextFloatsCompact(snap->text, ctx->text_host);

  // converted frame layout (network-res, interleaved). Assumes scaling pool is
  // RGBA(4ch) or RGB(3ch); pitch from batch.
  bool bgr = (batch->scaling_pool_format == NvDsPreProcessFormat_BGR ||
              batch->scaling_pool_format == NvDsPreProcessFormat_BGRx);
  int channels = (batch->scaling_pool_format == NvDsPreProcessFormat_RGBA ||
                  batch->scaling_pool_format == NvDsPreProcessFormat_BGRx) ? 4 : 3;
  int pitch = batch->pitch ? (int)batch->pitch : gdino::IMG_W * channels;

  for (size_t b = 0; b < batch->units.size(); ++b) {
    float* base = dst + (int64_t)b * gdino::PACKED_TOTAL;
    auto& u = batch->units[b];
    cudaError_t e = gdino::normalizeToPlanar(
        (const unsigned char*)u.converted_frame_ptr, channels, pitch,
        gdino::IMG_W, gdino::IMG_H, base + gdino::OFF_IMAGE,
        ctx->mean, ctx->stdv, ctx->inv_scale, bgr, ctx->stream);
    if (e != cudaSuccess) {
      fprintf(stderr, "[gdino] normalize failed: %s\n", cudaGetErrorString(e));
      acquirer->release(buf); return NVDSPREPROCESS_CUDA_ERROR;
    }
    cudaMemcpyAsync(base + gdino::OFF_INPUT_IDS, ctx->text_host,
                    gdino::TEXT_COUNT * sizeof(float),
                    cudaMemcpyHostToDevice, ctx->stream);
  }
  cudaStreamSynchronize(ctx->stream);

  if (!tensorParam.params.network_input_shape.empty())
    tensorParam.params.network_input_shape[0] = (int)batch->units.size();
  return NVDSPREPROCESS_SUCCESS;
}
