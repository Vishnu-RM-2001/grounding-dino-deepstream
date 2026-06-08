// Normalize interleaved uint8 (RGBA/RGB, pitched, on GPU) to planar float32
// [3,H,W] with per-channel (x/scale - mean)/std. Per-channel std is what
// Grounding-DINO needs and the stock nvdspreprocess doesn't provide.
#ifndef GDINO_NORMALIZE_H
#define GDINO_NORMALIZE_H
#include <cuda_runtime.h>

namespace gdino {

cudaError_t normalizeToPlanar(const unsigned char* converted, int channels,
                              int pitch_bytes, int width, int height,
                              float* out_planar, const float mean[3],
                              const float std[3], float inv_scale, bool bgr_order,
                              cudaStream_t stream);

} // namespace gdino
#endif
