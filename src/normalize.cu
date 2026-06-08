#include "normalize.h"

namespace gdino {

__global__ void normKernel(const unsigned char* src, int channels, int pitch,
                           int width, int height, float* out,
                           float m0, float m1, float m2,
                           float s0, float s1, float s2,
                           float inv_scale, int bgr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const unsigned char* px = src + (size_t)y * pitch + (size_t)x * channels;
  float c0 = px[0], c1 = px[1], c2 = px[2];      // interleaved channel 0,1,2
  float r, g, b;
  if (bgr) { b = c0; g = c1; r = c2; } else { r = c0; g = c1; b = c2; }

  const int hw = width * height;
  int idx = y * width + x;
  out[0 * hw + idx] = (r * inv_scale - m0) / s0;   // R plane
  out[1 * hw + idx] = (g * inv_scale - m1) / s1;   // G plane
  out[2 * hw + idx] = (b * inv_scale - m2) / s2;   // B plane
}

cudaError_t normalizeToPlanar(const unsigned char* converted, int channels,
                              int pitch_bytes, int width, int height,
                              float* out_planar, const float mean[3],
                              const float std[3], float inv_scale, bool bgr_order,
                              cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  normKernel<<<grid, block, 0, stream>>>(
      converted, channels, pitch_bytes, width, height, out_planar,
      mean[0], mean[1], mean[2], std[0], std[1], std[2],
      inv_scale, bgr_order ? 1 : 0);
  return cudaGetLastError();
}

} // namespace gdino
