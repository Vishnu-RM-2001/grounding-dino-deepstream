#include "gdino_decode.h"
#include <cmath>
#include <algorithm>

namespace gdino {

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static float iou(const Detection& a, const Detection& b) {
  float ax2 = a.left + a.width, ay2 = a.top + a.height;
  float bx2 = b.left + b.width, by2 = b.top + b.height;
  float ix1 = std::max(a.left, b.left), iy1 = std::max(a.top, b.top);
  float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  float inter = iw * ih;
  float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

// class-agnostic NMS: an overlapping box keeps only the highest-scoring label.
static void nms(std::vector<Detection>& dets, float iou_thr) {
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  for (const auto& d : dets) {
    bool suppressed = false;
    for (const auto& k : keep)
      if (iou(d, k) > iou_thr) { suppressed = true; break; }
    if (!suppressed) keep.push_back(d);
  }
  dets.swap(keep);
}

void decodeGDINO(const float* logits, int Q, int T, const float* boxes,
                 const PromptState& st, float box_thr, float nms_iou, float max_area,
                 int frameW, int frameH, std::vector<Detection>& out) {
  const auto& spans = st.text.phrase_spans;
  const auto& phrases = st.text.phrases;
  if (spans.empty()) return;

  for (int q = 0; q < Q; ++q) {
    const float* lq = logits + (size_t)q * T;
    int best_p = -1; float best_s = 0.0f;
    for (size_t p = 0; p < spans.size(); ++p) {
      float s = 0.0f;
      for (int t = spans[p].first; t < spans[p].second && t < T; ++t)
        s = std::max(s, sigmoid(lq[t]));
      if (s > best_s) { best_s = s; best_p = (int)p; }
    }
    if (best_p < 0 || best_s < box_thr) continue;

    const float* bq = boxes + (size_t)q * 4;
    float cx = bq[0], cy = bq[1], w = bq[2], h = bq[3];   // normalized cxcywh
    if (w * h > max_area) continue;   // drop the occasional whole-image box
    Detection d;
    d.left   = (cx - w * 0.5f) * frameW;
    d.top    = (cy - h * 0.5f) * frameH;
    d.width  = w * frameW;
    d.height = h * frameH;
    d.left = std::max(0.0f, d.left);
    d.top  = std::max(0.0f, d.top);
    d.score = best_s;
    d.class_id = best_p;
    d.label = (best_p < (int)phrases.size()) ? phrases[best_p] : std::string("object");
    out.push_back(std::move(d));
  }
  if (nms_iou > 0.0f) nms(out, nms_iou);
}

} // namespace gdino
