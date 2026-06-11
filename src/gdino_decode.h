// Grounding-DINO output decode: pred_logits[Q,T] + pred_boxes[Q,4] -> labeled
// detections. A query's score for a phrase = max sigmoid logit over that
// phrase's text-token span; the best phrase labels the box. Boxes are
// normalized cxcywh -> pixel xyxy in frame coords.
#ifndef GDINO_DECODE_H
#define GDINO_DECODE_H
#include "gdino_prompt_store.h"
#include <string>
#include <vector>

namespace gdino {

struct Detection {
  float left, top, width, height;   // pixels, frame coords
  float score;
  int   class_id;                   // phrase index
  std::string label;
};

// Decode raw GDINO outputs into labeled detections.
//   box_thr  : minimum score to keep a box (per-phrase max-sigmoid over its tokens).
//   nms_iou  : class-agnostic NMS IoU threshold; pass <= 0 to disable NMS.
//   max_area : drop boxes whose normalized area (w*h) exceeds this — the occasional
//              whole-image box; pass >= 1.0 to disable the filter.
void decodeGDINO(const float* logits, int Q, int T, const float* boxes,
                 const PromptState& st, float box_thr, float nms_iou, float max_area,
                 int frameW, int frameH, std::vector<Detection>& out);

} // namespace gdino
#endif
