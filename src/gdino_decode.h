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

void decodeGDINO(const float* logits, int Q, int T, const float* boxes,
                 const PromptState& st, float box_thr, float text_thr,
                 int frameW, int frameH, std::vector<Detection>& out);

} // namespace gdino
#endif
