// Custom Gst-nvinfer bbox parser (network-type=0). Decodes pred_logits/pred_boxes
// into boxes in network (960x544) space; nvinfer maps them to the frame via the
// preprocess ROI. class_id = phrase index (labelled by the app's probe).
#include "nvdsinfer_custom_impl.h"
#include "gdino_decode.h"
#include "gdino_prompt_store.h"
#include "gdino_layout.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// detection threshold: GDINO_THR env (default 0.25). GDINO confidences are
// low-calibrated; weak phrases like "man"/"pedestrian" need ~0.25.
static float boxThreshold() {
  const char* e = getenv("GDINO_THR");
  return e ? (float)atof(e) : 0.25f;
}

extern "C" bool NvDsInferParseCustomGDINO(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  static int calls = 0;
  const float* logits = nullptr; const float* boxes = nullptr;
  int Q = 900, T = 256;
  for (auto& l : outputLayersInfo) {
    if (!strcmp(l.layerName, "pred_logits")) {
      logits = (const float*)l.buffer;
      if (l.inferDims.numDims >= 2) { Q = l.inferDims.d[0]; T = l.inferDims.d[1]; }
    } else if (!strcmp(l.layerName, "pred_boxes")) {
      boxes = (const float*)l.buffer;
    }
  }
  if (calls < 3)
    fprintf(stderr, "[parser] call#%d layers=%zu logits=%p boxes=%p Q=%d T=%d "
            "first_logits=%.3f\n", calls, outputLayersInfo.size(),
            (const void*)logits, (const void*)boxes, Q, T,
            logits ? logits[0] : -999.0f);
  ++calls;
  if (!logits || !boxes) return true;

  auto snap = gdino::PromptStore::instance().current();
  if (!snap) return true;

  float thr = boxThreshold();
  std::vector<gdino::Detection> dets;
  gdino::decodeGDINO(logits, Q, T, boxes, *snap, thr, thr,
                     gdino::IMG_W, gdino::IMG_H, dets);  // boxes in 960x544
  for (auto& d : dets) {
    NvDsInferObjectDetectionInfo o;
    memset(&o, 0, sizeof(o));
    o.classId = d.class_id;
    o.left = d.left; o.top = d.top; o.width = d.width; o.height = d.height;
    o.detectionConfidence = d.score;
    objectList.push_back(o);
  }
  return true;
}
