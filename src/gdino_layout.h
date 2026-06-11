// Pack layout for the single-input engine. The packed fp32 tensor is always:
//   [ image (offset 0) | text region (offset IMG_COUNT, contiguous) ]
// Both supported models share the image region and total size; they differ ONLY
// in the ORDER of the 5 text tensors inside the text region (see writeTextRegion).
// Pack orders (must match onnx/build_single_input_onnx.py --variant):
//   tao        : input_ids, attention_mask[256], position_ids, token_type_ids,
//                text_token_mask[256,256]   (block-diagonal mask = text_token_mask)
//   gdino_b: input_ids, token_type_ids, attention_mask[256,256], position_ids,
//                text_token_mask[256]       (block-diagonal mask = attention_mask)
// input_ids is first in the text region for BOTH, so OFF_TEXT is variant-neutral.
#ifndef GDINO_LAYOUT_H
#define GDINO_LAYOUT_H
#include <cstdint>
#include <string>

namespace gdino {

// Image (same for both models).
// To change the model input resolution you must keep these in sync with the engine:
//   1) re-export/rebuild the ONNX at the new HxW (gdino_b: export/export_onnx.py
//      --height/--width; tao is fixed at 544x960 by the released model),
//   2) update IMG_H/IMG_W below and rebuild the libs,
//   3) set configs/config_preprocess_gdino.txt processing-width/height and
//      network-input-shape (= 1;PACKED_TOTAL;1;1) and the engine --*Shapes accordingly.
// H and W must be multiples of 32 (Swin backbone).
constexpr int   IMG_C = 3, IMG_H = 544, IMG_W = 960;
constexpr int   IMG_COUNT = IMG_C * IMG_H * IMG_W;     // 1,566,720
// Text
constexpr int   MAX_TOKENS = 256;
constexpr int   MASK_COUNT = MAX_TOKENS * MAX_TOKENS;  // 65,536 (block-diagonal)

constexpr int64_t OFF_IMAGE    = 0;
constexpr int64_t OFF_TEXT     = IMG_COUNT;            // text region start
constexpr int64_t TEXT_COUNT   = 4 * MAX_TOKENS + MASK_COUNT;   // 66,560
constexpr int64_t PACKED_TOTAL = IMG_COUNT + TEXT_COUNT;        // 1,633,280

// GroundingDINO special tokens (bert-base-uncased): [CLS] [SEP] '.' '?'
constexpr int CLS_ID = 101, SEP_ID = 102, DOT_ID = 1012, QUESTION_ID = 1029;
constexpr int PAD_ID = 0;

// Which model's packing/decoding to use.
enum class Variant { TAO, GDINO_B };
Variant variantFromString(const std::string& s);      // "tao" (default) | "gdino_b"
const char* variantName(Variant v);

} // namespace gdino
#endif
