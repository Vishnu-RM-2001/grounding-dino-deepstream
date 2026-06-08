// Pack layout for the single-input engine. Must match what
// onnx/build_single_input_onnx.py packs: the preprocess lib writes this float32
// layout, and the ONNX preamble splits it back into the 6 inputs.
#ifndef GDINO_LAYOUT_H
#define GDINO_LAYOUT_H
#include <cstdint>

namespace gdino {

// Image
constexpr int   IMG_C = 3, IMG_H = 544, IMG_W = 960;
constexpr int   IMG_COUNT = IMG_C * IMG_H * IMG_W;   // 1,566,720
// Text
constexpr int   MAX_TOKENS = 256;
constexpr int   MASK_COUNT = MAX_TOKENS * MAX_TOKENS; // 65,536

// Offsets (in float32 elements) into the packed tensor — pack order:
// inputs, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask
constexpr int64_t OFF_IMAGE          = 0;
constexpr int64_t OFF_INPUT_IDS      = OFF_IMAGE          + IMG_COUNT;   // 1,566,720
constexpr int64_t OFF_ATTENTION_MASK = OFF_INPUT_IDS      + MAX_TOKENS;  // 1,566,976
constexpr int64_t OFF_POSITION_IDS   = OFF_ATTENTION_MASK + MAX_TOKENS;  // 1,567,232
constexpr int64_t OFF_TOKEN_TYPE_IDS = OFF_POSITION_IDS   + MAX_TOKENS;  // 1,567,488
constexpr int64_t OFF_TEXT_MASK      = OFF_TOKEN_TYPE_IDS + MAX_TOKENS;  // 1,567,744
constexpr int64_t PACKED_TOTAL       = OFF_TEXT_MASK      + MASK_COUNT;  // 1,633,280
// The 5 text tensors are contiguous from OFF_INPUT_IDS to the end.
constexpr int64_t TEXT_COUNT         = PACKED_TOTAL - IMG_COUNT;         // 66,560

// GDINO special tokens (bert-base-uncased): [CLS] [SEP] '.' '?'
constexpr int CLS_ID = 101, SEP_ID = 102, DOT_ID = 1012, QUESTION_ID = 1029;
constexpr int PAD_ID = 0;

} // namespace gdino
#endif
