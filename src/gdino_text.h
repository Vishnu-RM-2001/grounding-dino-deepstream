// Grounding-DINO text encoding recipe (port of
// generate_masks_with_special_tokens_and_transfer_map + HF max_length padding).
// The recipe is identical for both models; only the packed write-order differs.
#ifndef GDINO_TEXT_H
#define GDINO_TEXT_H
#include "bert_tokenizer.h"
#include "gdino_layout.h"
#include <string>
#include <vector>
#include <cstdint>

namespace gdino {

struct TextTensors {
  int input_ids[MAX_TOKENS];
  int attention_mask[MAX_TOKENS];          // PADDING mask: 1 real token, 0 pad
  int position_ids[MAX_TOKENS];            // reset per phrase
  int token_type_ids[MAX_TOKENS];          // all 0
  std::vector<uint8_t> text_mask;          // [256*256] block-diagonal self-attn, 0/1
  int num_tokens = 0;                      // count of non-pad tokens

  // For the box decoder: each phrase -> its token-index span [start,end) in
  // input_ids, and the phrase text for labeling.
  std::vector<std::pair<int,int>> phrase_spans;
  std::vector<std::string> phrases;

  TextTensors() : text_mask(MASK_COUNT, 0) {}
};

std::string normalizeCaption(const std::string& text);

// Build all text tensors for a caption. Returns false if tokenizer not ready.
bool buildTextTensors(const BertTokenizer& tok, const std::string& caption,
                      TextTensors& out);

// Write the 5 text tensors contiguously into out[] (TEXT_COUNT floats) in the
// pack order for the given model variant. This region follows the image, so it
// can be H2D-copied straight to packed + OFF_TEXT.
void writeTextRegion(Variant v, const TextTensors& t, float* out);

} // namespace gdino
#endif
