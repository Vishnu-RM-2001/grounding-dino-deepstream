#include "gdino_text.h"
#include <algorithm>
#include <cctype>

namespace gdino {

std::string normalizeCaption(const std::string& text) {
  // The model separates phrases with '.', so accept commas and convert them.
  std::string t;
  for (char c : text) {
    if (c == ',') t += " . ";
    else t += (char)std::tolower((unsigned char)c);
  }
  // trim
  size_t b = t.find_first_not_of(" \t\n\r");
  size_t e = t.find_last_not_of(" \t\n\r");
  t = (b == std::string::npos) ? "" : t.substr(b, e - b + 1);
  if (t.empty() || t.back() != '.') t += " .";
  return t;
}

static bool isSpecial(int id) {
  return id == CLS_ID || id == SEP_ID || id == DOT_ID || id == QUESTION_ID;
}

// split normalized caption into phrase strings on '.'/'?'
static std::vector<std::string> splitPhrases(const std::string& cap) {
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    size_t b = cur.find_first_not_of(" \t");
    size_t e = cur.find_last_not_of(" \t");
    if (b != std::string::npos) out.push_back(cur.substr(b, e - b + 1));
    cur.clear();
  };
  for (char c : cap) {
    if (c == '.' || c == '?') flush();
    else cur += c;
  }
  flush();
  return out;
}

bool buildTextTensors(const BertTokenizer& tok, const std::string& caption,
                      TextTensors& out) {
  if (!tok.ok()) return false;
  std::string cap = normalizeCaption(caption);

  std::vector<int> pieces = tok.encodePieces(cap);   // includes '.' separators
  // [CLS] + pieces + [SEP], truncate to MAX_TOKENS
  std::vector<int> ids;
  ids.reserve(MAX_TOKENS);
  ids.push_back(CLS_ID);
  for (int p : pieces) {
    if ((int)ids.size() >= MAX_TOKENS - 1) break;     // leave room for [SEP]
    ids.push_back(p);
  }
  ids.push_back(SEP_ID);
  out.num_tokens = (int)ids.size();

  for (int i = 0; i < MAX_TOKENS; ++i) {
    out.input_ids[i]      = (i < (int)ids.size()) ? ids[i] : PAD_ID;
    out.attention_mask[i] = (i < (int)ids.size()) ? 1 : 0;
    out.token_type_ids[i] = 0;
    out.position_ids[i]   = 0;
  }
  std::fill(out.text_mask.begin(), out.text_mask.end(), (uint8_t)0);

  // ---- block-diagonal mask + position ids (port of GDINO transfer map) ----
  auto setMask = [&](int r, int c) { out.text_mask[(size_t)r * MAX_TOKENS + c] = 1; };
  for (int i = 0; i < MAX_TOKENS; ++i) setMask(i, i);   // identity (eye)

  out.phrase_spans.clear();
  int prev = 0;
  for (int col = 0; col < MAX_TOKENS; ++col) {
    if (!isSpecial(out.input_ids[col])) continue;
    if (col == 0 || col == MAX_TOKENS - 1) {
      setMask(col, col);
      out.position_ids[col] = 0;
    } else {
      for (int r = prev + 1; r <= col; ++r)
        for (int c = prev + 1; c <= col; ++c) setMask(r, c);
      for (int k = prev + 1; k <= col; ++k) out.position_ids[k] = k - (prev + 1);
      if (col - (prev + 1) > 0)                          // a real phrase segment
        out.phrase_spans.emplace_back(prev + 1, col);    // [start,end) tokens
    }
    prev = col;
  }

  out.phrases = splitPhrases(cap);
  // Align: phrase_spans correspond to caption phrases in order. If counts
  // mismatch (e.g. multi-word phrases), keep the smaller and let decoder fall
  // back to span index. (Documented: complex prompts may need span->phrase map.)
  return true;
}

void writeTextRegion(Variant v, const TextTensors& t, float* out) {
  int k = 0;
  auto put1 = [&](const int* a) { for (int i = 0; i < MAX_TOKENS; ++i) out[k++] = (float)a[i]; };
  auto putBlock = [&]() { for (int i = 0; i < MASK_COUNT; ++i) out[k++] = (float)t.text_mask[i]; };
  if (v == Variant::TAO) {
    // input_ids, attention_mask(pad)[256], position_ids, token_type_ids, text_token_mask(block)[256,256]
    put1(t.input_ids); put1(t.attention_mask); put1(t.position_ids); put1(t.token_type_ids); putBlock();
  } else {
    // input_ids, token_type_ids, attention_mask(block)[256,256], position_ids, text_token_mask(pad)[256]
    put1(t.input_ids); put1(t.token_type_ids); putBlock(); put1(t.position_ids); put1(t.attention_mask);
  }
}

Variant variantFromString(const std::string& s) {
  return (s == "gdino_b" || s == "b") ? Variant::GDINO_B : Variant::TAO;
}
const char* variantName(Variant v) { return v == Variant::GDINO_B ? "gdino_b" : "tao"; }

} // namespace gdino
