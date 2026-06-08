#include "bert_tokenizer.h"
#include <fstream>
#include <cctype>

namespace gdino {

// ---- utf8 helpers: split a utf8 string into a vector of utf8 "characters" ----
static std::vector<std::string> utf8Chars(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t len = (c < 0x80) ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3
                 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + len > s.size()) len = 1;
    out.emplace_back(s.substr(i, len));
    i += len;
  }
  return out;
}

// BERT punctuation: ascii ranges 33-47,58-64,91-96,123-126 plus anything that is
// not alnum/space and is single-byte. (Unicode P* categories approximated.)
static bool isPunct(const std::string& ch) {
  if (ch.size() != 1) return false;                 // treat multibyte as non-punct
  unsigned char c = ch[0];
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
      (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
  return false;
}
static bool isWhitespace(const std::string& ch) {
  if (ch.size() != 1) return false;
  unsigned char c = ch[0];
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == 0x0b;
}
static bool isControl(const std::string& ch) {
  if (ch.size() != 1) return false;
  unsigned char c = ch[0];
  if (c == '\t' || c == '\n' || c == '\r') return false;
  return c < 0x20 || c == 0x7f;
}
static std::string lowerAscii(const std::string& ch) {
  if (ch.size() == 1) { std::string r = ch; r[0] = (char)std::tolower((unsigned char)r[0]); return r; }
  return ch;  // non-ascii lowercasing not handled
}

bool BertTokenizer::load(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) return false;
  std::string line; int idx = 0;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    vocab_[line] = idx++;
  }
  auto it = vocab_.find("[UNK]");
  if (it != vocab_.end()) unk_id_ = it->second;
  loaded_ = !vocab_.empty();
  return loaded_;
}

int BertTokenizer::id(const std::string& tok) const {
  auto it = vocab_.find(tok);
  return it == vocab_.end() ? -1 : it->second;
}

// clean + lowercase + split on whitespace and punctuation (each punct = 1 token)
std::vector<std::string> BertTokenizer::basicTokenize(const std::string& text) const {
  std::vector<std::string> tokens;
  std::string cur;
  auto flush = [&]() { if (!cur.empty()) { tokens.push_back(cur); cur.clear(); } };
  for (const auto& ch : utf8Chars(text)) {
    if (isControl(ch)) continue;
    if (isWhitespace(ch)) { flush(); continue; }
    std::string lc = lowerAscii(ch);
    if (isPunct(lc)) { flush(); tokens.push_back(lc); }     // punctuation standalone
    else cur += lc;
  }
  flush();
  return tokens;
}

// greedy longest-match-first WordPiece
std::vector<std::string> BertTokenizer::wordpiece(const std::string& token) const {
  std::vector<std::string> out;
  auto chars = utf8Chars(token);
  if ((int)chars.size() > max_chars_per_word_) return {"[UNK]"};
  size_t start = 0;
  std::vector<std::string> subs;
  while (start < chars.size()) {
    size_t end = chars.size();
    std::string cur;
    bool found = false;
    while (start < end) {
      std::string sub;
      for (size_t k = start; k < end; ++k) sub += chars[k];
      if (start > 0) sub = "##" + sub;
      if (vocab_.count(sub)) { cur = sub; found = true; break; }
      --end;
    }
    if (!found) return {"[UNK]"};                            // is_bad -> whole word UNK
    subs.push_back(cur);
    start = end;
  }
  return subs;
}

std::vector<int> BertTokenizer::encodePieces(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& bt : basicTokenize(text))
    for (const auto& wp : wordpiece(bt)) {
      auto it = vocab_.find(wp);
      ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
    }
  return ids;
}

} // namespace gdino
