// BERT (bert-base-uncased) tokenizer: BasicTokenizer + WordPiece, STL only.
// Matches HuggingFace for lowercase English prompts (no accent/CJK handling).
#ifndef GDINO_BERT_TOKENIZER_H
#define GDINO_BERT_TOKENIZER_H
#include <string>
#include <vector>
#include <unordered_map>

namespace gdino {

class BertTokenizer {
public:
  bool load(const std::string& vocab_path);          // vocab.txt, id = line index
  // Encode raw caption into WordPiece ids (NO [CLS]/[SEP] added here).
  std::vector<int> encodePieces(const std::string& text) const;
  int id(const std::string& tok) const;              // -1 if absent
  bool ok() const { return loaded_; }
  size_t vocabSize() const { return vocab_.size(); }

private:
  std::vector<std::string> basicTokenize(const std::string& text) const;
  std::vector<std::string> wordpiece(const std::string& token) const;
  std::unordered_map<std::string, int> vocab_;
  int unk_id_ = 100;                                  // [UNK]
  int max_chars_per_word_ = 100;
  bool loaded_ = false;
};

} // namespace gdino
#endif
