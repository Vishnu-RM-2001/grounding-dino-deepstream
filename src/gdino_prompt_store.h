// Thread-safe store of the current prompt and its tokenized text tensors. A
// control thread reads new prompts from a FIFO and swaps atomically — this is
// what makes the text live. One shared instance for the preprocess lib and the
// decoder (both link this common lib).
#ifndef GDINO_PROMPT_STORE_H
#define GDINO_PROMPT_STORE_H
#include "bert_tokenizer.h"
#include "gdino_text.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace gdino {

struct PromptState {
  uint64_t version = 0;
  std::string caption;
  TextTensors text;
};

class PromptStore {
public:
  static PromptStore& instance();

  // vocab_path: bert vocab.txt; initial: first prompt; fifo_path: control FIFO
  // (created if missing; empty => no control thread). Safe to call once.
  bool init(const std::string& vocab_path, const std::string& initial,
            const std::string& fifo_path);

  std::shared_ptr<const PromptState> current() const;   // atomic snapshot
  bool setPrompt(const std::string& caption);           // tokenize + swap
  void shutdown();
  bool ready() const { return ready_.load(); }

private:
  PromptStore() = default;
  void controlLoop(std::string fifo_path);

  BertTokenizer tok_;
  mutable std::mutex mu_;
  std::shared_ptr<const PromptState> state_;
  std::atomic<uint64_t> version_{0};
  std::atomic<bool> ready_{false};
  std::atomic<bool> run_{false};
  std::thread ctrl_;
  std::string fifo_path_;
};

} // namespace gdino
#endif
