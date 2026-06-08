#include "gdino_prompt_store.h"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <cstring>

namespace gdino {

PromptStore& PromptStore::instance() {
  static PromptStore s;
  return s;
}

bool PromptStore::init(const std::string& vocab_path, const std::string& initial,
                       const std::string& fifo_path) {
  if (ready_.load()) return true;
  if (!tok_.load(vocab_path)) {
    fprintf(stderr, "[gdino] FATAL: cannot load vocab '%s'\n", vocab_path.c_str());
    return false;
  }
  if (!setPrompt(initial.empty() ? "object ." : initial)) return false;
  ready_.store(true);

  if (!fifo_path.empty()) {
    fifo_path_ = fifo_path;
    run_.store(true);
    ctrl_ = std::thread(&PromptStore::controlLoop, this, fifo_path);
  }
  fprintf(stderr, "[gdino] prompt store ready. initial='%s' fifo='%s'\n",
          initial.c_str(), fifo_path.c_str());
  return true;
}

bool PromptStore::setPrompt(const std::string& caption) {
  auto st = std::make_shared<PromptState>();
  st->caption = caption;
  if (!buildTextTensors(tok_, caption, st->text)) return false;
  st->version = ++version_;
  {
    std::lock_guard<std::mutex> lk(mu_);
    state_ = st;
  }
  fprintf(stderr, "[gdino] prompt v%llu := '%s' (%zu phrases, %d tokens)\n",
          (unsigned long long)st->version, caption.c_str(),
          st->text.phrases.size(), st->text.num_tokens);
  return true;
}

std::shared_ptr<const PromptState> PromptStore::current() const {
  std::lock_guard<std::mutex> lk(mu_);
  return state_;
}

void PromptStore::controlLoop(std::string fifo_path) {
  if (mkfifo(fifo_path.c_str(), 0666) != 0 && errno != EEXIST) {
    fprintf(stderr, "[gdino] mkfifo('%s') failed: %s\n",
            fifo_path.c_str(), strerror(errno));
    return;
  }
  fprintf(stderr, "[gdino] control FIFO listening: echo \"car . person .\" > %s\n",
          fifo_path.c_str());
  std::string buf;
  while (run_.load()) {
    int fd = open(fifo_path.c_str(), O_RDONLY);          // blocks until a writer
    if (fd < 0) { if (!run_.load()) break; usleep(100000); continue; }
    char tmp[1024];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0) {
      buf.append(tmp, n);
      size_t nl;
      while ((nl = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, nl);
        buf.erase(0, nl + 1);
        if (!line.empty()) setPrompt(line);
      }
    }
    close(fd);
    if (!buf.empty()) { setPrompt(buf); buf.clear(); }    // line without newline
  }
}

void PromptStore::shutdown() {
  run_.store(false);
  if (ctrl_.joinable()) {
    // unblock a blocked O_RDONLY open()/read() by opening the write end
    int fd = open(fifo_path_.c_str(), O_WRONLY | O_NONBLOCK);
    if (fd >= 0) { ssize_t r = write(fd, "\n", 1); (void)r; close(fd); }
    ctrl_.join();
  }
}

} // namespace gdino
