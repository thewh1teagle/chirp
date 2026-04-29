#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

class qwen_tokenizer {
public:
    qwen_tokenizer();
    ~qwen_tokenizer();

    bool load_from_gguf(const std::string & model_path);
    bool tokenize_tts_text(const std::string & text, std::vector<int32_t> & tokens) const;
    const std::string & error() const { return error_; }

private:
    mutable std::string error_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};
