#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class qwen_tokenizer {
public:
    bool load_from_gguf(const std::string & model_path);
    bool tokenize_tts_text(const std::string & text, std::vector<int32_t> & tokens) const;
    const std::string & error() const { return error_; }

private:
    bool tokenize_piece(const std::string & piece, std::vector<int32_t> & tokens) const;
    std::vector<std::string> pre_tokenize_ascii(const std::string & text) const;

    mutable std::string error_;
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> bpe_ranks_;
};
