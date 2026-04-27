#include "qwen_tokenizer.h"

#include "gguf.h"

#include <exception>
#include <tokenizers_cpp.h>

qwen_tokenizer::qwen_tokenizer() = default;
qwen_tokenizer::~qwen_tokenizer() = default;

bool qwen_tokenizer::load_from_gguf(const std::string & model_path) {
    error_.clear();
    tokenizer_.reset();

    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_ = "failed to open GGUF tokenizer metadata: " + model_path;
        return false;
    }

    const int64_t tokenizer_json_id = gguf_find_key(ctx, "tokenizer.huggingface.json");
    if (tokenizer_json_id < 0) {
        error_ = "model GGUF does not contain tokenizer.huggingface.json";
        gguf_free(ctx);
        if (meta) ggml_free(meta);
        return false;
    }

    std::string tokenizer_json = gguf_get_val_str(ctx, tokenizer_json_id);
    gguf_free(ctx);
    if (meta) ggml_free(meta);

    try {
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);
    } catch (const std::exception & e) {
        error_ = std::string("failed to load HuggingFace tokenizer JSON: ") + e.what();
        return false;
    }
    if (!tokenizer_) {
        error_ = "failed to load HuggingFace tokenizer JSON";
        return false;
    }
    return true;
}

bool qwen_tokenizer::tokenize_tts_text(const std::string & text, std::vector<int32_t> & tokens) const {
    tokens.clear();
    if (!tokenizer_) {
        error_ = "tokenizer is not loaded";
        return false;
    }

    const std::string prompt = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    try {
        tokens = tokenizer_->Encode(prompt);
    } catch (const std::exception & e) {
        error_ = std::string("failed to tokenize text: ") + e.what();
        return false;
    }
    return true;
}
