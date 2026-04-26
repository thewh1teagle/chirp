#include "qwen_tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

static std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back((char) cp);
    } else if (cp <= 0x7ff) {
        out.push_back((char) (0xc0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3f)));
    } else {
        out.push_back((char) (0xe0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3f)));
        out.push_back((char) (0x80 | (cp & 0x3f)));
    }
    return out;
}

static std::vector<std::string> make_byte_encoder() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xa1; i <= 0xac; ++i) bs.push_back(i);
    for (int i = 0xae; i <= 0xff; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    std::vector<std::string> enc(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        enc[(uint8_t) bs[i]] = utf8_encode((uint32_t) cs[i]);
    }
    return enc;
}

static std::string pair_key(const std::string & a, const std::string & b) {
    return a + '\001' + b;
}

bool qwen_tokenizer::load_from_gguf(const std::string & model_path) {
    error_.clear();
    byte_encoder_ = make_byte_encoder();

    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_ = "failed to open GGUF tokenizer metadata: " + model_path;
        return false;
    }

    const int64_t toks_id = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const int64_t merges_id = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (toks_id < 0 || merges_id < 0) {
        error_ = "model GGUF does not contain tokenizer.ggml.tokens/merges";
        gguf_free(ctx);
        if (meta) ggml_free(meta);
        return false;
    }

    vocab_.clear();
    const size_t n_vocab = gguf_get_arr_n(ctx, toks_id);
    for (size_t i = 0; i < n_vocab; ++i) {
        vocab_[gguf_get_arr_str(ctx, toks_id, i)] = (int32_t) i;
    }

    bpe_ranks_.clear();
    const size_t n_merges = gguf_get_arr_n(ctx, merges_id);
    for (size_t i = 0; i < n_merges; ++i) {
        std::string merge = gguf_get_arr_str(ctx, merges_id, i);
        size_t sp = merge.find(' ');
        if (sp != std::string::npos) {
            bpe_ranks_[pair_key(merge.substr(0, sp), merge.substr(sp + 1))] = (int32_t) i;
        }
    }

    gguf_free(ctx);
    if (meta) ggml_free(meta);
    return true;
}

std::vector<std::string> qwen_tokenizer::pre_tokenize_ascii(const std::string & text) const {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c = (unsigned char) text[i];
        if (c == '\n' || c == '\r') {
            size_t j = i;
            while (j < text.size() && (text[j] == '\n' || text[j] == '\r')) j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if (c == ' ' && i + 1 < text.size() && std::isalpha((unsigned char) text[i + 1])) {
            size_t j = i + 1;
            while (j < text.size() && std::isalpha((unsigned char) text[j])) j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if (std::isalpha(c)) {
            size_t j = i;
            while (j < text.size() && std::isalpha((unsigned char) text[j])) j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else if (c == ' ' && i + 1 < text.size() && std::isdigit((unsigned char) text[i + 1])) {
            out.push_back(text.substr(i, 2));
            i += 2;
        } else if (std::isdigit(c)) {
            out.push_back(text.substr(i, 1));
            i++;
        } else if (std::isspace(c)) {
            size_t j = i;
            while (j < text.size() && std::isspace((unsigned char) text[j]) && text[j] != '\n' && text[j] != '\r') j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        } else {
            size_t j = i;
            if (text[j] == ' ') j++;
            while (j < text.size() && !std::isalnum((unsigned char) text[j]) && !std::isspace((unsigned char) text[j])) j++;
            out.push_back(text.substr(i, j - i));
            i = j;
        }
    }
    return out;
}

bool qwen_tokenizer::tokenize_piece(const std::string & piece, std::vector<int32_t> & tokens) const {
    std::vector<std::string> word;
    for (uint8_t b : piece) {
        word.push_back(byte_encoder_[b]);
    }
    while (word.size() > 1) {
        int32_t best_rank = INT32_MAX;
        size_t best = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = bpe_ranks_.find(pair_key(word[i], word[i + 1]));
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        word[best] += word[best + 1];
        word.erase(word.begin() + (long) best + 1);
    }
    for (const std::string & w : word) {
        auto it = vocab_.find(w);
        if (it == vocab_.end()) {
            error_ = "missing token in vocab: " + w;
            return false;
        }
        tokens.push_back(it->second);
    }
    return true;
}

bool qwen_tokenizer::tokenize_tts_text(const std::string & text, std::vector<int32_t> & tokens) const {
    tokens.clear();
    const std::string prompt = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string specials[] = {"<|im_start|>", "<|im_end|>"};
    size_t pos = 0;
    while (pos < prompt.size()) {
        size_t next_pos = std::string::npos;
        std::string next_special;
        for (const auto & sp : specials) {
            size_t p = prompt.find(sp, pos);
            if (p != std::string::npos && (next_pos == std::string::npos || p < next_pos)) {
                next_pos = p;
                next_special = sp;
            }
        }
        std::string normal = prompt.substr(pos, next_pos == std::string::npos ? std::string::npos : next_pos - pos);
        for (const std::string & piece : pre_tokenize_ascii(normal)) {
            if (!tokenize_piece(piece, tokens)) return false;
        }
        if (next_pos == std::string::npos) break;
        auto it = vocab_.find(next_special);
        if (it != vocab_.end()) {
            tokens.push_back(it->second);
        } else if (next_special == "<|im_start|>") {
            tokens.push_back(151644);
        } else if (next_special == "<|im_end|>") {
            tokens.push_back(151645);
        } else {
            error_ = "missing special token: " + next_special;
            return false;
        }
        pos = next_pos + next_special.size();
    }
    return true;
}
