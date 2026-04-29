#include "chunk.h"

#include <cctype>

namespace chirp_kokoro {
namespace {

bool is_ascii_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

std::string trim_ascii(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && is_ascii_space(text[begin])) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && is_ascii_space(text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

}

std::string normalize_text_utf8(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    bool pending_space = false;
    for (char c : text) {
        if (is_ascii_space(c)) {
            pending_space = true;
            continue;
        }
        if (pending_space && !out.empty()) {
            out.push_back(' ');
        }
        pending_space = false;
        out.push_back(c);
    }
    return trim_ascii(out);
}

std::vector<std::string> chunk_text(const std::string & text) {
    std::string normalized = normalize_text_utf8(text);
    std::vector<std::string> chunks;
    if (!normalized.empty()) {
        chunks.push_back(normalized);
    }
    return chunks;
}

}
