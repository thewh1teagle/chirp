#include "chunk.h"

#include <cctype>

namespace chirp_kokoro {
namespace {

bool is_ascii_space(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

bool is_chunk_boundary(char c) {
    return c == '?' || c == '!' || c == '.' || c == ',';
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
    std::string current;
    current.reserve(normalized.size());

    auto flush = [&]() {
        std::string chunk = trim_ascii(current);
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        current.clear();
    };

    for (char c : normalized) {
        current.push_back(c);
        if (is_chunk_boundary(c)) {
            flush();
        }
    }

    if (!current.empty()) {
        flush();
    }
    return chunks;
}

}
