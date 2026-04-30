#include "chunk.h"

#include <cctype>

namespace chirp_kokoro {
namespace {

constexpr size_t kMaxTextChunkChars = 510;

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
    std::string pending;
    current.reserve(normalized.size());
    pending.reserve(normalized.size());

    auto push_chunk = [&](std::string & value) {
        std::string chunk = trim_ascii(value);
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        value.clear();
    };

    auto append_pending = [&]() {
        std::string clause = trim_ascii(pending);
        if (clause.empty()) {
            pending.clear();
            return;
        }
        size_t separator = current.empty() ? 0 : 1;
        if (!current.empty() && current.size() + separator + clause.size() > kMaxTextChunkChars) {
            push_chunk(current);
        }
        if (!current.empty()) {
            current.push_back(' ');
        }
        current += clause;
        pending.clear();
    };

    for (char c : normalized) {
        pending.push_back(c);
        if (is_chunk_boundary(c)) {
            append_pending();
        }
    }

    append_pending();
    if (!current.empty()) {
        push_chunk(current);
    }
    return chunks;
}

}
