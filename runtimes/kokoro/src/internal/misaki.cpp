#include "misaki.h"

#include <array>
#include <string>

namespace chirp_kokoro {
namespace {

struct Replacement {
    const char * old_value;
    const char * new_value;
};

constexpr std::array<Replacement, 26> kFromEspeak = {{
    {"ʔˌn̩", "tᵊn"},
    {"a^ɪ", "I"},
    {"aɪ", "I"},
    {"a^ʊ", "W"},
    {"aʊ", "W"},
    {"d^ʒ", "ʤ"},
    {"dʒ", "ʤ"},
    {"e^ɪ", "A"},
    {"eɪ", "A"},
    {"t^ʃ", "ʧ"},
    {"tʃ", "ʧ"},
    {"ɔ^ɪ", "Y"},
    {"ɔɪ", "Y"},
    {"ə^l", "ᵊl"},
    {"ʔn", "tᵊn"},
    {"ʲO", "jO"},
    {"ʲQ", "jQ"},
    {"o^ʊ", "O"},
    {"oʊ", "O"},
    {"̃", ""},
    {"e", "A"},
    {"r", "ɹ"},
    {"x", "k"},
    {"ç", "k"},
    {"ɐ", "ə"},
    {"ɚ", "əɹ"},
}};

void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

void remove_all(std::string & s, const std::string & needle) {
    replace_all(s, needle, "");
}

}

std::string espeak_to_misaki(const std::string & phonemes, bool british) {
    std::string ps = phonemes;
    for (const auto & replacement : kFromEspeak) {
        replace_all(ps, replacement.old_value, replacement.new_value);
    }

    replace_all(ps, "ɬ", "l");
    replace_all(ps, "ʔ", "t");
    remove_all(ps, "ʲ");

    size_t combining = 0;
    while ((combining = ps.find("̩", combining)) != std::string::npos) {
        size_t char_start = combining;
        if (char_start > 0) {
            --char_start;
            while (char_start > 0 && (static_cast<unsigned char>(ps[char_start]) & 0xc0) == 0x80) {
                --char_start;
            }
        }
        ps.erase(combining, std::string("̩").size());
        ps.insert(char_start, "ᵊ");
        combining = char_start + std::string("ᵊ").size();
    }
    remove_all(ps, "̹");

    if (british) {
        replace_all(ps, "e^ə", "ɛː");
        replace_all(ps, "iə", "ɪə");
        replace_all(ps, "ə^ʊ", "Q");
        replace_all(ps, "əʊ", "Q");
    } else {
        replace_all(ps, "o^ʊ", "O");
        replace_all(ps, "oʊ", "O");
        replace_all(ps, "ɜːɹ", "ɜɹ");
        replace_all(ps, "ɜː", "ɜɹ");
        replace_all(ps, "ɪə", "iə");
        remove_all(ps, "ː");
    }

    remove_all(ps, "^");
    return ps;
}

}
