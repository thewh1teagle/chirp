#include "phonemize.h"

#include <espeak-ng/speak_lib.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>

namespace chirp_kokoro {
namespace {

std::once_flag g_init_once;
PhonemizeResult g_init_result;

std::string strip_lang_switches(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    int depth = 0;
    for (char c : input) {
        if (c == '(') {
            ++depth;
        } else if (c == ')') {
            if (depth > 0) {
                --depth;
            }
        } else if (depth == 0) {
            out.push_back(c);
        }
    }
    return out;
}

void init_espeak() {
    const char * data_path = std::getenv("ESPEAK_DATA_PATH");
    std::string init_path;
    if (data_path && data_path[0] != '\0') {
        std::filesystem::path path(data_path);
        init_path = path.filename() == "espeak-ng-data" ? path.parent_path().string() : path.string();
    }
    int sample_rate = espeak_Initialize(
        AUDIO_OUTPUT_RETRIEVAL,
        0,
        init_path.empty() ? nullptr : init_path.c_str(),
        espeakINITIALIZE_DONT_EXIT);
    if (sample_rate <= 0) {
        g_init_result.ok = false;
        g_init_result.error = "failed to initialize espeak-ng";
        return;
    }
    g_init_result.ok = true;
}

bool ends_sentence(const std::string & text) {
    for (auto it = text.rbegin(); it != text.rend(); ++it) {
        if (*it == ' ' || *it == '\n' || *it == '\t' || *it == '\r') {
            continue;
        }
        return *it == '.' || *it == '?' || *it == '!';
    }
    return false;
}

}

PhonemizeResult phonemize(const std::string & text, const std::string & language) {
    std::call_once(g_init_once, init_espeak);
    if (!g_init_result.ok) {
        return g_init_result;
    }

    if (espeak_SetVoiceByName(language.c_str()) != EE_OK) {
        return {false, "failed to set espeak-ng voice: " + language, {}};
    }

    PhonemizeResult result;
    result.ok = true;
    std::string current;
    int phoneme_mode = ('^' << 8) | espeakINITIALIZE_PHONEME_IPA;

    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        const void * text_ptr = line.c_str();

        while (text_ptr != nullptr) {
            const char * raw = espeak_TextToPhonemes(&text_ptr, espeakCHARS_UTF8, phoneme_mode);
            if (raw == nullptr) {
                continue;
            }
            std::string clause = strip_lang_switches(raw);
            if (clause.empty()) {
                continue;
            }
            if (!current.empty() && current.back() != ' ') {
                current.push_back(' ');
            }
            current += clause;
            if (ends_sentence(current)) {
                result.sentences.push_back(std::exchange(current, {}));
            }
        }

        if (!current.empty()) {
            result.sentences.push_back(std::exchange(current, {}));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return result;
}

}
