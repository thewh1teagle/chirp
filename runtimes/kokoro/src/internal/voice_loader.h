#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace chirp_kokoro {

struct VoiceData {
    std::vector<float> values;
    size_t rows = 0;
    size_t dims = 0;
};

bool load_voice_from_archive(
    const std::string & voices_path,
    const std::string & voice,
    VoiceData & out,
    std::string & error);

bool list_voices_from_archive(
    const std::string & voices_path,
    std::vector<std::string> & out,
    std::string & error);

}
