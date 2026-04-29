#pragma once

#include <string>
#include <vector>

namespace qwen3_tts {

struct wav_data {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

bool read_wav_mono(const std::string & path, wav_data & wav, std::string & error);
bool write_wav_mono16(const std::string & path, const std::vector<float> & samples, int sample_rate);

} // namespace qwen3_tts
