#include "codec_wav.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cstdint>

namespace qwen3_tts {

bool read_wav_mono(const std::string & path, wav_data & wav, std::string & error) {
    unsigned int channels = 0;
    unsigned int sample_rate = 0;
    drwav_uint64 n_frames = 0;
    float * pcm = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &sample_rate, &n_frames, nullptr);
    if (!pcm) {
        error = "failed to open WAV: " + path;
        return false;
    }
    if (channels == 0 || sample_rate == 0 || n_frames == 0) {
        drwav_free(pcm, nullptr);
        error = "unsupported or empty WAV";
        return false;
    }

    wav.samples.assign((size_t) n_frames, 0.0f);
    wav.sample_rate = (int) sample_rate;
    wav.channels = 1;
    for (drwav_uint64 i = 0; i < n_frames; ++i) {
        double sum = 0.0;
        for (unsigned int ch = 0; ch < channels; ++ch) {
            sum += pcm[i * channels + ch];
        }
        wav.samples[(size_t) i] = (float) (sum / channels);
    }
    drwav_free(pcm, nullptr);
    return true;
}

bool write_wav_mono16(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    drwav_data_format format = {};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sample_rate;
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        return false;
    }

    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = (int16_t) (s * 32767.0f);
    }
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, samples.size(), pcm.data());
    drwav_uninit(&wav);
    return written == samples.size();
}

} // namespace qwen3_tts
