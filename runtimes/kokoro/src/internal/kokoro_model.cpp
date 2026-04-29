#include "kokoro_model.h"

#include "chunk.h"
#include "misaki.h"
#include "phonemize.h"
#include "voice_loader.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace chirp_kokoro {
namespace {

constexpr int kSampleRate = 24000;
constexpr size_t kMaxPhonemeLength = 510;

void write_u16(std::ofstream & out, uint16_t v) {
    out.put(static_cast<char>(v & 0xff));
    out.put(static_cast<char>((v >> 8) & 0xff));
}

void write_u32(std::ofstream & out, uint32_t v) {
    out.put(static_cast<char>(v & 0xff));
    out.put(static_cast<char>((v >> 8) & 0xff));
    out.put(static_cast<char>((v >> 16) & 0xff));
    out.put(static_cast<char>((v >> 24) & 0xff));
}

bool write_wav(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * block_align);

    out.write("RIFF", 4);
    write_u32(out, 36 + data_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32(out, 16);
    write_u16(out, 1);
    write_u16(out, channels);
    write_u32(out, static_cast<uint32_t>(sample_rate));
    write_u32(out, byte_rate);
    write_u16(out, block_align);
    write_u16(out, bits_per_sample);
    out.write("data", 4);
    write_u32(out, data_size);

    for (float sample : samples) {
        float s = std::max(-1.0f, std::min(1.0f, sample));
        int16_t pcm = static_cast<int16_t>(std::lrint(s * 32767.0f));
        write_u16(out, static_cast<uint16_t>(pcm));
    }
    return true;
}

const std::map<uint32_t, int64_t> & vocab() {
    static const std::map<uint32_t, int64_t> table = {
        {U';', 1}, {U':', 2}, {U',', 3}, {U'.', 4}, {U'!', 5}, {U'?', 6}, {U'—', 9}, {U'…', 10},
        {U'"', 11}, {U'(', 12}, {U')', 13}, {U'“', 14}, {U'”', 15}, {U' ', 16}, {U'̃', 17},
        {U'ʣ', 18}, {U'ʥ', 19}, {U'ʦ', 20}, {U'ʨ', 21}, {U'ᵝ', 22}, {U'ꭧ', 23},
        {U'A', 24}, {U'I', 25}, {U'O', 31}, {U'Q', 33}, {U'S', 35}, {U'T', 36}, {U'W', 39}, {U'Y', 41},
        {U'ᵊ', 42}, {U'a', 43}, {U'b', 44}, {U'c', 45}, {U'd', 46}, {U'e', 47}, {U'f', 48},
        {U'h', 50}, {U'i', 51}, {U'j', 52}, {U'k', 53}, {U'l', 54}, {U'm', 55}, {U'n', 56},
        {U'o', 57}, {U'p', 58}, {U'q', 59}, {U'r', 60}, {U's', 61}, {U't', 62}, {U'u', 63},
        {U'v', 64}, {U'w', 65}, {U'x', 66}, {U'y', 67}, {U'z', 68}, {U'ɑ', 69}, {U'ɐ', 70},
        {U'ɒ', 71}, {U'æ', 72}, {U'β', 75}, {U'ɔ', 76}, {U'ɕ', 77}, {U'ç', 78}, {U'ɖ', 80},
        {U'ð', 81}, {U'ʤ', 82}, {U'ə', 83}, {U'ɚ', 85}, {U'ɛ', 86}, {U'ɜ', 87}, {U'ɟ', 90},
        {U'ɡ', 92}, {U'ɥ', 99}, {U'ɨ', 101}, {U'ɪ', 102}, {U'ʝ', 103}, {U'ɯ', 110}, {U'ɰ', 111},
        {U'ŋ', 112}, {U'ɳ', 113}, {U'ɲ', 114}, {U'ɴ', 115}, {U'ø', 116}, {U'ɸ', 118}, {U'θ', 119},
        {U'œ', 120}, {U'ɹ', 123}, {U'ɾ', 125}, {U'ɻ', 126}, {U'ʁ', 128}, {U'ɽ', 129}, {U'ʂ', 130},
        {U'ʃ', 131}, {U'ʈ', 132}, {U'ʧ', 133}, {U'ʊ', 135}, {U'ʋ', 136}, {U'ʌ', 138}, {U'ɣ', 139},
        {U'ɤ', 140}, {U'χ', 142}, {U'ʎ', 143}, {U'ʒ', 147}, {U'ʔ', 148}, {U'ˈ', 156}, {U'ˌ', 157},
        {U'ː', 158}, {U'ʰ', 162}, {U'ʲ', 164}, {U'↓', 169}, {U'→', 171}, {U'↗', 172}, {U'↘', 173},
        {U'ᵻ', 177},
    };
    return table;
}

bool next_utf8(const std::string & s, size_t & i, uint32_t & cp) {
    if (i >= s.size()) {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(s[i++]);
    if (c < 0x80) {
        cp = c;
        return true;
    }
    int extra = 0;
    cp = 0;
    if ((c & 0xe0) == 0xc0) {
        extra = 1;
        cp = c & 0x1f;
    } else if ((c & 0xf0) == 0xe0) {
        extra = 2;
        cp = c & 0x0f;
    } else if ((c & 0xf8) == 0xf0) {
        extra = 3;
        cp = c & 0x07;
    } else {
        return false;
    }
    for (int j = 0; j < extra; ++j) {
        if (i >= s.size()) {
            return false;
        }
        unsigned char cc = static_cast<unsigned char>(s[i++]);
        if ((cc & 0xc0) != 0x80) {
            return false;
        }
        cp = (cp << 6) | (cc & 0x3f);
    }
    return true;
}

std::vector<int64_t> tokenize_phonemes(const std::string & phonemes) {
    std::vector<int64_t> tokens;
    size_t i = 0;
    uint32_t cp = 0;
    while (next_utf8(phonemes, i, cp)) {
        auto it = vocab().find(cp);
        if (it != vocab().end()) {
            tokens.push_back(it->second);
        }
    }
    return tokens;
}

std::vector<std::string> split_misaki_by_token_limit(const std::string & phonemes, size_t max_tokens) {
    std::vector<std::string> chunks;
    std::string current;
    std::string last_good;

    size_t i = 0;
    while (i < phonemes.size()) {
        size_t char_start = i;
        uint32_t cp = 0;
        if (!next_utf8(phonemes, i, cp)) {
            ++i;
            continue;
        }

        size_t current_before = current.size();
        current.append(phonemes, char_start, i - char_start);
        size_t token_count = tokenize_phonemes(current).size();
        if (token_count <= max_tokens) {
            if (cp == U' ' || cp == U',' || cp == U'.' || cp == U'!' || cp == U'?' || cp == U':' || cp == U';') {
                last_good = current;
            }
            continue;
        }

        if (!last_good.empty()) {
            chunks.push_back(last_good);
            current.erase(0, last_good.size());
            last_good.clear();
        } else {
            std::string head = current.substr(0, current_before);
            if (!head.empty()) {
                chunks.push_back(head);
            }
            current = current.substr(current_before);
        }
    }

    if (!current.empty()) {
        chunks.push_back(current);
    }
    return chunks;
}

std::vector<std::string> pack_misaki_sentences(const std::vector<std::string> & sentences, size_t max_tokens) {
    std::vector<std::string> batches;
    std::string current;

    auto flush = [&]() {
        if (!current.empty()) {
            batches.push_back(current);
            current.clear();
        }
    };

    for (const auto & sentence : sentences) {
        for (const auto & part : split_misaki_by_token_limit(sentence, max_tokens)) {
            std::string candidate = current.empty() ? part : current + " " + part;
            if (tokenize_phonemes(candidate).size() <= max_tokens) {
                current = std::move(candidate);
            } else {
                flush();
                if (tokenize_phonemes(part).size() <= max_tokens) {
                    current = part;
                } else {
                    for (const auto & forced : split_misaki_by_token_limit(part, max_tokens)) {
                        if (tokenize_phonemes(forced).size() <= max_tokens) {
                            batches.push_back(forced);
                        }
                    }
                }
            }
        }
    }
    flush();
    return batches;
}

}

struct KokoroModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "chirp-kokoro-c"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    VoiceData voice;
    std::string language = "en-US";
    std::string error;
    float speed = 1.0f;

    Impl(const KokoroParams & params) {
        if (params.model_path == nullptr || params.model_path[0] == '\0') {
            error = "model path is required";
            return;
        }
        if (params.voices_path == nullptr || params.voices_path[0] == '\0') {
            error = "voices path is required";
            return;
        }
        language = params.language ? params.language : "en-US";
        speed = params.speed > 0.0f ? params.speed : 1.0f;
        std::string voice_name = params.voice ? params.voice : "af_heart";
        if (!load_voice_from_archive(params.voices_path, voice_name, voice, error)) {
            return;
        }
        if (voice.dims != 256) {
            error = "voice style dimension must be 256";
            return;
        }
        try {
            options.SetIntraOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = std::make_unique<Ort::Session>(env, params.model_path, options);
        } catch (const Ort::Exception & e) {
            error = e.what();
        }
    }

    std::vector<float> infer(const std::vector<int64_t> & phoneme_tokens) {
        std::vector<int64_t> tokens;
        tokens.reserve(phoneme_tokens.size() + 2);
        tokens.push_back(0);
        tokens.insert(tokens.end(), phoneme_tokens.begin(), phoneme_tokens.end());
        tokens.push_back(0);
        size_t style_row = std::min(phoneme_tokens.size(), voice.rows - 1);
        float * style = voice.values.data() + style_row * voice.dims;

        std::vector<int64_t> token_shape = {1, static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> style_shape = {1, static_cast<int64_t>(voice.dims)};
        std::vector<int64_t> speed_shape = {1};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(
            memory, tokens.data(), tokens.size(), token_shape.data(), token_shape.size());
        Ort::Value style_tensor = Ort::Value::CreateTensor<float>(
            memory, style, voice.dims, style_shape.data(), style_shape.size());
        Ort::Value speed_tensor = Ort::Value::CreateTensor<float>(
            memory, &speed, 1, speed_shape.data(), speed_shape.size());

        const char * input_names[] = {"tokens", "style", "speed"};
        const char * output_names[] = {"audio"};
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(token_tensor));
        inputs.emplace_back(std::move(style_tensor));
        inputs.emplace_back(std::move(speed_tensor));

        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            inputs.data(),
            inputs.size(),
            output_names,
            1);

        float * audio = outputs[0].GetTensorMutableData<float>();
        size_t sample_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return std::vector<float>(audio, audio + sample_count);
    }
};

KokoroModel::KokoroModel(const KokoroParams & params) : impl_(std::make_unique<Impl>(params)) {}
KokoroModel::~KokoroModel() = default;

bool KokoroModel::ok() const {
    return impl_ && impl_->error.empty() && impl_->session != nullptr;
}

const std::string & KokoroModel::error() const {
    return impl_->error;
}

bool KokoroModel::synthesize_to_file(const std::string & text, const std::string & output_path) {
    if (!ok()) {
        return false;
    }

    std::vector<std::string> misaki_sentences;
    for (const auto & chunk : chunk_text(text)) {
        auto phoneme_result = phonemize(chunk, impl_->language);
        if (!phoneme_result.ok) {
            impl_->error = phoneme_result.error;
            return false;
        }
        for (const auto & phonemes : phoneme_result.sentences) {
            std::string misaki = espeak_to_misaki(phonemes, impl_->language == "en" || impl_->language == "en-GB" || impl_->language == "en-gb");
            if (std::getenv("CHIRP_KOKORO_DEBUG")) {
                std::cerr << "text: " << chunk << "\n";
                std::cerr << "espeak: " << phonemes << "\n";
                std::cerr << "misaki: " << misaki << "\n";
            }
            misaki_sentences.push_back(std::move(misaki));
        }
    }

    const size_t max_tokens = std::min(kMaxPhonemeLength, impl_->voice.rows > 0 ? impl_->voice.rows - 1 : 0);
    std::vector<float> all_audio;
    for (const auto & batch : pack_misaki_sentences(misaki_sentences, max_tokens)) {
        std::vector<int64_t> tokens = tokenize_phonemes(batch);
        if (std::getenv("CHIRP_KOKORO_DEBUG")) {
            std::cerr << "batch: " << batch << "\n";
            std::cerr << "batch tokens: " << tokens.size() << " / " << max_tokens << "\n";
        }
        if (tokens.empty()) {
            continue;
        }
        try {
            std::vector<float> audio = impl_->infer(tokens);
            all_audio.insert(all_audio.end(), audio.begin(), audio.end());
        } catch (const Ort::Exception & e) {
            impl_->error = e.what();
            return false;
        }
    }

    if (all_audio.empty()) {
        impl_->error = "no audio generated";
        return false;
    }
    if (!write_wav(output_path, all_audio, kSampleRate)) {
        impl_->error = "failed to write WAV";
        return false;
    }
    return true;
}

}
