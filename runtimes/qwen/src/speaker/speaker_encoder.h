#pragma once

#include "codec_tensor_reader.h"

#include <string>
#include <vector>

class speaker_encoder {
public:
    bool load(const std::string & model_path);
    bool extract(const std::string & wav_path, std::vector<float> & embedding);
    const std::string & error() const { return error_; }

private:
    struct conv {
        qwen3_tts::cpu_tensor w;
        qwen3_tts::cpu_tensor b;
        int dilation = 1;
    };
    struct block {
        conv tdnn1;
        conv res2[7];
        conv tdnn2;
        conv se1;
        conv se2;
    };

    bool load_tensor(qwen3_tts::GGUFCPUReader & reader, const std::string & name, qwen3_tts::cpu_tensor & out);
    bool load_conv(qwen3_tts::GGUFCPUReader & reader, const std::string & base, conv & c, int dilation = 1);
    std::vector<float> conv1d_same_reflect(const std::vector<float> & x, int channels, int len,
                                           const conv & c, int & out_len) const;
    std::vector<float> relu(std::vector<float> x) const;
    std::vector<float> se_res2_block(const std::vector<float> & x, int len, const block & b) const;
    std::vector<float> attentive_pool(const std::vector<float> & x, int len) const;
    bool mel_spectrogram(const std::string & wav_path, std::vector<float> & mel, int & frames);

    conv conv0_;
    block blocks_[3];
    conv mfa_;
    conv asp_tdnn_;
    conv asp_conv_;
    conv fc_;
    std::string error_;
};
