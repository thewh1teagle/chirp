#pragma once

#include "codec_tensor_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

struct audio_encoder_config {
    int sample_rate = 24000;
    int valid_codebooks = 16;
    int codebook_size = 2048;
    int codebook_dim = 256;
    int hidden = 512;
    int heads = 8;
    int head_dim = 64;
    int layers = 8;
    int ffn = 2048;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

struct enc_conv {
    cpu_tensor w;
    cpu_tensor b;
    int stride = 1;
    int dilation = 1;
    std::string pad_mode = "constant";
};

struct enc_resblock {
    enc_conv conv1;
    enc_conv conv2;
};

struct enc_layer {
    cpu_tensor attn_norm_w, attn_norm_b;
    cpu_tensor ffn_norm_w, ffn_norm_b;
    cpu_tensor q_w, k_w, v_w, o_w;
    cpu_tensor attn_scale;
    cpu_tensor up_w, down_w, ffn_scale;
};

struct enc_vq {
    cpu_tensor input_proj;
    std::vector<cpu_tensor> codebooks;
};

class AudioTokenizerEncoder {
public:
    bool load_model(const std::string & model_path);
    bool encode(const std::vector<float> & samples_24k, std::vector<int32_t> & codes, int32_t & n_frames);

    const std::string & get_error() const { return error_; }
    const audio_encoder_config & get_config() const { return cfg_; }

private:
    bool load_conv(GGUFCPUReader & reader, const std::string & base, enc_conv & conv, int stride,
                   const std::string & pad_mode = "constant");
    bool load_required(GGUFCPUReader & reader, const std::string & name, cpu_tensor & out);

    std::vector<float> conv1d(const std::vector<float> & x, int in_ch, int len, const enc_conv & c,
                              int & out_len) const;
    std::vector<float> layer_norm(const std::vector<float> & x, int len, const cpu_tensor & w,
                                  const cpu_tensor & b) const;
    std::vector<float> linear(const std::vector<float> & x, int rows, int in_dim, const cpu_tensor & w) const;
    void transformer_layer(std::vector<float> & x, int len, const enc_layer & layer) const;
    std::vector<int32_t> quantize(const std::vector<float> & emb, int len) const;

    audio_encoder_config cfg_;
    enc_conv conv0, downsample;
    enc_resblock res[4];
    enc_conv downs[4];
    enc_conv final_conv;
    enc_layer layers[8];
    enc_vq semantic, acoustic;
    std::string error_;
};

} // namespace qwen3_tts
