#include "audio_tokenizer_encoder.h"

#include <cstdio>

namespace qwen3_tts {

bool AudioTokenizerEncoder::load_required(GGUFCPUReader & reader, const std::string & name, cpu_tensor & out) {
    if (!reader.read_tensor(name, out)) {
        error_ = reader.error();
        return false;
    }
    return true;
}

bool AudioTokenizerEncoder::load_conv(GGUFCPUReader & reader, const std::string & base, enc_conv & conv, int stride,
                                      const std::string & pad_mode) {
    conv.stride = stride;
    conv.dilation = 1;
    conv.pad_mode = pad_mode;
    if (!load_required(reader, base + ".weight", conv.w)) {
        return false;
    }
    if (!reader.read_tensor(base + ".bias", conv.b)) {
        conv.b.data.clear();
        conv.b.ne.clear();
    }
    return true;
}

bool AudioTokenizerEncoder::load_model(const std::string & model_path) {
    GGUFCPUReader reader;
    if (!reader.open(model_path)) {
        error_ = reader.error();
        return false;
    }

    if (!load_conv(reader, "tok_enc.conv.0", conv0, 1)) return false;

    const int res_idx[4] = {1, 4, 7, 10};
    const int down_idx[4] = {3, 6, 9, 12};
    const int strides[4] = {4, 5, 6, 8};
    for (int i = 0; i < 4; ++i) {
        if (!load_conv(reader, "tok_enc.res." + std::to_string(res_idx[i]) + ".blk.1", res[i].conv1, 1)) return false;
        if (!load_conv(reader, "tok_enc.res." + std::to_string(res_idx[i]) + ".blk.3", res[i].conv2, 1)) return false;
        if (!load_conv(reader, "tok_enc.conv." + std::to_string(down_idx[i]), downs[i], strides[i])) return false;
    }

    if (!load_conv(reader, "tok_enc.conv.14", final_conv, 1)) return false;
    if (!load_required(reader, "tok_enc.downsample.weight", downsample.w)) return false;
    downsample.stride = 2;
    downsample.dilation = 1;
    downsample.pad_mode = "replicate";

    for (int i = 0; i < cfg_.layers; ++i) {
        const std::string p = "tok_enc.blk." + std::to_string(i);
        if (!load_required(reader, p + ".attn_norm.weight", layers[i].attn_norm_w)) return false;
        if (!load_required(reader, p + ".attn_norm.bias", layers[i].attn_norm_b)) return false;
        if (!load_required(reader, p + ".ffn_norm.weight", layers[i].ffn_norm_w)) return false;
        if (!load_required(reader, p + ".ffn_norm.bias", layers[i].ffn_norm_b)) return false;
        if (!load_required(reader, p + ".attn_q.weight", layers[i].q_w)) return false;
        if (!load_required(reader, p + ".attn_k.weight", layers[i].k_w)) return false;
        if (!load_required(reader, p + ".attn_v.weight", layers[i].v_w)) return false;
        if (!load_required(reader, p + ".attn_output.weight", layers[i].o_w)) return false;
        if (!load_required(reader, p + ".attn_scale", layers[i].attn_scale)) return false;
        if (!load_required(reader, p + ".ffn_up.weight", layers[i].up_w)) return false;
        if (!load_required(reader, p + ".ffn_down.weight", layers[i].down_w)) return false;
        if (!load_required(reader, p + ".ffn_scale", layers[i].ffn_scale)) return false;
    }

    if (!load_required(reader, "tok_enc.vq_semantic.input_proj.weight", semantic.input_proj)) return false;
    if (!load_required(reader, "tok_enc.vq_acoustic.input_proj.weight", acoustic.input_proj)) return false;
    semantic.codebooks.resize(1);
    acoustic.codebooks.resize(15);
    if (!load_required(reader, "tok_enc.vq_semantic.0.codebook", semantic.codebooks[0])) return false;
    for (int i = 0; i < 15; ++i) {
        if (!load_required(reader, "tok_enc.vq_acoustic." + std::to_string(i) + ".codebook", acoustic.codebooks[i])) {
            return false;
        }
    }

    return true;
}

} // namespace qwen3_tts
