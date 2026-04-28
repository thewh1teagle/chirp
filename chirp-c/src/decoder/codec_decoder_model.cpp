#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    model_.config.n_codebooks = loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16);
    model_.config.codebook_size = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);
    
    int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }
    
    if (dec_tensor_count == 0) {
        error_msg_ = "No decoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }
    
    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
            continue;
        }
        

        
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        
        model_.tensors[name] = tensor;
        
        std::string sname(name);
        
        if (sname == "tok_dec.vq_first.input_proj.weight") model_.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model_.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model_.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model_.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model_.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model_.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model_.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model_.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model_.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model_.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model_.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model_.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model_.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model_.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model_.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model_.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model_.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model_.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        }
        else {
            int blk_idx, res_idx, cb_idx, n = 0;
            char suffix[64];
            size_t name_len = strlen(name);
            

            
            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)
            
            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_codebook[cb_idx] = tensor;
                }
            }
            else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_usage[cb_idx] = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].conv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].dwconv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].norm_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv1_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv2_b = tensor;
                }
            }
            else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model_.upsample[blk_idx].gamma = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_q.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_k.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_v.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_output.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_gate.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_up.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_down.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_alpha = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_beta = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_w = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_b = tensor;
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_b = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_b = tensor;
                }
            }
            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }
    
    ggml_backend_t load_backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg_);
    if (!load_backend) {
        return false;
    }

    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, load_backend);
    if (!model_.buffer) {
        error_msg_ = "Failed to allocate decoder tensor buffer";
        release_preferred_backend(load_backend);
        return false;
    }
    ggml_backend_buffer_set_usage(model_.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) {
        error_msg_ = "Failed to open file for reading: " + model_path;
        release_preferred_backend(load_backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(gguf_ctx);
    std::map<std::string, std::vector<uint8_t>> tensor_data;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t offset = gguf_get_tensor_offset(gguf_ctx, i);

        auto it = model_.tensors.find(name);
        if (it == model_.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        auto & data = tensor_data[name];
        data.resize(nbytes);

        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
            error_msg_ = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(load_backend);
            return false;
        }

        if (fread(data.data(), 1, nbytes, f) != nbytes) {
            error_msg_ = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(load_backend);
            return false;
        }
    }

    fclose(f);
    
    for (int i = 0; i < 4; ++i) {
        model_.dec_blocks[i].res[0].dilation = 1;
        model_.dec_blocks[i].res[1].dilation = 3;
        model_.dec_blocks[i].res[2].dilation = 9;
    }
    
    auto normalize_codebook_data = [&tensor_data](struct ggml_tensor * codebook, struct ggml_tensor * usage) {
        if (!codebook || !usage) {
            return;
        }
        auto cb_it = tensor_data.find(ggml_get_name(codebook));
        auto usage_it = tensor_data.find(ggml_get_name(usage));
        if (cb_it == tensor_data.end() || usage_it == tensor_data.end()) {
            return;
        }

        const float epsilon = 1e-5f;
        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];

        ggml_fp16_t * cb_data = reinterpret_cast<ggml_fp16_t *>(cb_it->second.data());
        const float * usage_data = reinterpret_cast<const float *>(usage_it->second.data());

        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) {
                u = epsilon;
            }
            float inv_u = 1.0f / u;

            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }
    };

    normalize_codebook_data(model_.vq_first_codebook, model_.vq_first_usage);
    for (int i = 0; i < 15; ++i) {
        normalize_codebook_data(model_.vq_rest_codebook[i], model_.vq_rest_usage[i]);
    }

    for (const auto & item : tensor_data) {
        auto it = model_.tensors.find(item.first);
        if (it != model_.tensors.end()) {
            ggml_backend_tensor_set(it->second, item.second.data(), 0, item.second.size());
        }
    }
    
    state_.backend = load_backend;
    if (!state_.backend) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);
    
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerDecoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_DEC_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

} // namespace qwen3_tts
