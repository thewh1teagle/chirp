#include "tts_transformer.h"
#include "gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sys/stat.h>
#include <unordered_set>

namespace qwen3_tts {

bool TTSTransformer::build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                                         const float * speaker_embd, int32_t language_id,
                                         std::vector<float> & prefill_embd,
                                         std::vector<float> & trailing_text_hidden,
                                         std::vector<float> & tts_pad_embed) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for prefill";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t hidden_size = cfg.hidden_size;

    int32_t special_tokens[3] = {
        cfg.tts_bos_token_id,
        cfg.tts_eos_token_id,
        cfg.tts_pad_token_id,
    };

    std::vector<float> special_proj;
    if (!project_text_tokens(special_tokens, 3, special_proj)) {
        return false;
    }

    std::vector<float> tts_bos_embed(hidden_size);
    std::vector<float> tts_eos_embed(hidden_size);
    tts_pad_embed.resize(hidden_size);
    memcpy(tts_bos_embed.data(), special_proj.data() + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_eos_embed.data(), special_proj.data() + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_pad_embed.data(), special_proj.data() + 2 * hidden_size, hidden_size * sizeof(float));

    std::vector<float> role_embed;
    if (!project_text_tokens(text_tokens, 3, role_embed)) {
        return false;
    }

    std::vector<int32_t> codec_prefill_tokens;
    if (language_id < 0) {
        codec_prefill_tokens = {
            cfg.codec_nothink_id,
            cfg.codec_think_bos_id,
            cfg.codec_think_eos_id,
        };
    } else {
        codec_prefill_tokens = {
            cfg.codec_think_id,
            cfg.codec_think_bos_id,
            language_id,
            cfg.codec_think_eos_id,
        };
    }

    std::vector<float> codec_prefill_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_prefill_tokens.data(),
                               (int32_t)codec_prefill_tokens.size(),
                               "inp_codec_prefill_tokens", "codec_prefill_rows",
                               codec_prefill_embed)) {
        return false;
    }

    int32_t codec_tail_tokens[2] = { cfg.codec_pad_id, cfg.codec_bos_id };
    std::vector<float> codec_tail_embed;
    if (!lookup_embedding_rows(model_.codec_embd, codec_tail_tokens, 2,
                               "inp_codec_tail_tokens", "codec_tail_rows",
                               codec_tail_embed)) {
        return false;
    }

    const bool has_speaker = (speaker_embd != nullptr);
    const int32_t codec_input_len = (int32_t)codec_prefill_tokens.size() + (has_speaker ? 1 : 0) + 2;
    std::vector<float> codec_input_embedding((size_t)codec_input_len * hidden_size);

    int32_t dst_token = 0;
    memcpy(codec_input_embedding.data(), codec_prefill_embed.data(), codec_prefill_embed.size() * sizeof(float));
    dst_token += (int32_t)codec_prefill_tokens.size();

    if (has_speaker) {
        memcpy(codec_input_embedding.data() + (size_t)dst_token * hidden_size,
               speaker_embd, hidden_size * sizeof(float));
        ++dst_token;
    }

    memcpy(codec_input_embedding.data() + (size_t)dst_token * hidden_size,
           codec_tail_embed.data(), codec_tail_embed.size() * sizeof(float));

    const int32_t codec_plus_overlay_len = codec_input_len - 1;
    std::vector<float> codec_plus_overlay((size_t)codec_plus_overlay_len * hidden_size);
    for (int32_t t = 0; t < codec_plus_overlay_len; ++t) {
        const float * overlay = (t == codec_plus_overlay_len - 1)
            ? tts_bos_embed.data()
            : tts_pad_embed.data();
        const float * codec_row = codec_input_embedding.data() + (size_t)t * hidden_size;
        float * out_row = codec_plus_overlay.data() + (size_t)t * hidden_size;
        for (int32_t h = 0; h < hidden_size; ++h) {
            out_row[h] = overlay[h] + codec_row[h];
        }
    }

    std::vector<float> first_text_embed;
    if (!project_text_tokens(text_tokens + 3, 1, first_text_embed)) {
        return false;
    }

    std::vector<float> first_text_plus_codec_bos(hidden_size);
    const float * codec_bos_embed = codec_input_embedding.data() + (size_t)(codec_input_len - 1) * hidden_size;
    for (int32_t h = 0; h < hidden_size; ++h) {
        first_text_plus_codec_bos[h] = first_text_embed[h] + codec_bos_embed[h];
    }

    const int32_t prefill_len = 3 + codec_plus_overlay_len + 1;
    prefill_embd.resize((size_t)prefill_len * hidden_size);
    memcpy(prefill_embd.data(), role_embed.data(), role_embed.size() * sizeof(float));
    memcpy(prefill_embd.data() + (size_t)3 * hidden_size,
           codec_plus_overlay.data(), codec_plus_overlay.size() * sizeof(float));
    memcpy(prefill_embd.data() + (size_t)(prefill_len - 1) * hidden_size,
           first_text_plus_codec_bos.data(), hidden_size * sizeof(float));

    const int32_t trailing_token_count = std::max(0, n_tokens - 9);
    std::vector<float> trailing_text_proj;
    if (trailing_token_count > 0) {
        if (!project_text_tokens(text_tokens + 4, trailing_token_count, trailing_text_proj)) {
            return false;
        }
    }

    const int32_t trailing_len = trailing_token_count + 1;
    trailing_text_hidden.resize((size_t)trailing_len * hidden_size);
    if (trailing_token_count > 0) {
        memcpy(trailing_text_hidden.data(), trailing_text_proj.data(), trailing_text_proj.size() * sizeof(float));
    }
    memcpy(trailing_text_hidden.data() + (size_t)(trailing_len - 1) * hidden_size,
           tts_eos_embed.data(), hidden_size * sizeof(float));

    return true;
}

struct ggml_cgraph * TTSTransformer::build_prefill_forward_graph(int32_t n_tokens, int32_t n_past) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_prefill_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(inp_prefill_embd, "inp_prefill_embd");
    ggml_set_input(inp_prefill_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * cur = inp_prefill_embd;
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        int n_kv = n_past + n_tokens;
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_kv,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_kv,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, ffn_down_f32, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

} // namespace qwen3_tts
