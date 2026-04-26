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

struct ggml_cgraph * TTSTransformer::build_code_pred_graph(int32_t n_prev_codes) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const int n_layer = cfg.code_pred_layers;
    const int n_codebooks = cfg.n_codebooks;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    struct ggml_tensor * inp_prev_codes = nullptr;
    if (n_prev_codes > 0) {
        inp_prev_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prev_codes);
        ggml_set_name(inp_prev_codes, "inp_prev_codes");
        ggml_set_input(inp_prev_codes);
    }
    
    struct ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_hidden, hidden_size, 1);
    
    if (n_prev_codes > 0 && inp_prev_codes) {
        for (int cb = 0; cb < n_prev_codes && cb < n_codebooks - 1; ++cb) {
            struct ggml_tensor * code_idx = ggml_view_1d(ctx0, inp_prev_codes, 1, cb * sizeof(int32_t));
            struct ggml_tensor * code_embd = ggml_get_rows(ctx0, model_.code_pred_embd[cb], code_idx);
            cur = ggml_add(ctx0, cur, code_embd);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
        
        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, 1);
        
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_soft_max(ctx0, KQ);
        
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, 1);
        
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        struct ggml_tensor * old_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, old_ffn_down_f32, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    
    std::vector<struct ggml_tensor *> all_logits;
    for (int cb = 0; cb < n_codebooks - 1; ++cb) {
        struct ggml_tensor * cb_logits = ggml_mul_mat(ctx0, model_.code_pred_head[cb], cur);
        ggml_format_name(cb_logits, "logits_cb%d", cb + 1);
        ggml_set_output(cb_logits);
        all_logits.push_back(cb_logits);
    }
    
    for (auto * logits : all_logits) {
        ggml_build_forward_expand(gf, logits);
    }
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_prefill_graph() {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 2;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    // Input: past_hidden from talker [hidden_size]
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    // Input: codebook 0 token embedding [hidden_size] (pre-computed using talker's codec_embd)
    struct ggml_tensor * inp_cb0_embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    ggml_set_name(inp_cb0_embd, "inp_cb0_embd");
    ggml_set_input(inp_cb0_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    // Concatenate [past_hidden, cb0_embd] -> [2, hidden_size]
    struct ggml_tensor * hidden_2d = ggml_reshape_2d(ctx0, inp_hidden, hidden_size, 1);
    struct ggml_tensor * cb0_2d = ggml_reshape_2d(ctx0, inp_cb0_embd, hidden_size, 1);
    struct ggml_tensor * cur = ggml_concat(ctx0, hidden_2d, cb0_2d, 1);
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
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
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
        // Store at position 0 (prefill starts fresh)
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, 0);
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
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
     struct ggml_tensor * last_hidden = ggml_view_2d(ctx0, cur, hidden_size, 1, 
                                                      cur->nb[1], hidden_size * sizeof(float));
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[0], last_hidden);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_step_graph(int32_t n_past, int32_t generation_step) {
    const auto & cfg = model_.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 1;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    struct ggml_tensor * inp_code = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code, "inp_code");
    ggml_set_input(inp_code);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    
    struct ggml_tensor * cur;
    if (generation_step == 0) {
        cur = ggml_reshape_2d(ctx0, inp_hidden, hidden_size, 1);
    } else {
        cur = ggml_get_rows(ctx0, model_.code_pred_embd[generation_step - 1], inp_code);
        cur = ggml_reshape_2d(ctx0, cur, hidden_size, 1);
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
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
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
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
        
        struct ggml_tensor * step_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, step_ffn_down_f32, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
     cur = inpL;
     
     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[generation_step], cur);
     ggml_set_name(logits, "logits");
     ggml_set_output(logits);
     
     ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

} // namespace qwen3_tts
