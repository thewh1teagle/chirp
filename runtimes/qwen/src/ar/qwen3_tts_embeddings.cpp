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

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_tts_kv_cache(state_.cache);
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_key_value_heads;
    state_.cache.n_layers = cfg.n_layers;
    
    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    if (!state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }
    
    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    if (!state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }
    
    return true;
}

void TTSTransformer::clear_kv_cache() {
    state_.cache.n_used = 0;
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_tts_kv_cache(state_.code_pred_cache);
    
    state_.code_pred_cache.n_ctx = n_ctx;
    state_.code_pred_cache.n_used = 0;
    state_.code_pred_cache.head_dim = cfg.head_dim;
    state_.code_pred_cache.n_kv_heads = cfg.n_key_value_heads;
    state_.code_pred_cache.n_layers = cfg.code_pred_layers;
    
    const size_t n_tensors = cfg.code_pred_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.code_pred_cache.ctx = ggml_init(params);
    if (!state_.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }
    
    state_.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    state_.code_pred_cache.v_cache.resize(cfg.code_pred_layers);
    
    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        state_.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);
        
        state_.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }
    
    state_.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(state_.code_pred_cache.ctx, state_.backend);
    if (!state_.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }
    
    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    state_.code_pred_cache.n_used = 0;
}

bool TTSTransformer::lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                                           int32_t n_tokens, const char * input_name,
                                           const char * output_name, std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    const int32_t embd_dim = (int32_t) embedding->ne[0];
    if (n_tokens <= 32 &&
        (embedding->type == GGML_TYPE_F16 || embedding->type == GGML_TYPE_F32)) {
        output.resize((size_t) embd_dim * n_tokens);
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (!lookup_single_embedding_row(embedding, token_ids[t],
                                             output.data() + (size_t) t * embd_dim)) {
                return false;
            }
        }
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, input_name);
    ggml_set_input(inp_tokens);

    struct ggml_tensor * rows = ggml_get_rows(ctx0, embedding, inp_tokens);
    rows = ggml_cast(ctx0, rows, GGML_TYPE_F32);
    ggml_set_name(rows, output_name);
    ggml_set_output(rows);

    ggml_build_forward_expand(gf, rows);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate embedding lookup graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, input_name);
    ggml_backend_tensor_set(inp, token_ids, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute embedding lookup graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, output_name);
    if (!out) {
        error_msg_ = "Failed to find embedding lookup output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)embedding->ne[0] * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

bool TTSTransformer::lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                                 float * out_row) {
    if (!embedding) {
        error_msg_ = "Embedding tensor not found";
        return false;
    }
    if (!out_row) {
        error_msg_ = "Embedding output row is null";
        return false;
    }

    const int64_t embd_dim = embedding->ne[0];
    const int64_t vocab_size = embedding->ne[1];
    if (token_id < 0 || token_id >= vocab_size) {
        error_msg_ = "Embedding token ID out of range";
        return false;
    }

    const size_t row_offset = (size_t) token_id * embedding->nb[1];
    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embedding, out_row, row_offset, (size_t) embd_dim * sizeof(float));
        return true;
    }
    if (embedding->type == GGML_TYPE_F16) {
        embd_row_fp16_scratch_.resize((size_t) embd_dim);
        ggml_backend_tensor_get(embedding, embd_row_fp16_scratch_.data(),
                                row_offset, (size_t) embd_dim * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < embd_dim; ++i) {
            out_row[i] = ggml_fp16_to_fp32(embd_row_fp16_scratch_[i]);
        }
        return true;
    }

    std::vector<int32_t> single_token = { token_id };
    std::vector<float> single_out;
    if (!lookup_embedding_rows(embedding, single_token.data(), 1,
                               "inp_compat_embed", "out_compat_embed", single_out)) {
        return false;
    }
    memcpy(out_row, single_out.data(), (size_t) embd_dim * sizeof(float));
    return true;
}

bool TTSTransformer::project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                                         std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_text_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * cur = ggml_get_rows(ctx0, model_.text_embd, inp_tokens);
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc1, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc1_bias);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model_.text_proj_fc2, cur);
    cur = ggml_add(ctx0, cur, model_.text_proj_fc2_bias);

    ggml_set_name(cur, "text_proj_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate text projection graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp_text_tokens");
    ggml_backend_tensor_set(inp, text_tokens, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute text projection graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "text_proj_out");
    if (!out) {
        error_msg_ = "Failed to find text projection output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t)model_.config.hidden_size * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

} // namespace qwen3_tts
