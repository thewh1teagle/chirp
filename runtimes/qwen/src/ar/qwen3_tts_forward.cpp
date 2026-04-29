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

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_prefill_forward_graph(n_tokens, n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t)n_tokens * model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    output.resize(n_tokens * model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));
    
    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(), 
                           (n_tokens - 1) * model_.config.hidden_size * sizeof(float),
                           model_.config.hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

        logits_out->resize(model_.config.codec_vocab_size);
        ggml_backend_tensor_get(logits, logits_out->data(),
                                (n_tokens - 1) * model_.config.codec_vocab_size * sizeof(float),
                                model_.config.codec_vocab_size * sizeof(float));
    }
    
    state_.cache.n_used = n_past + n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t n_past,
                                  std::vector<float> & output) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    std::vector<float> projected;
    if (!project_text_tokens(text_tokens, n_tokens, projected)) {
        return false;
    }

    if (speaker_embd) {
        const int32_t hidden_size = model_.config.hidden_size;
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * row = projected.data() + (size_t)t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) {
                row[h] += speaker_embd[h];
            }
        }
    }

    return forward_prefill(projected.data(), n_tokens, n_past, output, nullptr);
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!step_embd) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_step_graph(n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
    if (inp_step) {
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (hidden) {
        last_hidden_.resize(model_.config.hidden_size);
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0, 
                               model_.config.hidden_size * sizeof(float));
        if (hidden_out) {
            *hidden_out = last_hidden_;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize(model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + 1;
    
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                   std::vector<float> & output) {
    std::vector<float> codec_row;
    if (!lookup_embedding_rows(model_.codec_embd, &codec_token, 1,
                               "inp_legacy_codec_token", "legacy_codec_row",
                               codec_row)) {
        return false;
    }

    return forward_step(codec_row.data(), n_past, output, nullptr);
}

bool TTSTransformer::get_hidden_states(std::vector<float> & hidden) const {
    if (last_hidden_.empty()) {
        return false;
    }
    hidden = last_hidden_;
    return true;
}

bool TTSTransformer::predict_codes(const float * hidden, const int32_t * prev_codes,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    int n_prev = (prev_codes != nullptr) ? cfg.n_codebooks - 1 : 0;
    
    struct ggml_cgraph * gf = build_code_pred_graph(n_prev);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate code predictor graph";
        return false;
    }
    
    struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
    if (inp_hidden) {
        ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
    }
    
    if (n_prev > 0) {
        struct ggml_tensor * inp_prev = ggml_graph_get_tensor(gf, "inp_prev_codes");
        if (inp_prev) {
            ggml_backend_tensor_set(inp_prev, prev_codes, 0, n_prev * sizeof(int32_t));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute code predictor graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize((cfg.n_codebooks - 1) * cfg.code_pred_vocab_size);
    
    for (int cb = 0; cb < cfg.n_codebooks - 1; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "logits_cb%d", cb + 1);
        struct ggml_tensor * cb_logits = ggml_graph_get_tensor(gf, name);
        if (cb_logits) {
            ggml_backend_tensor_get(cb_logits, output.data() + cb * cfg.code_pred_vocab_size,
                                   0, cfg.code_pred_vocab_size * sizeof(float));
        }
    }
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

} // namespace qwen3_tts
