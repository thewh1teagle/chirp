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

static int32_t argmax(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bool TTSTransformer::predict_codes_autoregressive_coreml(const float * hidden,
                                                         int32_t codebook_0_token,
                                                         std::vector<int32_t> & output,
                                                         float temperature,
                                                         int32_t top_k) {
    if (!use_coreml_code_predictor_ || !coreml_code_predictor_.is_loaded()) {
        error_msg_ = "CoreML code predictor is not loaded";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t n_steps = cfg.n_codebooks - 1;

    output.resize(n_steps);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t)16 * cfg.hidden_size, 0.0f);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }

        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }

        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }

        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }

        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };

    memcpy(seq_embd.data(), hidden, (size_t)cfg.hidden_size * sizeof(float));
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token,
                                     seq_embd.data() + cfg.hidden_size)) {
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    for (int32_t step = 0; step < n_steps; ++step) {
        if (step > 0) {
            float * dst = seq_embd.data() + (size_t)(step + 1) * cfg.hidden_size;
            if (!lookup_single_embedding_row(model_.code_pred_embd[step - 1], output[step - 1], dst)) {
                return false;
            }
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!coreml_code_predictor_.predict_step(step, seq_embd.data(), step + 2, cfg.hidden_size, logits_data)) {
            error_msg_ = "CoreML predictor step failed: " + coreml_code_predictor_.get_error();
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_compute_ms += dt_ms;
        if (timing_) timing_->t_code_pred_coreml_ms += dt_ms;
#endif

        if ((int32_t)logits_data.size() != cfg.code_pred_vocab_size) {
            error_msg_ = "CoreML predictor returned unexpected logits size";
            return false;
        }
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

#ifdef QWEN3_TTS_TIMING
        if (timing_) {
            if (step == 0) {
                timing_->t_code_pred_prefill_ms += dt_ms;
            } else {
                timing_->t_code_pred_steps_ms += dt_ms;
            }
        }
#endif
    }

    return true;
}

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                   std::vector<int32_t> & output,
                                                   float temperature, int32_t top_k) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    if (use_coreml_code_predictor_ && coreml_code_predictor_.is_loaded()) {
        if (predict_codes_autoregressive_coreml(hidden, codebook_0_token, output, temperature, top_k)) {
            return true;
        }
        if (skip_ggml_code_pred_layers_) {
            return false;
        }
        fprintf(stderr, "  CoreML code predictor failed, falling back to GGML: %s\n", error_msg_.c_str());
        use_coreml_code_predictor_ = false;
    }
    
    if (state_.code_pred_cache.n_ctx < 16) {
        if (!init_code_pred_kv_cache(16)) {
            return false;
        }
    }
    clear_code_pred_kv_cache();
    
    output.resize(15);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    
    // Helper lambda: temperature + top-k sampling (or greedy if temperature <= 0)
    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }
        // Temperature scaling
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }
        // Top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }
        // Softmax
        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }
        // Sample
        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };
    
    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    // Prefill with 2 tokens [past_hidden, cb0_embd]
    {
#ifdef QWEN3_TTS_TIMING
        auto t_pf_start = clk::now();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_prefill_graph();
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor prefill graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_cb0_embd = ggml_graph_get_tensor(gf, "inp_cb0_embd");
        if (inp_cb0_embd) {
            ggml_backend_tensor_set(inp_cb0_embd, cb0_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));
        
        output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }
    
    // Generate 14 more tokens autoregressively
#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    for (int step = 1; step < 15; ++step) {
        int32_t n_past = step + 1;

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_step_graph(n_past, step);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor step graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_code = ggml_graph_get_tensor(gf, "inp_code");
        if (inp_code) {
            int32_t prev_code = output[step - 1];
            ggml_backend_tensor_set(inp_code, &prev_code, 0, sizeof(int32_t));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t pos = n_past;
            ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));
        
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }
#ifdef QWEN3_TTS_TIMING
    if (timing_) timing_->t_code_pred_steps_ms += std::chrono::duration<double, std::milli>(clk::now() - t_steps_start).count();
#endif
    
    return true;
}

} // namespace qwen3_tts
