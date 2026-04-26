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

bool TTSTransformer::generate(const int32_t * text_tokens, int32_t n_tokens,
                               const float * speaker_embd, int32_t max_len,
                               std::vector<int32_t> & output,
                               int32_t language_id,
                               float repetition_penalty,
                               float temperature,
                               int32_t top_k,
                               tts_generate_progress_callback progress_cb,
                               void * progress_user_data) {
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    tts_timing timing = {};
    auto t_gen_start = clk::now();
    auto t0 = t_gen_start, t1 = t_gen_start;
    timing_ = &timing;
#endif

    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for generation";
        return false;
    }
    if (max_len <= 0) {
        output.clear();
        return true;
    }
    
    const auto & cfg = model_.config;

    std::vector<float> prefill_embd;
    std::vector<float> trailing_text_hidden;
    std::vector<float> tts_pad_embed;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                             prefill_embd, trailing_text_hidden, tts_pad_embed)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    const int32_t prefill_len = (int32_t)(prefill_embd.size() / cfg.hidden_size);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / cfg.hidden_size);

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx || state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) {
            return false;
        }
    }
    clear_kv_cache();
    
    std::vector<float> hidden_out;
    std::vector<float> logits;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    output.clear();
    output.reserve(max_len * cfg.n_codebooks);
    
    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = cfg.codec_vocab_size - 1024;
    
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(cfg.hidden_size, 0.0f);
    std::vector<float> embd_row(cfg.hidden_size);
    
    for (int frame = 0; frame < max_len; ++frame) {
        // Suppress tokens in [codec_vocab_size - 1024, codec_vocab_size), except codec_eos_id
        for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i) {
            if (i != cfg.codec_eos_id) {
                logits[i] = -INFINITY;
            }
        }

        // Repetition penalty (HuggingFace style) on previously generated CB0 tokens
        if (repetition_penalty != 1.0f) {
            for (int32_t tok : generated_cb0_tokens) {
                if (tok >= 0 && tok < cfg.codec_vocab_size) {
                    if (logits[tok] > 0.0f) {
                        logits[tok] /= repetition_penalty;
                    } else {
                        logits[tok] *= repetition_penalty;
                    }
                }
            }
        }

        int32_t next_token;
        if (temperature <= 0.0f) {
            next_token = argmax(logits.data(), cfg.codec_vocab_size);
        } else {
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                logits[i] /= temperature;
            }

            if (top_k > 0 && top_k < cfg.codec_vocab_size) {
                std::vector<std::pair<float, int32_t>> scored(cfg.codec_vocab_size);
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    scored[i] = {logits[i], i};
                }
                std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                    [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                        return a.first > b.first;
                    });
                float threshold = scored[top_k - 1].first;
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    if (logits[i] < threshold) {
                        logits[i] = -INFINITY;
                    }
                }
            }

            float max_logit = *std::max_element(logits.data(), logits.data() + cfg.codec_vocab_size);
            double sum = 0.0;
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = expf(logits[i] - max_logit);
                sum += probs[i];
            }
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = (float)(probs[i] / sum);
            }

            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng_);
        }
        
        if (next_token == cfg.codec_eos_id) {
            break;
        }
        
        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);
        
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0], codes_1_15, temperature, top_k)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_code_pred_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            frame_codes[cb] = codes_1_15[cb - 1];
        }
        
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            output.push_back(frame_codes[cb]);
        }

        if (progress_cb && !progress_cb(progress_user_data, frame + 1, max_len)) {
            error_msg_ = "generation cancelled";
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        timing.n_frames = frame + 1;
#endif

        if (frame + 1 >= max_len) {
            break;
        }

        std::fill(step_embd.begin(), step_embd.end(), 0.0f);

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) {
            return false;
        }
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] = embd_row[h];
        }

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            int32_t code_token = frame_codes[cb];
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1], code_token, embd_row.data())) {
                return false;
            }
            for (int32_t h = 0; h < cfg.hidden_size; ++h) {
                step_embd[h] += embd_row[h];
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_embed_lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * cfg.hidden_size
            : tts_pad_embed.data();
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] += trailing_row[h];
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!forward_step(step_embd.data(), n_past, logits)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_talker_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        n_past++;
    }
    
#ifdef QWEN3_TTS_TIMING
    timing.t_generate_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_gen_start).count();
    timing_ = nullptr;
    const auto & t = timing;
    int nf = t.n_frames;
    fprintf(stderr, "\n=== Detailed Generation Timing (%d frames) ===\n", nf);
    fprintf(stderr, "\n  Prefill:\n");
    fprintf(stderr, "    Build graph:      %8.1f ms\n", t.t_prefill_build_ms);
    fprintf(stderr, "    Forward total:    %8.1f ms\n", t.t_prefill_forward_ms);
    fprintf(stderr, "      Graph build:    %8.1f ms\n", t.t_prefill_graph_build_ms);
    fprintf(stderr, "      Graph alloc:    %8.1f ms\n", t.t_prefill_graph_alloc_ms);
    fprintf(stderr, "      Compute:        %8.1f ms\n", t.t_prefill_compute_ms);
    fprintf(stderr, "      Data I/O:       %8.1f ms\n", t.t_prefill_data_ms);
    fprintf(stderr, "\n  Talker forward_step (total / per-frame):\n");
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_talker_forward_ms, nf > 0 ? t.t_talker_forward_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_build_ms, nf > 0 ? t.t_talker_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_alloc_ms, nf > 0 ? t.t_talker_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_talker_compute_ms, nf > 0 ? t.t_talker_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_talker_data_ms, nf > 0 ? t.t_talker_data_ms / nf : 0.0);
    fprintf(stderr, "\n  Code predictor (total / per-frame):\n");
    fprintf(stderr, "    Backend:          %s\n", use_coreml_code_predictor_ ? "CoreML (CPU+NE)" : "GGML");
    if (use_coreml_code_predictor_ && !coreml_code_predictor_path_.empty()) {
        fprintf(stderr, "    CoreML model:     %s\n", coreml_code_predictor_path_.c_str());
    }
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_ms, nf > 0 ? t.t_code_pred_ms / nf : 0.0);
    fprintf(stderr, "      Init/KV/embed:  %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_init_ms, nf > 0 ? t.t_code_pred_init_ms / nf : 0.0);
    fprintf(stderr, "      Prefill (2tok): %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_prefill_ms, nf > 0 ? t.t_code_pred_prefill_ms / nf : 0.0);
    fprintf(stderr, "      Steps (14):     %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_steps_ms, nf > 0 ? t.t_code_pred_steps_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_build_ms, nf > 0 ? t.t_code_pred_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_alloc_ms, nf > 0 ? t.t_code_pred_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_compute_ms, nf > 0 ? t.t_code_pred_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_data_ms, nf > 0 ? t.t_code_pred_data_ms / nf : 0.0);
    fprintf(stderr, "      CoreML total:   %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_coreml_ms, nf > 0 ? t.t_code_pred_coreml_ms / nf : 0.0);
    fprintf(stderr, "\n  Embed lookups:      %8.1f ms   (%.1f ms/frame)\n", t.t_embed_lookup_ms, nf > 0 ? t.t_embed_lookup_ms / nf : 0.0);
    double accounted = t.t_prefill_build_ms + t.t_prefill_forward_ms + t.t_talker_forward_ms + t.t_code_pred_ms + t.t_embed_lookup_ms;
    fprintf(stderr, "  Other/overhead:     %8.1f ms\n", t.t_generate_total_ms - accounted);
    fprintf(stderr, "  ─────────────────────────────────────────\n");
    fprintf(stderr, "  Total generate:     %8.1f ms\n", t.t_generate_total_ms);
    if (nf > 0) {
        fprintf(stderr, "  Throughput:         %8.1f ms/frame (%.1f frames/s)\n",
                t.t_generate_total_ms / nf, 1000.0 * nf / t.t_generate_total_ms);
    }
#endif

    return true;
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                              std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                         const float * audio_embd, int32_t n_audio,
                                         int32_t audio_start_pos, int32_t n_past,
                                         std::vector<float> & output) {
    (void)audio_embd;
    (void)n_audio;
    (void)audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts
