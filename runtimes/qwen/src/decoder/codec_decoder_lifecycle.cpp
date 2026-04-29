#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

AudioTokenizerDecoder::AudioTokenizerDecoder() = default;

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    unload_model();
}

void AudioTokenizerDecoder::unload_model() {
    free_audio_decoder_model(model_);
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }

    state_.compute_meta.clear();
    codes_buf_.clear();
}

void AudioTokenizerDecoder::normalize_codebooks() {
    const float epsilon = 1e-5f;
    
    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char *) {
        if (!codebook || !usage || !codebook->data || !usage->data) return;
        
        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];
        
        ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data;
        float * usage_data = (float *)usage->data;
        
        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) u = epsilon;
            float inv_u = 1.0f / u;
            
            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }
        
    };
    
    normalize_codebook(model_.vq_first_codebook, model_.vq_first_usage, "first");
    
    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model_.vq_rest_codebook[i], model_.vq_rest_usage[i], name);
    }
}

} // namespace qwen3_tts
