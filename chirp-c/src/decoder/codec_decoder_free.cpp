#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

void free_audio_decoder_model(audio_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
