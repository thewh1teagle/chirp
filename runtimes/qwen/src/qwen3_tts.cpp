#include "qwen3_tts.h"

#include "audio_tokenizer_decoder.h"
#include "codec_wav.h"
#include "tts_transformer.h"
#include "qwen_tokenizer.h"
#include "speaker_encoder.h"

#include "ggml-backend.h"

#include <string>
#include <vector>

struct qwen3_tts_context {
    qwen3_tts_params params = {};
    std::string model_path;
    std::string codec_path;
    std::string error;
    qwen_tokenizer tokenizer;
    qwen3_tts::TTSTransformer ar;
    qwen3_tts::AudioTokenizerDecoder decoder;
    speaker_encoder speaker;
};

qwen3_tts_params qwen3_tts_default_params(void) {
    qwen3_tts_params params = {};
    params.max_tokens = 8192;
    params.temperature = 0.9f;
    params.top_k = 50;
    return params;
}

qwen3_tts_context * qwen3_tts_init(const qwen3_tts_params * params) {
    if (!params || !params->model_path || !params->codec_path) {
        return nullptr;
    }

    ggml_backend_load_all();

    qwen3_tts_context * ctx = new qwen3_tts_context;
    ctx->params = *params;
    ctx->model_path = params->model_path;
    ctx->codec_path = params->codec_path;
    ctx->params.max_tokens = params->max_tokens > 0 ? params->max_tokens : 8192;
    ctx->params.temperature = params->temperature;
    ctx->params.top_k = params->top_k > 0 ? params->top_k : 50;

    if (!ctx->ar.load_model(ctx->model_path)) {
        ctx->error = ctx->ar.get_error();
        return ctx;
    }
    if (!ctx->tokenizer.load_from_gguf(ctx->model_path)) {
        ctx->error = ctx->tokenizer.error();
        return ctx;
    }
    if (!ctx->decoder.load_model(ctx->codec_path)) {
        ctx->error = ctx->decoder.get_error();
        return ctx;
    }
    if (!ctx->speaker.load(ctx->model_path)) {
        ctx->error = ctx->speaker.error();
        return ctx;
    }
    return ctx;
}

void qwen3_tts_free(qwen3_tts_context * ctx) {
    delete ctx;
}

const char * qwen3_tts_get_error(const qwen3_tts_context * ctx) {
    if (!ctx) {
        return "qwen3_tts context is null";
    }
    return ctx->error.c_str();
}

int32_t qwen3_tts_get_language_count(const qwen3_tts_context * ctx) {
    if (!ctx) return 0;
    return (int32_t) ctx->ar.get_config().languages.size();
}

const char * qwen3_tts_get_language_name(const qwen3_tts_context * ctx, int32_t index) {
    if (!ctx || index < 0 || index >= (int32_t) ctx->ar.get_config().languages.size()) {
        return nullptr;
    }
    return ctx->ar.get_config().languages[(size_t) index].name.c_str();
}

int32_t qwen3_tts_get_language_id(const qwen3_tts_context * ctx, int32_t index) {
    if (!ctx || index < 0 || index >= (int32_t) ctx->ar.get_config().languages.size()) {
        return -1;
    }
    return ctx->ar.get_config().languages[(size_t) index].id;
}

int qwen3_tts_synthesize_to_file(qwen3_tts_context * ctx, const char * text,
                                    const char * ref_wav_path, const char * output_wav_path,
                                    int32_t language_id) {
    if (!ctx) return 0;
    ctx->error.clear();
    if (!text || !*text || !output_wav_path || !*output_wav_path) {
        ctx->error = "text and output_wav_path are required";
        return 0;
    }

    const std::string ref = ref_wav_path ? ref_wav_path : "";
    std::vector<int32_t> tokens;
    std::vector<float> speaker;
    if (!ctx->tokenizer.tokenize_tts_text(text, tokens)) {
        ctx->error = ctx->tokenizer.error();
        return 0;
    }

    if (!ref.empty()) {
        if (!ctx->speaker.extract(ref, speaker)) {
            ctx->error = ctx->speaker.error();
            return 0;
        }
    } else {
        speaker.assign((size_t) ctx->ar.get_config().hidden_size, 0.0f);
    }

    if ((int32_t) speaker.size() != ctx->ar.get_config().hidden_size) {
        ctx->error = "speaker embedding size does not match model hidden size";
        return 0;
    }
    if (language_id >= 0) {
        bool supported = false;
        for (const auto & language : ctx->ar.get_config().languages) {
            if (language.id == language_id) {
                supported = true;
                break;
            }
        }
        if (!supported) {
            ctx->error = "unsupported language_id";
            return 0;
        }
    }

    std::vector<int32_t> codes;
    if (!ctx->ar.generate(tokens.data(), (int32_t) tokens.size(), speaker.data(), ctx->params.max_tokens, codes,
                          language_id, 1.05f, ctx->params.temperature,
                          ctx->params.top_k, ctx->params.progress_cb, ctx->params.progress_user_data)) {
        ctx->error = ctx->ar.get_error();
        return 0;
    }

    std::vector<float> samples;
    const int32_t n_frames = (int32_t) (codes.size() / (size_t) ctx->ar.get_config().n_codebooks);
    if (!ctx->decoder.decode(codes.data(), n_frames, samples)) {
        ctx->error = ctx->decoder.get_error();
        return 0;
    }
    if (!qwen3_tts::write_wav_mono16(output_wav_path, samples, ctx->decoder.get_config().sample_rate)) {
        ctx->error = std::string("failed to write WAV: ") + output_wav_path;
        return 0;
    }
    return 1;
}
