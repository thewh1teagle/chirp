#include "chirp_kokoro.h"

#include "internal/kokoro_model.h"

#include <memory>
#include <string>

struct chirp_kokoro_context {
    std::unique_ptr<chirp_kokoro::KokoroModel> model;
    std::string error;
};

chirp_kokoro_params chirp_kokoro_default_params(void) {
    chirp_kokoro_params params = {};
    params.voice = "af_heart";
    params.language = "en-us";
    params.speed = 1.0f;
    return params;
}

chirp_kokoro_context * chirp_kokoro_init(const chirp_kokoro_params * params) {
    auto * ctx = new chirp_kokoro_context();
    if (params == nullptr) {
        ctx->error = "params are required";
        return ctx;
    }
    ctx->model = std::make_unique<chirp_kokoro::KokoroModel>(chirp_kokoro::KokoroParams{
        params->model_path,
        params->voices_path,
        params->voice ? params->voice : "af_heart",
        params->language ? params->language : "en-us",
        params->speed,
    });
    if (!ctx->model->ok()) {
        ctx->error = ctx->model->error();
    }
    return ctx;
}

void chirp_kokoro_free(chirp_kokoro_context * ctx) {
    delete ctx;
}

const char * chirp_kokoro_get_error(chirp_kokoro_context * ctx) {
    if (ctx == nullptr) {
        return "chirp-kokoro context is null";
    }
    return ctx->error.c_str();
}

int32_t chirp_kokoro_synthesize_to_file(chirp_kokoro_context * ctx, const char * text, const char * output_path) {
    if (ctx == nullptr || ctx->model == nullptr) {
        return 0;
    }
    if (text == nullptr || text[0] == '\0') {
        ctx->error = "text is required";
        return 0;
    }
    if (output_path == nullptr || output_path[0] == '\0') {
        ctx->error = "output path is required";
        return 0;
    }
    if (!ctx->model->synthesize_to_file(text, output_path)) {
        ctx->error = ctx->model->error();
        return 0;
    }
    ctx->error.clear();
    return 1;
}
