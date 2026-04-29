#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chirp_kokoro_context chirp_kokoro_context;

typedef struct chirp_kokoro_params {
    const char * model_path;
    const char * voices_path;
    const char * voice;
    const char * language;
    float speed;
} chirp_kokoro_params;

chirp_kokoro_params chirp_kokoro_default_params(void);
chirp_kokoro_context * chirp_kokoro_init(const chirp_kokoro_params * params);
void chirp_kokoro_free(chirp_kokoro_context * ctx);
const char * chirp_kokoro_get_error(chirp_kokoro_context * ctx);
int32_t chirp_kokoro_synthesize_to_file(chirp_kokoro_context * ctx, const char * text, const char * output_path);

#ifdef __cplusplus
}
#endif
