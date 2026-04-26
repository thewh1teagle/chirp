#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3_tts_context qwen3_tts_context;

typedef int (*qwen3_tts_generate_progress_callback)(
    void * user_data,
    int32_t current_frame,
    int32_t max_frames
);

typedef struct qwen3_tts_params {
    const char * model_path;
    const char * codec_path;
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    qwen3_tts_generate_progress_callback progress_cb;
    void * progress_user_data;
} qwen3_tts_params;

qwen3_tts_params qwen3_tts_default_params(void);
qwen3_tts_context * qwen3_tts_init(const qwen3_tts_params * params);
void qwen3_tts_free(qwen3_tts_context * ctx);
const char * qwen3_tts_get_error(const qwen3_tts_context * ctx);

int qwen3_tts_synthesize_to_file(
    qwen3_tts_context * ctx,
    const char * text,
    const char * ref_wav_path,
    const char * output_wav_path
);

#ifdef __cplusplus
}
#endif
