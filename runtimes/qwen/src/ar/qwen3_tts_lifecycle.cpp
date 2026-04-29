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

TTSTransformer::TTSTransformer() = default;

TTSTransformer::~TTSTransformer() {
    unload_model();
}

void TTSTransformer::unload_model() {
    free_tts_kv_cache(state_.cache);
    free_tts_kv_cache(state_.code_pred_cache);
    free_transformer_model(model_);

    coreml_code_predictor_.unload();
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();
    skip_ggml_code_pred_layers_ = false;

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
    last_hidden_.clear();
    embd_row_fp16_scratch_.clear();
}

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    skip_ggml_code_pred_layers_ = false;
#if defined(__APPLE__)
    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (!coreml_disabled) {
        std::string coreml_path;
        const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
        if (override_env && override_env[0] != '\0') {
            coreml_path = override_env;
        } else {
            size_t slash = model_path.find_last_of("/\\");
            const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
            coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
        }

        struct stat st = {};
        if (stat(coreml_path.c_str(), &st) == 0) {
            // Skip GGML code-predictor weights when CoreML package is present.
            skip_ggml_code_pred_layers_ = true;
        } else if (use_coreml_env && use_coreml_env[0] != '\0') {
            // Explicit opt-in should remain strict to surface configuration errors.
            skip_ggml_code_pred_layers_ = true;
        }
    }
#endif

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_transformer_model(model_);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    state_.backend = init_preferred_backend("TTSTransformer", &error_msg_);
    if (!state_.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }
    
    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());

    if (!try_init_coreml_code_predictor(model_path)) {
        return false;
    }
    
    return true;
}

bool TTSTransformer::try_init_coreml_code_predictor(const std::string & model_path) {
    use_coreml_code_predictor_ = false;
    coreml_code_predictor_path_.clear();

    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (coreml_disabled) {
        return true;
    }

#if !defined(__APPLE__)
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        fprintf(stderr, "  CoreML code predictor requested but this build is not on Apple platform\n");
    }
    return true;
#else
    std::string coreml_path;
    const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
    if (override_env && override_env[0] != '\0') {
        coreml_path = override_env;
    } else {
        size_t slash = model_path.find_last_of("/\\");
        const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
        coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
    }

    if (!coreml_code_predictor_.load(coreml_path, model_.config.n_codebooks - 1)) {
        if (skip_ggml_code_pred_layers_) {
            error_msg_ = "CoreML code predictor load failed in strict mode: " + coreml_code_predictor_.get_error();
            return false;
        } else {
            fprintf(stderr, "  CoreML code predictor load failed: %s\n",
                    coreml_code_predictor_.get_error().c_str());
            fprintf(stderr, "  Falling back to GGML code predictor\n");
            return true;
        }
    }

    use_coreml_code_predictor_ = true;
    coreml_code_predictor_path_ = coreml_path;
    fprintf(stderr, "  CoreML code predictor enabled: %s\n", coreml_code_predictor_path_.c_str());
    return true;
#endif
}

} // namespace qwen3_tts
