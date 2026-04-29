#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --model qwen3-tts.gguf --codec qwen3-tts-codec.gguf --text TEXT --output out.wav [--ref ref.wav]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model <path>        Qwen3-TTS AR transformer GGUF\n");
    fprintf(stderr, "  --codec <path>        Qwen3-TTS/Mimi codec GGUF\n");
    fprintf(stderr, "  --text <text>         Text to synthesize\n");
    fprintf(stderr, "  --ref <path>          Optional voice reference WAV\n");
    fprintf(stderr, "  --output <path>       Output WAV\n");
    fprintf(stderr, "  --max-tokens <n>      Maximum generated frames (default: 8192)\n");
    fprintf(stderr, "  --temperature <v>     Sampling temperature; 0 = greedy (default: 0.9)\n");
    fprintf(stderr, "  --top-k <n>           Top-k sampling (default: 50)\n");
}

struct progress_state {
    int32_t current_frame = 0;
    int32_t max_frames = 0;
};

static int print_progress(void * user_data, int32_t current_frame, int32_t max_frames) {
    progress_state * state = static_cast<progress_state *>(user_data);
    if (state) {
        state->current_frame = current_frame;
        state->max_frames = max_frames;
    }
    fprintf(stderr, "\rgenerating %d/%d frames", current_frame, max_frames);
    if (current_frame >= max_frames) {
        fprintf(stderr, "\n");
    }
    return 1;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string codec_path;
    std::string text;
    std::string ref_path;
    std::string output_path;
    qwen3_tts_params params = qwen3_tts_default_params();
    progress_state progress = {};
    params.progress_cb = print_progress;
    params.progress_user_data = &progress;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (++i >= argc) {
                fprintf(stderr, "error: missing value for %s\n", name);
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" || arg == "-m") {
            model_path = next(arg.c_str());
        } else if (arg == "--codec") {
            codec_path = next(arg.c_str());
        } else if (arg == "--text") {
            text = next(arg.c_str());
        } else if (arg == "--ref") {
            ref_path = next(arg.c_str());
        } else if (arg == "--output" || arg == "-o") {
            output_path = next(arg.c_str());
        } else if (arg == "--max-tokens") {
            params.max_tokens = std::stoi(next(arg.c_str()));
        } else if (arg == "--temperature") {
            params.temperature = std::stof(next(arg.c_str()));
        } else if (arg == "--top-k") {
            params.top_k = std::stoi(next(arg.c_str()));
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || codec_path.empty() || text.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    params.model_path = model_path.c_str();
    params.codec_path = codec_path.c_str();

    fprintf(stderr, "loading model and codec\n");
    qwen3_tts_context * tts = qwen3_tts_init(&params);
    if (!tts || qwen3_tts_get_error(tts)[0] != '\0') {
        fprintf(stderr, "error: %s\n", qwen3_tts_get_error(tts));
        qwen3_tts_free(tts);
        return 1;
    }

    fprintf(stderr, "preparing inputs\n");
    const char * ref = ref_path.empty() ? nullptr : ref_path.c_str();
    if (!qwen3_tts_synthesize_to_file(tts, text.c_str(), ref, output_path.c_str(), -1)) {
        fprintf(stderr, "\nerror: %s\n", qwen3_tts_get_error(tts));
        qwen3_tts_free(tts);
        return 1;
    }

    qwen3_tts_free(tts);
    fprintf(stderr, "\n");
    if (progress.max_frames > 0 && progress.current_frame >= progress.max_frames) {
        fprintf(stderr, "warning: generation reached --max-tokens; audio may be truncated\n");
    }
    fprintf(stderr, "wrote %s\n", output_path.c_str());
    return 0;
}
