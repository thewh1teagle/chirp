#include "chirp_kokoro.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static void usage(const char * prog) {
    std::fprintf(stderr, "Usage: %s --model kokoro-v1.0.onnx --voices voices-v1.0.bin --voice af_heart --text TEXT --output out.wav\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string voices_path;
    std::string voice = "af_heart";
    std::string text = "Hello world! Chirp Kokoro is now using text chunking, espeak phonemes, and ONNX Runtime.";
    std::string output_path = "kokoro-basic.wav";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--model") {
            model_path = next("--model");
        } else if (arg == "--voices") {
            voices_path = next("--voices");
        } else if (arg == "--voice") {
            voice = next("--voice");
        } else if (arg == "--text") {
            text = next("--text");
        } else if (arg == "--output" || arg == "-o") {
            output_path = next("--output");
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || voices_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    chirp_kokoro_params params = chirp_kokoro_default_params();
    params.model_path = model_path.c_str();
    params.voices_path = voices_path.c_str();
    params.voice = voice.c_str();

    chirp_kokoro_context * ctx = chirp_kokoro_init(&params);
    if (!ctx || chirp_kokoro_get_error(ctx)[0] != '\0') {
        std::fprintf(stderr, "error: %s\n", chirp_kokoro_get_error(ctx));
        chirp_kokoro_free(ctx);
        return 1;
    }

    if (!chirp_kokoro_synthesize_to_file(ctx, text.c_str(), output_path.c_str())) {
        std::fprintf(stderr, "error: %s\n", chirp_kokoro_get_error(ctx));
        chirp_kokoro_free(ctx);
        return 1;
    }

    chirp_kokoro_free(ctx);
    std::fprintf(stderr, "wrote %s\n", output_path.c_str());
    return 0;
}
