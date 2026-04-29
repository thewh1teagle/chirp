#pragma once

#include <memory>
#include <string>

namespace chirp_kokoro {

struct KokoroParams {
    const char * model_path = nullptr;
    const char * voices_path = nullptr;
    const char * voice = "af_heart";
    const char * language = "en-us";
    float speed = 1.0f;
};

class KokoroModel {
public:
    explicit KokoroModel(const KokoroParams & params);
    ~KokoroModel();

    KokoroModel(const KokoroModel &) = delete;
    KokoroModel & operator=(const KokoroModel &) = delete;

    bool ok() const;
    const std::string & error() const;
    bool synthesize_to_file(const std::string & text, const std::string & output_path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
