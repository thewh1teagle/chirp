#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

struct cpu_tensor {
    std::string name;
    std::vector<int64_t> ne;
    std::vector<float> data;

    int64_t dim(int i) const { return i < (int) ne.size() ? ne[i] : 1; }
    int64_t elements() const;
};

class GGUFCPUReader {
public:
    ~GGUFCPUReader();

    bool open(const std::string & path);
    void close();
    bool read_tensor(const std::string & name, cpu_tensor & out);

    const std::string & error() const { return error_; }

private:
    bool read_raw_tensor(int64_t index, const ggml_tensor * meta, std::vector<uint8_t> & bytes);

    gguf_context * ctx_ = nullptr;
    ggml_context * meta_ctx_ = nullptr;
    std::string path_;
    std::string error_;
};

} // namespace qwen3_tts
