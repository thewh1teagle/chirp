#include "codec_tensor_reader.h"

#include <cstdio>

namespace qwen3_tts {

int64_t cpu_tensor::elements() const {
    int64_t n = 1;
    for (int64_t d : ne) {
        n *= d;
    }
    return n;
}

GGUFCPUReader::~GGUFCPUReader() {
    close();
}

bool GGUFCPUReader::open(const std::string & path) {
    close();
    path_ = path;
    gguf_init_params params = { true, &meta_ctx_ };
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_ = "failed to open GGUF: " + path;
        return false;
    }
    return true;
}

void GGUFCPUReader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    path_.clear();
}

bool GGUFCPUReader::read_raw_tensor(int64_t index, const ggml_tensor * meta, std::vector<uint8_t> & bytes) {
    FILE * f = fopen(path_.c_str(), "rb");
    if (!f) {
        error_ = "failed to open GGUF for tensor read: " + path_;
        return false;
    }

    const size_t nbytes = ggml_nbytes(meta);
    const size_t offset = gguf_get_data_offset(ctx_) + gguf_get_tensor_offset(ctx_, index);
    bytes.resize(nbytes);

    if (fseek(f, (long) offset, SEEK_SET) != 0) {
        fclose(f);
        error_ = "failed to seek tensor data";
        return false;
    }
    if (fread(bytes.data(), 1, nbytes, f) != nbytes) {
        fclose(f);
        error_ = "failed to read tensor data";
        return false;
    }

    fclose(f);
    return true;
}

bool GGUFCPUReader::read_tensor(const std::string & name, cpu_tensor & out) {
    if (!ctx_ || !meta_ctx_) {
        error_ = "GGUF reader is not open";
        return false;
    }

    const int64_t index = gguf_find_tensor(ctx_, name.c_str());
    if (index < 0) {
        error_ = "missing tensor: " + name;
        return false;
    }

    ggml_tensor * meta = ggml_get_tensor(meta_ctx_, name.c_str());
    if (!meta) {
        error_ = "missing tensor metadata: " + name;
        return false;
    }

    std::vector<uint8_t> bytes;
    if (!read_raw_tensor(index, meta, bytes)) {
        error_ += " (" + name + ")";
        return false;
    }

    out.name = name;
    out.ne.clear();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out.ne.push_back(meta->ne[i]);
    }

    const int64_t n = ggml_nelements(meta);
    out.data.resize(n);
    if (meta->type == GGML_TYPE_F32) {
        const float * src = reinterpret_cast<const float *>(bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            out.data[i] = src[i];
        }
    } else if (meta->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            out.data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        error_ = "unsupported CPU tensor type for " + name;
        return false;
    }
    return true;
}

} // namespace qwen3_tts
