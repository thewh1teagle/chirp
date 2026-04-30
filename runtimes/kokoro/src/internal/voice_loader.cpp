#include "voice_loader.h"

#include <miniz.h>
#include <npy.hpp>

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace chirp_kokoro {
namespace {

class TempFile {
public:
    TempFile() {
#ifdef _WIN32
        char temp_dir[MAX_PATH + 1] = {};
        char temp_path[MAX_PATH + 1] = {};
        if (GetTempPathA(MAX_PATH, temp_dir) > 0 && GetTempFileNameA(temp_dir, "ckv", 0, temp_path) != 0) {
            path_ = temp_path;
        }
#else
        char pattern[] = "/tmp/chirp-kokoro-voice-XXXXXX";
        int fd = mkstemp(pattern);
        if (fd >= 0) {
            close(fd);
            path_ = pattern;
        }
#endif
    }

    ~TempFile() {
        if (!path_.empty()) {
            std::remove(path_.c_str());
        }
    }

    const std::string & path() const {
        return path_;
    }

private:
    std::string path_;
};

bool extract_zip_entry(const std::string & zip_path, const std::string & entry, std::vector<unsigned char> & out) {
    mz_zip_archive zip = {};
    if (!mz_zip_reader_init_file(&zip, zip_path.c_str(), 0)) {
        return false;
    }

    int index = mz_zip_reader_locate_file(&zip, entry.c_str(), nullptr, 0);
    if (index < 0) {
        mz_zip_reader_end(&zip);
        return false;
    }

    size_t size = 0;
    void * data = mz_zip_reader_extract_to_heap(&zip, index, &size, 0);
    mz_zip_reader_end(&zip);
    if (data == nullptr) {
        return false;
    }

    out.resize(size);
    std::memcpy(out.data(), data, size);
    mz_free(data);
    return true;
}

}

bool load_voice_from_archive(
    const std::string & voices_path,
    const std::string & voice,
    VoiceData & out,
    std::string & error) {
    std::string entry = voice;
    if (entry.size() < 4 || entry.substr(entry.size() - 4) != ".npy") {
        entry += ".npy";
    }

    std::vector<unsigned char> npy_bytes;
    if (!extract_zip_entry(voices_path, entry, npy_bytes)) {
        error = "failed to extract voice `" + entry + "` from " + voices_path;
        return false;
    }

    TempFile temp;
    if (temp.path().empty()) {
        error = "failed to create temporary voice file";
        return false;
    }

    {
        std::ofstream file(temp.path(), std::ios::binary);
        if (!file) {
            error = "failed to write temporary voice file";
            return false;
        }
        file.write(reinterpret_cast<const char *>(npy_bytes.data()), static_cast<std::streamsize>(npy_bytes.size()));
    }

    try {
        npy::npy_data<float> array = npy::read_npy<float>(temp.path());
        if (array.fortran_order) {
            error = "fortran-order voice arrays are not supported: " + voice;
            return false;
        }
        if (array.shape.size() == 2) {
            out.rows = array.shape[0];
            out.dims = array.shape[1];
        } else if (array.shape.size() == 3 && array.shape[1] == 1) {
            out.rows = array.shape[0];
            out.dims = array.shape[2];
        } else {
            error = "unsupported voice shape: " + voice;
            return false;
        }
        if (out.rows == 0 || out.dims == 0) {
            error = "empty voice array: " + voice;
            return false;
        }
        out.values = std::move(array.data);
        return true;
    } catch (const std::exception & e) {
        error = "failed to parse voice `" + voice + "` from " + voices_path + ": " + e.what();
        return false;
    }
}

bool list_voices_from_archive(
    const std::string & voices_path,
    std::vector<std::string> & out,
    std::string & error) {
    mz_zip_archive zip = {};
    if (!mz_zip_reader_init_file(&zip, voices_path.c_str(), 0)) {
        error = "failed to open voices archive: " + voices_path;
        return false;
    }

    mz_uint count = mz_zip_reader_get_num_files(&zip);
    out.clear();
    out.reserve(count);
    for (mz_uint i = 0; i < count; ++i) {
        mz_zip_archive_file_stat stat = {};
        if (!mz_zip_reader_file_stat(&zip, i, &stat)) {
            mz_zip_reader_end(&zip);
            error = "failed to read voices archive entry";
            return false;
        }
        std::string name = stat.m_filename;
        if (name.size() <= 4 || name.substr(name.size() - 4) != ".npy") {
            continue;
        }
        name.resize(name.size() - 4);
        out.push_back(name);
    }
    mz_zip_reader_end(&zip);
    std::sort(out.begin(), out.end());
    return true;
}

}
