#pragma once

#include <string>
#include <vector>

namespace chirp_kokoro {

struct PhonemizeResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> sentences;
};

PhonemizeResult phonemize(const std::string & text, const std::string & language = "en-us");

}
