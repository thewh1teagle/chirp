#pragma once

#include <string>
#include <vector>

namespace chirp_kokoro {

std::string normalize_text_utf8(const std::string & text);
std::vector<std::string> chunk_text(const std::string & text);

}

