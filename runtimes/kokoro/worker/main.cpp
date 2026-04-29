#include "chirp_kokoro.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

struct Request {
    int id = 0;
    std::string method;
    std::string model_path;
    std::string voices_path;
    std::string espeak_data_path;
    std::string voice = "af_heart";
    std::string language = "en-US";
    std::string input;
    std::string output_path;
    float speed = 1.0f;
};

std::string normalize_language(std::string language) {
    if (language.empty() || language == "auto") return "en-US";
    if (language == "english" || language == "en") return "en-US";
    if (language == "british" || language == "en-GB" || language == "en-gb") return "en";
    if (language == "spanish" || language == "es") return "es";
    if (language == "french" || language == "fr") return "fr-fr";
    if (language == "hindi" || language == "hi") return "hi";
    if (language == "italian" || language == "it") return "it";
    if (language == "portuguese" || language == "pt") return "pt-br";
    return language;
}

std::string language_for_voice(const std::string & voice) {
    if (voice.rfind("bf_", 0) == 0 || voice.rfind("bm_", 0) == 0) return "en";
    if (voice.rfind("ef_", 0) == 0 || voice.rfind("em_", 0) == 0) return "es";
    if (voice.rfind("ff_", 0) == 0) return "fr-fr";
    if (voice.rfind("hf_", 0) == 0 || voice.rfind("hm_", 0) == 0) return "hi";
    if (voice.rfind("if_", 0) == 0 || voice.rfind("im_", 0) == 0) return "it";
    if (voice.rfind("pf_", 0) == 0 || voice.rfind("pm_", 0) == 0) return "pt-br";
    return "en-US";
}

std::string default_voice_for_language(const std::string & language) {
    if (language == "en") return "bf_emma";
    if (language == "es") return "ef_dora";
    if (language == "fr-fr") return "ff_siwis";
    if (language == "hi") return "hf_alpha";
    if (language == "it") return "if_sara";
    if (language == "pt-br") return "pf_dora";
    return "af_heart";
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(c); break;
        }
    }
    return out;
}

bool parse_string_field(const std::string & line, const std::string & key, std::string & out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    ++pos;
    std::string value;
    while (pos < line.size()) {
        char c = line[pos++];
        if (c == '"') {
            out = value;
            return true;
        }
        if (c == '\\' && pos < line.size()) {
            char e = line[pos++];
            switch (e) {
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default: value.push_back(e); break;
            }
        } else {
            value.push_back(c);
        }
    }
    return false;
}

bool parse_int_field(const std::string & line, const std::string & key, int & out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    std::stringstream ss(line.substr(pos + 1));
    ss >> out;
    return !ss.fail();
}

bool parse_float_field(const std::string & line, const std::string & key, float & out) {
    std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    std::stringstream ss(line.substr(pos + 1));
    ss >> out;
    return !ss.fail();
}

Request parse_request(const std::string & line) {
    Request req;
    parse_int_field(line, "id", req.id);
    parse_string_field(line, "method", req.method);
    parse_string_field(line, "model_path", req.model_path);
    parse_string_field(line, "voices_path", req.voices_path);
    parse_string_field(line, "espeak_data_path", req.espeak_data_path);
    parse_string_field(line, "voice", req.voice);
    parse_string_field(line, "language", req.language);
    parse_string_field(line, "input", req.input);
    parse_string_field(line, "output_path", req.output_path);
    parse_float_field(line, "speed", req.speed);
    req.language = normalize_language(req.language);
    if (req.voice.empty()) req.voice = default_voice_for_language(req.language);
    if (!req.voice.empty()) req.language = language_for_voice(req.voice);
    if (req.speed <= 0.0f) req.speed = 1.0f;
    return req;
}

void ok(int id, const std::string & extra = "") {
    std::cout << "{\"id\":" << id << ",\"ok\":true";
    if (!extra.empty()) std::cout << "," << extra;
    std::cout << "}" << std::endl;
}

void fail(int id, const std::string & error) {
    std::cout << "{\"id\":" << id << ",\"ok\":false,\"error\":\"" << json_escape(error) << "\"}" << std::endl;
}

}

int main() {
    std::unique_ptr<chirp_kokoro_context, decltype(&chirp_kokoro_free)> ctx(nullptr, chirp_kokoro_free);
    std::string model_path;
    std::string voices_path;
    std::string loaded_voice;
    std::string loaded_language;
    float loaded_speed = 1.0f;
    std::string line;
    while (std::getline(std::cin, line)) {
        Request req = parse_request(line);
        if (req.method == "shutdown") {
            ok(req.id);
            break;
        }
        if (req.method == "languages") {
            ok(req.id, "\"languages\":[\"auto\",\"en-US\",\"en\",\"es\",\"fr-fr\",\"hi\",\"it\",\"pt-br\"]");
            continue;
        }
        if (req.method == "load") {
            model_path = req.model_path;
            voices_path = req.voices_path;
            loaded_voice = req.voice;
            loaded_language = req.language;
            loaded_speed = req.speed;
            if (!req.espeak_data_path.empty()) {
                setenv("ESPEAK_DATA_PATH", req.espeak_data_path.c_str(), 1);
            }
            chirp_kokoro_params params = chirp_kokoro_default_params();
            params.model_path = req.model_path.c_str();
            params.voices_path = req.voices_path.c_str();
            params.voice = req.voice.c_str();
            params.language = req.language.c_str();
            params.speed = req.speed;
            ctx.reset(chirp_kokoro_init(&params));
            const char * err = ctx ? chirp_kokoro_get_error(ctx.get()) : "failed to initialize kokoro";
            if (err && err[0] != '\0') {
                ctx.reset();
                fail(req.id, err);
            } else {
                ok(req.id);
            }
            continue;
        }
        if (req.method == "synthesize") {
            if (!ctx) {
                fail(req.id, "no model loaded");
                continue;
            }
            if (!req.voice.empty() && req.voice != loaded_voice) {
                loaded_voice = req.voice;
                loaded_language = language_for_voice(req.voice);
                chirp_kokoro_params params = chirp_kokoro_default_params();
                params.model_path = model_path.c_str();
                params.voices_path = voices_path.c_str();
                params.voice = loaded_voice.c_str();
                params.language = loaded_language.c_str();
                params.speed = loaded_speed;
                ctx.reset(chirp_kokoro_init(&params));
                const char * err = ctx ? chirp_kokoro_get_error(ctx.get()) : "failed to initialize kokoro";
                if (err && err[0] != '\0') {
                    ctx.reset();
                    fail(req.id, err);
                    continue;
                }
            }
            if (!chirp_kokoro_synthesize_to_file(ctx.get(), req.input.c_str(), req.output_path.c_str())) {
                fail(req.id, chirp_kokoro_get_error(ctx.get()));
            } else {
                ok(req.id, "\"output_path\":\"" + json_escape(req.output_path) + "\"");
            }
            continue;
        }
        fail(req.id, "unknown method: " + req.method);
    }
    return 0;
}
