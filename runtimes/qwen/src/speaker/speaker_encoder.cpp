#include "speaker_encoder.h"

#include "codec_wav.h"

#include "kiss_fftr.h"
#include "soxr.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

using qwen3_tts::cpu_tensor;

static constexpr float k_pi = 3.14159265358979323846f;

static int reflect_index(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

bool speaker_encoder::load_tensor(qwen3_tts::GGUFCPUReader & reader, const std::string & name, cpu_tensor & out) {
    if (!reader.read_tensor(name, out)) {
        error_ = reader.error();
        return false;
    }
    return true;
}

bool speaker_encoder::load_conv(qwen3_tts::GGUFCPUReader & reader, const std::string & base, conv & c, int dilation) {
    c.dilation = dilation;
    return load_tensor(reader, base + ".weight", c.w) && load_tensor(reader, base + ".bias", c.b);
}

bool speaker_encoder::load(const std::string & model_path) {
    qwen3_tts::GGUFCPUReader reader;
    if (!reader.open(model_path)) {
        error_ = reader.error();
        return false;
    }
    if (!load_conv(reader, "spk_enc.conv0", conv0_, 1)) return false;
    const int dilations[3] = {2, 3, 4};
    for (int bi = 0; bi < 3; ++bi) {
        const int idx = bi + 1;
        const std::string p = "spk_enc.blk." + std::to_string(idx);
        if (!load_conv(reader, p + ".tdnn1", blocks_[bi].tdnn1, 1)) return false;
        for (int r = 0; r < 7; ++r) {
            if (!load_conv(reader, p + ".res2net." + std::to_string(r), blocks_[bi].res2[r], dilations[bi])) return false;
        }
        if (!load_conv(reader, p + ".tdnn2", blocks_[bi].tdnn2, 1)) return false;
        if (!load_conv(reader, p + ".se.conv1", blocks_[bi].se1, 1)) return false;
        if (!load_conv(reader, p + ".se.conv2", blocks_[bi].se2, 1)) return false;
    }
    return load_conv(reader, "spk_enc.mfa", mfa_, 1) &&
           load_conv(reader, "spk_enc.asp.tdnn", asp_tdnn_, 1) &&
           load_conv(reader, "spk_enc.asp.conv", asp_conv_, 1) &&
           load_conv(reader, "spk_enc.fc", fc_, 1);
}

std::vector<float> speaker_encoder::conv1d_same_reflect(const std::vector<float> & x, int channels, int len,
                                                        const conv & c, int & out_len) const {
    const int k = (int) c.w.dim(0);
    const int out_ch = (int) c.w.dim(2);
    const int pad = ((k - 1) * c.dilation) / 2;
    out_len = len;
    std::vector<float> y((size_t) out_ch * len);
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int t = 0; t < len; ++t) {
            float sum = c.b.data[oc];
            for (int ic = 0; ic < channels; ++ic) {
                for (int kk = 0; kk < k; ++kk) {
                    const int src = reflect_index(t + kk * c.dilation - pad, len);
                    const size_t wi = ((size_t) oc * channels + ic) * k + kk;
                    sum += c.w.data[wi] * x[(size_t) ic * len + src];
                }
            }
            y[(size_t) oc * len + t] = sum;
        }
    }
    return y;
}

std::vector<float> speaker_encoder::relu(std::vector<float> x) const {
    for (float & v : x) v = std::max(0.0f, v);
    return x;
}

std::vector<float> speaker_encoder::se_res2_block(const std::vector<float> & x, int len, const block & b) const {
    int out_len = 0;
    std::vector<float> h = relu(conv1d_same_reflect(x, 512, len, b.tdnn1, out_len));
    std::vector<float> res2(h.size());
    std::vector<float> prev(64 * len);
    for (int part = 0; part < 8; ++part) {
        std::vector<float> chunk(64 * len);
        for (int c = 0; c < 64; ++c) {
            std::copy_n(h.data() + (size_t) (part * 64 + c) * len, len, chunk.data() + (size_t) c * len);
        }
        std::vector<float> out = chunk;
        if (part > 0) {
            if (part > 1) {
                for (size_t i = 0; i < chunk.size(); ++i) chunk[i] += prev[i];
            }
            out = relu(conv1d_same_reflect(chunk, 64, len, b.res2[part - 1], out_len));
            prev = out;
        }
        for (int c = 0; c < 64; ++c) {
            std::copy_n(out.data() + (size_t) c * len, len, res2.data() + (size_t) (part * 64 + c) * len);
        }
    }
    h = relu(conv1d_same_reflect(res2, 512, len, b.tdnn2, out_len));

    std::vector<float> mean(512);
    for (int c = 0; c < 512; ++c) {
        double s = 0.0;
        for (int t = 0; t < len; ++t) s += h[(size_t) c * len + t];
        mean[c] = (float) (s / len);
    }
    int one = 0;
    std::vector<float> se = relu(conv1d_same_reflect(mean, 512, 1, b.se1, one));
    se = conv1d_same_reflect(se, 128, 1, b.se2, one);
    for (float & v : se) v = 1.0f / (1.0f + std::exp(-v));
    for (int c = 0; c < 512; ++c) {
        for (int t = 0; t < len; ++t) h[(size_t) c * len + t] = x[(size_t) c * len + t] + h[(size_t) c * len + t] * se[c];
    }
    return h;
}

std::vector<float> speaker_encoder::attentive_pool(const std::vector<float> & x, int len) const {
    std::vector<float> mean(1536), stdv(1536);
    for (int c = 0; c < 1536; ++c) {
        double m = 0.0;
        for (int t = 0; t < len; ++t) m += x[(size_t) c * len + t];
        m /= len;
        mean[c] = (float) m;
        double v = 0.0;
        for (int t = 0; t < len; ++t) {
            double d = x[(size_t) c * len + t] - m;
            v += d * d / len;
        }
        stdv[c] = (float) std::sqrt(std::max(v, 1e-12));
    }
    std::vector<float> att_in((size_t) 4608 * len);
    for (int c = 0; c < 1536; ++c) {
        for (int t = 0; t < len; ++t) {
            att_in[(size_t) c * len + t] = x[(size_t) c * len + t];
            att_in[(size_t) (1536 + c) * len + t] = mean[c];
            att_in[(size_t) (3072 + c) * len + t] = stdv[c];
        }
    }
    int out_len = 0;
    std::vector<float> att = relu(conv1d_same_reflect(att_in, 4608, len, asp_tdnn_, out_len));
    for (float & v : att) v = std::tanh(v);
    att = conv1d_same_reflect(att, 128, len, asp_conv_, out_len);
    for (int c = 0; c < 1536; ++c) {
        float mx = -INFINITY;
        for (int t = 0; t < len; ++t) mx = std::max(mx, att[(size_t) c * len + t]);
        double den = 0.0;
        for (int t = 0; t < len; ++t) {
            float e = std::exp(att[(size_t) c * len + t] - mx);
            att[(size_t) c * len + t] = e;
            den += e;
        }
        for (int t = 0; t < len; ++t) att[(size_t) c * len + t] /= (float) den;
    }
    std::vector<float> pooled(3072);
    for (int c = 0; c < 1536; ++c) {
        double m = 0.0;
        for (int t = 0; t < len; ++t) m += x[(size_t) c * len + t] * att[(size_t) c * len + t];
        pooled[c] = (float) m;
        double v = 0.0;
        for (int t = 0; t < len; ++t) {
            double d = x[(size_t) c * len + t] - m;
            v += att[(size_t) c * len + t] * d * d;
        }
        pooled[1536 + c] = (float) std::sqrt(std::max(v, 1e-12));
    }
    return pooled;
}

static float hz_to_mel(float hz) {
    constexpr float f_sp = 200.0f / 3.0f;
    if (hz < 1000.0f) {
        return hz / f_sp;
    }
    constexpr float min_log_mel = 1000.0f / f_sp;
    constexpr float min_log_hz = 1000.0f;
    const float logstep = std::log(6.4f) / 27.0f;
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
}

static float mel_to_hz(float mel) {
    constexpr float f_sp = 200.0f / 3.0f;
    constexpr float min_log_mel = 1000.0f / f_sp;
    constexpr float min_log_hz = 1000.0f;
    if (mel < min_log_mel) {
        return mel * f_sp;
    }
    const float logstep = std::log(6.4f) / 27.0f;
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

static bool resample_soxr(const std::vector<float> & input, int src_rate, int dst_rate,
                          std::vector<float> & output, std::string & error) {
    if (src_rate == dst_rate || input.empty()) {
        output = input;
        return true;
    }

    const size_t out_capacity = (size_t) std::ceil((double) input.size() * dst_rate / src_rate) + 256;
    output.assign(out_capacity, 0.0f);
    size_t idone = 0;
    size_t odone = 0;
    const soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    const soxr_error_t err = soxr_oneshot((double) src_rate, (double) dst_rate, 1,
                                          input.data(), input.size(), &idone,
                                          output.data(), output.size(), &odone,
                                          nullptr, &quality, nullptr);
    if (err) {
        error = std::string("soxr resample failed: ") + soxr_strerror(err);
        return false;
    }
    output.resize(odone);
    return true;
}

bool speaker_encoder::mel_spectrogram(const std::string & wav_path, std::vector<float> & mel, int & frames) {
    qwen3_tts::wav_data wav;
    if (!qwen3_tts::read_wav_mono(wav_path, wav, error_)) {
        return false;
    }
    std::vector<float> audio;
    if (!resample_soxr(wav.samples, wav.sample_rate, 24000, audio, error_)) {
        return false;
    }
    if (audio.empty()) {
        error_ = "speaker reference WAV is empty";
        return false;
    }

    constexpr int n_fft = 1024;
    constexpr int hop = 256;
    constexpr int n_mels = 128;
    constexpr int n_bins = n_fft / 2 + 1;
    constexpr int pad = (n_fft - hop) / 2;

    std::vector<float> padded(audio.size() + 2 * pad);
    for (int i = 0; i < (int) padded.size(); ++i) {
        padded[(size_t) i] = audio[(size_t) reflect_index(i - pad, (int) audio.size())];
    }
    if ((int) padded.size() < n_fft) {
        error_ = "speaker reference WAV is too short";
        return false;
    }
    frames = 1 + ((int) padded.size() - n_fft) / hop;

    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * k_pi * i / n_fft);
    }

    std::vector<float> mel_edges(n_mels + 2);
    const float mel_min = hz_to_mel(0.0f);
    const float mel_max = hz_to_mel(12000.0f);
    for (int i = 0; i < n_mels + 2; ++i) {
        const float a = (float) i / (float) (n_mels + 1);
        mel_edges[(size_t) i] = mel_to_hz(mel_min + a * (mel_max - mel_min));
    }

    std::vector<float> mel_basis((size_t) n_mels * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const float lower_hz = mel_edges[(size_t) m];
        const float center_hz = mel_edges[(size_t) m + 1];
        const float upper_hz = mel_edges[(size_t) m + 2];
        const float left = center_hz - lower_hz;
        const float right = upper_hz - center_hz;
        const float enorm = 2.0f / (upper_hz - lower_hz);
        for (int b = 0; b < n_bins; ++b) {
            const float hz = (float) b * 24000.0f / n_fft;
            float w = 0.0f;
            if (hz >= lower_hz && hz <= center_hz && left > 0.0f) {
                w = (hz - lower_hz) / left;
            } else if (hz > center_hz && hz <= upper_hz && right > 0.0f) {
                w = (upper_hz - hz) / right;
            }
            mel_basis[(size_t) m * n_bins + b] = std::max(0.0f, w) * enorm;
        }
    }

    mel.assign((size_t) n_mels * frames, 0.0f);
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(n_fft, 0, nullptr, nullptr);
    if (!fft_cfg) {
        error_ = "failed to allocate kissfft plan";
        return false;
    }

    std::vector<float> power((size_t) n_bins * frames);
    std::vector<float> frame(n_fft);
    std::vector<kiss_fft_cpx> spectrum(n_bins);
    for (int t = 0; t < frames; ++t) {
        const int start = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            frame[(size_t) i] = padded[(size_t) start + i] * window[(size_t) i];
        }
        kiss_fftr(fft_cfg, frame.data(), spectrum.data());
        for (int b = 0; b < n_bins; ++b) {
            const float re = spectrum[(size_t) b].r;
            const float im = spectrum[(size_t) b].i;
            power[(size_t) b * frames + t] = std::sqrt(re * re + im * im + 1e-9f);
        }
    }
    std::free(fft_cfg);

    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < frames; ++t) {
            double s = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                s += (double) mel_basis[(size_t) m * n_bins + b] * power[(size_t) b * frames + t];
            }
            mel[(size_t) m * frames + t] = std::log(std::max((float) s, 1e-5f));
        }
    }
    return true;
}

bool speaker_encoder::extract(const std::string & wav_path, std::vector<float> & embedding) {
    int frames = 0;
    std::vector<float> mel;
    if (!mel_spectrogram(wav_path, mel, frames)) {
        return false;
    }

    int out_len = 0;
    std::vector<float> h = relu(conv1d_same_reflect(mel, 128, frames, conv0_, out_len));
    std::vector<std::vector<float>> block_outputs;
    block_outputs.reserve(3);
    for (int i = 0; i < 3; ++i) {
        h = se_res2_block(h, frames, blocks_[i]);
        block_outputs.push_back(h);
    }

    std::vector<float> cat((size_t) 1536 * frames);
    for (int bi = 0; bi < 3; ++bi) {
        for (int c = 0; c < 512; ++c) {
            std::copy_n(block_outputs[(size_t) bi].data() + (size_t) c * frames, frames,
                        cat.data() + (size_t) (bi * 512 + c) * frames);
        }
    }

    h = relu(conv1d_same_reflect(cat, 1536, frames, mfa_, out_len));
    std::vector<float> pooled = attentive_pool(h, frames);
    std::vector<float> fc = conv1d_same_reflect(pooled, 3072, 1, fc_, out_len);
    embedding.assign(fc.begin(), fc.begin() + 1024);
    return true;
}
