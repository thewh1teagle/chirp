#include "audio_tokenizer_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace qwen3_tts {

static float elu(float x) {
    return x >= 0.0f ? x : std::exp(x) - 1.0f;
}

static void apply_elu(std::vector<float> & x) {
    for (float & v : x) {
        v = elu(v);
    }
}

std::vector<float> AudioTokenizerEncoder::conv1d(const std::vector<float> & x, int in_ch, int len,
                                                 const enc_conv & c, int & out_len) const {
    const int k = (int) c.w.dim(0);
    const int out_ch = (int) c.w.dim(2);
    const int eff_k = (k - 1) * c.dilation + 1;
    const int pad_total = eff_k - c.stride;
    const int n_frames = (int) std::ceil((len - eff_k + pad_total) / (float) c.stride + 1.0f) - 1;
    const int ideal_len = n_frames * c.stride + eff_k - pad_total;
    const int extra = ideal_len - len;
    const int pad_left = pad_total;
    out_len = (len + pad_left + extra - eff_k) / c.stride + 1;

    std::vector<float> y((size_t) out_ch * out_len, 0.0f);
    for (int oc = 0; oc < out_ch; ++oc) {
        for (int t = 0; t < out_len; ++t) {
            float sum = c.b.data.empty() ? 0.0f : c.b.data[oc];
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int kk = 0; kk < k; ++kk) {
                    int src = t * c.stride + kk * c.dilation - pad_left;
                    if (src < 0 || src >= len) {
                        if (c.pad_mode == "replicate") {
                            src = std::max(0, std::min(len - 1, src));
                        } else {
                            continue;
                        }
                    }
                    const size_t wi = ((size_t) oc * in_ch + ic) * k + kk;
                    sum += c.w.data[wi] * x[(size_t) ic * len + src];
                }
            }
            y[(size_t) oc * out_len + t] = sum;
        }
    }
    return y;
}

std::vector<float> AudioTokenizerEncoder::layer_norm(const std::vector<float> & x, int len, const cpu_tensor & w,
                                                     const cpu_tensor & b) const {
    std::vector<float> y(x.size());
    for (int t = 0; t < len; ++t) {
        double mean = 0.0;
        for (int h = 0; h < cfg_.hidden; ++h) {
            mean += x[(size_t) t * cfg_.hidden + h];
        }
        mean /= cfg_.hidden;
        double var = 0.0;
        for (int h = 0; h < cfg_.hidden; ++h) {
            const double d = x[(size_t) t * cfg_.hidden + h] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt((float) (var / cfg_.hidden) + cfg_.norm_eps);
        for (int h = 0; h < cfg_.hidden; ++h) {
            const size_t i = (size_t) t * cfg_.hidden + h;
            y[i] = (x[i] - (float) mean) * inv * w.data[h] + b.data[h];
        }
    }
    return y;
}

std::vector<float> AudioTokenizerEncoder::linear(const std::vector<float> & x, int rows, int in_dim,
                                                 const cpu_tensor & w) const {
    const int out_dim = (int) w.dim(1);
    std::vector<float> y((size_t) rows * out_dim, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                sum += x[(size_t) r * in_dim + i] * w.data[(size_t) o * in_dim + i];
            }
            y[(size_t) r * out_dim + o] = sum;
        }
    }
    return y;
}

static void apply_rope(std::vector<float> & q, std::vector<float> & k, int len, int heads, int head_dim,
                       float theta) {
    const int half = head_dim / 2;
    for (int t = 0; t < len; ++t) {
        for (int h = 0; h < heads; ++h) {
            float * qh = &q[((size_t) t * heads + h) * head_dim];
            float * kh = &k[((size_t) t * heads + h) * head_dim];
            for (int i = 0; i < half; ++i) {
                const float inv = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
                const float c = std::cos(t * inv);
                const float s = std::sin(t * inv);
                const float q0 = qh[i], q1 = qh[i + half];
                const float k0 = kh[i], k1 = kh[i + half];
                qh[i] = q0 * c - q1 * s;
                qh[i + half] = q1 * c + q0 * s;
                kh[i] = k0 * c - k1 * s;
                kh[i + half] = k1 * c + k0 * s;
            }
        }
    }
}

void AudioTokenizerEncoder::transformer_layer(std::vector<float> & x, int len, const enc_layer & layer) const {
    std::vector<float> norm = layer_norm(x, len, layer.attn_norm_w, layer.attn_norm_b);
    std::vector<float> q = linear(norm, len, cfg_.hidden, layer.q_w);
    std::vector<float> k = linear(norm, len, cfg_.hidden, layer.k_w);
    std::vector<float> v = linear(norm, len, cfg_.hidden, layer.v_w);
    apply_rope(q, k, len, cfg_.heads, cfg_.head_dim, cfg_.rope_theta);

    std::vector<float> att((size_t) len * cfg_.hidden, 0.0f);
    const float scale = 1.0f / std::sqrt((float) cfg_.head_dim);
    for (int t = 0; t < len; ++t) {
        for (int h = 0; h < cfg_.heads; ++h) {
            std::vector<float> scores(t + 1);
            float max_score = -std::numeric_limits<float>::infinity();
            for (int s = 0; s <= t; ++s) {
                float dot = 0.0f;
                for (int d = 0; d < cfg_.head_dim; ++d) {
                    dot += q[((size_t) t * cfg_.heads + h) * cfg_.head_dim + d] *
                           k[((size_t) s * cfg_.heads + h) * cfg_.head_dim + d];
                }
                scores[s] = dot * scale;
                max_score = std::max(max_score, scores[s]);
            }
            float denom = 0.0f;
            for (float & sc : scores) {
                sc = std::exp(sc - max_score);
                denom += sc;
            }
            for (int s = 0; s <= t; ++s) {
                const float p = scores[s] / denom;
                for (int d = 0; d < cfg_.head_dim; ++d) {
                    att[(size_t) t * cfg_.hidden + h * cfg_.head_dim + d] +=
                        p * v[((size_t) s * cfg_.heads + h) * cfg_.head_dim + d];
                }
            }
        }
    }

    att = linear(att, len, cfg_.hidden, layer.o_w);
    for (int t = 0; t < len; ++t) {
        for (int h = 0; h < cfg_.hidden; ++h) {
            x[(size_t) t * cfg_.hidden + h] += att[(size_t) t * cfg_.hidden + h] * layer.attn_scale.data[h];
        }
    }

    norm = layer_norm(x, len, layer.ffn_norm_w, layer.ffn_norm_b);
    std::vector<float> ffn = linear(norm, len, cfg_.hidden, layer.up_w);
    for (float & z : ffn) {
        z = 0.5f * z * (1.0f + std::erf(z / std::sqrt(2.0f)));
    }
    ffn = linear(ffn, len, cfg_.ffn, layer.down_w);
    for (int t = 0; t < len; ++t) {
        for (int h = 0; h < cfg_.hidden; ++h) {
            x[(size_t) t * cfg_.hidden + h] += ffn[(size_t) t * cfg_.hidden + h] * layer.ffn_scale.data[h];
        }
    }
}

std::vector<int32_t> AudioTokenizerEncoder::quantize(const std::vector<float> & emb, int len) const {
    auto project = [&](const enc_vq & vq) {
        std::vector<float> y((size_t) len * cfg_.codebook_dim, 0.0f);
        for (int t = 0; t < len; ++t) {
            for (int o = 0; o < cfg_.codebook_dim; ++o) {
                float sum = 0.0f;
                for (int i = 0; i < cfg_.hidden; ++i) {
                    sum += emb[(size_t) i * len + t] * vq.input_proj.data[(size_t) o * cfg_.hidden + i];
                }
                y[(size_t) t * cfg_.codebook_dim + o] = sum;
            }
        }
        return y;
    };

    std::vector<int32_t> out((size_t) len * cfg_.valid_codebooks);
    auto encode_vq = [&](const enc_vq & vq, int out_offset, int n_books) {
        std::vector<float> residual = project(vq);
        for (int book = 0; book < n_books; ++book) {
            const cpu_tensor & cb = vq.codebooks[book];
            for (int t = 0; t < len; ++t) {
                int best = 0;
                float best_d = std::numeric_limits<float>::infinity();
                for (int c = 0; c < cfg_.codebook_size; ++c) {
                    float dist = 0.0f;
                    for (int d = 0; d < cfg_.codebook_dim; ++d) {
                        const float diff = residual[(size_t) t * cfg_.codebook_dim + d] -
                                           cb.data[(size_t) c * cfg_.codebook_dim + d];
                        dist += diff * diff;
                    }
                    if (dist < best_d) {
                        best_d = dist;
                        best = c;
                    }
                }
                out[(size_t) t * cfg_.valid_codebooks + out_offset + book] = best;
                for (int d = 0; d < cfg_.codebook_dim; ++d) {
                    residual[(size_t) t * cfg_.codebook_dim + d] -=
                        cb.data[(size_t) best * cfg_.codebook_dim + d];
                }
            }
        }
    };

    encode_vq(semantic, 0, 1);
    encode_vq(acoustic, 1, 15);
    return out;
}

bool AudioTokenizerEncoder::encode(const std::vector<float> & samples_24k, std::vector<int32_t> & codes,
                                   int32_t & n_frames) {
    int len = (int) samples_24k.size();
    std::vector<float> x = samples_24k;
    int out_len = 0;
    x = conv1d(x, 1, len, conv0, out_len);
    len = out_len;
    int channels = 64;

    for (int i = 0; i < 4; ++i) {
        std::vector<float> residual = x;
        apply_elu(x);
        x = conv1d(x, channels, len, res[i].conv1, out_len);
        apply_elu(x);
        x = conv1d(x, channels / 2, out_len, res[i].conv2, out_len);
        for (size_t j = 0; j < x.size(); ++j) {
            x[j] += residual[j];
        }
        apply_elu(x);
        x = conv1d(x, channels, len, downs[i], out_len);
        len = out_len;
        channels *= 2;
    }

    apply_elu(x);
    x = conv1d(x, channels, len, final_conv, out_len);
    len = out_len;

    std::vector<float> seq((size_t) len * cfg_.hidden);
    for (int c = 0; c < cfg_.hidden; ++c) {
        for (int t = 0; t < len; ++t) {
            seq[(size_t) t * cfg_.hidden + c] = x[(size_t) c * len + t];
        }
    }
    for (int i = 0; i < cfg_.layers; ++i) {
        transformer_layer(seq, len, layers[i]);
    }
    for (int c = 0; c < cfg_.hidden; ++c) {
        for (int t = 0; t < len; ++t) {
            x[(size_t) c * len + t] = seq[(size_t) t * cfg_.hidden + c];
        }
    }

    x = conv1d(x, cfg_.hidden, len, downsample, out_len);
    len = out_len;
    const int wanted = (int) ((samples_24k.size() + 1919) / 1920);
    n_frames = std::min(len, wanted);
    codes = quantize(x, n_frames);
    return true;
}

} // namespace qwen3_tts
