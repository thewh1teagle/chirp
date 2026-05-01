use crate::audio::signal::speaker_mel_spectrogram_from_wav;
use crate::error::Result;
use crate::ggml_runtime::gguf::GgufModel;

use super::ops::{conv1d_same_reflect, relu};
use super::types::{Block, Conv1d, SpeakerEncoder};

impl SpeakerEncoder {
    pub fn load(model: &GgufModel) -> Result<Self> {
        let conv0 = Conv1d::load(model, "spk_enc.conv0", 128, 512, 5, 1)?;
        let mut blocks = Vec::with_capacity(3);
        for (idx, dilation) in [2, 3, 4].into_iter().enumerate() {
            let base = format!("spk_enc.blk.{}", idx + 1);
            let tdnn1 = Conv1d::load(model, &format!("{base}.tdnn1"), 512, 512, 1, 1)?;
            let mut res2 = Vec::with_capacity(7);
            for part in 0..7 {
                res2.push(Conv1d::load(
                    model,
                    &format!("{base}.res2net.{part}"),
                    64,
                    64,
                    3,
                    dilation,
                )?);
            }
            let tdnn2 = Conv1d::load(model, &format!("{base}.tdnn2"), 512, 512, 1, 1)?;
            let se1 = Conv1d::load(model, &format!("{base}.se.conv1"), 512, 128, 1, 1)?;
            let se2 = Conv1d::load(model, &format!("{base}.se.conv2"), 128, 512, 1, 1)?;
            blocks.push(Block {
                tdnn1,
                res2,
                tdnn2,
                se1,
                se2,
            });
        }

        Ok(Self {
            conv0,
            blocks,
            mfa: Conv1d::load(model, "spk_enc.mfa", 1536, 1536, 1, 1)?,
            asp_tdnn: Conv1d::load(model, "spk_enc.asp.tdnn", 4608, 128, 1, 1)?,
            asp_conv: Conv1d::load(model, "spk_enc.asp.conv", 128, 1536, 1, 1)?,
            fc: Conv1d::load(model, "spk_enc.fc", 3072, 1024, 1, 1)?,
        })
    }

    pub fn extract(&self, wav_path: impl AsRef<std::path::Path>) -> Result<Vec<f32>> {
        let mel = speaker_mel_spectrogram_from_wav(wav_path)?;
        let frames = mel.frames;
        let mut h = relu(conv1d_same_reflect(&mel.values, frames, &self.conv0));
        let mut block_outputs = Vec::with_capacity(3);
        for block in &self.blocks {
            h = self.se_res2_block(&h, frames, block);
            block_outputs.push(h.clone());
        }

        let mut cat = vec![0.0; 1536 * frames];
        for (block_idx, block_out) in block_outputs.iter().enumerate() {
            for c in 0..512 {
                let src = c * frames;
                let dst = (block_idx * 512 + c) * frames;
                cat[dst..dst + frames].copy_from_slice(&block_out[src..src + frames]);
            }
        }

        h = relu(conv1d_same_reflect(&cat, frames, &self.mfa));
        let pooled = self.attentive_pool(&h, frames);
        let fc = conv1d_same_reflect(&pooled, 1, &self.fc);
        Ok(fc[..1024].to_vec())
    }

    fn se_res2_block(&self, x: &[f32], len: usize, block: &Block) -> Vec<f32> {
        let h = relu(conv1d_same_reflect(x, len, &block.tdnn1));
        let mut res2 = vec![0.0; 512 * len];
        let mut prev = vec![0.0; 64 * len];
        for part in 0..8 {
            let mut chunk = vec![0.0; 64 * len];
            for c in 0..64 {
                let src = (part * 64 + c) * len;
                let dst = c * len;
                chunk[dst..dst + len].copy_from_slice(&h[src..src + len]);
            }
            let mut out = chunk.clone();
            if part > 0 {
                if part > 1 {
                    for (value, prev_value) in chunk.iter_mut().zip(&prev) {
                        *value += *prev_value;
                    }
                }
                out = relu(conv1d_same_reflect(&chunk, len, &block.res2[part - 1]));
                prev = out.clone();
            }
            for c in 0..64 {
                let src = c * len;
                let dst = (part * 64 + c) * len;
                res2[dst..dst + len].copy_from_slice(&out[src..src + len]);
            }
        }

        let mut h = relu(conv1d_same_reflect(&res2, len, &block.tdnn2));
        let mut mean = vec![0.0; 512];
        for c in 0..512 {
            mean[c] = h[c * len..(c + 1) * len].iter().sum::<f32>() / len as f32;
        }
        let mut se = relu(conv1d_same_reflect(&mean, 1, &block.se1));
        se = conv1d_same_reflect(&se, 1, &block.se2);
        for value in &mut se {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        for c in 0..512 {
            for t in 0..len {
                let idx = c * len + t;
                h[idx] = x[idx] + h[idx] * se[c];
            }
        }
        h
    }

    fn attentive_pool(&self, x: &[f32], len: usize) -> Vec<f32> {
        let mut mean = vec![0.0; 1536];
        let mut stdv = vec![0.0; 1536];
        for c in 0..1536 {
            let row = &x[c * len..(c + 1) * len];
            let m = row.iter().map(|v| *v as f64).sum::<f64>() / len as f64;
            mean[c] = m as f32;
            let var = row
                .iter()
                .map(|v| {
                    let d = *v as f64 - m;
                    d * d / len as f64
                })
                .sum::<f64>();
            stdv[c] = var.max(1e-12).sqrt() as f32;
        }

        let mut att_in = vec![0.0; 4608 * len];
        for c in 0..1536 {
            for t in 0..len {
                att_in[c * len + t] = x[c * len + t];
                att_in[(1536 + c) * len + t] = mean[c];
                att_in[(3072 + c) * len + t] = stdv[c];
            }
        }
        let mut att = relu(conv1d_same_reflect(&att_in, len, &self.asp_tdnn));
        for value in &mut att {
            *value = value.tanh();
        }
        att = conv1d_same_reflect(&att, len, &self.asp_conv);
        for c in 0..1536 {
            let row = &mut att[c * len..(c + 1) * len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0;
            for value in row.iter_mut() {
                *value = (*value - max).exp();
                den += *value as f64;
            }
            for value in row.iter_mut() {
                *value /= den as f32;
            }
        }

        let mut pooled = vec![0.0; 3072];
        for c in 0..1536 {
            let mut m = 0.0f64;
            for t in 0..len {
                m += x[c * len + t] as f64 * att[c * len + t] as f64;
            }
            pooled[c] = m as f32;
            let mut var = 0.0f64;
            for t in 0..len {
                let d = x[c * len + t] as f64 - m;
                var += att[c * len + t] as f64 * d * d;
            }
            pooled[1536 + c] = var.max(1e-12).sqrt() as f32;
        }
        pooled
    }
}
