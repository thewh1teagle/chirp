use crate::error::Result;
use crate::ggml_runtime::gguf::GgufModel;

use super::CodecConfig;

impl CodecConfig {
    pub fn load(model: &GgufModel) -> Result<Self> {
        Ok(Self {
            sample_rate: model.get_u32_or("qwen3-tts.tokenizer.sample_rate", 24_000)?,
            n_codebooks: model.get_u32_or("qwen3-tts.tokenizer.num_codebooks", 16)?,
            codebook_size: model.get_u32_or("qwen3-tts.tokenizer.codebook_size", 2048)?,
            codebook_dim: 256,
            latent_dim: 1024,
            hidden_dim: 512,
            n_pre_tfm_layers: 8,
            n_heads: 16,
            ffn_dim: 1024,
            decoder_dim: 1536,
            upsample_rates: [8, 5, 4, 3],
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        })
    }
}
