use crate::ggml_runtime::gguf::TensorInfo;

#[derive(Debug, Clone, PartialEq)]
pub struct CodecConfig {
    pub sample_rate: u32,
    pub n_codebooks: u32,
    pub codebook_size: u32,
    pub codebook_dim: u32,
    pub latent_dim: u32,
    pub hidden_dim: u32,
    pub n_pre_tfm_layers: u32,
    pub n_heads: u32,
    pub ffn_dim: u32,
    pub decoder_dim: u32,
    pub upsample_rates: [u32; 4],
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
pub struct CodecTensorMap {
    pub config: CodecConfig,
    pub tensors: Vec<CodecMappedTensor>,
}

#[derive(Debug, Clone)]
pub struct CodecMappedTensor {
    pub name: String,
    pub info: TensorInfo,
}
