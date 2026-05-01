use crate::ggml_runtime::gguf::TensorInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Language {
    pub name: String,
    pub id: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArConfig {
    pub text_vocab_size: u32,
    pub text_embd_dim: u32,
    pub hidden_size: u32,
    pub n_layers: u32,
    pub n_attention_heads: u32,
    pub n_key_value_heads: u32,
    pub intermediate_size: u32,
    pub head_dim: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub codec_vocab_size: u32,
    pub n_codebooks: u32,
    pub code_pred_layers: u32,
    pub code_pred_vocab_size: u32,
    pub codec_pad_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub languages: Vec<Language>,
}

#[derive(Debug, Clone)]
pub struct ArTensorMap {
    pub config: ArConfig,
    pub tensors: Vec<MappedTensor>,
}

#[derive(Debug, Clone)]
pub struct MappedTensor {
    pub name: String,
    pub info: TensorInfo,
    pub shape: Vec<usize>,
    pub role: ArTensorRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArTensorRole {
    TextEmbedding,
    TextProjFc1Weight,
    TextProjFc1Bias,
    TextProjFc2Weight,
    TextProjFc2Bias,
    CodecEmbedding,
    CodecHead,
    OutputNorm,
    TalkerLayer { layer: usize, kind: LayerTensorKind },
    CodePredLayer { layer: usize, kind: LayerTensorKind },
    CodePredEmbedding { codebook: usize },
    CodePredHead { codebook: usize },
    CodePredOutputNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerTensorKind {
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    AttnQNorm,
    AttnKNorm,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
}
