use std::path::PathBuf;

use ggml_rs_sys as ffi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("path contains an interior nul byte: {0:?}")]
    NulPath(PathBuf),
    #[error("failed to open GGUF model: {0}")]
    GgufOpen(PathBuf),
    #[error("tensor index {0} is out of range")]
    TensorIndex(i64),
    #[error("GGUF tensor {0} has no name")]
    MissingTensorName(i64),
    #[error("tensor data range is invalid for {path:?}: offset={offset} size={size}")]
    InvalidTensorRange {
        path: PathBuf,
        offset: usize,
        size: usize,
    },
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("unsupported tensor type {tensor_type} for {name}")]
    UnsupportedTensorType {
        name: String,
        tensor_type: ffi::ggml_type,
    },
    #[error("model config error: {0}")]
    ModelConfig(String),
    #[error("ggml error: {0}")]
    Ggml(String),
    #[error("metadata key contains an interior nul byte: {0}")]
    InvalidMetadataKey(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("codec path is required for synthesis")]
    MissingCodec,
    #[error("qwen rust synthesis is not implemented yet")]
    SynthesisNotImplemented,
    #[error("audio processing error: {0}")]
    AudioProcessing(String),
    #[error("unsupported wav format: {sample_format} {bits_per_sample} bits")]
    UnsupportedWav {
        sample_format: String,
        bits_per_sample: u16,
    },
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
