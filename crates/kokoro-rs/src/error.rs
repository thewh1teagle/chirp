#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("onnx runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("espeak error: {0}")]
    Espeak(#[from] espeak_rs::ESpeakError),
    #[error("invalid voice `{voice}`: {reason}")]
    InvalidVoice { voice: String, reason: String },
    #[error("missing voice `{0}`")]
    MissingVoice(String),
    #[error("no audio generated")]
    NoAudio,
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;
