use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

pub struct QwenTokenizer {
    inner: Tokenizer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedText {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

impl QwenTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(|err| Error::Tokenizer(err.to_string()))?;
        Ok(Self { inner })
    }

    pub fn from_gguf(model: &crate::ggml_runtime::gguf::GgufModel) -> Result<Self> {
        let json = model
            .get_string("tokenizer.huggingface.json")?
            .ok_or_else(|| {
                Error::Tokenizer("model GGUF does not contain tokenizer.huggingface.json".into())
            })?;
        let inner = Tokenizer::from_bytes(json.as_bytes())
            .map_err(|err| Error::Tokenizer(err.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<TokenizedText> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| Error::Tokenizer(err.to_string()))?;
        Ok(TokenizedText {
            ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
        })
    }

    pub fn encode_tts_text(&self, text: &str) -> Result<TokenizedText> {
        self.encode(&tts_prompt(text))
    }

    pub fn encode_instruct_text(&self, instruct: &str) -> Result<TokenizedText> {
        self.encode(&instruct_prompt(instruct))
    }
}

pub fn tts_prompt(text: &str) -> String {
    format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn instruct_prompt(instruct: &str) -> String {
    // VoiceDesign uses the same separate user turn as the upstream Python wrapper.
    format!("<|im_start|>user\n{instruct}<|im_end|>\n")
}
