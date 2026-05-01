use crate::error::{Error, Result};
use crate::ggml_runtime::gguf::GgufModel;

use super::{ArConfig, Language};

impl ArConfig {
    pub fn load(model: &GgufModel) -> Result<Self> {
        let language_count = model.get_u32_or("qwen3-tts.language.count", 0)?;
        if language_count == 0 {
            return Err(Error::ModelConfig(
                "missing qwen3-tts language metadata".into(),
            ));
        }
        let mut languages = Vec::with_capacity(language_count as usize);
        for idx in 0..language_count {
            let name_key = format!("qwen3-tts.language.{idx}.name");
            let id_key = format!("qwen3-tts.language.{idx}.id");
            let name = model
                .get_string(&name_key)?
                .ok_or_else(|| Error::ModelConfig(format!("missing {name_key}")))?
                .to_lowercase();
            let id = model
                .get_u32(&id_key)?
                .ok_or_else(|| Error::ModelConfig(format!("missing {id_key}")))?;
            languages.push(Language {
                name,
                id: id as i32,
            });
        }

        Ok(Self {
            text_vocab_size: model.get_u32_any(
                &["qwen3-tts.text.vocab_size", "qwen3-tts.text_vocab_size"],
                151_936,
            )?,
            text_embd_dim: model.get_u32_any(
                &["qwen3-tts.text.embedding_dim", "qwen3-tts.text_hidden_size"],
                2048,
            )?,
            hidden_size: model.get_u32_any(
                &[
                    "qwen3-tts.talker.embedding_length",
                    "qwen3-tts.embedding_length",
                ],
                1024,
            )?,
            n_layers: model.get_u32_any(
                &["qwen3-tts.talker.block_count", "qwen3-tts.block_count"],
                28,
            )?,
            n_attention_heads: model.get_u32_any(
                &[
                    "qwen3-tts.talker.attention.head_count",
                    "qwen3-tts.attention.head_count",
                ],
                16,
            )?,
            n_key_value_heads: model.get_u32_any(
                &[
                    "qwen3-tts.talker.attention.head_count_kv",
                    "qwen3-tts.attention.head_count_kv",
                ],
                8,
            )?,
            intermediate_size: model.get_u32_any(
                &[
                    "qwen3-tts.talker.feed_forward_length",
                    "qwen3-tts.feed_forward_length",
                ],
                3072,
            )?,
            head_dim: model.get_u32_any(
                &[
                    "qwen3-tts.talker.attention.key_length",
                    "qwen3-tts.attention.key_length",
                ],
                128,
            )?,
            rms_norm_eps: model.get_f32_any(
                &[
                    "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
                    "qwen3-tts.attention.layer_norm_rms_epsilon",
                ],
                1e-6,
            )?,
            rope_theta: model.get_f32_any(
                &[
                    "qwen3-tts.talker.rope.freq_base",
                    "qwen3-tts.rope.freq_base",
                ],
                1_000_000.0,
            )?,
            codec_vocab_size: model.get_u32_any(
                &["qwen3-tts.talker.codec_vocab_size", "qwen3-tts.vocab_size"],
                3072,
            )?,
            n_codebooks: model.get_u32_any(
                &[
                    "qwen3-tts.talker.num_codebooks",
                    "qwen3-tts.num_code_groups",
                ],
                16,
            )?,
            code_pred_layers: model.get_u32_any(
                &[
                    "qwen3-tts.code_pred.layer_count",
                    "qwen3-tts.code_predictor.layer_count",
                ],
                5,
            )?,
            code_pred_vocab_size: model.get_u32_any(
                &[
                    "qwen3-tts.code_pred.vocab_size",
                    "qwen3-tts.code_predictor.vocab_size",
                ],
                2048,
            )?,
            codec_pad_id: model.get_u32_or("qwen3-tts.codec.pad_id", 2148)?,
            codec_bos_id: model.get_u32_or("qwen3-tts.codec.bos_id", 2149)?,
            codec_eos_id: model.get_u32_any(
                &["qwen3-tts.codec.eos_id", "qwen3-tts.codec.eos_token_id"],
                2150,
            )?,
            tts_bos_token_id: model.get_u32_any(
                &[
                    "qwen3-tts.tts_bos_token_id",
                    "qwen3-tts.tts.bos_token_id",
                    "qwen3-tts.tts.bos_id",
                ],
                151_672,
            )?,
            tts_eos_token_id: model.get_u32_any(
                &[
                    "qwen3-tts.tts_eos_token_id",
                    "qwen3-tts.tts.eos_token_id",
                    "qwen3-tts.tts.eos_id",
                ],
                151_673,
            )?,
            tts_pad_token_id: model.get_u32_any(
                &[
                    "qwen3-tts.tts_pad_token_id",
                    "qwen3-tts.tts.pad_token_id",
                    "qwen3-tts.tts.pad_id",
                ],
                151_671,
            )?,
            codec_think_id: model.get_u32_any(
                &["qwen3-tts.codec.think_id", "qwen3-tts.codec_think_id"],
                2154,
            )?,
            codec_nothink_id: model.get_u32_any(
                &["qwen3-tts.codec.nothink_id", "qwen3-tts.codec_nothink_id"],
                2155,
            )?,
            codec_think_bos_id: model.get_u32_any(
                &[
                    "qwen3-tts.codec.think_bos_id",
                    "qwen3-tts.codec_think_bos_id",
                ],
                2156,
            )?,
            codec_think_eos_id: model.get_u32_any(
                &[
                    "qwen3-tts.codec.think_eos_id",
                    "qwen3-tts.codec_think_eos_id",
                ],
                2157,
            )?,
            languages,
        })
    }
}
