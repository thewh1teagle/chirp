use crate::error::{Error, Result};
use crate::ggml_runtime::gguf::{GgufModel, TensorInfo};

use super::{ArConfig, ArTensorRole, LayerTensorKind, MappedTensor};

pub(super) fn classify_ar_tensor(name: &str, cfg: &ArConfig) -> Option<(ArTensorRole, Vec<usize>)> {
    let h = cfg.hidden_size as usize;
    let cp_h = cfg.code_pred_hidden_size as usize;
    let text_dim = cfg.text_embd_dim as usize;
    let text_vocab = cfg.text_vocab_size as usize;
    let codec_vocab = cfg.codec_vocab_size as usize;
    let q_dim = (cfg.n_attention_heads * cfg.head_dim) as usize;
    let kv_dim = (cfg.n_key_value_heads * cfg.head_dim) as usize;
    let ffn = cfg.intermediate_size as usize;
    // The code predictor usually matches the talker, but 1.7B VoiceDesign has
    // its own narrower transformer behind code_pred.input_proj.
    let cp_q_dim = (cfg.code_pred_attention_heads * cfg.code_pred_head_dim) as usize;
    let cp_kv_dim = (cfg.code_pred_key_value_heads * cfg.code_pred_head_dim) as usize;
    let cp_ffn = cfg.code_pred_intermediate_size as usize;

    Some(match name {
        "talker.text_embd.weight" => (ArTensorRole::TextEmbedding, vec![text_dim, text_vocab]),
        "talker.text_proj.fc1.weight" => {
            (ArTensorRole::TextProjFc1Weight, vec![text_dim, text_dim])
        }
        "talker.text_proj.fc1.bias" => (ArTensorRole::TextProjFc1Bias, vec![text_dim]),
        "talker.text_proj.fc2.weight" => (ArTensorRole::TextProjFc2Weight, vec![text_dim, h]),
        "talker.text_proj.fc2.bias" => (ArTensorRole::TextProjFc2Bias, vec![h]),
        "talker.codec_embd.weight" => (ArTensorRole::CodecEmbedding, vec![h, codec_vocab]),
        "talker.codec_head.weight" => (ArTensorRole::CodecHead, vec![h, codec_vocab]),
        "talker.output_norm.weight" => (ArTensorRole::OutputNorm, vec![h]),
        "code_pred.output_norm.weight" => (ArTensorRole::CodePredOutputNorm, vec![cp_h]),
        "code_pred.input_proj.weight" => (ArTensorRole::CodePredInputProjWeight, vec![h, cp_h]),
        "code_pred.input_proj.bias" => (ArTensorRole::CodePredInputProjBias, vec![cp_h]),
        _ => {
            if let Some((layer, suffix)) = parse_indexed_suffix(name, "talker.blk.") {
                return layer_kind(suffix, h, q_dim, kv_dim, ffn, cfg.head_dim as usize)
                    .map(|(kind, shape)| (ArTensorRole::TalkerLayer { layer, kind }, shape));
            }
            if let Some((layer, suffix)) = parse_indexed_suffix(name, "code_pred.blk.") {
                return layer_kind(
                    suffix,
                    cp_h,
                    cp_q_dim,
                    cp_kv_dim,
                    cp_ffn,
                    cfg.code_pred_head_dim as usize,
                )
                .map(|(kind, shape)| (ArTensorRole::CodePredLayer { layer, kind }, shape));
            }
            if let Some(cb) = parse_indexed_terminal(name, "code_pred.codec_embd.", ".weight") {
                return Some((
                    ArTensorRole::CodePredEmbedding { codebook: cb },
                    vec![h, cfg.code_pred_vocab_size as usize],
                ));
            }
            if let Some(cb) = parse_indexed_terminal(name, "code_pred.lm_head.", ".weight") {
                return Some((
                    ArTensorRole::CodePredHead { codebook: cb },
                    vec![cp_h, cfg.code_pred_vocab_size as usize],
                ));
            }
            return None;
        }
    })
}

fn layer_kind(
    suffix: &str,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn: usize,
    head_dim: usize,
) -> Option<(LayerTensorKind, Vec<usize>)> {
    Some(match suffix {
        "attn_norm.weight" => (LayerTensorKind::AttnNorm, vec![h]),
        "attn_q_norm.weight" => (LayerTensorKind::AttnQNorm, vec![head_dim]),
        "attn_k_norm.weight" => (LayerTensorKind::AttnKNorm, vec![head_dim]),
        "attn_q.weight" => (LayerTensorKind::AttnQ, vec![h, q_dim]),
        "attn_k.weight" => (LayerTensorKind::AttnK, vec![h, kv_dim]),
        "attn_v.weight" => (LayerTensorKind::AttnV, vec![h, kv_dim]),
        "attn_output.weight" => (LayerTensorKind::AttnOutput, vec![q_dim, h]),
        "ffn_norm.weight" => (LayerTensorKind::FfnNorm, vec![h]),
        "ffn_gate.weight" => (LayerTensorKind::FfnGate, vec![h, ffn]),
        "ffn_up.weight" => (LayerTensorKind::FfnUp, vec![h, ffn]),
        "ffn_down.weight" => (LayerTensorKind::FfnDown, vec![ffn, h]),
        _ => return None,
    })
}

fn parse_indexed_suffix<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let dot = rest.find('.')?;
    let index = rest[..dot].parse().ok()?;
    Some((index, &rest[dot + 1..]))
}

fn parse_indexed_terminal(name: &str, prefix: &str, suffix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

pub(super) fn validate_tensor_size(
    model: &GgufModel,
    tensor: &TensorInfo,
    shape: &[usize],
) -> Result<()> {
    let elements = shape.iter().product::<usize>();
    let bytes_per_element = match tensor.tensor_type.raw() {
        0 => 4,
        1 => 2,
        8 => 18,
        _ => return Ok(()),
    };
    if tensor.tensor_type.raw() == 8 {
        return Ok(());
    }
    let expected = elements * bytes_per_element;
    if tensor.size != expected {
        return Err(Error::ModelConfig(format!(
            "tensor {} has {} bytes, expected {} for shape {:?} in {}",
            tensor.name,
            tensor.size,
            expected,
            shape,
            model.path().display()
        )));
    }
    Ok(())
}

pub(super) fn validate_ar_coverage(cfg: &ArConfig, tensors: &[MappedTensor]) -> Result<()> {
    let need = 8
        + cfg.n_layers as usize * 11
        + cfg.code_pred_layers as usize * 11
        + (cfg.n_codebooks as usize - 1) * 2
        + 1;
    if tensors.len() < need {
        return Err(Error::ModelConfig(format!(
            "AR tensor map incomplete: mapped {} tensors, expected at least {need}",
            tensors.len()
        )));
    }
    Ok(())
}
