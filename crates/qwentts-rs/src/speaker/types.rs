use crate::error::{Error, Result};
use crate::ggml_runtime::gguf::GgufModel;

#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) in_ch: usize,
    pub(super) out_ch: usize,
    pub(super) kernel: usize,
    pub(super) dilation: usize,
}

impl Conv1d {
    pub(super) fn load(
        model: &GgufModel,
        base: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        let weight = model.tensor_f32_by_name(&format!("{base}.weight"))?;
        let bias = model.tensor_f32_by_name(&format!("{base}.bias"))?;
        let expected = in_ch * out_ch * kernel;
        if weight.len() != expected || bias.len() != out_ch {
            return Err(Error::AudioProcessing(format!(
                "unexpected speaker tensor shape for {base}: weight={} expected={expected} bias={} expected={out_ch}",
                weight.len(),
                bias.len()
            )));
        }
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            dilation,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct Block {
    pub(super) tdnn1: Conv1d,
    pub(super) res2: Vec<Conv1d>,
    pub(super) tdnn2: Conv1d,
    pub(super) se1: Conv1d,
    pub(super) se2: Conv1d,
}

#[derive(Debug, Clone)]
pub struct SpeakerEncoder {
    pub(super) conv0: Conv1d,
    pub(super) blocks: Vec<Block>,
    pub(super) mfa: Conv1d,
    pub(super) asp_tdnn: Conv1d,
    pub(super) asp_conv: Conv1d,
    pub(super) fc: Conv1d,
}
