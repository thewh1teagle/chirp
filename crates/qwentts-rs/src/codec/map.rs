use crate::error::{Error, Result};
use crate::ggml_runtime::gguf::GgufModel;

use super::{CodecConfig, CodecMappedTensor, CodecTensorMap};

impl CodecTensorMap {
    pub fn load(model: &GgufModel) -> Result<Self> {
        let config = CodecConfig::load(model)?;
        let tensors = model
            .tensors()
            .filter_map(|tensor| match tensor {
                Ok(info) if info.name.starts_with("tok_dec.") => Some(Ok(CodecMappedTensor {
                    name: info.name.clone(),
                    info,
                })),
                Ok(_) => None,
                Err(err) => Some(Err(err)),
            })
            .collect::<Result<Vec<_>>>()?;
        if tensors.is_empty() {
            return Err(Error::ModelConfig("no tok_dec tensors found".into()));
        }
        validate_codec_coverage(&tensors)?;
        Ok(Self { config, tensors })
    }
}
fn validate_codec_coverage(tensors: &[CodecMappedTensor]) -> Result<()> {
    for required in [
        "tok_dec.vq_first.input_proj.weight",
        "tok_dec.vq_first.output_proj.weight",
        "tok_dec.vq_first.0.codebook",
        "tok_dec.vq_rest.input_proj.weight",
        "tok_dec.vq_rest.output_proj.weight",
        "tok_dec.pre_tfm.input_proj.weight",
        "tok_dec.pre_tfm.output_proj.weight",
        "tok_dec.pre_conv.weight",
        "tok_dec.dec.0.conv.weight",
        "tok_dec.dec.6.conv.weight",
    ] {
        if !tensors.iter().any(|tensor| tensor.name == required) {
            return Err(Error::ModelConfig(format!(
                "codec tensor map missing {required}"
            )));
        }
    }
    Ok(())
}
