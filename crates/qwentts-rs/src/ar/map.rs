use crate::error::Result;
use crate::ggml_runtime::gguf::GgufModel;

use super::classify::{classify_ar_tensor, validate_ar_coverage, validate_tensor_size};
use super::{ArConfig, ArTensorMap, MappedTensor};

impl ArTensorMap {
    pub fn load(model: &GgufModel) -> Result<Self> {
        let config = ArConfig::load(model)?;
        let mut tensors = Vec::new();
        for tensor in model.tensors() {
            let info = tensor?;
            if let Some((role, shape)) = classify_ar_tensor(&info.name, &config) {
                validate_tensor_size(model, &info, &shape)?;
                tensors.push(MappedTensor {
                    name: info.name.clone(),
                    info,
                    shape,
                    role,
                });
            }
        }
        validate_ar_coverage(&config, &tensors)?;
        Ok(Self { config, tensors })
    }
}
