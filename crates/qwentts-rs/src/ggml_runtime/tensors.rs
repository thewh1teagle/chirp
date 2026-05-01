use ggml_rs_sys as ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorType(pub ffi::ggml_type);

impl TensorType {
    pub fn raw(self) -> ffi::ggml_type {
        self.0
    }
}
