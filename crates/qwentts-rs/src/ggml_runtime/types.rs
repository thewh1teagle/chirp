struct TensorSpec<'a> {
    name: &'a str,
    tensor_type: ffi::ggml_type,
    shape: &'a [usize],
    index: i64,
}

#[derive(Clone, Copy)]
enum BackendChoice {
    Best,
    Cpu,
}

pub struct PrefillEmbeddings {
    pub prefill: Vec<f32>,
    pub trailing_text_hidden: Vec<f32>,
    pub tts_pad_embed: Vec<f32>,
    pub prefill_len: usize,
    pub trailing_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub language_id: Option<i32>,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_k: i32,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            language_id: None,
            repetition_penalty: 1.0,
            temperature: 0.0,
            top_k: 1,
        }
    }
}

struct KvCache {
    ctx: *mut ffi::ggml_context,
    buffer: ffi::ggml_backend_buffer_t,
    k_cache: Vec<*mut ffi::ggml_tensor>,
    v_cache: Vec<*mut ffi::ggml_tensor>,
    n_ctx: usize,
}

impl KvCache {
    fn new(
        backend: ffi::ggml_backend_t,
        n_layers: usize,
        n_ctx: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Result<Self> {
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: n_layers * 2 * ffi::ggml_tensor_overhead(),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(Error::Ggml("failed to create KV cache context".into()));
            }
            let mut k_cache = Vec::with_capacity(n_layers);
            let mut v_cache = Vec::with_capacity(n_layers);
            for il in 0..n_layers {
                let k = ffi::ggml_new_tensor_3d(
                    ctx,
                    ffi::ggml_type_GGML_TYPE_F16,
                    head_dim as i64,
                    n_kv_heads as i64,
                    n_ctx as i64,
                );
                let v = ffi::ggml_new_tensor_3d(
                    ctx,
                    ffi::ggml_type_GGML_TYPE_F16,
                    head_dim as i64,
                    n_kv_heads as i64,
                    n_ctx as i64,
                );
                let k_name = CString::new(format!("k_cache_{il}")).unwrap();
                let v_name = CString::new(format!("v_cache_{il}")).unwrap();
                ffi::ggml_set_name(k, k_name.as_ptr());
                ffi::ggml_set_name(v, v_name.as_ptr());
                k_cache.push(k);
                v_cache.push(v);
            }
            let buffer = ffi::ggml_backend_alloc_ctx_tensors(ctx, backend);
            if buffer.is_null() {
                ffi::ggml_free(ctx);
                return Err(Error::Ggml("failed to allocate KV cache buffer".into()));
            }
            ffi::ggml_backend_buffer_clear(buffer, 0);
            Ok(Self {
                ctx,
                buffer,
                k_cache,
                v_cache,
                n_ctx,
            })
        }
    }
}

impl Drop for KvCache {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
