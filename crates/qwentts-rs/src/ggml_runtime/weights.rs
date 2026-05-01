pub struct GgmlWeights {
    ctx: *mut ffi::ggml_context,
    backend: ffi::ggml_backend_t,
    backend_cpu: ffi::ggml_backend_t,
    sched: ffi::ggml_backend_sched_t,
    buffer: ffi::ggml_backend_buffer_t,
    compute_meta: Vec<u8>,
    tensors: HashMap<String, *mut ffi::ggml_tensor>,
    kv: Option<KvCache>,
    code_kv: Option<KvCache>,
}

impl GgmlWeights {
    pub fn load_ar(model: &GgufModel, map: &ArTensorMap) -> Result<Self> {
        let specs = map
            .tensors
            .iter()
            .map(|mapped| TensorSpec {
                name: mapped.name.as_str(),
                tensor_type: mapped.info.tensor_type.raw(),
                shape: mapped.shape.as_slice(),
                index: mapped.info.index,
            })
            .collect::<Vec<_>>();
        Self::load(model, &specs, BackendChoice::Best)
    }

    pub fn load_codec(model: &GgufModel, map: &CodecTensorMap) -> Result<Self> {
        let specs = map
            .tensors
            .iter()
            .map(|mapped| TensorSpec {
                name: mapped.name.as_str(),
                tensor_type: mapped.info.tensor_type.raw(),
                shape: &[],
                index: mapped.info.index,
            })
            .collect::<Vec<_>>();
        let mut weights = Self::load_existing_shapes(model, &specs, BackendChoice::Cpu)?;
        weights.normalize_codec_codebooks(model)?;
        Ok(weights)
    }

    pub fn tensor(&self, name: &str) -> Result<*mut ffi::ggml_tensor> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| Error::MissingTensor(name.into()))
    }

    pub fn backend_name(&self) -> String {
        unsafe {
            let name = ffi::ggml_backend_name(self.backend);
            if name.is_null() {
                return "unknown".into();
            }
            CStr::from_ptr(name).to_string_lossy().into_owned()
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn project_text_tokens(
        &mut self,
        text_tokens: &[i32],
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        if text_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let inp_name = CString::new("inp_text_tokens").unwrap();
        let out_name = CString::new("text_proj_out").unwrap();
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: self.compute_meta.len(),
                mem_buffer: self.compute_meta.as_mut_ptr().cast(),
                no_alloc: true,
            };
            let ctx0 = ffi::ggml_init(params);
            if ctx0.is_null() {
                return Err(Error::Ggml("failed to create compute context".into()));
            }
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            if gf.is_null() {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to create compute graph".into()));
            }

            let inp = ffi::ggml_new_tensor_1d(
                ctx0,
                ffi::ggml_type_GGML_TYPE_I32,
                text_tokens.len() as i64,
            );
            ffi::ggml_set_name(inp, inp_name.as_ptr());
            ffi::ggml_set_input(inp);

            let mut cur = ffi::ggml_get_rows(ctx0, self.tensor("talker.text_embd.weight")?, inp);
            cur = ffi::ggml_mul_mat(ctx0, self.tensor("talker.text_proj.fc1.weight")?, cur);
            cur = ffi::ggml_add(ctx0, cur, self.tensor("talker.text_proj.fc1.bias")?);
            cur = ffi::ggml_silu(ctx0, cur);
            cur = ffi::ggml_mul_mat(ctx0, self.tensor("talker.text_proj.fc2.weight")?, cur);
            cur = ffi::ggml_add(ctx0, cur, self.tensor("talker.text_proj.fc2.bias")?);
            ffi::ggml_set_name(cur, out_name.as_ptr());
            ffi::ggml_set_output(cur);
            ffi::ggml_build_forward_expand(gf, cur);

            if !ffi::ggml_backend_sched_alloc_graph(self.sched, gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate text projection graph".into(),
                ));
            }
            let inp_graph = ffi::ggml_graph_get_tensor(gf, inp_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                inp_graph,
                text_tokens.as_ptr().cast(),
                0,
                std::mem::size_of_val(text_tokens),
            );
            let status = ffi::ggml_backend_sched_graph_compute(self.sched, gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "failed to compute text projection graph: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            if out.is_null() {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("missing text projection output".into()));
            }
            let mut output = vec![0.0f32; hidden_size * text_tokens.len()];
            ffi::ggml_backend_tensor_get(
                out,
                output.as_mut_ptr().cast(),
                0,
                output.len() * std::mem::size_of::<f32>(),
            );
            ffi::ggml_backend_sched_reset(self.sched);
            ffi::ggml_free(ctx0);
            Ok(output)
        }
    }

    pub fn lookup_embedding_rows(
        &mut self,
        embedding_name: &str,
        token_ids: &[i32],
        embd_dim: usize,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let inp_name = CString::new("inp_lookup_tokens").unwrap();
        let out_name = CString::new("lookup_rows").unwrap();
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: self.compute_meta.len(),
                mem_buffer: self.compute_meta.as_mut_ptr().cast(),
                no_alloc: true,
            };
            let ctx0 = ffi::ggml_init(params);
            if ctx0.is_null() {
                return Err(Error::Ggml("failed to create lookup context".into()));
            }
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            let inp =
                ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, token_ids.len() as i64);
            ffi::ggml_set_name(inp, inp_name.as_ptr());
            ffi::ggml_set_input(inp);
            let rows = ffi::ggml_get_rows(ctx0, self.tensor(embedding_name)?, inp);
            let rows = ffi::ggml_cast(ctx0, rows, ffi::ggml_type_GGML_TYPE_F32);
            ffi::ggml_set_name(rows, out_name.as_ptr());
            ffi::ggml_set_output(rows);
            ffi::ggml_build_forward_expand(gf, rows);
            if !ffi::ggml_backend_sched_alloc_graph(self.sched, gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate embedding lookup graph".into(),
                ));
            }
            let inp_graph = ffi::ggml_graph_get_tensor(gf, inp_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                inp_graph,
                token_ids.as_ptr().cast(),
                0,
                std::mem::size_of_val(token_ids),
            );
            let status = ffi::ggml_backend_sched_graph_compute(self.sched, gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "embedding lookup failed: status={status}"
                )));
            }
            let out = ffi::ggml_graph_get_tensor(gf, out_name.as_ptr());
            let mut output = vec![0.0f32; embd_dim * token_ids.len()];
            ffi::ggml_backend_tensor_get(
                out,
                output.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(output.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.sched);
            ffi::ggml_free(ctx0);
            Ok(output)
        }
    }


}
