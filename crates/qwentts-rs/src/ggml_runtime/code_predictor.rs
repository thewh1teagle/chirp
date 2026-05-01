impl GgmlWeights {
    fn init_code_kv_cache(&mut self, cfg: &crate::ar::ArConfig) -> Result<()> {
        self.code_kv = Some(KvCache::new(
            self.backend,
            cfg.code_pred_layers as usize,
            16,
            cfg.code_pred_head_dim as usize,
            cfg.code_pred_key_value_heads as usize,
        )?);
        Ok(())
    }

    fn code_pred_prefill(
        &mut self,
        hidden: &[f32],
        cb0_embd: &[f32],
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<f32>> {
        self.init_code_kv_cache(cfg)?;
        self.forward_code_pred(Some(cb0_embd), hidden, None, 0, 0, cfg)
    }

    fn code_pred_step(
        &mut self,
        hidden: &[f32],
        prev_code: i32,
        step: usize,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<f32>> {
        self.forward_code_pred(None, hidden, Some(prev_code), step + 1, step, cfg)
    }

    fn forward_code_pred(
        &mut self,
        cb0_embd: Option<&[f32]>,
        hidden: &[f32],
        code: Option<i32>,
        n_past: usize,
        generation_step: usize,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<f32>> {
        let h = cfg.hidden_size as usize;
        let cp_h = cfg.code_pred_hidden_size as usize;
        let n_tokens = if cb0_embd.is_some() { 2usize } else { 1usize };
        let code_kv = self
            .code_kv
            .as_ref()
            .ok_or_else(|| Error::Ggml("code predictor KV cache missing".into()))?;
        unsafe {
            let hidden_name = CString::new("inp_hidden").unwrap();
            let cb0_name = CString::new("inp_cb0_embd").unwrap();
            let code_name = CString::new("inp_code").unwrap();
            let pos_name = CString::new("inp_pos").unwrap();
            let logits_name = CString::new("logits").unwrap();
            let params = ffi::ggml_init_params {
                mem_size: self.compute_meta.len(),
                mem_buffer: self.compute_meta.as_mut_ptr().cast(),
                no_alloc: true,
            };
            let ctx0 = ffi::ggml_init(params);
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            let inp_hidden = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_F32, h as i64);
            ffi::ggml_set_name(inp_hidden, hidden_name.as_ptr());
            ffi::ggml_set_input(inp_hidden);
            let inp_pos =
                ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_tokens as i64);
            ffi::ggml_set_name(inp_pos, pos_name.as_ptr());
            ffi::ggml_set_input(inp_pos);
            let mut cur;
            if cb0_embd.is_some() {
                let inp_cb0 = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_F32, h as i64);
                ffi::ggml_set_name(inp_cb0, cb0_name.as_ptr());
                ffi::ggml_set_input(inp_cb0);
                let hidden_2d = ffi::ggml_reshape_2d(ctx0, inp_hidden, h as i64, 1);
                let cb0_2d = ffi::ggml_reshape_2d(ctx0, inp_cb0, h as i64, 1);
                cur = ffi::ggml_concat(ctx0, hidden_2d, cb0_2d, 1);
            } else if generation_step == 0 {
                cur = ffi::ggml_reshape_2d(ctx0, inp_hidden, h as i64, 1);
            } else {
                let inp_code = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, 1);
                ffi::ggml_set_name(inp_code, code_name.as_ptr());
                ffi::ggml_set_input(inp_code);
                let name = format!("code_pred.codec_embd.{}.weight", generation_step - 1);
                cur = ffi::ggml_get_rows(ctx0, self.tensor(&name)?, inp_code);
                cur = ffi::ggml_reshape_2d(ctx0, cur, h as i64, 1);
            }
            if self.tensors.contains_key("code_pred.input_proj.weight") {
                // 1.7B checkpoints use a narrower code predictor behind this projection.
                cur = ffi::ggml_mul_mat(ctx0, self.tensor("code_pred.input_proj.weight")?, cur);
                cur = ffi::ggml_add(ctx0, cur, self.tensor("code_pred.input_proj.bias")?);
            }
            let mut inp_l = cur;
            let head_dim = cfg.code_pred_head_dim as i64;
            let n_head = cfg.code_pred_attention_heads as i64;
            let n_kv_head = cfg.code_pred_key_value_heads as i64;
            let kq_scale = 1.0f32 / (cfg.code_pred_head_dim as f32).sqrt();
            for il in 0..cfg.code_pred_layers as usize {
                let prefix = format!("code_pred.blk.{il}.");
                cur = ffi::ggml_rms_norm(ctx0, inp_l, cfg.rms_norm_eps);
                cur = ffi::ggml_mul(
                    ctx0,
                    cur,
                    self.tensor(&(prefix.clone() + "attn_norm.weight"))?,
                );
                let mut q =
                    ffi::ggml_mul_mat(ctx0, self.tensor(&(prefix.clone() + "attn_q.weight"))?, cur);
                let mut k =
                    ffi::ggml_mul_mat(ctx0, self.tensor(&(prefix.clone() + "attn_k.weight"))?, cur);
                let mut v =
                    ffi::ggml_mul_mat(ctx0, self.tensor(&(prefix.clone() + "attn_v.weight"))?, cur);
                q = ffi::ggml_reshape_3d(ctx0, q, head_dim, n_head, n_tokens as i64);
                k = ffi::ggml_reshape_3d(ctx0, k, head_dim, n_kv_head, n_tokens as i64);
                v = ffi::ggml_reshape_3d(ctx0, v, head_dim, n_kv_head, n_tokens as i64);
                if self
                    .tensors
                    .contains_key(&(prefix.clone() + "attn_q_norm.weight"))
                {
                    q = ffi::ggml_mul(
                        ctx0,
                        ffi::ggml_rms_norm(ctx0, q, cfg.rms_norm_eps),
                        self.tensor(&(prefix.clone() + "attn_q_norm.weight"))?,
                    );
                }
                if self
                    .tensors
                    .contains_key(&(prefix.clone() + "attn_k_norm.weight"))
                {
                    k = ffi::ggml_mul(
                        ctx0,
                        ffi::ggml_rms_norm(ctx0, k, cfg.rms_norm_eps),
                        self.tensor(&(prefix.clone() + "attn_k_norm.weight"))?,
                    );
                }
                q = ffi::ggml_rope_ext(
                    ctx0,
                    q,
                    inp_pos,
                    ptr::null_mut(),
                    cfg.code_pred_head_dim as i32,
                    ffi::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    cfg.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                k = ffi::ggml_rope_ext(
                    ctx0,
                    k,
                    inp_pos,
                    ptr::null_mut(),
                    cfg.code_pred_head_dim as i32,
                    ffi::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    cfg.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                let k_cache = code_kv.k_cache[il];
                let v_cache = code_kv.v_cache[il];
                let k_view = ffi::ggml_view_3d(
                    ctx0,
                    k_cache,
                    head_dim,
                    n_kv_head,
                    n_tokens as i64,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    n_past * (*k_cache).nb[2],
                );
                let v_view = ffi::ggml_view_3d(
                    ctx0,
                    v_cache,
                    head_dim,
                    n_kv_head,
                    n_tokens as i64,
                    (*v_cache).nb[1],
                    (*v_cache).nb[2],
                    n_past * (*v_cache).nb[2],
                );
                ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx0, k, k_view));
                ffi::ggml_build_forward_expand(gf, ffi::ggml_cpy(ctx0, v, v_view));
                let n_kv = (n_past + n_tokens) as i64;
                let mut k_all = ffi::ggml_view_3d(
                    ctx0,
                    k_cache,
                    head_dim,
                    n_kv_head,
                    n_kv,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    0,
                );
                let mut v_all = ffi::ggml_view_3d(
                    ctx0,
                    v_cache,
                    head_dim,
                    n_kv_head,
                    n_kv,
                    (*v_cache).nb[1],
                    (*v_cache).nb[2],
                    0,
                );
                let q_perm = ffi::ggml_permute(ctx0, q, 0, 2, 1, 3);
                k_all = ffi::ggml_permute(ctx0, k_all, 0, 2, 1, 3);
                v_all = ffi::ggml_permute(ctx0, v_all, 0, 2, 1, 3);
                let mut kq = ffi::ggml_mul_mat(ctx0, k_all, q_perm);
                kq = ffi::ggml_scale(ctx0, kq, kq_scale);
                kq = ffi::ggml_diag_mask_inf(ctx0, kq, n_past as i32);
                kq = ffi::ggml_soft_max(ctx0, kq);
                v_all = ffi::ggml_cont(ctx0, ffi::ggml_transpose(ctx0, v_all));
                let mut kqv = ffi::ggml_mul_mat(ctx0, v_all, kq);
                kqv = ffi::ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                cur = ffi::ggml_cont_2d(ctx0, kqv, n_head * head_dim, n_tokens as i64);
                cur = ffi::ggml_mul_mat(
                    ctx0,
                    self.tensor(&(prefix.clone() + "attn_output.weight"))?,
                    cur,
                );
                cur = ffi::ggml_add(ctx0, cur, inp_l);
                let inp_ff = cur;
                cur = ffi::ggml_mul(
                    ctx0,
                    ffi::ggml_rms_norm(ctx0, inp_ff, cfg.rms_norm_eps),
                    self.tensor(&(prefix.clone() + "ffn_norm.weight"))?,
                );
                let mut gate = ffi::ggml_mul_mat(
                    ctx0,
                    self.tensor(&(prefix.clone() + "ffn_gate.weight"))?,
                    cur,
                );
                let up =
                    ffi::ggml_mul_mat(ctx0, self.tensor(&(prefix.clone() + "ffn_up.weight"))?, cur);
                gate = ffi::ggml_silu(ctx0, gate);
                cur = ffi::ggml_mul(ctx0, gate, up);
                let down = ffi::ggml_cast(
                    ctx0,
                    self.tensor(&(prefix.clone() + "ffn_down.weight"))?,
                    ffi::ggml_type_GGML_TYPE_F32,
                );
                cur = ffi::ggml_mul_mat(ctx0, down, cur);
                inp_l = ffi::ggml_add(ctx0, cur, inp_ff);
            }
            cur = ffi::ggml_mul(
                ctx0,
                ffi::ggml_rms_norm(ctx0, inp_l, cfg.rms_norm_eps),
                self.tensor("code_pred.output_norm.weight")?,
            );
            if n_tokens == 2 {
                cur = ffi::ggml_view_2d(
                    ctx0,
                    cur,
                    cp_h as i64,
                    1,
                    (*cur).nb[1],
                    cp_h * std::mem::size_of::<f32>(),
                );
            }
            let head_name = format!("code_pred.lm_head.{generation_step}.weight");
            let logits = ffi::ggml_mul_mat(ctx0, self.tensor(&head_name)?, cur);
            ffi::ggml_set_name(logits, logits_name.as_ptr());
            ffi::ggml_set_output(logits);
            ffi::ggml_build_forward_expand(gf, logits);
            if !ffi::ggml_backend_sched_alloc_graph(self.sched, gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "failed to allocate code predictor graph".into(),
                ));
            }
            let hidden_tensor = ffi::ggml_graph_get_tensor(gf, hidden_name.as_ptr());
            if hidden_tensor.is_null() && cb0_embd.is_some() {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(
                    "code predictor graph missing inp_hidden".into(),
                ));
            }
            if !hidden_tensor.is_null() {
                ffi::ggml_backend_tensor_set(
                    hidden_tensor,
                    hidden.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(hidden),
                );
            }
            if let Some(cb0) = cb0_embd {
                let cb0_tensor = ffi::ggml_graph_get_tensor(gf, cb0_name.as_ptr());
                if cb0_tensor.is_null() {
                    ffi::ggml_backend_sched_reset(self.sched);
                    ffi::ggml_free(ctx0);
                    return Err(Error::Ggml(
                        "code predictor graph missing inp_cb0_embd".into(),
                    ));
                }
                ffi::ggml_backend_tensor_set(
                    cb0_tensor,
                    cb0.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(cb0),
                );
                let positions = [0_i32, 1_i32];
                let pos_tensor = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
                if pos_tensor.is_null() {
                    ffi::ggml_backend_sched_reset(self.sched);
                    ffi::ggml_free(ctx0);
                    return Err(Error::Ggml("code predictor graph missing inp_pos".into()));
                }
                ffi::ggml_backend_tensor_set(
                    pos_tensor,
                    positions.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(&positions),
                );
            } else {
                let pos = [n_past as i32];
                let pos_tensor = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
                if pos_tensor.is_null() {
                    ffi::ggml_backend_sched_reset(self.sched);
                    ffi::ggml_free(ctx0);
                    return Err(Error::Ggml("code predictor graph missing inp_pos".into()));
                }
                ffi::ggml_backend_tensor_set(
                    pos_tensor,
                    pos.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(&pos),
                );
                if let Some(code) = code {
                    let code_arr = [code];
                    let code_tensor = ffi::ggml_graph_get_tensor(gf, code_name.as_ptr());
                    if code_tensor.is_null() {
                        ffi::ggml_backend_sched_reset(self.sched);
                        ffi::ggml_free(ctx0);
                        return Err(Error::Ggml("code predictor graph missing inp_code".into()));
                    }
                    ffi::ggml_backend_tensor_set(
                        code_tensor,
                        code_arr.as_ptr().cast(),
                        0,
                        std::mem::size_of_val(&code_arr),
                    );
                }
            }
            let status = ffi::ggml_backend_sched_graph_compute(self.sched, gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "code predictor compute failed: status={status}"
                )));
            }
            let mut out = vec![0.0f32; cfg.code_pred_vocab_size as usize];
            ffi::ggml_backend_tensor_get(
                ffi::ggml_graph_get_tensor(gf, logits_name.as_ptr()),
                out.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(out.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.sched);
            ffi::ggml_free(ctx0);
            Ok(out)
        }
    }
}
