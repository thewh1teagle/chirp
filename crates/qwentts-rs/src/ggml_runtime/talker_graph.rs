impl GgmlWeights {
    pub fn init_kv_cache(
        &mut self,
        n_layers: usize,
        n_ctx: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        self.kv = Some(KvCache::new(
            self.backend,
            n_layers,
            n_ctx,
            head_dim,
            n_kv_heads,
        )?);
        Ok(())
    }

    pub fn forward_prefill(
        &mut self,
        prefill_embd: &[f32],
        n_tokens: usize,
        n_past: usize,
        cfg: &crate::ar::ArConfig,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if n_tokens == 0 {
            return Err(Error::Ggml("n_tokens must be > 0".into()));
        }
        if prefill_embd.len() != n_tokens * cfg.hidden_size as usize {
            return Err(Error::Ggml("prefill embedding length mismatch".into()));
        }
        if self.kv.is_none() {
            self.init_kv_cache(
                cfg.n_layers as usize,
                (n_past + n_tokens + 16).max(256),
                cfg.head_dim as usize,
                cfg.n_key_value_heads as usize,
            )?;
        }
        let kv = self.kv.as_ref().unwrap();
        if n_past + n_tokens > kv.n_ctx {
            return Err(Error::Ggml("context length exceeded".into()));
        }

        unsafe {
            let inp_name = CString::new("inp_prefill_embd").unwrap();
            let pos_name = CString::new("inp_pos").unwrap();
            let hidden_name = CString::new("hidden_states").unwrap();
            let logits_name = CString::new("logits").unwrap();
            let params = ffi::ggml_init_params {
                mem_size: self.compute_meta.len(),
                mem_buffer: self.compute_meta.as_mut_ptr().cast(),
                no_alloc: true,
            };
            let ctx0 = ffi::ggml_init(params);
            if ctx0.is_null() {
                return Err(Error::Ggml("failed to create prefill context".into()));
            }
            let gf = ffi::ggml_new_graph_custom(ctx0, 16384, false);
            let hidden = cfg.hidden_size as i64;
            let n_tokens_i = n_tokens as i64;
            let inp =
                ffi::ggml_new_tensor_2d(ctx0, ffi::ggml_type_GGML_TYPE_F32, hidden, n_tokens_i);
            ffi::ggml_set_name(inp, inp_name.as_ptr());
            ffi::ggml_set_input(inp);
            let pos = ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_tokens_i);
            ffi::ggml_set_name(pos, pos_name.as_ptr());
            ffi::ggml_set_input(pos);

            let mut inp_l = inp;
            let head_dim = cfg.head_dim as i64;
            let n_head = cfg.n_attention_heads as i64;
            let n_kv_head = cfg.n_key_value_heads as i64;
            let kq_scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

            for il in 0..cfg.n_layers as usize {
                let prefix = format!("talker.blk.{il}.");
                let mut cur = ffi::ggml_rms_norm(ctx0, inp_l, cfg.rms_norm_eps);
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
                q = ffi::ggml_reshape_3d(ctx0, q, head_dim, n_head, n_tokens_i);
                k = ffi::ggml_reshape_3d(ctx0, k, head_dim, n_kv_head, n_tokens_i);
                v = ffi::ggml_reshape_3d(ctx0, v, head_dim, n_kv_head, n_tokens_i);
                if self
                    .tensors
                    .contains_key(&(prefix.clone() + "attn_q_norm.weight"))
                {
                    q = ffi::ggml_rms_norm(ctx0, q, cfg.rms_norm_eps);
                    q = ffi::ggml_mul(
                        ctx0,
                        q,
                        self.tensor(&(prefix.clone() + "attn_q_norm.weight"))?,
                    );
                }
                if self
                    .tensors
                    .contains_key(&(prefix.clone() + "attn_k_norm.weight"))
                {
                    k = ffi::ggml_rms_norm(ctx0, k, cfg.rms_norm_eps);
                    k = ffi::ggml_mul(
                        ctx0,
                        k,
                        self.tensor(&(prefix.clone() + "attn_k_norm.weight"))?,
                    );
                }
                q = ffi::ggml_rope_ext(
                    ctx0,
                    q,
                    pos,
                    ptr::null_mut(),
                    cfg.head_dim as i32,
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
                    pos,
                    ptr::null_mut(),
                    cfg.head_dim as i32,
                    ffi::GGML_ROPE_TYPE_NEOX as i32,
                    0,
                    cfg.rope_theta,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                );
                let k_cache = kv.k_cache[il];
                let v_cache = kv.v_cache[il];
                let k_view = ffi::ggml_view_3d(
                    ctx0,
                    k_cache,
                    head_dim,
                    n_kv_head,
                    n_tokens_i,
                    (*k_cache).nb[1],
                    (*k_cache).nb[2],
                    n_past * (*k_cache).nb[2],
                );
                let v_view = ffi::ggml_view_3d(
                    ctx0,
                    v_cache,
                    head_dim,
                    n_kv_head,
                    n_tokens_i,
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
                cur = ffi::ggml_cont_2d(ctx0, kqv, n_head * head_dim, n_tokens_i);
                cur = ffi::ggml_mul_mat(
                    ctx0,
                    self.tensor(&(prefix.clone() + "attn_output.weight"))?,
                    cur,
                );
                cur = ffi::ggml_add(ctx0, cur, inp_l);
                let inp_ff = cur;
                cur = ffi::ggml_rms_norm(ctx0, inp_ff, cfg.rms_norm_eps);
                cur = ffi::ggml_mul(
                    ctx0,
                    cur,
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
            let mut cur = ffi::ggml_rms_norm(ctx0, inp_l, cfg.rms_norm_eps);
            cur = ffi::ggml_mul(ctx0, cur, self.tensor("talker.output_norm.weight")?);
            ffi::ggml_set_name(cur, hidden_name.as_ptr());
            ffi::ggml_set_output(cur);
            let logits = ffi::ggml_mul_mat(ctx0, self.tensor("talker.codec_head.weight")?, cur);
            ffi::ggml_set_name(logits, logits_name.as_ptr());
            ffi::ggml_set_output(logits);
            ffi::ggml_build_forward_expand(gf, logits);

            if !ffi::ggml_backend_sched_alloc_graph(self.sched, gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to allocate prefill graph".into()));
            }
            let inp_graph = ffi::ggml_graph_get_tensor(gf, inp_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                inp_graph,
                prefill_embd.as_ptr().cast(),
                0,
                std::mem::size_of_val(prefill_embd),
            );
            let positions = (0..n_tokens)
                .map(|idx| (n_past + idx) as i32)
                .collect::<Vec<_>>();
            let pos_graph = ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr());
            ffi::ggml_backend_tensor_set(
                pos_graph,
                positions.as_ptr().cast(),
                0,
                std::mem::size_of_val(positions.as_slice()),
            );
            let status = ffi::ggml_backend_sched_graph_compute(self.sched, gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!(
                    "prefill compute failed: status={status}"
                )));
            }
            let hidden_tensor = ffi::ggml_graph_get_tensor(gf, hidden_name.as_ptr());
            let logits_tensor = ffi::ggml_graph_get_tensor(gf, logits_name.as_ptr());
            let mut hidden_out = vec![0.0; n_tokens * cfg.hidden_size as usize];
            ffi::ggml_backend_tensor_get(
                hidden_tensor,
                hidden_out.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(hidden_out.as_slice()),
            );
            let mut logits_out = vec![0.0; cfg.codec_vocab_size as usize];
            ffi::ggml_backend_tensor_get(
                logits_tensor,
                logits_out.as_mut_ptr().cast(),
                (n_tokens - 1) * cfg.codec_vocab_size as usize * std::mem::size_of::<f32>(),
                std::mem::size_of_val(logits_out.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.sched);
            ffi::ggml_free(ctx0);
            Ok((hidden_out, logits_out))
        }
    }
}
