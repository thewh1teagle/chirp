impl GgmlWeights {
    pub fn decode_codec_codes(
        &mut self,
        codes: &[i32],
        n_frames: usize,
        cfg: &crate::codec::CodecConfig,
    ) -> Result<Vec<f32>> {
        if codes.len() != n_frames * cfg.n_codebooks as usize {
            return Err(Error::Ggml("codec code length mismatch".into()));
        }
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: self
                    .compute_meta
                    .len()
                    .max(ffi::ggml_tensor_overhead() * 32768 + ffi::ggml_graph_overhead()),
                mem_buffer: self.compute_meta.as_mut_ptr().cast(),
                no_alloc: true,
            };
            let ctx0 = ffi::ggml_init(params);
            let gf = ffi::ggml_new_graph_custom(ctx0, 32768, false);
            let mut cb_tensors = Vec::with_capacity(16);
            for cb in 0..16 {
                let name = CString::new(format!("codes_cb{cb}")).unwrap();
                let tensor =
                    ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_frames as i64);
                ffi::ggml_set_name(tensor, name.as_ptr());
                ffi::ggml_set_input(tensor);
                cb_tensors.push((name, tensor));
            }

            let first_emb = ffi::ggml_get_rows(
                ctx0,
                self.tensor("tok_dec.vq_first.0.codebook")?,
                cb_tensors[0].1,
            );
            let first_emb_2d =
                ffi::ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim as i64, n_frames as i64);
            let first_proj_weight = ffi::ggml_reshape_2d(
                ctx0,
                self.tensor("tok_dec.vq_first.output_proj.weight")?,
                cfg.codebook_dim as i64,
                cfg.hidden_dim as i64,
            );
            let first_proj = ffi::ggml_mul_mat(ctx0, first_proj_weight, first_emb_2d);

            let rest_proj_weight = ffi::ggml_reshape_2d(
                ctx0,
                self.tensor("tok_dec.vq_rest.output_proj.weight")?,
                cfg.codebook_dim as i64,
                cfg.hidden_dim as i64,
            );
            let mut rest_proj: *mut ffi::ggml_tensor = ptr::null_mut();
            for cb in 0..15 {
                let cb_emb = ffi::ggml_get_rows(
                    ctx0,
                    self.tensor(&format!("tok_dec.vq_rest.{cb}.codebook"))?,
                    cb_tensors[cb + 1].1,
                );
                let cb_emb =
                    ffi::ggml_reshape_2d(ctx0, cb_emb, cfg.codebook_dim as i64, n_frames as i64);
                let cb_proj = ffi::ggml_mul_mat(ctx0, rest_proj_weight, cb_emb);
                rest_proj = if rest_proj.is_null() {
                    cb_proj
                } else {
                    ffi::ggml_add(ctx0, rest_proj, cb_proj)
                };
            }
            let latent_2d = ffi::ggml_add(ctx0, first_proj, rest_proj);
            let latent = ffi::ggml_reshape_3d(
                ctx0,
                ffi::ggml_cont(ctx0, ffi::ggml_transpose(ctx0, latent_2d)),
                n_frames as i64,
                cfg.hidden_dim as i64,
                1,
            );

            let mut cur =
                ffi::ggml_pad_ext(ctx0, ffi::ggml_cont(ctx0, latent), 2, 0, 0, 0, 0, 0, 0, 0);
            cur = ffi::ggml_conv_1d(ctx0, self.tensor("tok_dec.pre_conv.weight")?, cur, 1, 0, 1);
            cur = add_bias_3d(
                ctx0,
                cur,
                self.tensor("tok_dec.pre_conv.bias")?,
                cfg.latent_dim as i64,
            );
            cur = ffi::ggml_cont(
                ctx0,
                ffi::ggml_transpose(
                    ctx0,
                    ffi::ggml_reshape_2d(ctx0, cur, n_frames as i64, cfg.latent_dim as i64),
                ),
            );
            cur = ffi::ggml_mul_mat(ctx0, self.tensor("tok_dec.pre_tfm.input_proj.weight")?, cur);
            cur = add_bias_2d(ctx0, cur, self.tensor("tok_dec.pre_tfm.input_proj.bias")?);

            let pos_name = CString::new("positions").unwrap();
            let positions =
                ffi::ggml_new_tensor_1d(ctx0, ffi::ggml_type_GGML_TYPE_I32, n_frames as i64);
            ffi::ggml_set_name(positions, pos_name.as_ptr());
            ffi::ggml_set_input(positions);
            for layer in 0..cfg.n_pre_tfm_layers as usize {
                cur = self.codec_pre_tfm_layer(ctx0, cur, layer, n_frames, positions, cfg)?;
            }
            cur = rms_mul(
                ctx0,
                cur,
                self.tensor("tok_dec.pre_tfm.norm.weight")?,
                cfg.rms_norm_eps,
            );
            cur = ffi::ggml_mul_mat(
                ctx0,
                self.tensor("tok_dec.pre_tfm.output_proj.weight")?,
                cur,
            );
            cur = add_bias_2d(ctx0, cur, self.tensor("tok_dec.pre_tfm.output_proj.bias")?);
            cur = ffi::ggml_reshape_3d(
                ctx0,
                ffi::ggml_cont(ctx0, ffi::ggml_permute(ctx0, cur, 1, 0, 2, 3)),
                n_frames as i64,
                cfg.latent_dim as i64,
                1,
            );

            for block in 0..2 {
                cur = self.codec_upsample_block(ctx0, cur, block)?;
            }
            cur = ffi::ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
            cur = ffi::ggml_conv_1d(
                ctx0,
                self.tensor("tok_dec.dec.0.conv.weight")?,
                cur,
                1,
                0,
                1,
            );
            cur = add_bias_3d(
                ctx0,
                cur,
                self.tensor("tok_dec.dec.0.conv.bias")?,
                cfg.decoder_dim as i64,
            );
            for (idx, rate) in cfg.upsample_rates.iter().enumerate() {
                cur = self.codec_decoder_block(ctx0, cur, idx + 1, *rate as i32)?;
            }
            cur = self.codec_snake(
                ctx0,
                cur,
                self.tensor("tok_dec.dec.5.snake.alpha")?,
                self.tensor("tok_dec.dec.5.snake.beta")?,
            );
            cur = ffi::ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
            cur = ffi::ggml_conv_1d(
                ctx0,
                self.tensor("tok_dec.dec.6.conv.weight")?,
                cur,
                1,
                0,
                1,
            );
            cur = add_bias_3d(ctx0, cur, self.tensor("tok_dec.dec.6.conv.bias")?, 1);
            cur = ffi::ggml_tanh(ctx0, cur);
            cur = ffi::ggml_reshape_1d(ctx0, cur, (*cur).ne[0]);
            let audio_name = CString::new("audio").unwrap();
            ffi::ggml_set_name(cur, audio_name.as_ptr());
            ffi::ggml_set_output(cur);
            ffi::ggml_build_forward_expand(gf, cur);

            if !ffi::ggml_backend_sched_alloc_graph(self.sched, gf) {
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml("failed to allocate codec graph".into()));
            }
            let mut cb_codes = vec![0i32; n_frames];
            for cb in 0..16 {
                for f in 0..n_frames {
                    cb_codes[f] = codes[f * 16 + cb];
                }
                let tensor = ffi::ggml_graph_get_tensor(gf, cb_tensors[cb].0.as_ptr());
                ffi::ggml_backend_tensor_set(
                    tensor,
                    cb_codes.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(cb_codes.as_slice()),
                );
            }
            let positions_data = (0..n_frames as i32).collect::<Vec<_>>();
            ffi::ggml_backend_tensor_set(
                ffi::ggml_graph_get_tensor(gf, pos_name.as_ptr()),
                positions_data.as_ptr().cast(),
                0,
                std::mem::size_of_val(positions_data.as_slice()),
            );
            let status = ffi::ggml_backend_sched_graph_compute(self.sched, gf);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                ffi::ggml_backend_sched_reset(self.sched);
                ffi::ggml_free(ctx0);
                return Err(Error::Ggml(format!("codec decode failed: status={status}")));
            }
            let audio = ffi::ggml_graph_get_tensor(gf, audio_name.as_ptr());
            let n_samples = (*audio).ne[0] as usize;
            let mut samples = vec![0.0f32; n_samples];
            ffi::ggml_backend_tensor_get(
                audio,
                samples.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(samples.as_slice()),
            );
            ffi::ggml_backend_sched_reset(self.sched);
            ffi::ggml_free(ctx0);
            Ok(samples)
        }
    }
}
