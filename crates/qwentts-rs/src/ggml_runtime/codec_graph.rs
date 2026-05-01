unsafe fn add_bias_2d(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    bias: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    ffi::ggml_add(ctx, x, bias)
}

unsafe fn add_bias_3d(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    bias: *mut ffi::ggml_tensor,
    channels: i64,
) -> *mut ffi::ggml_tensor {
    ffi::ggml_add(ctx, x, ffi::ggml_reshape_3d(ctx, bias, 1, channels, 1))
}

unsafe fn rms_mul(
    ctx: *mut ffi::ggml_context,
    x: *mut ffi::ggml_tensor,
    weight: *mut ffi::ggml_tensor,
    eps: f32,
) -> *mut ffi::ggml_tensor {
    ffi::ggml_mul(ctx, ffi::ggml_rms_norm(ctx, x, eps), weight)
}

impl GgmlWeights {
    unsafe fn codec_snake(
        &self,
        ctx: *mut ffi::ggml_context,
        x: *mut ffi::ggml_tensor,
        alpha: *mut ffi::ggml_tensor,
        beta: *mut ffi::ggml_tensor,
    ) -> *mut ffi::ggml_tensor {
        let seq_len = (*x).ne[0];
        let channels = (*x).ne[1];
        let batch = (*x).ne[2];
        let alpha = ffi::ggml_exp(ctx, alpha);
        let alpha = ffi::ggml_repeat(
            ctx,
            ffi::ggml_reshape_3d(ctx, alpha, 1, channels, 1),
            ffi::ggml_new_tensor_3d(ctx, ffi::ggml_type_GGML_TYPE_F32, seq_len, channels, batch),
        );
        let sin_sq = ffi::ggml_sqr(ctx, ffi::ggml_sin(ctx, ffi::ggml_mul(ctx, x, alpha)));
        let inv_beta = ffi::ggml_exp(ctx, ffi::ggml_scale(ctx, beta, -1.0));
        let inv_beta = ffi::ggml_repeat(
            ctx,
            ffi::ggml_reshape_3d(ctx, inv_beta, 1, channels, 1),
            ffi::ggml_new_tensor_3d(ctx, ffi::ggml_type_GGML_TYPE_F32, seq_len, channels, batch),
        );
        ffi::ggml_add(ctx, x, ffi::ggml_mul(ctx, sin_sq, inv_beta))
    }

    unsafe fn codec_pre_tfm_layer(
        &self,
        ctx: *mut ffi::ggml_context,
        x: *mut ffi::ggml_tensor,
        layer: usize,
        n_frames: usize,
        positions: *mut ffi::ggml_tensor,
        cfg: &crate::codec::CodecConfig,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("tok_dec.pre_tfm.blk.{layer}.");
        let head_dim = (cfg.latent_dim / cfg.n_heads) as i64;
        let mut normed = rms_mul(
            ctx,
            x,
            self.tensor(&(p.clone() + "attn_norm.weight"))?,
            cfg.rms_norm_eps,
        );
        let mut q = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "attn_q.weight"))?, normed);
        let mut k = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "attn_k.weight"))?, normed);
        let mut v = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "attn_v.weight"))?, normed);
        q = ffi::ggml_reshape_3d(ctx, q, head_dim, cfg.n_heads as i64, n_frames as i64);
        k = ffi::ggml_reshape_3d(ctx, k, head_dim, cfg.n_heads as i64, n_frames as i64);
        v = ffi::ggml_reshape_3d(ctx, v, head_dim, cfg.n_heads as i64, n_frames as i64);
        q = ffi::ggml_rope_ext(
            ctx,
            q,
            positions,
            ptr::null_mut(),
            head_dim as i32,
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
            ctx,
            k,
            positions,
            ptr::null_mut(),
            head_dim as i32,
            ffi::GGML_ROPE_TYPE_NEOX as i32,
            0,
            cfg.rope_theta,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        let q = ffi::ggml_permute(ctx, q, 0, 2, 1, 3);
        let k = ffi::ggml_permute(ctx, k, 0, 2, 1, 3);
        let mut v = ffi::ggml_permute(ctx, v, 0, 2, 1, 3);
        let mut kq = ffi::ggml_mul_mat(ctx, k, q);
        kq = ffi::ggml_scale(ctx, kq, 1.0 / (head_dim as f32).sqrt());
        kq = ffi::ggml_diag_mask_inf(ctx, kq, 0);
        kq = ffi::ggml_soft_max(ctx, kq);
        v = ffi::ggml_cont(ctx, ffi::ggml_transpose(ctx, v));
        let mut attn = ffi::ggml_mul_mat(ctx, v, kq);
        attn = ffi::ggml_permute(ctx, attn, 0, 2, 1, 3);
        attn = ffi::ggml_cont_2d(ctx, attn, cfg.latent_dim as i64, n_frames as i64);
        attn = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "attn_output.weight"))?, attn);
        attn = ffi::ggml_mul(ctx, attn, self.tensor(&(p.clone() + "attn_scale"))?);
        let x = ffi::ggml_add(ctx, x, attn);
        normed = rms_mul(
            ctx,
            x,
            self.tensor(&(p.clone() + "ffn_norm.weight"))?,
            cfg.rms_norm_eps,
        );
        let mut gate =
            ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "ffn_gate.weight"))?, normed);
        let up = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "ffn_up.weight"))?, normed);
        gate = ffi::ggml_silu(ctx, gate);
        let mut ffn = ffi::ggml_mul(ctx, gate, up);
        ffn = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "ffn_down.weight"))?, ffn);
        ffn = ffi::ggml_mul(ctx, ffn, self.tensor(&(p + "ffn_scale"))?);
        Ok(ffi::ggml_add(ctx, x, ffn))
    }

    unsafe fn codec_upsample_block(
        &self,
        ctx: *mut ffi::ggml_context,
        mut x: *mut ffi::ggml_tensor,
        block: usize,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("tok_dec.upsample.{block}.");
        let seq_len = (*x).ne[0];
        let in_channels = (*x).ne[1];
        let x2d = ffi::ggml_reshape_2d(ctx, x, seq_len, in_channels);
        let x2d = ffi::ggml_conv_transpose_1d(
            ctx,
            self.tensor(&(p.clone() + "conv.weight"))?,
            x2d,
            2,
            0,
            1,
        );
        let new_seq = (*x2d).ne[0];
        let channels = (*x2d).ne[1];
        x = ffi::ggml_reshape_3d(ctx, x2d, new_seq, channels, 1);
        x = add_bias_3d(ctx, x, self.tensor(&(p.clone() + "conv.bias"))?, channels);
        let residual = x;
        x = ffi::ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        let dw = self.tensor(&(p.clone() + "dwconv.weight"))?;
        x = ffi::ggml_conv_1d_dw(ctx, dw, x, 1, 0, 1);
        x = add_bias_3d(ctx, x, self.tensor(&(p.clone() + "dwconv.bias"))?, channels);
        x = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, x, 1, 0, 2, 3));
        x = ffi::ggml_norm(ctx, x, 1e-6);
        x = ffi::ggml_add(
            ctx,
            ffi::ggml_mul(ctx, x, self.tensor(&(p.clone() + "norm.weight"))?),
            self.tensor(&(p.clone() + "norm.bias"))?,
        );
        x = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "pwconv1.weight"))?, x);
        x = add_bias_2d(ctx, x, self.tensor(&(p.clone() + "pwconv1.bias"))?);
        x = ffi::ggml_gelu(ctx, x);
        x = ffi::ggml_mul_mat(ctx, self.tensor(&(p.clone() + "pwconv2.weight"))?, x);
        x = add_bias_2d(ctx, x, self.tensor(&(p.clone() + "pwconv2.bias"))?);
        x = ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, x, 1, 0, 2, 3));
        let gamma = ffi::ggml_repeat(
            ctx,
            ffi::ggml_reshape_3d(ctx, self.tensor(&(p + "gamma"))?, 1, channels, 1),
            ffi::ggml_new_tensor_3d(ctx, ffi::ggml_type_GGML_TYPE_F32, new_seq, channels, 1),
        );
        x = ffi::ggml_mul(ctx, x, gamma);
        Ok(ffi::ggml_add(ctx, residual, x))
    }

    unsafe fn codec_residual_block(
        &self,
        ctx: *mut ffi::ggml_context,
        mut x: *mut ffi::ggml_tensor,
        block: usize,
        res: usize,
        dilation: i32,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("tok_dec.dec.{block}.res.{res}.");
        let residual = x;
        x = self.codec_snake(
            ctx,
            x,
            self.tensor(&(p.clone() + "act1.alpha"))?,
            self.tensor(&(p.clone() + "act1.beta"))?,
        );
        let conv1 = self.tensor(&(p.clone() + "conv1.weight"))?;
        let out_channels = (*conv1).ne[2];
        x = ffi::ggml_pad_ext(ctx, x, 6 * dilation, 0, 0, 0, 0, 0, 0, 0);
        x = ffi::ggml_conv_1d(ctx, conv1, x, 1, 0, dilation);
        x = add_bias_3d(
            ctx,
            x,
            self.tensor(&(p.clone() + "conv1.bias"))?,
            out_channels,
        );
        x = self.codec_snake(
            ctx,
            x,
            self.tensor(&(p.clone() + "act2.alpha"))?,
            self.tensor(&(p.clone() + "act2.beta"))?,
        );
        let conv2 = self.tensor(&(p.clone() + "conv2.weight"))?;
        let out_channels = (*conv2).ne[2];
        x = ffi::ggml_conv_1d(ctx, conv2, x, 1, 0, 1);
        x = add_bias_3d(ctx, x, self.tensor(&(p + "conv2.bias"))?, out_channels);
        Ok(ffi::ggml_add(ctx, residual, x))
    }

    unsafe fn codec_decoder_block(
        &self,
        ctx: *mut ffi::ggml_context,
        mut x: *mut ffi::ggml_tensor,
        block: usize,
        rate: i32,
    ) -> Result<*mut ffi::ggml_tensor> {
        let p = format!("tok_dec.dec.{block}.");
        x = self.codec_snake(
            ctx,
            x,
            self.tensor(&(p.clone() + "snake.alpha"))?,
            self.tensor(&(p.clone() + "snake.beta"))?,
        );
        let seq_len = (*x).ne[0];
        let in_channels = (*x).ne[1];
        let conv_t = self.tensor(&(p.clone() + "conv_t.weight"))?;
        let out_channels = (*conv_t).ne[1];
        let kernel = (*conv_t).ne[0] as i32;
        let x2d = ffi::ggml_reshape_2d(ctx, x, seq_len, in_channels);
        let x2d = ffi::ggml_conv_transpose_1d(ctx, conv_t, x2d, rate, 0, 1);
        let new_seq = (*x2d).ne[0];
        x = ffi::ggml_reshape_3d(ctx, x2d, new_seq, out_channels, 1);
        let pad = (kernel - rate) as i64;
        x = ffi::ggml_cont(
            ctx,
            ffi::ggml_view_3d(
                ctx,
                x,
                new_seq - pad * 2,
                out_channels,
                1,
                (*x).nb[1],
                (*x).nb[2],
                pad as usize * (*x).nb[0],
            ),
        );
        x = add_bias_3d(ctx, x, self.tensor(&(p + "conv_t.bias"))?, out_channels);
        for (idx, dilation) in [1, 3, 9].into_iter().enumerate() {
            x = self.codec_residual_block(ctx, x, block, idx + 2, dilation)?;
        }
        Ok(x)
    }
}
