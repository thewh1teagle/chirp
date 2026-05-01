impl GgmlWeights {
    pub fn forward_step(
        &mut self,
        step_embd: &[f32],
        n_past: usize,
        cfg: &crate::ar::ArConfig,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_prefill(step_embd, 1, n_past, cfg)
    }

    pub fn predict_codes_greedy(
        &mut self,
        hidden: &[f32],
        cb0: i32,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<i32>> {
        let h = cfg.hidden_size as usize;
        if hidden.len() != h {
            return Err(Error::Ggml("hidden size mismatch".into()));
        }
        let mut out = Vec::with_capacity((cfg.n_codebooks - 1) as usize);
        let cb0_emb = self.lookup_embedding_rows("talker.codec_embd.weight", &[cb0], h)?;
        let logits = self.code_pred_prefill(hidden, &cb0_emb, cfg)?;
        out.push(argmax(&logits) as i32);
        for step in 1..(cfg.n_codebooks as usize - 1) {
            let logits = self.code_pred_step(hidden, out[step - 1], step, cfg)?;
            out.push(argmax(&logits) as i32);
        }
        Ok(out)
    }

    pub fn predict_codes_autoregressive(
        &mut self,
        hidden: &[f32],
        cb0: i32,
        temperature: f32,
        top_k: i32,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<i32>> {
        let h = cfg.hidden_size as usize;
        if hidden.len() != h {
            return Err(Error::Ggml("hidden size mismatch".into()));
        }
        let mut rng = rand::rng();
        let mut out = Vec::with_capacity((cfg.n_codebooks - 1) as usize);
        let cb0_emb = self.lookup_embedding_rows("talker.codec_embd.weight", &[cb0], h)?;
        let mut logits = self.code_pred_prefill(hidden, &cb0_emb, cfg)?;
        out.push(sample_or_argmax_with_rng(
            &mut logits,
            temperature,
            top_k,
            &mut rng,
        )? as i32);
        for step in 1..(cfg.n_codebooks as usize - 1) {
            let mut logits = self.code_pred_step(hidden, out[step - 1], step, cfg)?;
            out.push(sample_or_argmax_with_rng(
                &mut logits,
                temperature,
                top_k,
                &mut rng,
            )? as i32);
        }
        Ok(out)
    }

    pub fn generate_codes(
        &mut self,
        text_tokens: &[i32],
        speaker: Option<&[f32]>,
        max_len: usize,
        options: GenerateOptions,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<i32>> {
        if max_len == 0 {
            return Ok(Vec::new());
        }
        let prefill =
            self.build_qwen_prefill_embeddings(text_tokens, speaker, options.language_id, cfg)?;
        self.init_kv_cache(
            cfg.n_layers as usize,
            prefill.prefill_len + max_len + 8,
            cfg.head_dim as usize,
            cfg.n_key_value_heads as usize,
        )?;
        let (hidden, mut logits) =
            self.forward_prefill(&prefill.prefill, prefill.prefill_len, 0, cfg)?;
        let h = cfg.hidden_size as usize;
        let mut last_hidden =
            hidden[(prefill.prefill_len - 1) * h..prefill.prefill_len * h].to_vec();
        let mut codes = Vec::with_capacity(max_len * cfg.n_codebooks as usize);
        let mut n_past = prefill.prefill_len;
        let mut generated_cb0_tokens = HashSet::new();
        let mut rng = rand::rng();

        for frame in 0..max_len {
            suppress_codec_tail(&mut logits, cfg);
            apply_repetition_penalty(
                &mut logits,
                &generated_cb0_tokens,
                options.repetition_penalty,
            );
            let cb0 = sample_or_argmax_with_rng(
                &mut logits,
                options.temperature,
                options.top_k,
                &mut rng,
            )? as i32;
            if cb0 == cfg.codec_eos_id as i32 {
                break;
            }
            generated_cb0_tokens.insert(cb0);

            let rest = self.predict_codes_autoregressive(
                &last_hidden,
                cb0,
                options.temperature,
                options.top_k,
                cfg,
            )?;
            codes.push(cb0);
            codes.extend(rest.iter().copied());
            if frame + 1 >= max_len {
                break;
            }

            let mut step_embd =
                self.lookup_embedding_rows("talker.codec_embd.weight", &[cb0], h)?;
            for (idx, code) in rest.iter().enumerate() {
                let name = format!("code_pred.codec_embd.{idx}.weight");
                let row = self.lookup_embedding_rows(&name, &[*code], h)?;
                for i in 0..h {
                    step_embd[i] += row[i];
                }
            }
            let trailing = if frame < prefill.trailing_len {
                &prefill.trailing_text_hidden[frame * h..(frame + 1) * h]
            } else {
                &prefill.tts_pad_embed
            };
            for i in 0..h {
                step_embd[i] += trailing[i];
            }
            let (hidden, next_logits) = self.forward_step(&step_embd, n_past, cfg)?;
            last_hidden = hidden[..h].to_vec();
            logits = next_logits;
            n_past += 1;
        }
        Ok(codes)
    }

    pub fn generate_codes_greedy(
        &mut self,
        text_tokens: &[i32],
        speaker: &[f32],
        max_len: usize,
        cfg: &crate::ar::ArConfig,
    ) -> Result<Vec<i32>> {
        let prefill = self.build_qwen_prefill_embeddings(text_tokens, Some(speaker), None, cfg)?;
        self.init_kv_cache(
            cfg.n_layers as usize,
            prefill.prefill_len + max_len + 8,
            cfg.head_dim as usize,
            cfg.n_key_value_heads as usize,
        )?;
        let (hidden, mut logits) =
            self.forward_prefill(&prefill.prefill, prefill.prefill_len, 0, cfg)?;
        let h = cfg.hidden_size as usize;
        let mut last_hidden =
            hidden[(prefill.prefill_len - 1) * h..prefill.prefill_len * h].to_vec();
        let mut codes = Vec::with_capacity(max_len * cfg.n_codebooks as usize);
        let mut n_past = prefill.prefill_len;
        for frame in 0..max_len {
            let suppress_start = cfg.codec_vocab_size as usize - 1024;
            for (idx, value) in logits.iter_mut().enumerate().skip(suppress_start) {
                if idx != cfg.codec_eos_id as usize {
                    *value = f32::NEG_INFINITY;
                }
            }
            let cb0 = argmax(&logits) as i32;
            if cb0 == cfg.codec_eos_id as i32 {
                break;
            }
            let rest = self.predict_codes_greedy(&last_hidden, cb0, cfg)?;
            codes.push(cb0);
            codes.extend(rest.iter().copied());
            if frame + 1 >= max_len {
                break;
            }
            let mut step_embd =
                self.lookup_embedding_rows("talker.codec_embd.weight", &[cb0], h)?;
            for (idx, code) in rest.iter().enumerate() {
                let name = format!("code_pred.codec_embd.{idx}.weight");
                let row = self.lookup_embedding_rows(&name, &[*code], h)?;
                for i in 0..h {
                    step_embd[i] += row[i];
                }
            }
            let trailing = if frame < prefill.trailing_len {
                &prefill.trailing_text_hidden[frame * h..(frame + 1) * h]
            } else {
                &prefill.tts_pad_embed
            };
            for i in 0..h {
                step_embd[i] += trailing[i];
            }
            let (hidden, next_logits) = self.forward_step(&step_embd, n_past, cfg)?;
            last_hidden = hidden[..h].to_vec();
            logits = next_logits;
            n_past += 1;
        }
        Ok(codes)
    }
}

fn suppress_codec_tail(logits: &mut [f32], cfg: &crate::ar::ArConfig) {
    let suppress_start = cfg.codec_vocab_size as usize - 1024;
    for (idx, value) in logits.iter_mut().enumerate().skip(suppress_start) {
        if idx != cfg.codec_eos_id as usize {
            *value = f32::NEG_INFINITY;
        }
    }
}

fn apply_repetition_penalty(
    logits: &mut [f32],
    generated_cb0_tokens: &HashSet<i32>,
    repetition_penalty: f32,
) {
    if repetition_penalty == 1.0 {
        return;
    }
    for &token in generated_cb0_tokens {
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
    }
}

fn sample_or_argmax_with_rng(
    logits: &mut [f32],
    temperature: f32,
    top_k: i32,
    rng: &mut impl rand::Rng,
) -> Result<usize> {
    if temperature <= 0.0 {
        return Ok(argmax(logits));
    }
    for value in logits.iter_mut() {
        *value /= temperature;
    }
    if top_k > 0 && (top_k as usize) < logits.len() {
        let mut scored = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| (value, idx))
            .collect::<Vec<_>>();
        scored.select_nth_unstable_by(top_k as usize - 1, |a, b| b.0.total_cmp(&a.0));
        let threshold = scored[top_k as usize - 1].0;
        for value in logits.iter_mut() {
            if *value < threshold {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    let max_logit = logits
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap_or(f32::NEG_INFINITY);
    let probs = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect::<Vec<_>>();
    let dist = rand::distr::weighted::WeightedIndex::new(&probs)
        .map_err(|err| Error::Ggml(format!("failed to sample logits: {err}")))?;
    Ok(rand::distr::Distribution::sample(&dist, rng))
}
