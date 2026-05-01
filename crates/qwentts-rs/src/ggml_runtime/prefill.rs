impl GgmlWeights {
    pub fn build_qwen_prefill_embeddings(
        &mut self,
        text_tokens: &[i32],
        speaker: Option<&[f32]>,
        language_id: Option<i32>,
        cfg: &crate::ar::ArConfig,
    ) -> Result<PrefillEmbeddings> {
        if text_tokens.len() < 4 {
            return Err(Error::Ggml(
                "need at least 4 text tokens for prefill".into(),
            ));
        }
        let h = cfg.hidden_size as usize;
        let special = [
            cfg.tts_bos_token_id as i32,
            cfg.tts_eos_token_id as i32,
            cfg.tts_pad_token_id as i32,
        ];
        let special_proj = self.project_text_tokens(&special, h)?;
        let tts_bos = &special_proj[0..h];
        let tts_eos = &special_proj[h..2 * h];
        let tts_pad = special_proj[2 * h..3 * h].to_vec();
        let role_embed = self.project_text_tokens(&text_tokens[..3], h)?;

        let codec_prefill_tokens = if let Some(language_id) = language_id {
            vec![
                cfg.codec_think_id as i32,
                cfg.codec_think_bos_id as i32,
                language_id,
                cfg.codec_think_eos_id as i32,
            ]
        } else {
            vec![
                cfg.codec_nothink_id as i32,
                cfg.codec_think_bos_id as i32,
                cfg.codec_think_eos_id as i32,
            ]
        };
        let codec_prefill =
            self.lookup_embedding_rows("talker.codec_embd.weight", &codec_prefill_tokens, h)?;
        let codec_tail = self.lookup_embedding_rows(
            "talker.codec_embd.weight",
            &[cfg.codec_pad_id as i32, cfg.codec_bos_id as i32],
            h,
        )?;

        let has_speaker = speaker.is_some();
        let codec_input_len = codec_prefill_tokens.len() + usize::from(has_speaker) + 2;
        let mut codec_input = vec![0.0f32; codec_input_len * h];
        let mut dst = 0;
        codec_input[..codec_prefill.len()].copy_from_slice(&codec_prefill);
        dst += codec_prefill_tokens.len();
        if let Some(speaker) = speaker {
            if speaker.len() != h {
                return Err(Error::Ggml("speaker embedding size mismatch".into()));
            }
            codec_input[dst * h..(dst + 1) * h].copy_from_slice(speaker);
            dst += 1;
        }
        codec_input[dst * h..dst * h + codec_tail.len()].copy_from_slice(&codec_tail);

        let codec_plus_overlay_len = codec_input_len - 1;
        let mut codec_plus_overlay = vec![0.0f32; codec_plus_overlay_len * h];
        for t in 0..codec_plus_overlay_len {
            let overlay = if t == codec_plus_overlay_len - 1 {
                tts_bos
            } else {
                &tts_pad
            };
            for i in 0..h {
                codec_plus_overlay[t * h + i] = overlay[i] + codec_input[t * h + i];
            }
        }

        let first_text = self.project_text_tokens(&text_tokens[3..4], h)?;
        let codec_bos = &codec_input[(codec_input_len - 1) * h..codec_input_len * h];
        let mut first_text_plus_codec_bos = vec![0.0f32; h];
        for i in 0..h {
            first_text_plus_codec_bos[i] = first_text[i] + codec_bos[i];
        }

        let prefill_len = 3 + codec_plus_overlay_len + 1;
        let mut prefill = vec![0.0f32; prefill_len * h];
        prefill[..role_embed.len()].copy_from_slice(&role_embed);
        prefill[3 * h..3 * h + codec_plus_overlay.len()].copy_from_slice(&codec_plus_overlay);
        prefill[(prefill_len - 1) * h..prefill_len * h].copy_from_slice(&first_text_plus_codec_bos);

        let trailing_token_count = text_tokens.len().saturating_sub(9);
        let trailing_proj = if trailing_token_count > 0 {
            self.project_text_tokens(&text_tokens[4..4 + trailing_token_count], h)?
        } else {
            Vec::new()
        };
        let trailing_len = trailing_token_count + 1;
        let mut trailing = vec![0.0f32; trailing_len * h];
        if trailing_token_count > 0 {
            trailing[..trailing_proj.len()].copy_from_slice(&trailing_proj);
        }
        trailing[(trailing_len - 1) * h..trailing_len * h].copy_from_slice(tts_eos);

        Ok(PrefillEmbeddings {
            prefill,
            trailing_text_hidden: trailing,
            tts_pad_embed: tts_pad,
            prefill_len,
            trailing_len,
        })
    }
}
