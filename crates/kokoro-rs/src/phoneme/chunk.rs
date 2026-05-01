use super::tokenize_phonemes;

const MAX_TEXT_CHUNK_CHARS: usize = 510;

pub fn chunk_text(text: &str) -> Vec<String> {
    let normalized = normalize_text_utf8(text);
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut pending = String::new();

    let push_chunk = |value: &mut String, chunks: &mut Vec<String>| {
        let chunk = trim_ascii(value);
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        value.clear();
    };

    let append_pending = |current: &mut String, pending: &mut String, chunks: &mut Vec<String>| {
        let clause = trim_ascii(pending);
        if clause.is_empty() {
            pending.clear();
            return;
        }
        let separator = usize::from(!current.is_empty());
        if !current.is_empty() && current.len() + separator + clause.len() > MAX_TEXT_CHUNK_CHARS {
            push_chunk(current, chunks);
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&clause);
        pending.clear();
    };

    for ch in normalized.chars() {
        pending.push(ch);
        if is_chunk_boundary(ch) {
            append_pending(&mut current, &mut pending, &mut chunks);
        }
    }

    append_pending(&mut current, &mut pending, &mut chunks);
    if !current.is_empty() {
        push_chunk(&mut current, &mut chunks);
    }
    chunks
}

pub fn pack_misaki_sentences(sentences: &[String], max_tokens: usize) -> Vec<String> {
    let mut batches = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        for part in split_misaki_by_token_limit(sentence, max_tokens) {
            let candidate = if current.is_empty() {
                part.clone()
            } else {
                format!("{current} {part}")
            };
            if tokenize_phonemes(&candidate).len() <= max_tokens {
                current = candidate;
            } else {
                if !current.is_empty() {
                    batches.push(std::mem::take(&mut current));
                }
                if tokenize_phonemes(&part).len() <= max_tokens {
                    current = part;
                } else {
                    batches.extend(split_misaki_by_token_limit(&part, max_tokens));
                }
            }
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn normalize_text_utf8(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_ascii_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        pending_space = false;
        out.push(ch);
    }
    trim_ascii(&out)
}

fn trim_ascii(text: &str) -> String {
    text.trim_matches(|ch: char| ch.is_ascii_whitespace())
        .to_string()
}

fn is_chunk_boundary(ch: char) -> bool {
    matches!(ch, '?' | '!' | '.' | ',')
}

fn split_misaki_by_token_limit(phonemes: &str, max_tokens: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut last_good = String::new();
    for ch in phonemes.chars() {
        let current_before = current.clone();
        current.push(ch);
        if tokenize_phonemes(&current).len() <= max_tokens {
            if matches!(ch, ' ' | ',' | '.' | '!' | '?' | ':' | ';') {
                last_good = current.clone();
            }
            continue;
        }
        if !last_good.is_empty() {
            chunks.push(last_good.clone());
            current = current[last_good.len()..].to_string();
            last_good.clear();
        } else {
            if !current_before.is_empty() {
                chunks.push(current_before);
            }
            current = ch.to_string();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
