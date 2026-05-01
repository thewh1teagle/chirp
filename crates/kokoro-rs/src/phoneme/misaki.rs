const REPLACEMENTS: &[(&str, &str)] = &[
    ("ʔˌn̩", "tᵊn"),
    ("a^ɪ", "I"),
    ("aɪ", "I"),
    ("a^ʊ", "W"),
    ("aʊ", "W"),
    ("d^ʒ", "ʤ"),
    ("dʒ", "ʤ"),
    ("e^ɪ", "A"),
    ("eɪ", "A"),
    ("t^ʃ", "ʧ"),
    ("tʃ", "ʧ"),
    ("ɔ^ɪ", "Y"),
    ("ɔɪ", "Y"),
    ("ə^l", "ᵊl"),
    ("ʔn", "tᵊn"),
    ("ʲO", "jO"),
    ("ʲQ", "jQ"),
    ("o^ʊ", "O"),
    ("oʊ", "O"),
    ("̃", ""),
    ("e", "A"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɐ", "ə"),
    ("ɚ", "əɹ"),
];

pub fn espeak_to_misaki(phonemes: &str, british: bool) -> String {
    let mut ps = phonemes.to_string();
    for (from, to) in REPLACEMENTS {
        ps = ps.replace(from, to);
    }
    ps = ps.replace("ɬ", "l").replace("ʔ", "t").replace("ʲ", "");
    ps = replace_syllabic_marker(ps);
    ps = ps.replace("̹", "");
    if british {
        ps = ps
            .replace("e^ə", "ɛː")
            .replace("iə", "ɪə")
            .replace("ə^ʊ", "Q")
            .replace("əʊ", "Q");
    } else {
        ps = ps
            .replace("o^ʊ", "O")
            .replace("oʊ", "O")
            .replace("ɜːɹ", "ɜɹ")
            .replace("ɜː", "ɜɹ")
            .replace("ɪə", "iə")
            .replace("ː", "");
    }
    ps.replace('^', "")
}

fn replace_syllabic_marker(input: String) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut out = String::new();
    let mut idx = 0;
    while idx < chars.len() {
        if idx + 1 < chars.len() && chars[idx + 1] == '\u{0329}' {
            out.push('ᵊ');
            out.push(chars[idx]);
            idx += 2;
        } else if chars[idx] != '\u{0329}' {
            out.push(chars[idx]);
            idx += 1;
        } else {
            idx += 1;
        }
    }
    out
}
