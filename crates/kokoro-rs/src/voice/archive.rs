use std::io::{Cursor, Read};
use std::path::Path;

use npyz::{NpyFile, Order};
use zip::ZipArchive;

use crate::{Error, Result};

use super::VoiceData;

pub fn load_voice(path: impl AsRef<Path>, voice: &str) -> Result<VoiceData> {
    let entry = if voice.ends_with(".npy") {
        voice.to_string()
    } else {
        format!("{voice}.npy")
    };
    let bytes =
        read_zip_entry(path.as_ref(), &entry)?.ok_or_else(|| Error::MissingVoice(entry.clone()))?;
    parse_voice_npy(&bytes, voice)
}

pub fn list_voices(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut voices = Vec::new();
    for index in 0..archive.len() {
        let file = archive.by_index(index)?;
        let name = file.name();
        if let Some(stem) = name.strip_suffix(".npy") {
            voices.push(stem.to_string());
        }
    }
    voices.sort();
    Ok(voices)
}

fn read_zip_entry(path: &Path, entry: &str) -> Result<Option<Vec<u8>>> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let Ok(mut file) = archive.by_name(entry) else {
        return Ok(None);
    };
    let mut bytes = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

fn parse_voice_npy(bytes: &[u8], voice: &str) -> Result<VoiceData> {
    let npy = NpyFile::new(Cursor::new(bytes)).map_err(|err| Error::InvalidVoice {
        voice: voice.into(),
        reason: err.to_string(),
    })?;
    if npy.order() == Order::Fortran {
        return Err(Error::InvalidVoice {
            voice: voice.into(),
            reason: "fortran-order voice arrays are not supported".into(),
        });
    }
    let shape = npy.shape().to_vec();
    let (rows, dims) = match shape.as_slice() {
        [rows, dims] => (*rows as usize, *dims as usize),
        [rows, 1, dims] => (*rows as usize, *dims as usize),
        _ => {
            return Err(Error::InvalidVoice {
                voice: voice.into(),
                reason: format!("unsupported shape {shape:?}"),
            })
        }
    };
    if rows == 0 || dims == 0 {
        return Err(Error::InvalidVoice {
            voice: voice.into(),
            reason: "empty voice array".into(),
        });
    }
    let values = npy.into_vec::<f32>().map_err(|err| Error::InvalidVoice {
        voice: voice.into(),
        reason: err.to_string(),
    })?;
    if values.len() != rows * dims {
        return Err(Error::InvalidVoice {
            voice: voice.into(),
            reason: format!(
                "data length {} does not match shape {shape:?}",
                values.len()
            ),
        });
    }
    Ok(VoiceData { rows, dims, values })
}
