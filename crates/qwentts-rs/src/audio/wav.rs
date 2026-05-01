use std::path::Path;

use crate::error::Result;
use crate::AudioSamples;

pub fn write_mono_f32_wav(path: impl AsRef<Path>, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(sample.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn write_mono_i16_wav(path: impl AsRef<Path>, audio: &AudioSamples) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn read_mono_wav(path: impl AsRef<Path>) -> Result<AudioSamples> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        _ => {
            return Err(crate::error::Error::UnsupportedWav {
                sample_format: format!("{:?}", spec.sample_format),
                bits_per_sample: spec.bits_per_sample,
            })
        }
    };

    let samples = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };

    Ok(AudioSamples {
        sample_rate: spec.sample_rate,
        samples,
    })
}
