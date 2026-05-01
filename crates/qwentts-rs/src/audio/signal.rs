use rubato::{FftFixedIn, Resampler};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use crate::audio::wav::read_mono_wav;
use crate::error::{Error, Result};

pub const SPEAKER_SAMPLE_RATE: u32 = 24_000;
pub const SPEAKER_N_FFT: usize = 1024;
pub const SPEAKER_HOP: usize = 256;
pub const SPEAKER_N_MELS: usize = 128;

const PI: f32 = std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    pub channels: usize,
    pub frames: usize,
    pub values: Vec<f32>,
}

impl MelSpectrogram {
    pub fn at(&self, channel: usize, frame: usize) -> f32 {
        self.values[channel * self.frames + frame]
    }
}

pub fn read_wav_resampled(path: impl AsRef<std::path::Path>, sample_rate: u32) -> Result<Vec<f32>> {
    let audio = read_mono_wav(path)?;
    resample_mono(&audio.samples, audio.sample_rate, sample_rate)
}

pub fn resample_mono(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate || input.is_empty() {
        return Ok(input.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler =
        FftFixedIn::<f32>::new(src_rate as usize, dst_rate as usize, chunk_size, 1, 1)
            .map_err(|err| Error::AudioProcessing(err.to_string()))?;

    let mut output = Vec::with_capacity(
        ((input.len() as f64 * dst_rate as f64 / src_rate as f64).ceil() as usize) + 256,
    );
    let mut pos = 0;
    while pos + chunk_size <= input.len() {
        let channels = [&input[pos..pos + chunk_size]];
        let chunk = resampler
            .process(&channels, None)
            .map_err(|err| Error::AudioProcessing(err.to_string()))?;
        output.extend_from_slice(&chunk[0]);
        pos += chunk_size;
    }

    if pos < input.len() {
        let channels = [&input[pos..]];
        let chunk = resampler
            .process_partial(Some(&channels), None)
            .map_err(|err| Error::AudioProcessing(err.to_string()))?;
        output.extend_from_slice(&chunk[0]);
    }

    Ok(output)
}

pub fn speaker_mel_spectrogram_from_wav(
    path: impl AsRef<std::path::Path>,
) -> Result<MelSpectrogram> {
    let audio = read_wav_resampled(path, SPEAKER_SAMPLE_RATE)?;
    speaker_mel_spectrogram(&audio)
}

pub fn speaker_mel_spectrogram(audio: &[f32]) -> Result<MelSpectrogram> {
    if audio.is_empty() {
        return Err(Error::AudioProcessing(
            "speaker reference WAV is empty".into(),
        ));
    }

    let pad = (SPEAKER_N_FFT - SPEAKER_HOP) / 2;
    let padded_len = audio.len() + 2 * pad;
    if padded_len < SPEAKER_N_FFT {
        return Err(Error::AudioProcessing(
            "speaker reference WAV is too short".into(),
        ));
    }

    let mut padded = vec![0.0; padded_len];
    for (i, sample) in padded.iter_mut().enumerate() {
        *sample = audio[reflect_index(i as isize - pad as isize, audio.len())];
    }

    let frames = 1 + (padded.len() - SPEAKER_N_FFT) / SPEAKER_HOP;
    let window = hann_window(SPEAKER_N_FFT);
    let mel_basis = mel_basis(SPEAKER_N_MELS, SPEAKER_N_FFT / 2 + 1, SPEAKER_SAMPLE_RATE);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(SPEAKER_N_FFT);
    let mut fft_buf = vec![Complex32::ZERO; SPEAKER_N_FFT];
    let mut power = vec![0.0; (SPEAKER_N_FFT / 2 + 1) * frames];

    for frame_idx in 0..frames {
        let start = frame_idx * SPEAKER_HOP;
        for i in 0..SPEAKER_N_FFT {
            fft_buf[i] = Complex32::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut fft_buf);
        for bin in 0..=(SPEAKER_N_FFT / 2) {
            let value = fft_buf[bin];
            power[bin * frames + frame_idx] =
                (value.re * value.re + value.im * value.im + 1e-9).sqrt();
        }
    }

    let mut values = vec![0.0; SPEAKER_N_MELS * frames];
    let n_bins = SPEAKER_N_FFT / 2 + 1;
    for mel in 0..SPEAKER_N_MELS {
        for frame_idx in 0..frames {
            let mut sum = 0.0f64;
            for bin in 0..n_bins {
                sum +=
                    mel_basis[mel * n_bins + bin] as f64 * power[bin * frames + frame_idx] as f64;
            }
            values[mel * frames + frame_idx] = (sum as f32).max(1e-5).ln();
        }
    }

    Ok(MelSpectrogram {
        channels: SPEAKER_N_MELS,
        frames,
        values,
    })
}

fn reflect_index(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}

fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / len as f32).cos())
        .collect()
}

fn mel_basis(n_mels: usize, n_bins: usize, sample_rate: u32) -> Vec<f32> {
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f32 / 2.0);
    let mut edges = vec![0.0; n_mels + 2];
    for (i, edge) in edges.iter_mut().enumerate() {
        let a = i as f32 / (n_mels + 1) as f32;
        *edge = mel_to_hz(mel_min + a * (mel_max - mel_min));
    }

    let mut basis = vec![0.0; n_mels * n_bins];
    for mel in 0..n_mels {
        let lower = edges[mel];
        let center = edges[mel + 1];
        let upper = edges[mel + 2];
        let left = center - lower;
        let right = upper - center;
        let enorm = 2.0 / (upper - lower);
        for bin in 0..n_bins {
            let hz = bin as f32 * sample_rate as f32 / SPEAKER_N_FFT as f32;
            let weight = if hz >= lower && hz <= center && left > 0.0 {
                (hz - lower) / left
            } else if hz > center && hz <= upper && right > 0.0 {
                (upper - hz) / right
            } else {
                0.0
            };
            basis[mel * n_bins + bin] = weight.max(0.0) * enorm;
        }
    }
    basis
}

fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    if hz < 1000.0 {
        hz / F_SP
    } else {
        const MIN_LOG_MEL: f32 = 1000.0 / F_SP;
        const MIN_LOG_HZ: f32 = 1000.0;
        let logstep = 6.4f32.ln() / 27.0;
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_MEL: f32 = 1000.0 / F_SP;
    const MIN_LOG_HZ: f32 = 1000.0;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        let logstep = 6.4f32.ln() / 27.0;
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_shape_matches_cpp_framing() {
        let audio = vec![0.0; SPEAKER_SAMPLE_RATE as usize];
        let mel = speaker_mel_spectrogram(&audio).unwrap();
        assert_eq!(mel.channels, 128);
        assert_eq!(mel.frames, 93);
        assert_eq!(mel.values.len(), 128 * 93);
    }

    #[test]
    fn same_rate_resample_is_identity() {
        let input = vec![0.0, 0.25, -0.5, 1.0];
        assert_eq!(resample_mono(&input, 24_000, 24_000).unwrap(), input);
    }
}
