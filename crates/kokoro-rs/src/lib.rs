pub mod audio;
pub mod error;
mod model;
mod phoneme;
mod voice;

pub use audio::wav::write_mono_i16_wav;
pub use audio::AudioSamples;
pub use error::{Error, Result};
pub use model::{Kokoro, KokoroConfig, SynthesizeRequest};
pub use voice::list_voices;
