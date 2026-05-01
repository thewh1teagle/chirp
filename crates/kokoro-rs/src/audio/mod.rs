pub mod wav;

#[derive(Debug, Clone)]
pub struct AudioSamples {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}
