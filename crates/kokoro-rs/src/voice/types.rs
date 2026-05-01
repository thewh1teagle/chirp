#[derive(Debug, Clone)]
pub struct VoiceData {
    pub rows: usize,
    pub dims: usize,
    pub values: Vec<f32>,
}

impl VoiceData {
    pub fn style_for_token_count(&self, token_count: usize) -> &[f32] {
        let row = token_count.min(self.rows.saturating_sub(1));
        &self.values[row * self.dims..(row + 1) * self.dims]
    }
}
