impl Drop for GgmlWeights {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.sched.is_null() {
                ffi::ggml_backend_sched_free(self.sched);
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
            if !self.backend_cpu.is_null() {
                ffi::ggml_backend_free(self.backend_cpu);
            }
            if !self.backend.is_null() {
                ffi::ggml_backend_free(self.backend);
            }
        }
    }
}
