impl GgmlWeights {
    fn load(model: &GgufModel, specs: &[TensorSpec<'_>], backend: BackendChoice) -> Result<Self> {
        let mut weights = Self::allocate_context(specs.len(), backend)?;
        for spec in specs {
            weights.create_tensor(spec)?;
        }
        weights.allocate_buffer()?;
        weights.load_tensor_data(model, specs)?;
        Ok(weights)
    }

    fn load_existing_shapes(
        model: &GgufModel,
        specs: &[TensorSpec<'_>],
        backend: BackendChoice,
    ) -> Result<Self> {
        let mut weights = Self::allocate_context(specs.len(), backend)?;
        for spec in specs {
            let shape = model
                .tensor_shape_by_name(spec.name)?
                .ok_or_else(|| Error::MissingTensor(spec.name.into()))?;
            let shape = if spec.name.contains(".dwconv.weight") && shape.len() == 2 {
                vec![shape[0], 1, shape[1]]
            } else {
                shape
            };
            let spec = TensorSpec {
                name: spec.name,
                tensor_type: spec.tensor_type,
                shape: &shape,
                index: spec.index,
            };
            weights.create_tensor(&spec)?;
        }
        weights.allocate_buffer()?;
        weights.load_tensor_data(model, specs)?;
        Ok(weights)
    }

    fn allocate_context(n_tensors: usize, backend_choice: BackendChoice) -> Result<Self> {
        unsafe {
            ffi::ggml_backend_load_all();
            let backend = match backend_choice {
                BackendChoice::Best => ffi::ggml_backend_init_best(),
                BackendChoice::Cpu => ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    ptr::null(),
                ),
            };
            if backend.is_null() {
                return Err(Error::Ggml("failed to initialize ggml backend".into()));
            }
            let params = ffi::ggml_init_params {
                mem_size: n_tensors * ffi::ggml_tensor_overhead(),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                ffi::ggml_backend_free(backend);
                return Err(Error::Ggml("failed to initialize ggml context".into()));
            }
            Ok(Self {
                ctx,
                backend,
                backend_cpu: ptr::null_mut(),
                sched: ptr::null_mut(),
                buffer: ptr::null_mut(),
                compute_meta: Vec::new(),
                tensors: HashMap::new(),
                kv: None,
                code_kv: None,
            })
        }
    }

    fn create_tensor(&mut self, spec: &TensorSpec<'_>) -> Result<()> {
        let c_name =
            CString::new(spec.name).map_err(|_| Error::InvalidMetadataKey(spec.name.into()))?;
        let mut ne = [1_i64; 4];
        for (idx, dim) in spec.shape.iter().enumerate().take(4) {
            ne[idx] = *dim as i64;
        }
        let n_dims = spec.shape.len().max(1).min(4) as i32;
        let tensor =
            unsafe { ffi::ggml_new_tensor(self.ctx, spec.tensor_type, n_dims, ne.as_ptr()) };
        if tensor.is_null() {
            return Err(Error::Ggml(format!(
                "failed to create tensor {}",
                spec.name
            )));
        }
        unsafe {
            ffi::ggml_set_name(tensor, c_name.as_ptr());
        }
        self.tensors.insert(spec.name.to_string(), tensor);
        Ok(())
    }

    fn allocate_buffer(&mut self) -> Result<()> {
        self.buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(self.ctx, self.backend) };
        if self.buffer.is_null() {
            return Err(Error::Ggml("failed to allocate ggml weight buffer".into()));
        }
        unsafe {
            ffi::ggml_backend_buffer_set_usage(
                self.buffer,
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
        }
        Ok(())
    }

    fn load_tensor_data(&mut self, model: &GgufModel, specs: &[TensorSpec<'_>]) -> Result<()> {
        for spec in specs {
            let tensor = self.tensor(spec.name)?;
            let bytes = model.tensor_bytes(spec.index)?;
            let expected = unsafe { ffi::ggml_nbytes(tensor) };
            if bytes.len() != expected {
                return Err(Error::Ggml(format!(
                    "tensor {} data has {} bytes, ggml expects {}",
                    spec.name,
                    bytes.len(),
                    expected
                )));
            }
            unsafe {
                ffi::ggml_backend_tensor_set(tensor, bytes.as_ptr().cast(), 0, bytes.len());
            }
        }
        unsafe {
            ffi::ggml_backend_synchronize(self.backend);
        }
        self.init_scheduler()?;
        Ok(())
    }

    fn normalize_codec_codebooks(&mut self, model: &GgufModel) -> Result<()> {
        for (codebook, usage) in std::iter::once((
            "tok_dec.vq_first.0.codebook".to_string(),
            "tok_dec.vq_first.0.usage".to_string(),
        ))
        .chain((0..15).map(|idx| {
            (
                format!("tok_dec.vq_rest.{idx}.codebook"),
                format!("tok_dec.vq_rest.{idx}.usage"),
            )
        })) {
            let tensor = self.tensor(&codebook)?;
            if model.tensor_by_name(&usage)?.is_none() {
                continue;
            }
            let mut bytes = model
                .tensor_by_name(&codebook)?
                .ok_or_else(|| Error::MissingTensor(codebook.clone()))
                .and_then(|info| model.tensor_bytes(info.index))?;
            let usage = model.tensor_f32_by_name(&usage)?;
            let codebook_size = usage.len();
            if codebook_size == 0 || bytes.len() % (2 * codebook_size) != 0 {
                continue;
            }
            let codebook_dim = bytes.len() / 2 / codebook_size;
            for emb in 0..codebook_size {
                let inv = 1.0 / usage[emb].max(1e-5);
                for dim in 0..codebook_dim {
                    let offset = (emb * codebook_dim + dim) * 2;
                    let value =
                        f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
                            .to_f32();
                    let bits = f16::from_f32(value * inv).to_bits().to_le_bytes();
                    bytes[offset] = bits[0];
                    bytes[offset + 1] = bits[1];
                }
            }
            unsafe {
                ffi::ggml_backend_tensor_set(tensor, bytes.as_ptr().cast(), 0, bytes.len());
            }
        }
        Ok(())
    }

    fn init_scheduler(&mut self) -> Result<()> {
        unsafe {
            let device = ffi::ggml_backend_get_device(self.backend);
            let mut backends = vec![self.backend];
            if !device.is_null()
                && ffi::ggml_backend_dev_type(device)
                    != ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU
            {
                self.backend_cpu = ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    ptr::null(),
                );
                if !self.backend_cpu.is_null() {
                    backends.push(self.backend_cpu);
                }
            }
            self.sched = ffi::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                16384,
                false,
                true,
            );
            if self.sched.is_null() {
                return Err(Error::Ggml("failed to create ggml scheduler".into()));
            }
            self.compute_meta =
                vec![0; ffi::ggml_tensor_overhead() * 16384 + ffi::ggml_graph_overhead()];
            Ok(())
        }
    }
}
