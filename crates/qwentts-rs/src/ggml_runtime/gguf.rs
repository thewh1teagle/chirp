use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::ptr;

use ggml_rs_sys as ffi;
use half::f16;

use crate::error::{Error, Result};
use crate::ggml_runtime::metadata::ModelSummary;
use crate::ggml_runtime::tensors::TensorType;

pub struct GgufModel {
    ctx: *mut ffi::gguf_context,
    meta_ctx: *mut ffi::ggml_context,
    path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub index: i64,
    pub name: String,
    pub tensor_type: TensorType,
    pub offset: usize,
    pub size: usize,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let c_path = path_to_cstring(path)?;
        let mut meta_ctx = ptr::null_mut();
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: &mut meta_ctx,
        };
        let ctx = unsafe { ffi::gguf_init_from_file(c_path.as_ptr(), params) };
        if ctx.is_null() {
            return Err(Error::GgufOpen(path.to_path_buf()));
        }

        Ok(Self {
            ctx,
            meta_ctx,
            path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            tensor_count: self.tensor_count(),
        }
    }

    pub fn tensor_count(&self) -> i64 {
        unsafe { ffi::gguf_get_n_tensors(self.ctx) }
    }

    pub fn tensor(&self, index: i64) -> Result<TensorInfo> {
        if index < 0 || index >= self.tensor_count() {
            return Err(Error::TensorIndex(index));
        }

        let name = unsafe { ffi::gguf_get_tensor_name(self.ctx, index) };
        if name.is_null() {
            return Err(Error::MissingTensorName(index));
        }
        let name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .to_string();
        let tensor_type = TensorType(unsafe { ffi::gguf_get_tensor_type(self.ctx, index) });
        let offset = unsafe { ffi::gguf_get_tensor_offset(self.ctx, index) };
        let size = unsafe { ffi::gguf_get_tensor_size(self.ctx, index) };

        Ok(TensorInfo {
            index,
            name,
            tensor_type,
            offset,
            size,
        })
    }

    pub fn data_offset(&self) -> usize {
        unsafe { ffi::gguf_get_data_offset(self.ctx) }
    }

    pub fn tensor_by_name(&self, name: &str) -> Result<Option<TensorInfo>> {
        let c_name = CString::new(name).map_err(|_| Error::InvalidMetadataKey(name.into()))?;
        let index = unsafe { ffi::gguf_find_tensor(self.ctx, c_name.as_ptr()) };
        if index < 0 {
            return Ok(None);
        }
        self.tensor(index).map(Some)
    }

    pub fn tensor_shape_by_name(&self, name: &str) -> Result<Option<Vec<usize>>> {
        let c_name = CString::new(name).map_err(|_| Error::InvalidMetadataKey(name.into()))?;
        let tensor = unsafe { ffi::ggml_get_tensor(self.meta_ctx, c_name.as_ptr()) };
        if tensor.is_null() {
            return Ok(None);
        }
        let tensor = unsafe { &*tensor };
        let mut shape = Vec::new();
        for dim in tensor.ne {
            if dim > 1 || shape.is_empty() {
                shape.push(dim as usize);
            }
        }
        while shape.len() > 1 && shape.last() == Some(&1) {
            shape.pop();
        }
        Ok(Some(shape))
    }

    pub fn tensor_bytes(&self, index: i64) -> Result<Vec<u8>> {
        let tensor = self.tensor(index)?;
        let absolute_offset = self
            .data_offset()
            .checked_add(tensor.offset)
            .ok_or_else(|| Error::InvalidTensorRange {
                path: self.path.clone(),
                offset: tensor.offset,
                size: tensor.size,
            })?;

        let mut file = File::open(&self.path)?;
        let file_len = file.metadata()?.len() as usize;
        let end =
            absolute_offset
                .checked_add(tensor.size)
                .ok_or_else(|| Error::InvalidTensorRange {
                    path: self.path.clone(),
                    offset: absolute_offset,
                    size: tensor.size,
                })?;
        if end > file_len {
            return Err(Error::InvalidTensorRange {
                path: self.path.clone(),
                offset: absolute_offset,
                size: tensor.size,
            });
        }

        let mut bytes = vec![0; tensor.size];
        file.seek(SeekFrom::Start(absolute_offset as u64))?;
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn tensor_checksum(&self, index: i64) -> Result<u64> {
        Ok(fnv1a64(&self.tensor_bytes(index)?))
    }

    pub fn tensor_f32_by_name(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensor_by_name(name)?
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        let bytes = self.tensor_bytes(tensor.index)?;
        match tensor.tensor_type.raw() {
            0 => {
                if bytes.len() % 4 != 0 {
                    return Err(Error::InvalidTensorRange {
                        path: self.path.clone(),
                        offset: tensor.offset,
                        size: tensor.size,
                    });
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            1 => {
                if bytes.len() % 2 != 0 {
                    return Err(Error::InvalidTensorRange {
                        path: self.path.clone(),
                        offset: tensor.offset,
                        size: tensor.size,
                    });
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                    .collect())
            }
            tensor_type => Err(Error::UnsupportedTensorType {
                name: name.into(),
                tensor_type,
            }),
        }
    }

    pub fn find_key(&self, key: &str) -> Result<Option<i64>> {
        let c_key = CString::new(key).map_err(|_| Error::InvalidMetadataKey(key.into()))?;
        let index = unsafe { ffi::gguf_find_key(self.ctx, c_key.as_ptr()) };
        Ok((index >= 0).then_some(index))
    }

    pub fn get_u32(&self, key: &str) -> Result<Option<u32>> {
        Ok(self
            .find_key(key)?
            .map(|index| unsafe { ffi::gguf_get_val_u32(self.ctx, index) }))
    }

    pub fn get_u32_or(&self, key: &str, default: u32) -> Result<u32> {
        Ok(self.get_u32(key)?.unwrap_or(default))
    }

    pub fn get_u32_any(&self, keys: &[&str], default: u32) -> Result<u32> {
        for key in keys {
            if let Some(value) = self.get_u32(key)? {
                return Ok(value);
            }
        }
        Ok(default)
    }

    pub fn get_f32(&self, key: &str) -> Result<Option<f32>> {
        Ok(self
            .find_key(key)?
            .map(|index| unsafe { ffi::gguf_get_val_f32(self.ctx, index) }))
    }

    pub fn get_f32_any(&self, keys: &[&str], default: f32) -> Result<f32> {
        for key in keys {
            if let Some(value) = self.get_f32(key)? {
                return Ok(value);
            }
        }
        Ok(default)
    }

    pub fn get_string(&self, key: &str) -> Result<Option<String>> {
        let Some(index) = self.find_key(key)? else {
            return Ok(None);
        };
        let value = unsafe { ffi::gguf_get_val_str(self.ctx, index) };
        if value.is_null() {
            return Ok(None);
        }
        Ok(Some(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .to_string(),
        ))
    }

    pub fn tensors(&self) -> impl Iterator<Item = Result<TensorInfo>> + '_ {
        (0..self.tensor_count()).map(|index| self.tensor(index))
    }

    pub fn first_tensors(&self, limit: usize) -> Result<Vec<TensorInfo>> {
        self.tensors().take(limit).collect()
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Drop for GgufModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                ffi::gguf_free(self.ctx);
            }
            if !self.meta_ctx.is_null() {
                ffi::ggml_free(self.meta_ctx);
            }
        }
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let bytes = path.to_string_lossy().into_owned();
    CString::new(bytes).map_err(|_| Error::NulPath(path.to_path_buf()))
}
