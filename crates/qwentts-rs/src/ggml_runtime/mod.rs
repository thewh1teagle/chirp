use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::ptr;

use ggml_rs_sys as ffi;
use half::f16;

use crate::ar::ArTensorMap;
use crate::codec::CodecTensorMap;
use crate::error::{Error, Result};

pub mod gguf;
pub mod metadata;
pub mod tensors;

use gguf::GgufModel;

include!("weights.rs");
include!("prefill.rs");
include!("talker_graph.rs");
include!("code_predictor.rs");
include!("loading.rs");
include!("drop.rs");
include!("codec_graph.rs");
include!("types.rs");
include!("generate.rs");
include!("codec_decode.rs");
