use std::path::PathBuf;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::runtime::{kokoro_language, RuntimeParams};

use super::super::{
    dto::{LoadBody, LoadResponse},
    errors::write_error,
    state::{LoadParams, SharedServer},
    util::{empty_to_none, first_non_empty, first_non_zero, first_non_zero_float},
};

#[utoipa::path(
    post,
    path = "/v1/models/load",
    request_body = LoadBody,
    responses((status = 200, body = LoadResponse), (status = 400), (status = 500))
)]
pub async fn model_load(State(server): State<SharedServer>, body: Option<Json<LoadBody>>) -> Response {
    let mut body = body.map(|Json(body)| body).unwrap_or_default();
    set_default_runtime(&mut body);

    let params = match body.runtime.as_str() {
        "qwen" => qwen_load_params(body),
        "kokoro" => kokoro_load_params(body),
        _ => return write_error(StatusCode::BAD_REQUEST, "invalid_request", "unsupported runtime"),
    };
    let params = match params {
        Ok(params) => params,
        Err(message) => return write_error(StatusCode::BAD_REQUEST, "invalid_request", message),
    };

    if let Err(err) = server.load_model(params).await {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("failed to load model: {err}"),
        );
    }
    let inner = server.inner.lock().await;
    Json(LoadResponse {
        status: "loaded".into(),
        model: inner.model_name.clone(),
    })
    .into_response()
}

fn set_default_runtime(body: &mut LoadBody) {
    if body.runtime.is_empty() {
        body.runtime = std::env::var("CHIRP_RUNTIME").unwrap_or_default();
    }
    if body.runtime.is_empty() && !body.kokoro.model_path.is_empty() {
        body.runtime = "kokoro".into();
    }
    if body.runtime.is_empty() && !body.qwen.codec_path.is_empty() {
        body.runtime = "qwen".into();
    }
    if body.runtime.is_empty() {
        body.runtime = "qwen".into();
    }
}

fn qwen_load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    let model_path = first_non_empty([
        body.qwen.model_path,
        body.model_path,
        std::env::var("CHIRP_MODEL_PATH").unwrap_or_default(),
    ]);
    let codec_path = first_non_empty([
        body.qwen.codec_path,
        body.codec_path,
        std::env::var("CHIRP_CODEC_PATH").unwrap_or_default(),
    ]);
    if model_path.is_empty() || codec_path.is_empty() {
        return Err("qwen runtime requires model_path and codec_path");
    }
    Ok(LoadParams {
        runtime: "qwen".into(),
        qwen: RuntimeParams::Qwen {
            model_path: model_path.into(),
            codec_path: codec_path.into(),
            max_tokens: first_non_zero([body.qwen.max_tokens, body.max_tokens]),
            temperature: first_non_zero_float([body.qwen.temperature, body.temperature]),
            top_k: first_non_zero([body.qwen.top_k, body.top_k]),
        },
        kokoro: None,
    })
}

fn kokoro_load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    let model_path = first_non_empty([
        body.kokoro.model_path,
        std::env::var("CHIRP_KOKORO_MODEL_PATH").unwrap_or_default(),
    ]);
    let voices_path = first_non_empty([
        body.kokoro.voices_path,
        std::env::var("CHIRP_KOKORO_VOICES_PATH").unwrap_or_default(),
    ]);
    if model_path.is_empty() || voices_path.is_empty() {
        return Err("kokoro runtime requires model_path and voices_path");
    }
    if !body.kokoro.espeak_data_path.is_empty() {
        unsafe {
            std::env::set_var(
                "PIPER_ESPEAKNG_DATA_DIRECTORY",
                body.kokoro.espeak_data_path,
            );
        }
    }
    Ok(LoadParams {
        runtime: "kokoro".into(),
        qwen: RuntimeParams::Qwen {
            model_path: PathBuf::new(),
            codec_path: PathBuf::new(),
            max_tokens: 0,
            temperature: 0.0,
            top_k: 0,
        },
        kokoro: Some(RuntimeParams::Kokoro {
            model_path: model_path.into(),
            voices_path: voices_path.into(),
            voice: empty_to_none(body.kokoro.voice),
            language: Some(kokoro_language(&body.kokoro.language)),
            speed: body.kokoro.speed,
        }),
    })
}
