use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use super::super::{
    dto::SpeechBody,
    errors::write_error,
    state::SharedServer,
    util::{first_non_empty, path_option},
};

#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechBody,
    responses((status = 200, content_type = "audio/wav"), (status = 400), (status = 503), (status = 500))
)]
pub async fn speech(State(server): State<SharedServer>, Json(body): Json<SpeechBody>) -> Response {
    if body.input.is_empty() {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        );
    }
    if !body.response_format.is_empty() && body.response_format != "wav" {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "only wav response_format is supported",
        );
    }

    let Ok(tmp) = tempfile::Builder::new()
        .prefix("chirp-speech-")
        .suffix(".wav")
        .tempfile()
    else {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "failed to create temp output",
        );
    };

    let out_path = tmp.path().to_path_buf();
    let voice = first_non_empty([body.voice_reference, body.voice]);
    {
        let mut inner = server.inner.lock().await;
        let Some(ctx) = inner.ctx.as_mut() else {
            return write_error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "no model loaded");
        };
        if let Err(err) =
            ctx.synthesize_to_file(&body.input, path_option(&voice), &out_path, &body.language)
        {
            return write_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", err.to_string());
        }
    }

    let Ok(data) = std::fs::read(&out_path) else {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "failed to read output WAV",
        );
    };
    wav_response(data)
}

fn wav_response(data: Vec<u8>) -> Response {
    let mut response = Bytes::from(data).into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"speech.wav\""),
    );
    response
}
