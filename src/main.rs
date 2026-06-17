// SPDX-License-Identifier: GPL-3.0-only
//! Whisper subprocess backend: serves the Super STT `/v1` contract over a
//! pathname Unix socket (`SUPER_STT_BACKEND_SOCKET`), loading the model from
//! `SUPER_STT_BACKEND_DIR/models/<name>`. Self-contained — no super-stt deps.

#![allow(clippy::doc_markdown)]

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::UnixListener;
use tokio::sync::mpsc;

use super_stt_backend_whisper::inference::WhisperEngine;

/// This backend implements exactly one provider (the model routing class every
/// one of its models declares). A `/v1/load` naming any other provider is
/// `400 invalid_model` per docs/protocol/backend/contract.md.
const PROVIDER: &str = "local_whisper";

#[derive(Clone, Copy)]
enum LoadState {
    Starting,
    Loading,
    Ready,
    Error,
}

impl LoadState {
    fn as_str(self) -> &'static str {
        match self {
            LoadState::Starting => "starting",
            LoadState::Loading => "loading",
            LoadState::Ready => "ready",
            LoadState::Error => "error",
        }
    }
}

struct Status {
    state: LoadState,
    model: Option<String>,
    device: Option<String>,
    reason: Option<String>,
}

impl Default for Status {
    fn default() -> Self {
        Self {
            state: LoadState::Starting,
            model: None,
            device: None,
            reason: None,
        }
    }
}

struct AppState {
    backend_dir: PathBuf,
    status: Mutex<Status>,
    engine: Mutex<Option<WhisperEngine>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let socket = std::env::var("SUPER_STT_BACKEND_SOCKET")
        .context("SUPER_STT_BACKEND_SOCKET must be set")?;
    let backend_dir =
        std::env::var("SUPER_STT_BACKEND_DIR").context("SUPER_STT_BACKEND_DIR must be set")?;

    let state = Arc::new(AppState {
        backend_dir: PathBuf::from(backend_dir),
        status: Mutex::new(Status::default()),
        engine: Mutex::new(None),
    });

    let app = router(state);

    if let Some(parent) = std::path::Path::new(&socket).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = std::fs::remove_file(&socket);
    let listener = UnixListener::bind(&socket).with_context(|| format!("bind {socket}"))?;
    log::info!("whisper backend serving /v1 on {socket}");

    loop {
        let (stream, _) = listener.accept().await?;
        let app = app.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = TowerToHyperService::new(app);
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}

/// Build the `/v1` router. Extracted from `main` so handlers can be exercised
/// in-process by the tests below without spawning the binary.
fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(get_status))
        .route("/v1/load", post(load))
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/cancel", post(cancel))
        // Audio payloads (f32 arrays as JSON) easily exceed the 2 MB default.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

async fn get_status(State(s): State<Arc<AppState>>) -> Json<Value> {
    let st = s.status.lock().unwrap();
    let mut out = json!({ "status": "success", "state": st.state.as_str() });
    if let Some(m) = &st.model {
        // The contract's model identity is (name, provider); this backend's
        // provider is fixed, so report it alongside the name.
        out["model"] = json!({ "name": m, "provider": PROVIDER });
    }
    if let Some(d) = &st.device {
        out["device"] = json!(d);
    }
    if let Some(r) = &st.reason {
        out["reason"] = json!(r);
    }
    Json(out)
}

#[derive(Deserialize)]
struct LoadReq {
    name: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    device: Option<String>,
}

async fn load(State(s): State<Arc<AppState>>, Json(req): Json<LoadReq>) -> impl IntoResponse {
    // Contract: an unimplemented (name, provider) is a client error. This
    // backend serves only PROVIDER, so a mismatched provider is `invalid_model`.
    if let Some(provider) = &req.provider
        && provider != PROVIDER
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_model" })),
        );
    }
    {
        let mut st = s.status.lock().unwrap();
        // Contract: reject a concurrent load. A model switch is a fresh load
        // after the daemon tears this backend down, so only an in-flight load on
        // this instance trips `already_loading`.
        if matches!(st.state, LoadState::Loading) {
            return (
                StatusCode::CONFLICT,
                Json(json!({ "status": "error", "message": "already_loading" })),
            );
        }
        st.state = LoadState::Loading;
        st.model = Some(req.name.clone());
        st.device = None;
        st.reason = None;
    }
    let dir = s.backend_dir.join("models").join(&req.name);
    let force_cpu = req.device.as_deref() == Some("cpu");
    let model_name = req.name.clone();
    let s2 = Arc::clone(&s);
    tokio::spawn(async move {
        let res =
            tokio::task::spawn_blocking(move || WhisperEngine::load(&dir, &model_name, force_cpu))
                .await;
        match res {
            Ok(Ok(engine)) => {
                let label = engine.device_label().to_string();
                *s2.engine.lock().unwrap() = Some(engine);
                let mut st = s2.status.lock().unwrap();
                st.device = Some(label);
                st.state = LoadState::Ready;
                log::info!("model loaded; ready");
            }
            Ok(Err(e)) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("{e:#}"));
                log::error!("model load failed: {e:#}");
            }
            Err(e) => {
                let mut st = s2.status.lock().unwrap();
                st.state = LoadState::Error;
                st.reason = Some(format!("load task panicked: {e}"));
            }
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
}

#[derive(Deserialize, Default)]
struct TranscribeOptions {
    #[serde(default)]
    stream_realtime: bool,
}

#[derive(Deserialize)]
struct TranscribeReq {
    audio_data: Vec<f32>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    options: TranscribeOptions,
}

async fn transcribe(
    State(s): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(req): Json<TranscribeReq>,
) -> Response {
    if !matches!(s.status.lock().unwrap().state, LoadState::Ready) {
        return (
            StatusCode::CONFLICT,
            Json(json!({ "status": "error", "message": "not_ready" })),
        )
            .into_response();
    }
    // Contract: empty `audio_data` is a client error, not an inference failure
    // (docs/protocol/backend/contract.md → 400 invalid_audio). Guarding here also
    // keeps an empty buffer out of mel extraction (0 frames → empty transcription).
    if req.audio_data.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "message": "invalid_audio" })),
        )
            .into_response();
    }
    let sample_rate = req.sample_rate.unwrap_or(16000);
    let audio = req.audio_data;
    let language = req.language;

    if req.options.stream_realtime {
        transcribe_streaming(s, audio, sample_rate, language)
    } else {
        transcribe_oneshot(s, audio, sample_rate, language).await
    }
}

async fn transcribe_oneshot(
    s: Arc<AppState>,
    audio: Vec<f32>,
    sample_rate: u32,
    language: Option<String>,
) -> Response {
    let s2 = Arc::clone(&s);
    let result = tokio::task::spawn_blocking(move || {
        let mut guard = s2.engine.lock().unwrap();
        let engine = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("engine not loaded"))?;
        engine.transcribe(&audio, sample_rate, language.as_deref())
    })
    .await;
    match result {
        Ok(Ok(text)) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "transcription": text })),
        )
            .into_response(),
        Ok(Err(e)) => {
            let msg = format!("{e:#}");
            let code = if msg.contains("unsupported_language") {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            let body = if msg.contains("unsupported_language") {
                json!({ "status": "error", "message": "unsupported_language" })
            } else {
                json!({ "status": "error", "message": "inference_failed", "detail": msg })
            };
            (code, Json(body)).into_response()
        }
        // A task panic is still an inference failure; the contract documents
        // `inference_failed` for 500, so report that (the panic is in `detail`).
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                json!({ "status": "error", "message": "inference_failed", "detail": format!("panicked: {e}") }),
            ),
        )
            .into_response(),
    }
}

// Owned `s` mirrors `transcribe_oneshot`'s signature for symmetry; clippy
// nags because the body only borrows it.
#[allow(clippy::needless_pass_by_value)]
fn transcribe_streaming(
    s: Arc<AppState>,
    audio: Vec<f32>,
    sample_rate: u32,
    language: Option<String>,
) -> Response {
    let (tx, mut rx) = mpsc::unbounded_channel::<SseFrame>();
    let s2 = Arc::clone(&s);
    let preview_tx = tx.clone();

    tokio::task::spawn_blocking(move || {
        let mut guard = s2.engine.lock().unwrap();
        let Some(engine) = guard.as_mut() else {
            let _ = tx.send(SseFrame::Error("engine not loaded".to_string()));
            return;
        };
        let result = engine.transcribe_streaming(&audio, sample_rate, language.as_deref(), |t| {
            let _ = preview_tx.send(SseFrame::Preview(t.to_string()));
        });
        match result {
            Ok(text) => {
                let _ = tx.send(SseFrame::Done(text));
            }
            Err(e) => {
                let _ = tx.send(SseFrame::Error(format!("{e:#}")));
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(frame) = rx.recv().await {
            let terminal = frame.is_terminal();
            yield Ok::<_, Infallible>(frame.encode());
            if terminal {
                break;
            }
        }
    };

    let body = Body::from_stream(stream);
    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    resp.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache"),
    );
    resp
}

enum SseFrame {
    Preview(String),
    Done(String),
    Error(String),
}

impl SseFrame {
    fn encode(&self) -> bytes::Bytes {
        let s = match self {
            Self::Preview(t) => format!(
                "event: preview\ndata: {}\n\n",
                serde_json::to_string(&json!({ "text": t })).unwrap()
            ),
            Self::Done(t) => format!(
                "event: done\ndata: {}\n\n",
                serde_json::to_string(&json!({ "transcription": t })).unwrap()
            ),
            Self::Error(m) => format!(
                "event: error\ndata: {}\n\n",
                serde_json::to_string(&json!({ "message": m })).unwrap()
            ),
        };
        bytes::Bytes::from(s)
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::Done(_) | Self::Error(_))
    }
}

async fn cancel() -> Json<Value> {
    Json(json!({ "status": "success", "message": "Cancelled" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            backend_dir: std::env::temp_dir(),
            status: Mutex::new(Status::default()),
            engine: Mutex::new(None),
        })
    }

    async fn json_body(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn ping_returns_pong() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/ping").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "pong");
    }

    #[tokio::test]
    async fn status_is_starting_before_load() {
        let resp = router(test_state())
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(json_body(resp).await["state"], "starting");
    }

    #[tokio::test]
    async fn cancel_acks() {
        let resp = router(test_state())
            .oneshot(Request::post("/v1/cancel").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(json_body(resp).await["message"], "Cancelled");
    }

    #[tokio::test]
    async fn transcribe_before_ready_conflicts() {
        let body =
            serde_json::to_vec(&json!({ "audio_data": [0.0f32, 0.1], "sample_rate": 16000 }))
                .unwrap();
        let resp = router(test_state())
            .oneshot(
                Request::post("/v1/transcribe")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "not_ready");
    }

    #[tokio::test]
    async fn transcribe_empty_audio_is_invalid() {
        // Ready state, but empty audio_data → 400 invalid_audio (contract), before
        // the engine is ever touched (so no model needed to exercise it).
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Ready;
        let body = serde_json::to_vec(&json!({ "audio_data": [], "sample_rate": 16000 })).unwrap();
        let resp = router(state)
            .oneshot(
                Request::post("/v1/transcribe")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["message"], "invalid_audio");
    }

    #[test]
    fn load_state_wire_strings() {
        assert_eq!(LoadState::Starting.as_str(), "starting");
        assert_eq!(LoadState::Loading.as_str(), "loading");
        assert_eq!(LoadState::Ready.as_str(), "ready");
        assert_eq!(LoadState::Error.as_str(), "error");
    }

    #[tokio::test]
    async fn status_includes_populated_fields() {
        let state = test_state();
        {
            let mut st = state.status.lock().unwrap();
            st.state = LoadState::Ready;
            st.model = Some("whisper-tiny".to_string());
            st.device = Some("cuda".to_string());
            st.reason = Some("recovered".to_string());
        }
        let resp = router(state)
            .oneshot(Request::get("/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let v = json_body(resp).await;
        assert_eq!(v["state"], "ready");
        assert_eq!(v["model"]["name"], "whisper-tiny");
        assert_eq!(v["model"]["provider"], "local_whisper");
        assert_eq!(v["device"], "cuda");
        assert_eq!(v["reason"], "recovered");
    }

    #[tokio::test]
    async fn load_rejects_mismatched_provider() {
        let body =
            serde_json::to_vec(&json!({ "name": "whisper-tiny", "provider": "openai" })).unwrap();
        let resp = router(test_state())
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(resp).await["message"], "invalid_model");
    }

    #[tokio::test]
    async fn load_rejects_concurrent_load() {
        let state = test_state();
        state.status.lock().unwrap().state = LoadState::Loading;
        let body = serde_json::to_vec(&json!({ "name": "whisper-tiny" })).unwrap();
        let resp = router(state)
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
        assert_eq!(json_body(resp).await["message"], "already_loading");
    }

    #[tokio::test]
    async fn load_sets_model_name_synchronously() {
        // The handler sets the model name before spawning the load task, and the
        // error path leaves it intact — so it's readable right after the 202.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "whisper-tiny", "device": "cpu" })).unwrap();
        let resp = router(Arc::clone(&state))
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        assert_eq!(
            state.status.lock().unwrap().model.as_deref(),
            Some("whisper-tiny")
        );
    }

    #[tokio::test]
    async fn load_missing_weights_transitions_to_error() {
        // backend_dir is a temp dir with no models/, so the load fails fast.
        let state = test_state();
        let body = serde_json::to_vec(&json!({ "name": "whisper-tiny", "device": "cpu" })).unwrap();
        let resp = router(Arc::clone(&state))
            .oneshot(
                Request::post("/v1/load")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        // The load runs in a spawned task; with no weights it must reach `error`.
        for _ in 0..250 {
            if matches!(state.status.lock().unwrap().state, LoadState::Error) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("load did not reach error state");
    }

    #[test]
    fn sse_frame_encoding_and_terminality() {
        // Preview frames are non-terminal; done/error frames end the stream.
        let preview = SseFrame::Preview("hi".to_string());
        assert!(!preview.is_terminal());
        let s = String::from_utf8(preview.encode().to_vec()).unwrap();
        assert!(s.starts_with("event: preview\n"), "{s:?}");
        assert!(s.contains(r#""text":"hi""#), "{s:?}");
        assert!(s.ends_with("\n\n"), "{s:?}");

        let done = SseFrame::Done("final text".to_string());
        assert!(done.is_terminal());
        let s = String::from_utf8(done.encode().to_vec()).unwrap();
        assert!(s.starts_with("event: done\n"), "{s:?}");
        assert!(s.contains(r#""transcription":"final text""#), "{s:?}");

        let err = SseFrame::Error("boom".to_string());
        assert!(err.is_terminal());
        let s = String::from_utf8(err.encode().to_vec()).unwrap();
        assert!(s.starts_with("event: error\n"), "{s:?}");
        assert!(s.contains(r#""message":"boom""#), "{s:?}");
    }
}
