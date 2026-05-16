//! SSE streaming fallback for OpenAI & Anthropic chat.
//!
//! Verifies the two paths added in `llm/provider.rs`:
//!   1. Non-stream returns 5xx → retry with `stream: true`, parse SSE → Ok(text).
//!   2. Non-stream returns 4xx → bubble up, no fallback.

use contribai::core::config::LlmConfig;
use contribai::llm::provider::{
    create_llm_provider_raw, AnthropicProvider, ChatMessage, LlmProvider, OpenAIProvider,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn cfg(provider: &str, base_url: &str) -> LlmConfig {
    LlmConfig {
        provider: provider.into(),
        api_key: "test-key".into(),
        model: "test-model".into(),
        temperature: 0.0,
        max_tokens: 64,
        base_url: Some(base_url.into()),
        vertex_project: String::new(),
        vertex_location: "global".into(),
        cache_enabled: false,
        cache_ttl_days: 7,
        small_model: None,
        copilot: false,
        fallback: vec![],
    }
}

/// Responder that branches on whether the request body has `"stream": true`.
struct StreamAware {
    non_stream: ResponseTemplate,
    streaming: ResponseTemplate,
    saw_stream: Arc<AtomicUsize>,
    saw_non_stream: Arc<AtomicUsize>,
}

impl Respond for StreamAware {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let body: serde_json::Value =
            serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
        if body["stream"].as_bool() == Some(true) {
            self.saw_stream.fetch_add(1, Ordering::SeqCst);
            self.streaming.clone()
        } else {
            self.saw_non_stream.fetch_add(1, Ordering::SeqCst);
            self.non_stream.clone()
        }
    }
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn openai_falls_back_to_sse_on_5xx() {
    let server = MockServer::start().await;

    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
               data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
               data: [DONE]\n\n";

    let saw_stream = Arc::new(AtomicUsize::new(0));
    let saw_non_stream = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(StreamAware {
            non_stream: ResponseTemplate::new(503).set_body_string("upstream down"),
            streaming: ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
            saw_stream: saw_stream.clone(),
            saw_non_stream: saw_non_stream.clone(),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new(&cfg("openai", &server.uri())).unwrap();
    let out = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect("SSE fallback should succeed");

    assert_eq!(out, "Hello world");
    assert_eq!(saw_non_stream.load(Ordering::SeqCst), 1);
    assert_eq!(saw_stream.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn openai_does_not_fallback_on_4xx() {
    let server = MockServer::start().await;

    let saw_stream = Arc::new(AtomicUsize::new(0));
    let saw_non_stream = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(StreamAware {
            non_stream: ResponseTemplate::new(400)
                .set_body_json(serde_json::json!({ "error": { "message": "bad input" }})),
            streaming: ResponseTemplate::new(500), // should never be hit
            saw_stream: saw_stream.clone(),
            saw_non_stream: saw_non_stream.clone(),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new(&cfg("openai", &server.uri())).unwrap();
    let err = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect_err("4xx must not trigger SSE fallback");

    let msg = err.to_string();
    assert!(msg.contains("400"), "got: {}", msg);
    assert!(msg.contains("bad input"), "got: {}", msg);
    assert_eq!(saw_stream.load(Ordering::SeqCst), 0, "stream must not be tried on 4xx");
    assert_eq!(saw_non_stream.load(Ordering::SeqCst), 1);
}

// ── Anthropic ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_falls_back_to_sse_on_5xx() {
    let server = MockServer::start().await;

    let sse = "event: content_block_delta\n\
               data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
               event: content_block_delta\n\
               data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n\
               event: message_stop\n\
               data: {\"type\":\"message_stop\"}\n\n";

    let saw_stream = Arc::new(AtomicUsize::new(0));
    let saw_non_stream = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(StreamAware {
            non_stream: ResponseTemplate::new(502).set_body_string("bad gateway"),
            streaming: ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
            saw_stream: saw_stream.clone(),
            saw_non_stream: saw_non_stream.clone(),
        })
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(&cfg("anthropic", &server.uri())).unwrap();
    let out = provider
        .chat(&[ChatMessage::user("hi")], Some("be terse"), None, None)
        .await
        .expect("SSE fallback should succeed");

    assert_eq!(out, "Hi there");
    assert_eq!(saw_non_stream.load(Ordering::SeqCst), 1);
    assert_eq!(saw_stream.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn anthropic_does_not_fallback_on_4xx() {
    let server = MockServer::start().await;

    let saw_stream = Arc::new(AtomicUsize::new(0));
    let saw_non_stream = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(StreamAware {
            non_stream: ResponseTemplate::new(401)
                .set_body_json(serde_json::json!({ "error": { "message": "invalid key" }})),
            streaming: ResponseTemplate::new(500),
            saw_stream: saw_stream.clone(),
            saw_non_stream: saw_non_stream.clone(),
        })
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(&cfg("anthropic", &server.uri())).unwrap();
    let err = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect_err("4xx must not trigger SSE fallback");

    let msg = err.to_string();
    assert!(msg.contains("401"), "got: {}", msg);
    assert!(msg.contains("invalid key"), "got: {}", msg);
    assert_eq!(saw_stream.load(Ordering::SeqCst), 0);
    assert_eq!(saw_non_stream.load(Ordering::SeqCst), 1);
}

// Sanity: the public factory still wires the providers used above.
#[tokio::test]
async fn factory_creates_openai_and_anthropic() {
    assert!(create_llm_provider_raw(&cfg("openai", "http://example.invalid")).is_ok());
    assert!(create_llm_provider_raw(&cfg("anthropic", "http://example.invalid")).is_ok());
}
