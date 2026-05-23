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
    assert_eq!(
        saw_stream.load(Ordering::SeqCst),
        0,
        "stream must not be tried on 4xx"
    );
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

// ── UTF-8 split across SSE chunk boundaries ───────────────────────────────────

/// Responder that streams the SSE body byte-by-byte, forcing every multi-byte
/// UTF-8 character to be split across reqwest chunk boundaries.
struct ByteByByte {
    body: Vec<u8>,
}

impl Respond for ByteByByte {
    fn respond(&self, _req: &Request) -> ResponseTemplate {
        // wiremock collapses set_body_bytes into one HTTP body, but reqwest's
        // bytes_stream will still chunk at the network layer. To guarantee
        // mid-codepoint splits, we don't rely on chunking — instead we emit a
        // payload where multi-byte chars fall on every position, and rely on
        // the parser to handle a single contiguous slice plus a forced "split"
        // produced by trimming the stream into two halves below.
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_bytes(self.body.clone())
    }
}

/// SSE payload exercise: Vietnamese + emoji + CJK so any mishandled split
/// would corrupt visible characters.
fn unicode_openai_sse() -> Vec<u8> {
    let frame1 = "data: {\"choices\":[{\"delta\":{\"content\":\"Xin chào 🌏\"}}]}\n\n";
    let frame2 = "data: {\"choices\":[{\"delta\":{\"content\":\" 你好\"}}]}\n\n";
    let done = "data: [DONE]\n\n";
    let mut out = Vec::new();
    out.extend_from_slice(frame1.as_bytes());
    out.extend_from_slice(frame2.as_bytes());
    out.extend_from_slice(done.as_bytes());
    out
}

#[tokio::test]
async fn openai_sse_handles_multibyte_utf8_payload() {
    let server = MockServer::start().await;

    let saw_stream = Arc::new(AtomicUsize::new(0));
    let saw_non_stream = Arc::new(AtomicUsize::new(0));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(StreamAware {
            non_stream: ResponseTemplate::new(503),
            streaming: ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_bytes(unicode_openai_sse()),
            saw_stream: saw_stream.clone(),
            saw_non_stream: saw_non_stream.clone(),
        })
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new(&cfg("openai", &server.uri())).unwrap();
    let out = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect("SSE fallback should succeed with multibyte text");

    assert_eq!(out, "Xin chào 🌏 你好");
}

/// Direct unit-style check: feed the byte parser a payload split mid-codepoint
/// and ensure no characters are dropped. This pins the bug Codex flagged.
#[tokio::test]
async fn drain_sse_events_preserves_split_codepoints() {
    use contribai::llm::provider::__test_only::drain_sse_events;

    let frame = "data: {\"choices\":[{\"delta\":{\"content\":\"é🌏你\"}}]}\n\n";
    let bytes = frame.as_bytes();

    // Force a split inside the 4-byte 🌏 (U+1F30F = F0 9F 8C 8F).
    let emoji_start = frame.find('🌏').expect("emoji present");
    let split = emoji_start + 2;

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&bytes[..split]);
    let first = drain_sse_events(&mut buf);
    assert!(first.is_empty(), "no complete event yet, got {:?}", first);

    buf.extend_from_slice(&bytes[split..]);
    let second = drain_sse_events(&mut buf);
    assert_eq!(second.len(), 1, "expected one event, got {:?}", second);

    let v: serde_json::Value = serde_json::from_str(&second[0]).unwrap();
    assert_eq!(
        v["choices"][0]["delta"]["content"].as_str().unwrap(),
        "é🌏你"
    );
}

// ── Proxies that always return SSE ────────────────────────────────────────────
//
// Some OpenAI-compatible routers (e.g. nip.io reverse proxies in front of
// Claude) ignore `stream:false` and respond with `Content-Type: text/event-stream`
// + 200. The provider must recognize the header and parse SSE instead of
// trying to decode JSON, otherwise every call fails with "JSON parse error".

#[tokio::test]
async fn openai_handles_sse_on_2xx_when_stream_false() {
    let server = MockServer::start().await;

    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
               data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
               data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = OpenAIProvider::new(&cfg("openai", &server.uri())).unwrap();
    let out = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect("must parse SSE on 2xx success when proxy ignores stream:false");

    assert_eq!(out, "Hi there");
}

#[tokio::test]
async fn anthropic_handles_sse_on_2xx_when_stream_false() {
    let server = MockServer::start().await;

    let sse = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
               data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n\
               data: {\"type\":\"message_stop\"}\n\n";

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(&cfg("anthropic", &server.uri())).unwrap();
    let out = provider
        .chat(&[ChatMessage::user("hi")], None, None, None)
        .await
        .expect("must parse SSE on 2xx success when proxy ignores stream:false");

    assert_eq!(out, "Hi there");
}
