use super::{otlp_exporter, OtlpProtocol};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Response, Server};
use opentelemetry::trace::{Span, SpanId, TraceId, Tracer, TracerProvider};
use opentelemetry::KeyValue;
use opentelemetry_proto::tonic::collector::trace::v1::{
    trace_service_server::{TraceService, TraceServiceServer},
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::any_value::Value;
use opentelemetry_sdk::{trace, Resource};
use prost::Message;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_rustls::{rustls, TlsAcceptor};
use tokio_stream::wrappers::TcpListenerStream;

const TRACE_ID: [u8; 16] = [0x12; 16];
const SPAN_ID: [u8; 8] = [0x34; 8];
const TEST_CASE: &str = "TEI_OTLP_TEST_CASE";
const TLS_DIRECTORY: &str = "TEI_OTLP_TEST_TLS_DIRECTORY";

// Each child gets its own exporter environment, even when Cargo runs tests in parallel.
fn run_isolated(case: &str) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        "logging::tests::otlp_export_child",
        "--ignored",
        "--nocapture",
    ]);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("OTEL_") {
            command.env_remove(name);
        }
    }
    command
        .env(TEST_CASE, case)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .env(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "authorization=Bearer%20generic%2Ctoken%3Dvalue%2520,x-generic=generic",
        );
    if case.ends_with("signal") {
        command.env(
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
            "authorization=Bearer%20trace%2Ctoken%3Dvalue%2520,x-signal=trace",
        );
    }
    let _certificates = if case.starts_with("https-") {
        let directory = tempfile::tempdir().unwrap();
        let certificate = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_owned()]).unwrap();
        std::fs::write(
            directory.path().join("certificate.pem"),
            certificate.serialize_pem().unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("certificate.der"),
            certificate.serialize_der().unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("key.der"),
            certificate.serialize_private_key_der(),
        )
        .unwrap();
        command
            .env("SSL_CERT_FILE", directory.path().join("certificate.pem"))
            .env(TLS_DIRECTORY, directory.path());
        Some(directory)
    } else {
        None
    };
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{case} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn http_exports_protobuf_to_the_trace_path() {
    run_isolated("http-generic");
}

#[test]
fn http_trace_headers_replace_generic_headers() {
    run_isolated("http-signal");
}

#[test]
fn http_generic_endpoint_overrides_the_cli_endpoint() {
    run_isolated("http-generic-endpoint");
}

#[test]
fn http_trace_endpoint_overrides_generic_and_cli_endpoints() {
    run_isolated("http-trace-endpoint");
}

#[test]
fn https_exports_with_a_trusted_certificate() {
    run_isolated("https-generic");
}

#[test]
fn grpc_exports_spans_with_decoded_headers() {
    run_isolated("grpc-generic");
}

#[test]
fn grpc_trace_headers_replace_generic_headers() {
    run_isolated("grpc-signal");
}

#[test]
fn invalid_http_endpoints_do_not_expose_credentials() {
    for endpoint in [
        "ftp://user:password@collector?token=secret",
        "http:///v1/traces?token=secret",
        "http://user:password@collector:invalid?token=secret",
        "http://collector:99999?token=secret",
        "http://collector/v1/traces?token=secret#fragment",
    ] {
        let error = otlp_exporter(endpoint, OtlpProtocol::HttpProto)
            .unwrap_err()
            .to_string();
        assert!(!error.contains("secret"), "credentials leaked: {error}");
        assert!(!error.contains("password"), "credentials leaked: {error}");
        assert!(!error.contains(endpoint), "endpoint leaked: {error}");
    }
}

#[test]
#[ignore = "invoked by the exporter tests with an isolated environment"]
fn otlp_export_child() {
    let case = std::env::var(TEST_CASE).expect("exporter test case must be set");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    // Endpoint variables belong only to this child and are set before starting Tokio.
    if case.ends_with("-endpoint") {
        let address = listener.local_addr().unwrap();
        std::env::set_var(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            format!("http://{address}/environment"),
        );
        if case == "http-trace-endpoint" {
            std::env::set_var(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                format!("http://{address}/custom-traces?tenant=blue"),
            );
        }
    }
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(20), async {
                match case.as_str() {
                    "http-generic"
                    | "http-signal"
                    | "http-generic-endpoint"
                    | "http-trace-endpoint" => check_http(&case, listener).await,
                    "https-generic" => check_https(&case).await,
                    "grpc-generic" | "grpc-signal" => check_grpc(&case).await,
                    _ => panic!("unknown exporter test case: {case}"),
                }
            })
            .await
            .expect("exporter did not finish before the timeout");
        });
}

type HttpExport = (String, BTreeMap<String, String>, ExportTraceServiceRequest);

async fn receive_http(
    request: hyper::Request<Body>,
    sender: mpsc::Sender<HttpExport>,
) -> Result<Response<Body>, Infallible> {
    assert_eq!(request.method(), hyper::Method::POST);
    let path = request.uri().to_string();
    let headers = request
        .headers()
        .iter()
        .map(|(key, value)| (key.as_str().to_owned(), value.to_str().unwrap().to_owned()))
        .collect();
    let body = hyper::body::to_bytes(request.into_body()).await.unwrap();
    let message = ExportTraceServiceRequest::decode(body).unwrap();
    sender.send((path, headers, message)).await.unwrap();
    Ok(Response::builder()
        .header("content-type", "application/x-protobuf")
        .body(Body::from(
            ExportTraceServiceResponse::default().encode_to_vec(),
        ))
        .unwrap())
}

async fn export_span(endpoint: &str, protocol: OtlpProtocol) {
    let exporter = otlp_exporter(endpoint, protocol)
        .unwrap()
        .build_span_exporter()
        .unwrap();
    let provider = trace::TracerProvider::builder()
        .with_config(
            trace::config().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                "tei-export-test",
            )])),
        )
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .build();
    let tracer = provider.tracer("tei-export-test");
    tracer
        .span_builder("embedding-request")
        .with_trace_id(TraceId::from_bytes(TRACE_ID))
        .with_span_id(SpanId::from_bytes(SPAN_ID))
        .start(&tracer)
        .end();
    drop(tracer);
    // Flush and shutdown are blocking; the runtime must keep driving the exporter.
    tokio::task::spawn_blocking(move || {
        for result in provider.force_flush() {
            result.unwrap();
        }
        drop(provider);
    })
    .await
    .unwrap();
}

fn assert_export(request: &ExportTraceServiceRequest) {
    assert_eq!(request.resource_spans.len(), 1);
    let resource_spans = &request.resource_spans[0];
    let service_name = resource_spans
        .resource
        .as_ref()
        .unwrap()
        .attributes
        .iter()
        .find(|attribute| attribute.key == "service.name")
        .unwrap();
    assert_eq!(
        service_name.value.as_ref().unwrap().value,
        Some(Value::StringValue("tei-export-test".to_owned()))
    );
    assert_eq!(resource_spans.scope_spans.len(), 1);
    let spans = &resource_spans.scope_spans[0].spans;
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].name, "embedding-request");
    assert_eq!(spans[0].trace_id, TRACE_ID);
    assert_eq!(spans[0].span_id, SPAN_ID);
}

fn assert_headers(headers: &BTreeMap<String, String>, case: &str) {
    if case.ends_with("signal") {
        assert_eq!(headers["authorization"], "Bearer trace,token=value%20");
        assert_eq!(headers["x-signal"], "trace");
        assert!(!headers.contains_key("x-generic"));
    } else {
        assert_eq!(headers["authorization"], "Bearer generic,token=value%20");
        assert_eq!(headers["x-generic"], "generic");
        assert!(!headers.contains_key("x-signal"));
    }
}

async fn check_http(case: &str, listener: std::net::TcpListener) {
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, mut receiver) = mpsc::channel(1);
    let (shutdown, shutdown_signal) = oneshot::channel();
    let service = make_service_fn(move |_| {
        let sender = sender.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |request: hyper::Request<Body>| {
                receive_http(request, sender.clone())
            }))
        }
    });
    let server = tokio::spawn(
        Server::from_tcp(listener)
            .unwrap()
            .serve(service)
            .with_graceful_shutdown(async {
                let _ = shutdown_signal.await;
            }),
    );
    let paths: &[(&str, &str)] = match case {
        "http-generic-endpoint" => &[("/cli", "/environment/v1/traces")],
        "http-trace-endpoint" => &[("/cli", "/custom-traces?tenant=blue")],
        _ => &[
            ("", "/v1/traces"),
            ("/", "/v1/traces"),
            ("/api/public/otel", "/api/public/otel/v1/traces"),
            ("/api/public/otel/", "/api/public/otel/v1/traces"),
            ("/api/public/otel/v1/traces", "/api/public/otel/v1/traces"),
            ("/api/public/otel/v1/traces/", "/api/public/otel/v1/traces"),
            ("/otel?tenant=blue", "/otel/v1/traces?tenant=blue"),
        ],
    };
    for &(suffix, expected_path) in paths {
        export_span(
            &format!("http://{address}{suffix}"),
            OtlpProtocol::HttpProto,
        )
        .await;
        let (path, headers, message) = receiver.recv().await.unwrap();
        assert_eq!(path, expected_path);
        assert_eq!(headers["content-type"], "application/x-protobuf");
        assert_headers(&headers, case);
        assert_export(&message);
    }
    shutdown.send(()).unwrap();
    server.await.unwrap().unwrap();
}

async fn check_https(case: &str) {
    let directory = std::path::PathBuf::from(std::env::var_os(TLS_DIRECTORY).unwrap());
    let certificate =
        rustls::Certificate(std::fs::read(directory.join("certificate.der")).unwrap());
    let private_key = rustls::PrivateKey(std::fs::read(directory.join("key.der")).unwrap());
    let config = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(vec![certificate], private_key)
        .unwrap();
    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, mut receiver) = mpsc::channel(1);
    let server = tokio::spawn(async move {
        let (connection, _) = listener.accept().await.unwrap();
        let connection = acceptor.accept(connection).await.unwrap();
        hyper::server::conn::Http::new()
            .serve_connection(
                connection,
                service_fn(move |request| receive_http(request, sender.clone())),
            )
            .await
            .unwrap();
    });
    export_span(&format!("https://{address}/otel"), OtlpProtocol::HttpProto).await;
    let (path, headers, message) = receiver.recv().await.unwrap();
    assert_eq!(path, "/otel/v1/traces");
    assert_eq!(headers["content-type"], "application/x-protobuf");
    assert_headers(&headers, case);
    assert_export(&message);
    server.await.unwrap();
}

struct TraceCollector {
    sender: mpsc::Sender<(BTreeMap<String, String>, ExportTraceServiceRequest)>,
}

#[tonic::async_trait]
impl TraceService for TraceCollector {
    async fn export(
        &self,
        request: tonic::Request<ExportTraceServiceRequest>,
    ) -> Result<tonic::Response<ExportTraceServiceResponse>, tonic::Status> {
        let headers = request
            .metadata()
            .iter()
            .filter_map(|entry| match entry {
                tonic::metadata::KeyAndValueRef::Ascii(key, value) => {
                    Some((key.as_str().to_owned(), value.to_str().unwrap().to_owned()))
                }
                tonic::metadata::KeyAndValueRef::Binary(_, _) => None,
            })
            .collect();
        self.sender
            .send((headers, request.into_inner()))
            .await
            .unwrap();
        Ok(tonic::Response::new(ExportTraceServiceResponse::default()))
    }
}

async fn check_grpc(case: &str) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, mut receiver) = mpsc::channel(1);
    let (shutdown, shutdown_signal) = oneshot::channel();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(TraceServiceServer::new(TraceCollector { sender }))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = shutdown_signal.await;
            }),
    );
    export_span(&format!("http://{address}"), OtlpProtocol::Grpc).await;
    let (headers, message) = receiver.recv().await.unwrap();
    assert_headers(&headers, case);
    assert_export(&message);
    shutdown.send(()).unwrap();
    server.await.unwrap().unwrap();
}
