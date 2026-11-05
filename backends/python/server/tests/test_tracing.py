import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from threading import Thread

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

from text_embeddings_server.utils.tracing import _http_trace_endpoint, setup_tracing


EXPORT_SPAN = """
import sys
from opentelemetry import trace
from text_embeddings_server.utils.tracing import setup_tracing

setup_tracing(sys.argv[1], "tei-test-service", *sys.argv[2:])
with trace.get_tracer("tei-export-test").start_as_current_span("embed"):
    pass
provider = trace.get_tracer_provider()
assert provider.force_flush(timeout_millis=5000)
provider.shutdown()
"""


def export_span(endpoint, protocol=None, headers=None):
    # Isolate the SDK's process-global provider and environment for each export.
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["OTEL_EXPORTER_OTLP_TIMEOUT"] = "2"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env.update(headers or {})
    args = [sys.executable, "-c", EXPORT_SPAN, endpoint]
    if protocol is not None:
        args.append(protocol)
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def assert_trace(request):
    assert len(request.resource_spans) == 1
    resource_spans = request.resource_spans[0]
    attributes = {
        attribute.key: attribute.value.string_value
        for attribute in resource_spans.resource.attributes
    }
    assert attributes["service.name"] == "tei-test-service"
    spans = [span for scope in resource_spans.scope_spans for span in scope.spans]
    assert len(spans) == 1
    assert spans[0].name == "embed"
    assert len(spans[0].trace_id) == 16
    assert len(spans[0].span_id) == 8


@pytest.fixture
def http_collector():
    received = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.put((self.path, self.headers, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize(
    "endpoint_path, request_path",
    [
        ("", "/v1/traces"),
        ("/", "/v1/traces"),
        ("/api/public/otel", "/api/public/otel/v1/traces"),
        ("/api/public/otel/", "/api/public/otel/v1/traces"),
        ("/api/public/otel/v1/traces", "/api/public/otel/v1/traces"),
        ("/api/public/otel/v1/traces/", "/api/public/otel/v1/traces"),
        ("/collector?tenant=test", "/collector/v1/traces?tenant=test"),
    ],
)
def test_http_exports_protobuf(http_collector, endpoint_path, request_path):
    endpoint, received = http_collector
    export_span(
        endpoint + endpoint_path,
        "http-proto",
        {
            "OTEL_EXPORTER_OTLP_HEADERS": (
                "authorization=Basic%20test%3D,x-tenant=one%2Ctwo"
            )
        },
    )
    path, headers, body = received.get(timeout=5)
    assert path == request_path
    assert headers["Content-Type"] == "application/x-protobuf"
    assert headers["Authorization"] == "Basic test="
    assert headers["x-tenant"] == "one,two"
    assert_trace(trace_service_pb2.ExportTraceServiceRequest.FromString(body))


def test_http_trace_headers_replace_general_headers(http_collector):
    endpoint, received = http_collector
    export_span(
        endpoint,
        "http-proto",
        {
            "OTEL_EXPORTER_OTLP_HEADERS": "authorization=wrong,x-general=unused",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "authorization=Bearer%20trace-token",
        },
    )
    _, headers, body = received.get(timeout=5)
    assert headers["Authorization"] == "Bearer trace-token"
    assert "x-general" not in headers
    assert_trace(trace_service_pb2.ExportTraceServiceRequest.FromString(body))


def test_http_general_endpoint_overrides_cli(http_collector):
    endpoint, received = http_collector
    export_span(
        "http://127.0.0.1:1/cli",
        "http-proto",
        {"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint + "/generic"},
    )
    path, _, body = received.get(timeout=5)
    assert path == "/generic/v1/traces"
    assert_trace(trace_service_pb2.ExportTraceServiceRequest.FromString(body))


def test_http_trace_endpoint_overrides_general_endpoint_and_cli(http_collector):
    endpoint, received = http_collector
    export_span(
        "http://127.0.0.1:1/cli",
        "http-proto",
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint + "/generic",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint + "/custom/traces?tenant=test",
        },
    )
    path, _, body = received.get(timeout=5)
    assert path == "/custom/traces?tenant=test"
    assert_trace(trace_service_pb2.ExportTraceServiceRequest.FromString(body))


@pytest.mark.parametrize("protocol", [None, "grpc"])
@pytest.mark.parametrize("trace_headers", [False, True])
def test_grpc_exports_protobuf(protocol, trace_headers):
    received = Queue()

    class Collector(trace_service_pb2_grpc.TraceServiceServicer):
        def Export(self, request, context):
            received.put((request, dict(context.invocation_metadata())))
            return trace_service_pb2.ExportTraceServiceResponse()

    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(Collector(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        headers = {
            "OTEL_EXPORTER_OTLP_HEADERS": (
                "authorization=Basic%20test%3D,x-tenant=one%2Ctwo"
            )
        }
        if trace_headers:
            headers["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
                "authorization=Bearer%20trace-token"
            )
        export_span(
            f"http://127.0.0.1:{port}",
            protocol,
            headers,
        )
        request, headers = received.get(timeout=5)
        if trace_headers:
            assert headers["authorization"] == "Bearer trace-token"
            assert "x-tenant" not in headers
        else:
            assert headers["authorization"] == "Basic test="
            assert headers["x-tenant"] == "one,two"
        assert_trace(request)
    finally:
        server.stop(0).wait()


@pytest.mark.parametrize(
    "endpoint",
    [
        "collector:4318",
        "ftp://collector:4318",
        "http:///collector",
        "http://collector:invalid",
        "http://collector:65536",
        "http://[invalid",
        "http://user:secret@collector/#fragment",
        "http://collector/#",
    ],
)
def test_invalid_http_endpoint_does_not_expose_credentials(endpoint):
    with pytest.raises(ValueError, match="^Invalid OTLP HTTP endpoint$"):
        setup_tracing(endpoint, "tei-test-service", "http-proto")


def test_https_endpoint_preserves_scheme_and_query():
    assert _http_trace_endpoint("https://collector/otel?tenant=test") == (
        "https://collector/otel/v1/traces?tenant=test"
    )


def test_invalid_protocol():
    with pytest.raises(ValueError, match="^Unsupported OTLP protocol$"):
        setup_tracing("http://localhost:4318", "tei-test-service", "invalid")
