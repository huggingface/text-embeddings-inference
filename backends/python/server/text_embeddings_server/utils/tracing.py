import os
from urllib.parse import urlsplit, urlunsplit

import grpc
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.instrumentation.grpc._aio_server import (
    OpenTelemetryAioServerInterceptor,
)
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)


class UDSOpenTelemetryAioServerInterceptor(OpenTelemetryAioServerInterceptor):
    def __init__(self):
        super().__init__(trace.get_tracer(__name__))

    def _start_span(self, handler_call_details, context, set_status_on_exception=False):
        """
        Rewrite _start_span method to support Unix Domain Socket gRPC contexts
        """

        # standard attributes
        attributes = {
            SpanAttributes.RPC_SYSTEM: "grpc",
            SpanAttributes.RPC_GRPC_STATUS_CODE: grpc.StatusCode.OK.value[0],
        }

        # if we have details about the call, split into service and method
        if handler_call_details.method:
            service, method = handler_call_details.method.lstrip("/").split("/", 1)
            attributes.update(
                {
                    SpanAttributes.RPC_METHOD: method,
                    SpanAttributes.RPC_SERVICE: service,
                }
            )

        # add some attributes from the metadata
        metadata = dict(context.invocation_metadata())
        if "user-agent" in metadata:
            attributes["rpc.user_agent"] = metadata["user-agent"]

        # We use gRPC over a UNIX socket
        attributes.update({SpanAttributes.NET_TRANSPORT: "unix"})

        return self._tracer.start_as_current_span(
            name=handler_call_details.method,
            kind=trace.SpanKind.SERVER,
            attributes=attributes,
            set_status_on_exception=set_status_on_exception,
        )


def _http_trace_endpoint(endpoint: str) -> str:
    try:
        url = urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or "#" in endpoint
        ):
            raise ValueError
        # Accessing port also validates its syntax and range.
        url.port
    except ValueError:
        raise ValueError("Invalid OTLP HTTP endpoint") from None

    path = url.path.rstrip("/")
    if not path.endswith("/v1/traces"):
        path += "/v1/traces"
    return urlunsplit(url._replace(path=path))


def setup_tracing(
    otlp_endpoint: str, otlp_service_name: str, otlp_protocol: str = "grpc"
):
    resource = Resource.create(attributes={"service.name": otlp_service_name})
    if otlp_protocol == "http-proto":
        endpoint = _http_trace_endpoint(otlp_endpoint)
        if os.environ.get(OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) or os.environ.get(
            OTEL_EXPORTER_OTLP_ENDPOINT
        ):
            # Let the SDK apply its endpoint precedence and per-signal path rules.
            endpoint = None
        span_exporter = HttpOTLPSpanExporter(endpoint=endpoint)
    elif otlp_protocol == "grpc":
        span_exporter = GrpcOTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    else:
        raise ValueError("Unsupported OTLP protocol")
    span_processor = BatchSpanProcessor(span_exporter)

    trace.set_tracer_provider(TracerProvider(resource=resource))
    trace.get_tracer_provider().add_span_processor(span_processor)
