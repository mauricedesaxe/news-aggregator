"""OpenTelemetry setup and pure-ASGI HTTP request tracing."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from romanian_news.config import BETTERSTACK_INGESTING_HOST, BETTERSTACK_SOURCE_TOKEN

SERVICE = "romanian-news-pipeline"
SERVER_SERVICE = "romanian-news-reader"
TRACER = trace.get_tracer(SERVICE)
DEFAULT_EXCLUDED_PATHS = frozenset({"/health", "/healthz", "/livez", "/readyz"})
_telemetry_configured = False


@runtime_checkable
class _RouteMatcher(Protocol):
    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...


class HttpTracingMiddleware:
    """Create bounded server spans for HTTP requests in any ASGI application."""

    def __init__(
        self,
        app: ASGIApp,
        tracer: Tracer | None = None,
        excluded_paths: frozenset[str] = DEFAULT_EXCLUDED_PATHS,
        routes: Sequence[object] | None = None,
    ) -> None:
        self.app = app
        self.tracer = tracer or TRACER
        self.excluded_paths = excluded_paths
        self.routes = routes if routes is not None else ()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        parent_context = propagate.extract(_header_carrier(scope.get("headers", [])))
        with self.tracer.start_as_current_span(
            f"HTTP {method}", context=parent_context, kind=SpanKind.SERVER
        ) as current:
            _set_request_attributes(current, scope, method)

            async def traced_send(message: Message) -> None:
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    current.set_attribute("http.response.status_code", status_code)
                    route = _resolve_route_template(scope, self.routes)
                    if route:
                        current.set_attribute("http.route", route)
                        current.update_name(f"{method} {route}")
                    if status_code >= 500:
                        current.set_status(Status(StatusCode.ERROR))
                await send(message)

            try:
                await self.app(scope, receive, traced_send)
            except Exception as error:
                current.record_exception(error)
                current.set_attribute("error.type", type(error).__name__)
                current.set_status(Status(StatusCode.ERROR))
                raise


def configure_telemetry(service_name: str = SERVICE) -> None:
    """Export traces to Better Stack once when the source is configured."""
    global _telemetry_configured
    if _telemetry_configured or not BETTERSTACK_INGESTING_HOST or not BETTERSTACK_SOURCE_TOKEN:
        return
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    exporter = OTLPSpanExporter(
        endpoint=f"https://{BETTERSTACK_INGESTING_HOST}/v1/traces",
        headers={"Authorization": f"Bearer {BETTERSTACK_SOURCE_TOKEN}"},
        compression=Compression.Gzip,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _telemetry_configured = True


@contextmanager
def span(name: str, **attributes: str | int | float | bool) -> Iterator[Span]:
    """Create a trace span with scalar attributes safe to export."""
    with TRACER.start_as_current_span(name) as current:
        current.set_attributes(attributes)
        try:
            yield current
        except Exception as error:
            current.record_exception(error)
            current.set_status(Status(StatusCode.ERROR, str(error)))
            raise


def event(name: str, **attributes: str | int | float | bool) -> None:
    """Attach a structured event to the active pipeline span."""
    trace.get_current_span().add_event(name, attributes)


def fail(current: Span, description: str) -> None:
    """Mark an expected terminal failure on a span that does not raise."""
    current.set_status(Status(StatusCode.ERROR, description))


def _header_carrier(headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    return {name.decode("latin-1"): value.decode("latin-1") for name, value in headers}


def _set_request_attributes(current: Span, scope: Scope, method: str) -> None:
    current.set_attribute("http.request.method", method)
    scheme = scope.get("scheme")
    if isinstance(scheme, str):
        current.set_attribute("url.scheme", scheme)
    server = scope.get("server")
    if isinstance(server, tuple) and len(server) == 2:
        address, port = server
        if isinstance(address, str):
            current.set_attribute("server.address", address)
        if isinstance(port, int):
            current.set_attribute("server.port", port)


def _resolve_route_template(scope: Scope, routes: Sequence[object]) -> str | None:
    template = _route_template(scope.get("route"))
    if template:
        return template
    for route in routes:
        if not isinstance(route, _RouteMatcher):
            continue
        matcher = route.matches
        if callable(matcher) and matcher(scope)[0] is Match.FULL:
            return _route_template(route)
    return None


def _route_template(route: object) -> str | None:
    if route is None:
        return None
    path_format = getattr(route, "path_format", None)
    if isinstance(path_format, str):
        return path_format
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else None
