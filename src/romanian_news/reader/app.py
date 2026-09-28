from __future__ import annotations

import hmac
import re
import secrets
import time
from collections import deque
from collections.abc import Callable
from datetime import date
from ipaddress import ip_address
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal, TypeVar, cast
from urllib.parse import quote, urlsplit
from uuid import UUID

from fasthtml.common import (
    H1,
    A,
    Beforeware,
    Div,
    FastHTML,
    FtResponse,
    Link,
    Main,
    Meta,
    P,
    RedirectResponse,
    Script,
    Style,
    Title,
    fast_app,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.weekly_status import (
    WeeklyStatusNotFound,
)
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.config import (
    APP_PASSWORD,
    COOKIE_SECURE,
    NEWS_PUBLIC_MEDIA_BASE_URL,
    SESSION_SECRET,
    TRUST_PROXY_HEADERS,
)
from romanian_news.current_report import (
    CurrentDailyReport,
)
from romanian_news.feedback import (
    DailyReportSummary,
    DailyReportVersionNotFound,
    ThemeFeedbackTarget,
)
from romanian_news.reader import clock
from romanian_news.reader.components import (
    _feedback_command,
    _feedback_control,
    _feedback_target_from_form,
    _research_flag_control,
)
from romanian_news.reader.domain import PRODUCTION_DOMAIN, ReaderDomain
from romanian_news.reader.report_status import (
    _check_report_status,
    _poll_report_status,
    _valid_csrf,
)
from romanian_news.reader.status import (
    render_archive_progress,
    render_status,
    render_status_archive,
)
from romanian_news.reader.styles import _STYLES
from romanian_news.reader.views import (
    _is_htmx,
    _load_exact_report,
    _load_report_archive,
    _login_page,
    _not_found_page,
    _not_ready_page,
    _report_page,
    _safe_next,
    _site_header,
    _unavailable_page,
    _video_feedback_control,
)
from romanian_news.reports import DailyReportDocument
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
)
from romanian_news.telemetry import HttpTracingMiddleware, configure_telemetry
from romanian_news.video_digest.errors import VideoDigestCatalogError
from romanian_news.video_digest.models import EditionId, StoryId
from romanian_news.video_digest_feedback import (
    VideoDigestFeedbackCommand,
)
from romanian_news.weekly_status import WeeklyStatusRead

SESSION_COOKIE = "romanian_news_session"
SESSION_MAX_AGE = 14 * 24 * 60 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_CLIENTS = 1024
READINESS_CACHE_SECONDS = 10


class ReaderSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    app_password: Annotated[str, Field(min_length=12)]
    session_secret: Annotated[str, Field(min_length=32)]
    cookie_secure: bool = False
    trust_proxy_headers: bool = False
    public_media_origin: str | None = None

    @field_validator("public_media_origin", mode="before")
    @classmethod
    def validate_public_media_origin(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if not isinstance(value, str) or value != value.strip():
            raise ValueError("public_media_origin must be an exact HTTPS origin")
        parsed = urlsplit(value)
        try:
            _ = parsed.port
        except ValueError:
            raise ValueError("public_media_origin must be an exact HTTPS origin") from None
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or re.search(r"[\s;'\"\\]", parsed.netloc)
        ):
            raise ValueError("public_media_origin must be an exact HTTPS origin")
        return f"https://{parsed.netloc}"

    @classmethod
    def from_environment(cls) -> ReaderSettings:
        return cls.model_validate(
            {
                "app_password": APP_PASSWORD,
                "session_secret": SESSION_SECRET,
                "cookie_secure": COOKIE_SECURE,
                "trust_proxy_headers": TRUST_PROXY_HEADERS,
                "public_media_origin": NEWS_PUBLIC_MEDIA_BASE_URL,
            }
        )


_CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'inline-speculation-rules'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"


def _security_headers(public_media_origin: str | None) -> tuple[tuple[bytes, bytes], ...]:
    media = f"; media-src 'self' {public_media_origin}" if public_media_origin else ""
    return (
        (b"content-security-policy", f"{_CONTENT_SECURITY_POLICY}{media}".encode()),
        (b"cache-control", b"no-store"),
        (b"referrer-policy", b"no-referrer"),
        (b"strict-transport-security", b"max-age=63072000; includeSubDomains"),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
    )


_HTMX_CONFIG = '{"responseHandling":[{"code":"204","swap":false},{"code":"[23]..","swap":true},{"code":"[45]..","swap":true,"error":false}]}'


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, headers: tuple[tuple[bytes, bytes], ...]) -> None:
        self.app = app
        self.headers = headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                present = {name.lower() for name, _ in headers}
                headers.extend((name, value) for name, value in self.headers if name not in present)
            await send(message)

        await self.app(scope, receive, send_with_headers)


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., object])


_HandlerT = TypeVar("_HandlerT", bound=Callable[..., object])
_HandlerDecorator = Callable[[_HandlerT], _HandlerT]


def _route(
    app: FastHTML, method: Literal["get", "post"], path: str
) -> _HandlerDecorator[_HandlerT]:
    """Return the route decorator FastHTML attaches to itself at import time."""
    return cast("_HandlerDecorator[_HandlerT]", getattr(app, method)(path))


def decode_session_cookie(app: FastHTML, cookie: str) -> dict[str, object]:
    """Decode a signed session cookie through the method FastHTML attaches at import time."""
    decoder = cast("Callable[[str], dict[str, object]]", app.decode_session)
    return decoder(cookie)


def create_app(
    settings: ReaderSettings | None = None,
    domain: ReaderDomain = PRODUCTION_DOMAIN,
) -> FastHTML:
    settings = settings or ReaderSettings.from_environment()
    configure_telemetry("romanian-news-reader")
    app, _ = fast_app(
        title="Press review",
        before=Beforeware(
            _require_owner,
            skip=[
                r"/healthz",
                r"/livez",
                r"/readyz",
                r"/login",
                r"/favicon.ico",
                r"/htmx\.min\.js",
            ],
        ),
        default_hdrs=False,
        htmx=False,
        pico=False,
        surreal=False,
        canonical=False,
        static_path=str(Path(__file__).parent / "static"),
        secret_key=settings.session_secret,
        session_cookie=SESSION_COOKIE,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        sess_https_only=settings.cookie_secure,
        htmlkw={"lang": "en"},
        hdrs=(
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Meta(name="color-scheme", content="light dark"),
            Meta(name="theme-color", content="#f6f3ea", media="(prefers-color-scheme: light)"),
            Meta(name="theme-color", content="#171714", media="(prefers-color-scheme: dark)"),
            Meta(name="htmx-config", content=_HTMX_CONFIG),
            Link(rel="icon", href="data:,"),
            Script(src="/htmx.min.js", defer=True),
            Style(_STYLES),
        ),
    )

    _register_base_routes(app, settings, domain)
    _register_daily_routes(app, settings, domain)
    _register_status_routes(app, domain)
    _register_action_routes(app, domain)

    app.add_middleware(
        SecurityHeadersMiddleware, headers=_security_headers(settings.public_media_origin)
    )
    app.add_middleware(HttpTracingMiddleware, routes=app.router.routes)
    return app


def _register_base_routes(app: FastHTML, settings: ReaderSettings, domain: ReaderDomain) -> None:
    login_failures: dict[str, deque[float]] = {}
    readiness_cache: tuple[float, bool] | None = None
    readiness_lock = Lock()

    @_route(app, "get", "/healthz")
    def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @_route(app, "get", "/livez")
    def livez() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @_route(app, "get", "/readyz")
    def readyz() -> PlainTextResponse:
        nonlocal readiness_cache
        with readiness_lock:
            now = time.monotonic()
            if readiness_cache is None or now - readiness_cache[0] >= READINESS_CACHE_SECONDS:
                try:
                    domain.check_readiness()
                except (ResearchCatalogError, ResearchObjectUnavailable):
                    readiness_cache = (now, False)
                else:
                    readiness_cache = (now, True)
            available = readiness_cache[1]
        return PlainTextResponse(
            "ok" if available else "unavailable",
            status_code=200 if available else 503,
        )

    @_route(app, "get", "/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @_route(app, "get", "/login")
    def login_form(request: Request) -> tuple[Any, ...]:
        return _login_page(_safe_next(request.query_params.get("next", "/")))

    @_route(app, "post", "/login")
    async def login_submit(request: Request) -> Response | FtResponse:
        return await _submit_login(request, settings, login_failures)


def _register_daily_routes(app: FastHTML, settings: ReaderSettings, domain: ReaderDomain) -> None:
    @_route(app, "get", "/")
    def home(request: Request) -> tuple[Any, ...] | FtResponse:
        return _load_home(request, domain, settings.public_media_origin)

    @_route(app, "get", "/today")
    def today(request: Request) -> tuple[Any, ...] | FtResponse:
        return _load_today(request, domain, settings.public_media_origin)

    @_route(app, "get", "/reports")
    def report_archive(request: Request) -> tuple[Any, ...] | FtResponse:
        return _load_report_archive(request, domain)

    @_route(app, "get", "/reports/backfill")
    def archive_progress(request: Request) -> tuple[Any, ...] | FtResponse:
        try:
            discovery = domain.list_archive_discovery()
            archive_reports = domain.list_archive_reports(ARCHIVE_START, ARCHIVE_END)
        except ResearchCatalogError:
            return _unavailable_page(request)
        return render_archive_progress(
            discovery, archive_reports, _site_header(str(request.session["csrf_token"]))
        )

    @_route(app, "get", "/reports/exact/{report_version_id}")
    def exact_report_page(request: Request, report_version_id: str) -> tuple[Any, ...] | FtResponse:
        return _load_exact_report(request, report_version_id, domain)

    @_route(app, "get", "/reports/{report_version_id}")
    def report_page(
        request: Request, report_version_id: str
    ) -> tuple[Any, ...] | Response | FtResponse:
        return _load_report(request, report_version_id, domain, settings.public_media_origin)


def _register_status_routes(app: FastHTML, domain: ReaderDomain) -> None:
    @_route(app, "get", "/status")
    def status(request: Request) -> tuple[Any, ...] | Response | FtResponse:
        try:
            latest = domain.list_status(1, 0)
        except ResearchCatalogError:
            return _unavailable_page(request)
        if not latest:
            return FtResponse(_status_not_ready_page(request), status_code=503)
        return RedirectResponse(
            f"/status/weeks/{latest[0].week_start.isoformat()}", status_code=303
        )

    @_route(app, "get", "/status/archive")
    def status_archive(request: Request) -> tuple[Any, ...] | FtResponse:
        raw_page = request.query_params.get("page", "1")
        if not raw_page.isdecimal() or not 1 <= int(raw_page) <= 1000:
            return FtResponse(_not_found_page(request), status_code=404)
        page = int(raw_page)
        try:
            rows = domain.list_status(21, (page - 1) * 20)
        except ResearchCatalogError:
            return _unavailable_page(request)
        return render_status_archive(
            rows[:20],
            page,
            len(rows) > 20,
            _site_header(str(request.session["csrf_token"])),
        )

    @_route(app, "get", "/status/weeks/{week_start}")
    def status_week(request: Request, week_start: str) -> tuple[Any, ...] | FtResponse:
        if domain.read_status is None:
            return FtResponse(_status_not_ready_page(request), status_code=503)
        try:
            version_id, read = domain.read_status(date.fromisoformat(week_start))
        except (WeeklyStatusNotFound, ValueError):
            return FtResponse(_not_found_page(request), status_code=404)
        except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
            return _unavailable_page(request)
        try:
            source_reports = _read_status_source_reports(read, domain)
        except (ResearchObjectUnavailable, ResearchObjectIntegrityError, ValueError):
            return _unavailable_page(request)
        return render_status(
            read,
            version_id,
            _site_header(str(request.session["csrf_token"])),
            source_reports=source_reports,
        )

    @_route(app, "get", "/status/versions/{version_id}")
    def status_version(request: Request, version_id: str) -> tuple[Any, ...] | FtResponse:
        if domain.read_status_version is None:
            return FtResponse(_status_not_ready_page(request), status_code=503)
        try:
            exact_id, read = domain.read_status_version(version_id)
        except WeeklyStatusNotFound:
            return FtResponse(_not_found_page(request), status_code=404)
        except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
            return _unavailable_page(request)
        try:
            source_reports = _read_status_source_reports(read, domain)
        except (ResearchObjectUnavailable, ResearchObjectIntegrityError, ValueError):
            return _unavailable_page(request)
        return render_status(
            read,
            exact_id,
            _site_header(str(request.session["csrf_token"])),
            exact_version=True,
            source_reports=source_reports,
        )


def _read_status_source_reports(
    read: WeeklyStatusRead, domain: ReaderDomain
) -> tuple[DailyReportDocument, ...]:
    reports: list[DailyReportDocument] = []
    for slot in read.days:
        if slot.report is None:
            continue
        report = domain.read_report_reference(slot.report)
        if report.day != slot.day:
            raise ResearchObjectIntegrityError(
                f"Weekly source report date does not match {slot.day.isoformat()}"
            )
        reports.append(report)
    return tuple(reports)


def _register_action_routes(app: FastHTML, domain: ReaderDomain) -> None:
    @_route(app, "post", "/report-status/check")
    async def report_status_check(request: Request) -> FtResponse:
        return await _check_report_status(request, domain)

    @_route(app, "get", "/report-status")
    async def report_status(request: Request) -> FtResponse:
        return await _poll_report_status(request, domain)

    @_route(app, "post", "/feedback")
    async def feedback_submit(request: Request) -> Response | FtResponse:
        return await _submit_feedback(request, domain)

    @_route(app, "post", "/video-feedback")
    async def video_feedback_submit(request: Request) -> Response | FtResponse:
        return await _submit_video_feedback(request, domain)

    @_route(app, "post", "/logout")
    async def logout(request: Request) -> Response | FtResponse:
        return await _submit_logout(request)


def _require_owner(request: Request) -> Response | None:
    if ".." in request.url.path.split("/"):
        return Response(status_code=404)
    if request.session.get("authenticated") is True:
        return None
    next_url = request.url.path
    if request.url.query:
        next_url = f"{next_url}?{request.url.query}"
    login_url = f"/login?next={quote(next_url, safe='')}"
    if "hx-request" in request.headers:
        return Response(status_code=401, headers={"HX-Redirect": login_url})
    return RedirectResponse(login_url, status_code=303)


async def _submit_login(
    request: Request,
    settings: ReaderSettings,
    login_failures: dict[str, deque[float]],
) -> Response | FtResponse:
    form = await request.form()
    password = str(form.get("password", ""))
    next_url = _safe_next(str(form.get("next", "/")))
    client_id = _login_client_id(request, settings.trust_proxy_headers)
    now = time.monotonic()
    recent = login_failures.get(client_id, deque())
    while recent and now - recent[0] >= LOGIN_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= LOGIN_MAX_FAILURES:
        return FtResponse(
            _login_page(next_url, "Too many attempts. Try again in a few minutes."),
            status_code=429,
        )
    if hmac.compare_digest(password, settings.app_password):
        login_failures.pop(client_id, None)
        request.session.clear()
        request.session.update({"authenticated": True, "csrf_token": secrets.token_urlsafe(32)})
        return RedirectResponse(next_url, status_code=303)
    if client_id not in login_failures and len(login_failures) >= LOGIN_MAX_CLIENTS:
        login_failures.pop(next(iter(login_failures)))
    recent.append(now)
    login_failures[client_id] = recent
    return FtResponse(_login_page(next_url, "The password is incorrect."), status_code=401)


def _login_client_id(request: Request, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            try:
                return ip_address(real_ip.strip()).compressed
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def _load_home(
    request: Request, domain: ReaderDomain, public_media_origin: str | None
) -> tuple[Any, ...] | FtResponse:
    return _load_current_day(request, domain, public_media_origin)


def _load_today(
    request: Request, domain: ReaderDomain, public_media_origin: str | None
) -> tuple[Any, ...] | FtResponse:
    return _load_current_day(request, domain, public_media_origin)


def _load_current_day(
    request: Request, domain: ReaderDomain, public_media_origin: str | None
) -> tuple[Any, ...] | FtResponse:
    try:
        reports = domain.list_reports(30)
    except (ResearchCatalogError, ValidationError, ValueError):
        return _unavailable_page(request)
    try:
        current = domain.read_current_report(clock.bucharest_today())
    except (
        ResearchCatalogError,
        ResearchObjectUnavailable,
        ResearchObjectIntegrityError,
        ValidationError,
        ValueError,
    ):
        return _unavailable_page(request, reports)
    if current is None:
        return _not_ready_page(request, reports)
    summaries = (
        _live_summary(current),
        *(report for report in reports if report.report_version_id != current.head.version_id),
    )
    return _report_page(
        request,
        current.head.version_id,
        summaries,
        domain,
        live=current,
        public_media_origin=public_media_origin,
    )


def _live_summary(live: CurrentDailyReport) -> DailyReportSummary:
    return DailyReportSummary(report_version_id=live.head.version_id, day=live.head.day)


def _load_report(
    request: Request,
    report_version_id: str,
    domain: ReaderDomain,
    public_media_origin: str | None,
) -> tuple[Any, ...] | Response | FtResponse:
    try:
        current_version_id = domain.resolve_current_report_version(report_version_id)
    except DailyReportVersionNotFound:
        return FtResponse(_not_found_page(request), status_code=404)
    except ResearchCatalogError:
        return _unavailable_page(request)
    if current_version_id != report_version_id:
        return RedirectResponse(f"/reports/{current_version_id}", status_code=307)
    try:
        reports = domain.list_reports(30)
    except (ResearchCatalogError, ValueError):
        return _unavailable_page(request)
    try:
        return _report_page(
            request,
            report_version_id,
            reports,
            domain,
            public_media_origin=public_media_origin,
        )
    except DailyReportVersionNotFound:
        return FtResponse(_not_found_page(request), status_code=404)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return _unavailable_page(request, reports)


def _status_not_ready_page(request: Request) -> tuple[Any, ...]:
    return (
        Title("Weekly status is not ready | Press review"),
        _site_header(str(request.session["csrf_token"])),
        Main(
            Div(
                H1("Weekly status is not ready yet"),
                P(
                    "The completed-week read will appear here when enough daily reports are available."
                ),
                A("Open daily reports", href="/reports"),
                cls="empty",
            )
        ),
    )


async def _submit_feedback(request: Request, domain: ReaderDomain) -> Response | FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return _feedback_error_response(
            request,
            form,
            "The request expired. Reload the page.",
            403,
        )
    try:
        command = _feedback_command(form)
        event = await run_in_threadpool(domain.submit_feedback, command)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return _feedback_error_response(
            request,
            form,
            "The service is unavailable. Feedback was not saved.",
            503,
        )
    except (DailyReportVersionNotFound, ValidationError, ValueError, TypeError):
        return _feedback_error_response(request, form, "Feedback could not be saved.", 400)
    if _is_htmx(request):
        if str(form.get("flag_vote", "")) == "1" and isinstance(
            command.target, ThemeFeedbackTarget
        ):
            return FtResponse(
                _research_flag_control(command.target, None, request.session["csrf_token"], event)
            )
        return FtResponse(
            _feedback_control(
                command.target, request.session["csrf_token"], event, open_details=True
            )
        )
    return RedirectResponse(f"/reports/{event.target.report_version_id}", status_code=303)


async def _submit_video_feedback(request: Request, domain: ReaderDomain) -> Response | FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return _video_feedback_error_response(
            request,
            form,
            "The request expired. Reload the page.",
            403,
        )
    try:
        command = VideoDigestFeedbackCommand.model_validate(
            {
                "feedback_id": UUID(str(form.get("feedback_id", ""))),
                "edition_id": str(form.get("edition_id", "")),
                "story_id": str(form.get("story_id", "")).strip() or None,
                "rating": str(form.get("rating", "")).strip() or None,
                "note": str(form.get("note", "")),
                "actor": "owner",
            }
        )
        event = await run_in_threadpool(domain.submit_video_feedback, command)
    except (
        ResearchCatalogError,
        ResearchObjectUnavailable,
        ResearchObjectIntegrityError,
        VideoDigestCatalogError,
    ):
        return _video_feedback_error_response(
            request,
            form,
            "The service is unavailable. Feedback was not saved.",
            503,
        )
    except (ValidationError, ValueError, TypeError):
        return _video_feedback_error_response(
            request,
            form,
            "Feedback could not be saved.",
            400,
        )
    return_to = _safe_next(str(request.query_params.get("return_to", "/")))
    if _is_htmx(request):
        return FtResponse(
            _video_feedback_control(
                event.edition_id,
                event.story_id,
                str(request.session["csrf_token"]),
                return_to,
                event=event,
                open_details=True,
            )
        )
    return RedirectResponse(return_to, status_code=303)


def _video_feedback_error_response(
    request: Request,
    form: Any,
    message: str,
    status_code: int,
) -> FtResponse:
    edition_id = str(form.get("edition_id", ""))
    story_value = str(form.get("story_id", "")).strip()
    story_id = StoryId(story_value) if story_value else None
    return_to = _safe_next(str(request.query_params.get("return_to", "/")))
    if (
        _is_htmx(request)
        and re.fullmatch(r"[0-9a-f]{64}", edition_id)
        and (story_id is None or re.fullmatch(r"[0-9a-f]{64}", story_id))
    ):
        return FtResponse(
            _video_feedback_control(
                EditionId(edition_id),
                story_id,
                str(request.session["csrf_token"]),
                return_to,
                error=message,
                note=str(form.get("note", "")),
                open_details=True,
            ),
            status_code=status_code,
        )
    return FtResponse(P(message), status_code=status_code)


def _feedback_error_response(
    request: Request,
    form: Any,
    message: str,
    status_code: int,
) -> FtResponse:
    target = _feedback_target_from_form(form)
    if _is_htmx(request) and target is not None:
        if str(form.get("flag_vote", "")) == "1" and isinstance(target, ThemeFeedbackTarget):
            return FtResponse(
                _research_flag_control(
                    target,
                    None,
                    request.session["csrf_token"],
                    error=message,
                ),
                status_code=status_code,
            )
        return FtResponse(
            _feedback_control(
                target,
                request.session["csrf_token"],
                error=message,
                note=str(form.get("note", "")),
                open_details=True,
            ),
            status_code=status_code,
        )
    return FtResponse(P(message), status_code=status_code)


async def _submit_logout(request: Request) -> Response | FtResponse:
    form = await request.form()
    if not _valid_csrf(request, str(form.get("csrf_token", ""))):
        return FtResponse(P("The request expired. Reload the page."), status_code=403)
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
