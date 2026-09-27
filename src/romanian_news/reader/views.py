from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, timedelta
from itertools import count
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
from uuid import uuid4

from fasthtml.common import (
    FT,
    H1,
    H2,
    H3,
    A,
    Article,
    Button,
    Details,
    Div,
    Form,
    FtResponse,
    Header,
    Hidden,
    Input,
    Label,
    Li,
    Main,
    Nav,
    Ol,
    P,
    Script,
    Section,
    Small,
    Span,
    Strong,
    Summary,
    Textarea,
    Time,
    Title,
    Track,
    Ul,
    Video,
)
from pydantic import ValidationError
from starlette.requests import Request

from romanian_news import Sha256
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.current_report import (
    CurrentDailyReport,
)
from romanian_news.feedback import (
    DailyReportSummary,
    DailyReportVersionNotFound,
    NewsFeedbackEvent,
    ReportFeedbackTarget,
)
from romanian_news.reader import clock
from romanian_news.reader.components import (
    _feedback_control,
    _report_section,
    _target_key,
)
from romanian_news.reader.report_status import _initial_status_strip
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    RetrospectiveDailyReport,
)
from romanian_news.research_triggers import SubjectResearchFlag
from romanian_news.storage import (
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
)
from romanian_news.video_digest.errors import VideoDigestCatalogError
from romanian_news.video_digest.models import EditionId, StoryId
from romanian_news.video_digest.reader import (
    ReaderVideoDigest,
    ReaderVideoEdition,
    edition_label,
)
from romanian_news.video_digest_feedback import (
    VideoDigestFeedbackEvent,
)

if TYPE_CHECKING:
    from romanian_news.reader.domain import ReaderDomain


def _report_page(
    request: Request,
    report_version_id: Sha256,
    reports: tuple[DailyReportSummary, ...],
    domain: ReaderDomain,
    *,
    live: CurrentDailyReport | None = None,
    public_media_origin: str | None = None,
) -> tuple[Any, ...]:
    report = live.report if live is not None else domain.read_report(report_version_id)
    latest_feedback = {
        _target_key(event.target): event for event in domain.read_feedback(report_version_id)
    }
    csrf_token = str(request.session["csrf_token"])
    selected_index = next(
        (
            index
            for index, value in enumerate(reports)
            if value.report_version_id == report_version_id
        ),
        None,
    )
    report_target = ReportFeedbackTarget(report_version_id=report_version_id)
    research_flags = _research_flags(domain, report_version_id)
    video_digest = _reader_video_digest(
        request,
        domain,
        report.day,
        report_version_id,
        public_media_origin,
    )
    next_live = live is None and report.day < clock.bucharest_today()
    if isinstance(report, DailyReport):
        meta, sections_body = _tiered_report_body(
            report, report_version_id, csrf_token, latest_feedback, research_flags
        )
    else:
        meta = (
            f"{report.theme_count if isinstance(report, DailyReport) else report.group_count}"
            " subjects"
            f" · {report.group_count} events"
            f" · {report.accepted_article_count} articles in this report"
        )
        sections_body = tuple(
            _report_section(
                section,
                position,
                report_version_id,
                csrf_token,
                latest_feedback,
                research_flags,
            )
            for position, section in enumerate(report.sections, start=1)
        )
    report_tools = Div(
        Div(Span(meta), cls="report-meta"),
        Div(
            _feedback_control(
                report_target,
                csrf_token,
                latest_feedback.get(_target_key(report_target)),
            ),
            cls="report-feedback",
        ),
        cls="report-tools",
    )
    reader = Div(
        *_date_navigation(reports, selected_index, report.day, next_live=next_live),
        _initial_status_strip(report.day, report_version_id, csrf_token)
        if live is not None
        else None,
        _video_digest(video_digest, request.url.path, csrf_token)
        if video_digest is not None
        else None,
        _retrospective_notice(report) if isinstance(report, RetrospectiveDailyReport) else None,
        report_tools,
        *sections_body,
        cls="reader",
    )
    return (
        Title(f"{_format_date(report.day)} | Press review"),
        _site_header(csrf_token),
        Main(reader),
    )


def _retrospective_notice(report: RetrospectiveDailyReport) -> Any:
    coverage = report.retrospective
    started = coverage.capture_started_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ended = coverage.capture_ended_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return Section(
        H2("Historical report"),
        P(
            f"This report covers {report.day.isoformat()} using publisher pages captured "
            f"later, from {started} to {ended}."
        ),
        P(
            f"Archive sources: {', '.join(coverage.included_outlets)}. "
            f"{coverage.discovered_url_count:,} URLs found; "
            f"{coverage.verified_page_count:,} pages date-verified; "
            f"{coverage.captured_article_count:,} articles captured."
        ),
        P(coverage.coverage_note),
        cls="retrospective-notice",
    )


def _reader_video_digest(
    request: Request,
    domain: ReaderDomain,
    day: date,
    report_version_id: Sha256,
    public_media_origin: str | None,
) -> ReaderVideoDigest | None:
    if public_media_origin is None:
        return None
    selected = request.query_params.get("edition")
    try:
        return domain.read_video_digest(
            day,
            report_version_id,
            EditionId(selected) if selected else None,
            public_media_origin,
        )
    except (
        ResearchCatalogError,
        ResearchObjectUnavailable,
        ResearchObjectIntegrityError,
        VideoDigestCatalogError,
        ValidationError,
        ValueError,
        TypeError,
    ):
        return None


def _video_digest(value: ReaderVideoDigest, report_path: str, csrf_token: str) -> FT:
    selected = value.selected
    selector = Nav(
        *(
            A(
                edition_label(option.slot_name, option.published_at),
                href=f"{report_path}?edition={option.edition_id}",
                aria_current=(
                    "true"
                    if selected is not None and option.edition_id == selected.edition_id
                    else None
                ),
            )
            for option in value.editions
        ),
        aria_label="Video edition",
        cls="edition-selector",
    )
    if selected is None:
        return Section(
            Div(H2("Video digest"), selector, cls="video-digest-head"),
            P("The selected video edition is unavailable. The written report is below."),
            cls="video-digest",
        )
    return_to = f"{report_path}?edition={selected.edition_id}"
    track = (
        Track(
            src=str(selected.subtitle.url),
            kind="captions",
            srclang="ro",
            label="Romanian",
            default=True,
        )
        if selected.subtitle.kind == "available"
        else None
    )
    return Section(
        Div(H2("Video digest"), selector, cls="video-digest-head"),
        Video(
            track,
            P(
                "Your browser cannot play this video. ",
                A("Open the video directly", href=str(selected.video_url)),
            ),
            src=str(selected.video_url),
            controls=True,
            preload="metadata",
            crossorigin="anonymous" if track is not None else None,
        ),
        P(
            A(
                "Open video directly",
                href=str(selected.video_url),
                target="_blank",
                rel="noreferrer",
            ),
            Span("Captions are unavailable for this edition.", cls="subtitle-status")
            if selected.subtitle.kind == "failed"
            else Span("Romanian captions available.", cls="subtitle-status"),
            cls="video-links",
        ),
        _video_feedback_control(
            selected.edition_id,
            None,
            csrf_token,
            return_to,
        ),
        _video_transcript(selected, csrf_token, return_to),
        cls="video-digest",
    )


def _video_transcript(
    edition: ReaderVideoEdition,
    csrf_token: str,
    return_to: str,
) -> FT:
    return Section(
        H3("Transcript"),
        Ol(
            *(
                Li(
                    Article(
                        H3(story.title),
                        P(story.narration),
                        _video_feedback_control(
                            edition.edition_id,
                            StoryId(story.story_id),
                            csrf_token,
                            return_to,
                        ),
                    )
                )
                for story in edition.stories
            )
        ),
        aria_label="Video transcript",
        cls="transcript",
    )


def _video_feedback_control(
    edition_id: EditionId,
    story_id: StoryId | None,
    csrf_token: str,
    return_to: str,
    event: VideoDigestFeedbackEvent | None = None,
    *,
    error: str | None = None,
    note: str | None = None,
    open_details: bool = False,
) -> FT:
    suffix = story_id or "edition"
    control_id = f"video-feedback-{suffix}"
    action = f"/video-feedback?return_to={quote(return_to, safe='')}"
    rating = (
        {"positive": "Positive", "negative": "Negative", None: "Note"}[event.rating]
        if event is not None
        else None
    )
    return Div(
        Div(
            Strong(f"Feedback saved: {rating}"),
            Small(event.note) if event and event.note else None,
            cls="feedback-state",
        )
        if rating is not None
        else None,
        Details(
            Summary("Feedback on this video" if story_id is None else "Feedback on this story"),
            P(error, cls="error") if error else None,
            Form(
                Hidden(str(uuid4()), name="feedback_id"),
                Hidden(edition_id, name="edition_id"),
                Hidden(story_id, name="story_id") if story_id is not None else None,
                Hidden(csrf_token, name="csrf_token"),
                Label("Optional note", fr=f"{control_id}-note", cls="note-label"),
                Textarea(
                    event.note if event and event.note else (note or ""),
                    id=f"{control_id}-note",
                    name="note",
                    maxlength="2000",
                    placeholder="What was useful, or what should change?",
                ),
                Div(
                    Button("Positive", type="submit", name="rating", value="positive"),
                    Button("Negative", type="submit", name="rating", value="negative"),
                    Button("Save note", type="submit", name="rating", value=""),
                    cls="rating-actions",
                ),
                method="post",
                action=action,
                hx_post=action,
                hx_target=f"#{control_id}",
                hx_swap="outerHTML",
                hx_disabled_elt=f"#{control_id} button",
                cls="feedback-form",
            ),
            open=True if open_details else None,
        ),
        id=control_id,
        cls="feedback-control",
    )


def _research_flags(
    domain: ReaderDomain, report_version_id: Sha256
) -> dict[Sha256, SubjectResearchFlag]:
    """Load flagged subjects for one report; the reader works without them."""
    try:
        trigger_set = domain.read_research_flags(report_version_id)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return {}
    if trigger_set is None:
        return {}
    return {flag.theme_id: flag for flag in trigger_set.triggers if flag.flagged}


def _tiered_report_body(
    report: DailyReport,
    report_version_id: Sha256,
    csrf_token: str,
    latest_feedback: dict[tuple[str, str, str, str], NewsFeedbackEvent],
    research_flags: dict[Sha256, SubjectResearchFlag],
) -> tuple[str, tuple[FT, ...]]:
    main_sections = tuple(section for section in report.sections if section.tier == "main")
    worth_sections = tuple(
        section for section in report.sections if section.tier == "worth_knowing"
    )
    meta = (
        f"{len(main_sections)} main {'subject' if len(main_sections) == 1 else 'subjects'}"
        f" · {len(worth_sections)} in worth knowing"
        f" · {report.group_count} events"
        f" · {report.accepted_article_count} articles in this report"
    )
    position = _subject_positions()
    main_body = tuple(
        _report_section(
            section,
            next(position),
            report_version_id,
            csrf_token,
            latest_feedback,
            research_flags,
        )
        for section in main_sections
    )
    worth_body = tuple(
        _report_section(
            section,
            next(position),
            report_version_id,
            csrf_token,
            latest_feedback,
            research_flags,
        )
        for section in worth_sections
    )
    worth_block = _worth_knowing_section(len(worth_sections), worth_body)
    body = (*main_body, worth_block) if worth_block is not None else main_body
    return meta, body


def _subject_positions() -> Iterator[int]:
    """Number every visible subject continuously across tiers."""
    return iter(count(1))


def _worth_knowing_section(count_value: int, sections: tuple[FT, ...]) -> FT | None:
    if not sections:
        return None
    return Details(
        Summary(
            Span("Worth knowing", cls="wk-title"),
            Span(
                f"{count_value} {'subject' if count_value == 1 else 'subjects'}",
                cls="event-article-count",
            ),
        ),
        *sections,
        cls="worth-knowing",
    )


def _site_header(csrf_token: str) -> FT:
    return Header(
        Div(
            A("Press review", href="/", cls="brand"),
            Nav(
                A("Status", href="/status"),
                A("Daily reports", href="/reports"),
                aria_label="Main navigation",
                cls="site-links",
            ),
            Form(
                Hidden(csrf_token, name="csrf_token"),
                Button("Log out", type="submit"),
                method="post",
                action="/logout",
                cls="logout",
            ),
            cls="header-actions",
        ),
        cls="site-header",
    )


def _date_navigation(
    reports: tuple[DailyReportSummary, ...],
    selected_index: int | None,
    report_day: date,
    *,
    next_live: bool = False,
) -> tuple[FT, ...]:
    if selected_index is None:
        return (
            Nav(
                Span(),
                Time(_format_date(report_day), datetime=report_day.isoformat()),
                A("Latest report →", href="/"),
                aria_label="Report date navigation",
                cls="date-nav",
            ),
        )
    newer = reports[selected_index - 1] if selected_index > 0 else None
    older = reports[selected_index + 1] if selected_index + 1 < len(reports) else None
    newer_path = (
        f"/reports/{newer.report_version_id}" if newer else ("/today" if next_live else None)
    )
    older_path = f"/reports/{older.report_version_id}" if older else None
    adjacent_paths = tuple(
        path for path in (older_path, newer_path) if path is not None and path != "/today"
    )
    selected = reports[selected_index]
    navigation = Nav(
        A("← Previous day", href=older_path) if older_path else Span(),
        Time(_format_date(selected.day), datetime=selected.day.isoformat()),
        A("Next day →", href=newer_path) if newer_path else Span(),
        aria_label="Report date navigation",
        cls="date-nav",
    )
    if not adjacent_paths:
        return (navigation,)
    speculation_rules = Script(
        json.dumps(
            {
                "prefetch": [
                    {
                        "source": "list",
                        "urls": adjacent_paths,
                        "eagerness": "immediate",
                    }
                ]
            },
            separators=(",", ":"),
        ),
        type="speculationrules",
    )
    return speculation_rules, navigation


def _is_htmx(request: Request) -> bool:
    return "hx-request" in request.headers


def _login_page(next_url: str, error: str | None = None) -> tuple[Any, ...]:
    return (
        Title("Sign in | Press review"),
        Main(
            Div(
                P("Private access", cls="eyebrow"),
                H1("Press review"),
                P("Enter the password to open the daily report."),
                P(error, cls="error") if error else None,
                Form(
                    Hidden(next_url, name="next"),
                    Label("Password", fr="password"),
                    Input(
                        type="password",
                        id="password",
                        name="password",
                        required=True,
                        autocomplete="current-password",
                    ),
                    Button("Sign in", type="submit"),
                    method="post",
                    action="/login",
                ),
                cls="login-card",
            ),
            cls="login-shell",
        ),
    )


def _not_ready_page(request: Request, reports: tuple[DailyReportSummary, ...]) -> FtResponse:
    day = clock.bucharest_today()
    prior_href = _yesterday_report_href(reports)
    csrf_token = str(request.session["csrf_token"])
    return FtResponse(
        (
            Title("Today's report is not ready | Press review"),
            _site_header(csrf_token),
            Main(
                Div(
                    H1("Today's report is not ready yet"),
                    P("The report will stay on this page while its inputs become ready."),
                    A("Open yesterday's report →", href=prior_href) if prior_href else None,
                    _initial_status_strip(day, None, csrf_token),
                    cls="empty",
                )
            ),
        )
    )


def _not_found_page(request: Request) -> tuple[Any, ...]:
    return (
        Title("Report unavailable | Press review"),
        _site_header(str(request.session["csrf_token"])),
        Main(
            Div(
                H1("Report unavailable"),
                P("The requested version is not in the recent archive."),
                A("Open the latest report", href="/"),
                cls="empty",
            )
        ),
    )


def _unavailable_page(
    request: Request,
    reports: tuple[DailyReportSummary, ...] = (),
) -> FtResponse:
    prior_href = _yesterday_report_href(reports)
    return FtResponse(
        (
            Title("Report unavailable | Press review"),
            _site_header(str(request.session["csrf_token"])),
            Main(
                Div(
                    H1("The report could not be loaded"),
                    P("The data is temporarily unavailable. Try again in a few minutes."),
                    A("Retry report →", href="/"),
                    A("Open yesterday's report →", href=prior_href) if prior_href else None,
                    cls="empty",
                )
            ),
        ),
        status_code=503,
    )


def _yesterday_report_href(reports: tuple[DailyReportSummary, ...]) -> str | None:
    yesterday = clock.bucharest_today() - timedelta(days=1)
    prior = next((report for report in reports if report.day == yesterday), None)
    return f"/reports/{prior.report_version_id}" if prior is not None else None


def _safe_next(value: str) -> str:
    if value.startswith("/") and not value.startswith(("//", "/\\")):
        return value
    return "/"


def _format_date(value: date) -> str:
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    return f"{value.day} {months[value.month - 1]} {value.year}"


def _load_exact_report(
    request: Request, report_version_id: str, domain: ReaderDomain
) -> tuple[Any, ...] | FtResponse:
    try:
        report = domain.read_report(report_version_id)
    except DailyReportVersionNotFound:
        return FtResponse(_not_found_page(request), status_code=404)
    except (ResearchCatalogError, ResearchObjectUnavailable, ResearchObjectIntegrityError):
        return _unavailable_page(request)
    return (
        Title(f"{_format_date(report.day)} | Press review"),
        _site_header(str(request.session["csrf_token"])),
        Main(
            Div(
                P(A("← All reports", href="/reports"), cls="eyebrow"),
                P("Saved version. This page will keep showing the same daily report."),
                H1(_format_date(report.day)),
                *(
                    _report_section(
                        section,
                        position,
                        report_version_id,
                        str(request.session["csrf_token"]),
                        {},
                        {},
                    )
                    for position, section in enumerate(report.sections, start=1)
                    if not isinstance(section, DailyReportSection) or section.tier != "excluded"
                ),
                cls="reader",
            )
        ),
    )


def _load_report_archive(request: Request, domain: ReaderDomain) -> tuple[Any, ...] | FtResponse:
    raw_page = request.query_params.get("page", "1")
    if not raw_page.isdecimal() or int(raw_page) < 1 or int(raw_page) > 1000:
        return FtResponse(_not_found_page(request), status_code=404)
    page = int(raw_page)
    page_size = 30
    try:
        rows = domain.list_report_archive(page_size + 1, (page - 1) * page_size)
    except (ResearchCatalogError, ValueError):
        return _unavailable_page(request)
    visible = rows[:page_size]
    return (
        Title("Daily reports | Press review"),
        _site_header(str(request.session["csrf_token"])),
        Main(
            Div(
                P("Archive", cls="eyebrow"),
                H1("Daily reports"),
                P("Open any saved daily report."),
                Ul(
                    *(
                        Li(
                            A(
                                _format_date(report.day),
                                href=f"/reports/{report.report_version_id}",
                            )
                        )
                        for report in visible
                    ),
                    cls="archive-list",
                ),
                Nav(
                    A("← Newer", href=f"/reports?page={page - 1}") if page > 1 else None,
                    A("Older →", href=f"/reports?page={page + 1}")
                    if len(rows) > page_size
                    else None,
                    aria_label="Archive pages",
                    cls="archive-pages",
                ),
                cls="reader",
            )
        ),
    )
