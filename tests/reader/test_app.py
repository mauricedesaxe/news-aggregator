from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import pytest
from fasthtml.common import FastHTML
from starlette.requests import Request
from starlette.testclient import TestClient

from romanian_news import BUCHAREST
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.current_report import (
    CurrentDailyReport,
    CurrentDailyReportHead,
    DailyReportFreshness,
    ReportFresh,
    ReportInputsNotReady,
    ReportMissing,
    ReportStale,
)
from romanian_news.feedback import (
    ArticleFeedbackTarget,
    DailyReportSummary,
    GroupFeedbackTarget,
    NewsFeedbackCommand,
    NewsFeedbackEvent,
    ThemeFeedbackTarget,
)
from romanian_news.reader.app import (
    _STYLES,
    SESSION_COOKIE,
    ReaderDomain,
    ReaderSettings,
    _require_owner,
    create_app,
    decode_session_cookie,
)
from romanian_news.reader.dagster_repair import (
    DagsterRepairError,
    DagsterRepairUnavailable,
    RepairRun,
)
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportDocument,
    DailyReportSection,
    DailyReportSectionV2,
    DailyReportV2,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
    RetrospectiveCoverage,
    RetrospectiveDailyReport,
)
from romanian_news.research_triggers import (
    DailyResearchTriggerSet,
    EmptyResearchTriggerConstruction,
    ResearchTriggerPolicy,
    SubjectResearchFlag,
)
from romanian_news.storage import ResearchObjectUnavailable
from romanian_news.video_digest.models import EditionId, SlotName, StoryId
from romanian_news.video_digest.reader import (
    ReaderEditionOption,
    ReaderSubtitleAvailable,
    ReaderSubtitleFailed,
    ReaderVideoDigest,
    ReaderVideoEdition,
    ReaderVideoStory,
)
from romanian_news.video_digest_feedback import (
    VideoDigestFeedbackCommand,
    VideoDigestFeedbackEvent,
)

REPORT_VERSION = "a" * 64
OLDER_REPORT_VERSION = "b" * 64
NEWER_REPORT_VERSION = "e" * 64
UNRELATED_REPORT_VERSION = "1" * 64
LIVE_VERSION = "9" * 64
GROUP_ID = "c" * 64
ARTICLE_VERSION = "d" * 64
THEME_ID = "f" * 64
FEEDBACK_ID = "00000000-0000-4000-8000-000000000001"
VIDEO_EDITION = "2" * 64
OLDER_VIDEO_EDITION = "3" * 64
VIDEO_STORY = "4" * 64
TEST_SESSION_SECRET = "s" * 32


@dataclass
class DomainState:
    commands: list[NewsFeedbackCommand] = field(default_factory=list)
    events: list[NewsFeedbackEvent] = field(default_factory=list)


@dataclass
class Harness:
    app: Any
    state: DomainState


def _not_ready(_day: date) -> CurrentDailyReport | None:
    return None


def _live_report(day: date) -> CurrentDailyReport:
    return CurrentDailyReport(
        head=CurrentDailyReportHead(
            day=day,
            version_id=LIVE_VERSION,
            run_id="7" * 64,
            content_digest="6" * 64,
            r2_key="news/reports/live.json",
            input_time=datetime(2026, 9, 14, 6, 41, tzinfo=UTC),
        ),
        report=_daily_report().model_copy(update={"day": day}),
    )


@pytest.fixture
def harness() -> Harness:
    state = DomainState()

    def submit(command: NewsFeedbackCommand) -> NewsFeedbackEvent:
        state.commands.append(command)
        event = NewsFeedbackEvent(
            **command.model_dump(),
            created_at=datetime(2026, 8, 31, 9, 30, tzinfo=UTC),
        )
        state.events.append(event)
        return event

    domain = ReaderDomain(
        list_reports=lambda _limit: _report_summaries(),
        resolve_current_report_version=lambda version: (
            REPORT_VERSION if version == "e" * 64 else version
        ),
        read_report=lambda _version: _daily_report(),
        read_current_report=_live_report,
        read_feedback=lambda _version: tuple(state.events),
        submit_feedback=submit,
        read_research_flags=lambda _version: None,
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
            trust_proxy_headers=True,
        ),
        domain,
    )
    return Harness(app=app, state=state)


def test_anonymous_reader_redirects_to_login(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=%2F"


def test_login_rejects_wrong_password_and_sets_signed_session_on_success(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        failed = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            follow_redirects=False,
        )
        succeeded = client.post(
            "/login",
            data={"password": "correct horse", "next": "/"},
            follow_redirects=False,
        )

    assert failed.status_code == 401
    assert "The password is incorrect" in failed.text
    assert succeeded.status_code == 303
    assert succeeded.headers["location"] == "/"
    cookie = succeeded.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert SESSION_COOKIE in succeeded.headers["set-cookie"]


def test_login_limits_repeated_failures(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        for _attempt in range(5):
            response = client.post(
                "/login",
                data={"password": "wrong", "next": "/"},
                headers={"x-real-ip": "198.51.100.10"},
                follow_redirects=False,
            )
            assert response.status_code == 401

        locked = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            headers={"x-real-ip": "198.51.100.10"},
            follow_redirects=False,
        )
        owner = client.post(
            "/login",
            data={"password": "correct horse", "next": "/"},
            headers={"x-real-ip": "203.0.113.10"},
            follow_redirects=False,
        )

    assert locked.status_code == 429
    assert "Too many attempts" in locked.text
    assert owner.status_code == 303


def test_login_rejects_invalid_trusted_proxy_address(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        for attempt in range(5):
            response = client.post(
                "/login",
                data={"password": "wrong", "next": "/"},
                headers={"x-real-ip": f"invalid-{attempt}"},
            )
            assert response.status_code == 401
        locked = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            headers={"x-real-ip": "invalid-final"},
        )

    assert locked.status_code == 429


def test_login_ignores_proxy_identity_headers_unless_trusted() -> None:
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            trust_proxy_headers=False,
        ),
        ReaderDomain(
            list_reports=lambda _limit: (),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_not_ready,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        for attempt in range(5):
            response = client.post(
                "/login",
                data={"password": "wrong", "next": "/"},
                headers={"x-real-ip": f"203.0.113.{attempt + 1}"},
            )
            assert response.status_code == 401
        locked = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            headers={"x-real-ip": "203.0.113.99"},
        )

    assert locked.status_code == 429


def test_cookie_secure_setting_adds_secure_flag(harness: Harness) -> None:
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=True,
        ),
        ReaderDomain(
            list_reports=lambda _limit: _report_summaries(),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_live_report,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/login",
            data={"password": "correct horse", "next": "/"},
            follow_redirects=False,
        )

    assert "secure" in response.headers["set-cookie"].lower()


def test_reader_settings_reject_weak_authentication_secrets() -> None:
    with pytest.raises(ValueError):
        ReaderSettings(app_password="short", session_secret=TEST_SESSION_SECRET)
    with pytest.raises(ValueError):
        ReaderSettings(app_password="long-enough-password", session_secret="short")


def test_environment_settings_use_cookie_secure_configuration(monkeypatch) -> None:
    monkeypatch.setattr("romanian_news.reader.app.APP_PASSWORD", "correct horse")
    monkeypatch.setattr(
        "romanian_news.reader.app.SESSION_SECRET",
        TEST_SESSION_SECRET,
    )
    monkeypatch.setattr("romanian_news.reader.app.COOKIE_SECURE", False)
    monkeypatch.setattr("romanian_news.reader.app.TRUST_PROXY_HEADERS", False)
    monkeypatch.setattr("romanian_news.reader.app.NEWS_PUBLIC_MEDIA_BASE_URL", "")
    assert ReaderSettings.from_environment().cookie_secure is False
    assert ReaderSettings.from_environment().trust_proxy_headers is False

    monkeypatch.setattr("romanian_news.reader.app.COOKIE_SECURE", True)
    monkeypatch.setattr("romanian_news.reader.app.TRUST_PROXY_HEADERS", True)
    assert ReaderSettings.from_environment().cookie_secure is True
    assert ReaderSettings.from_environment().trust_proxy_headers is True


def test_report_uses_english_reader_copy_without_a_promotional_hero(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.get("/")

    assert response.status_code == 200
    assert '<html lang="en">' in response.text
    assert "Press review" in response.text
    assert "Previous report" in response.text
    assert "Subject 01" in response.text
    assert "Key points" in response.text
    assert "Tone assessment" in response.text
    assert "Reviewed articles" in response.text
    assert 'class="report-header"' not in response.text
    assert 'class="deck"' not in response.text
    assert response.text.index('class="date-nav"') < response.text.index('class="story-section"')
    assert 'hx-post="/report-status/check"' in response.text
    assert LIVE_VERSION in response.text
    assert f'href="/reports/{REPORT_VERSION}"' in response.text
    assert 'href="/today"' not in response.text
    assert (
        response.text.count('<meta name="viewport" content="width=device-width, initial-scale=1">')
        == 1
    )
    assert "Research suggested" not in response.text


def test_home_shows_waiting_page_when_today_is_not_ready() -> None:
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        ReaderDomain(
            list_reports=lambda _limit: (
                DailyReportSummary(
                    report_version_id=REPORT_VERSION,
                    day=datetime.now(BUCHAREST).date() - timedelta(days=1),
                ),
            ),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_not_ready,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/")

    assert response.status_code == 200
    assert "Today's report is not ready yet" in response.text
    assert LIVE_VERSION not in response.text
    assert f'href="/reports/{REPORT_VERSION}"' in response.text
    assert "Open yesterday's report" in response.text
    assert 'hx-post="/report-status/check"' in response.text


def test_today_route_reports_storage_unavailability() -> None:
    yesterday = datetime.now(BUCHAREST).date() - timedelta(days=1)

    def broken(_day: date) -> CurrentDailyReport:
        raise ResearchCatalogError("catalog unavailable")

    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        ReaderDomain(
            list_reports=lambda _limit: (
                DailyReportSummary(report_version_id=REPORT_VERSION, day=yesterday),
            ),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=broken,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/today")

    assert response.status_code == 503
    assert "Open yesterday's report" in response.text
    assert f'href="/reports/{REPORT_VERSION}"' in response.text


def test_newest_published_report_links_to_today(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.get(f"/reports/{REPORT_VERSION}")

    assert response.status_code == 200
    assert 'href="/today"' in response.text
    assert "Today →" in response.text
    assert f'href="/reports/{OLDER_REPORT_VERSION}"' in response.text


def test_sparse_report_navigation_names_published_reports() -> None:
    reports = (
        DailyReportSummary(report_version_id=REPORT_VERSION, day=date(2026, 8, 31)),
        DailyReportSummary(report_version_id=OLDER_REPORT_VERSION, day=date(2025, 10, 1)),
    )
    response = _read_report_response(
        _daily_report().model_copy(update={"day": date(2025, 10, 1)}),
        reports=reports,
        path=f"/reports/{OLDER_REPORT_VERSION}",
    )

    assert response.status_code == 200
    assert f'href="/reports/{REPORT_VERSION}">Next report →</a>' in response.text
    assert "Next day" not in response.text


def test_current_report_metadata_uses_typed_counts_in_display_order() -> None:
    report = _daily_report().model_copy(
        update={"theme_count": 2, "group_count": 3, "accepted_article_count": 4}
    )

    response = _read_report_response(report)

    assert re.search(
        r'<div class="report-meta">\s*<span>'
        r"1 main subject · 0 in worth knowing · 3 events · 4 articles in this report</span>"
        r"\s*</div>",
        response.text,
    )


def test_provisional_notice_is_visible_on_report_and_exact_version() -> None:
    report = _daily_report().model_copy(update={"coverage_status": "provisional"})

    current = _read_report_response(report)
    exact = _read_report_response(report, path=f"/reports/exact/{REPORT_VERSION}")

    assert current.status_code == exact.status_code == 200
    assert "Provisional report" in current.text
    assert "Provisional report" in exact.text
    assert "The latest report may change" in current.text
    assert "Saved version" in exact.text


def test_legacy_report_discloses_unknown_coverage() -> None:
    response = _read_report_response(_daily_report())

    assert "Coverage unknown" in response.text
    assert "Coverage status was not recorded" in response.text


def test_complete_report_has_no_coverage_warning() -> None:
    report = _daily_report().model_copy(update={"coverage_status": "complete"})

    response = _read_report_response(report)

    assert "Provisional report" not in response.text
    assert "Coverage unknown" not in response.text


def test_archived_report_metadata_uses_group_count_for_subjects_and_events() -> None:
    report = _archived_report().model_copy(update={"group_count": 5, "accepted_article_count": 6})

    response = _read_report_response(report)

    assert re.search(
        r'<div class="report-meta">\s*<span>'
        r"5 subjects · 5 events · 6 articles in this report</span>\s*</div>",
        response.text,
    )


def test_historical_report_discloses_later_capture_and_limited_sources() -> None:
    report = RetrospectiveDailyReport(
        **_daily_report().model_dump(exclude={"schema_version"}),
        retrospective=RetrospectiveCoverage(
            capture_started_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC),
            capture_ended_at=datetime(2026, 9, 27, 10, 5, tzinfo=UTC),
            included_outlets=("hotnews", "digi24"),
            discovered_url_count=200,
            verified_page_count=80,
            captured_article_count=40,
            coverage_note="Other configured outlets were not included.",
        ),
    )

    response = _read_report_response(report)

    assert response.status_code == 200
    assert "Historical report" in response.text
    assert "pages captured later" in response.text
    assert "2026-09-27 10:00 UTC" in response.text
    assert "Archive sources: hotnews, digi24" in response.text
    assert "Other configured outlets were not included." in response.text


def _tiered_report() -> DailyReport:
    main = _daily_report().sections[0]
    worth_event = main.events[0].model_copy(
        update={"group_id": "9" * 64, "title_ro": "Tram network expansion"}
    )
    worth = main.model_copy(
        update={
            "theme_id": "8" * 64,
            "title": "Public transport modernization",
            "summary": "The tram network expansion remains useful for the reader.",
            "tier": "worth_knowing",
            "semantic_rank": 1,
            "consequence_rationale": "Useful regionally, with no direct national impact.",
            "citations": (),
            "events": (worth_event,),
        }
    )
    excluded = main.model_copy(
        update={
            "theme_id": "7" * 64,
            "title": "Uncorroborated opinion subject",
            "summary": "An opinion subject that should not appear.",
            "tier": "excluded",
            "semantic_rank": 1,
            "consequence_rationale": "Opinion without a corroborated event.",
            "citations": (),
            "events": (),
        }
    )
    return _daily_report().model_copy(
        update={"theme_count": 3, "sections": (main, worth, excluded)}
    )


def test_tiered_report_orders_main_first_and_collapses_worth_knowing() -> None:
    response = _read_report_response(_tiered_report())

    assert response.status_code == 200
    opening = re.search(r'<details class="worth-knowing"[^>]*>', response.text)
    assert opening is not None
    assert "open" not in opening.group(0)
    assert "Worth knowing" in response.text
    assert "1 subject" in response.text
    main_title = "Budget policy"
    worth_title = "Public transport modernization"
    assert response.text.index(f"<h2>{main_title}</h2>") < response.text.index(
        '<details class="worth-knowing">'
    )
    assert response.text.index('<details class="worth-knowing">') < response.text.index(
        f"<h2>{worth_title}</h2>"
    )
    worth_block = response.text.split('<details class="worth-knowing">', 1)[1]
    assert '<details class="subject-events"><summary>1 article group</summary>' in worth_block
    assert "Uncorroborated opinion subject" not in response.text
    assert "Subject 01" in response.text
    assert "Subject 02" in response.text
    assert "Subject 03" not in response.text
    assert "Budget decisions with direct national effects." in response.text
    assert 'title="Guvernul a publicat proiectul."' in response.text
    assert "Cited evidence" in response.text
    assert "Cited evidence" not in worth_block


def test_schema_v2_report_keeps_the_flat_rendering_without_tiers() -> None:
    section = DailyReportSectionV2(
        theme_id=THEME_ID,
        title="Budget policy",
        summary="The draft budget and reactions form the subject of the day.",
        events=(_daily_report().sections[0].events[0],),
    )
    report = DailyReportV2(
        day=date(2026, 8, 31),
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=(section,),
    )

    response = _read_report_response(report)

    assert response.status_code == 200
    assert '<details class="worth-knowing">' not in response.text
    assert "in worth knowing" not in response.text
    assert "Budget policy" in response.text
    assert "Subject 01" in response.text
    assert '<details class="subject-events"><summary>1 article group</summary>' in response.text


def test_report_prefetches_both_adjacent_reports_immediately() -> None:
    reports = (
        DailyReportSummary(
            report_version_id=NEWER_REPORT_VERSION,
            day=date(2026, 9, 1),
        ),
        DailyReportSummary(
            report_version_id=REPORT_VERSION,
            day=date(2026, 8, 31),
        ),
        DailyReportSummary(
            report_version_id=OLDER_REPORT_VERSION,
            day=date(2026, 8, 30),
        ),
        DailyReportSummary(
            report_version_id=UNRELATED_REPORT_VERSION,
            day=date(2026, 8, 29),
        ),
    )

    response = _read_report_response(
        _daily_report(),
        reports=reports,
        path=f"/reports/{REPORT_VERSION}",
    )
    rules = _speculation_rules(response.text)

    assert rules == {
        "prefetch": [
            {
                "source": "list",
                "urls": [
                    f"/reports/{OLDER_REPORT_VERSION}",
                    f"/reports/{NEWER_REPORT_VERSION}",
                ],
                "eagerness": "immediate",
            }
        ]
    }
    assert f"/reports/{UNRELATED_REPORT_VERSION}" not in json.dumps(rules)
    script_source = next(
        directive
        for directive in response.headers["content-security-policy"].split("; ")
        if directive.startswith("script-src ")
    )
    assert script_source == "script-src 'self' 'inline-speculation-rules'"
    assert response.headers["cache-control"] == "no-store"


def test_report_prefetches_only_the_available_boundary_neighbor() -> None:
    reports = (
        DailyReportSummary(
            report_version_id=REPORT_VERSION,
            day=date(2026, 8, 31),
        ),
        DailyReportSummary(
            report_version_id=OLDER_REPORT_VERSION,
            day=date(2026, 8, 30),
        ),
        DailyReportSummary(
            report_version_id=UNRELATED_REPORT_VERSION,
            day=date(2026, 8, 29),
        ),
    )

    response = _read_report_response(
        _daily_report(),
        reports=reports,
        path=f"/reports/{REPORT_VERSION}",
    )
    rules = _speculation_rules(response.text)

    assert rules == {
        "prefetch": [
            {
                "source": "list",
                "urls": [f"/reports/{OLDER_REPORT_VERSION}"],
                "eagerness": "immediate",
            }
        ]
    }
    assert f"/reports/{UNRELATED_REPORT_VERSION}" not in json.dumps(rules)


def test_subject_article_groups_are_closed_below_visible_subject_content() -> None:
    response = _read_report_response(_daily_report())
    subject_content, marker, groups_content = response.text.partition(
        '<details class="subject-events"><summary>1 article group</summary>'
    )

    assert marker
    assert "Subject 01" in subject_content
    assert "<h2>Budget policy</h2>" in subject_content
    assert "The draft budget and reactions form the subject of the day." in subject_content
    assert "Budget decisions with direct national effects." in subject_content
    assert "Feedback on this subject" in subject_content
    assert "The public budget enters debate" in groups_content
    assert groups_content.lstrip().startswith('<details class="event-disclosure event-section">')
    assert '<details open class="subject-events">' not in response.text


def test_event_disclosure_is_closed_with_its_title_and_singular_count() -> None:
    response = _read_report_response(_daily_report())
    opening = '<details class="event-disclosure event-section">'
    _, marker, event_content = response.text.partition(opening)
    summary, summary_end, expanded_content = event_content.partition("</summary>")

    assert marker
    assert summary_end
    assert re.fullmatch(
        r"<summary>\s*<h3>The public budget enters debate</h3>\s*"
        r"<span class=\"event-article-count\">1 article</span>\s*",
        summary,
    )
    assert '<details open class="event-disclosure event-section">' not in response.text
    assert "Event 1" in expanded_content
    assert "The government published the draft" in expanded_content
    assert "Key points" in expanded_content
    assert "Tone assessment" in expanded_content
    assert "Uncertainty:" in expanded_content
    assert "Feedback on this event" in expanded_content
    assert "Reviewed articles (1 article)" in expanded_content
    assert "Feedback on this article" in expanded_content
    assert ".event-disclosure > summary { cursor: pointer; }" in response.text
    assert ".event-disclosure > summary:focus-visible" in response.text
    assert ".event-disclosure > summary h3 { display: inline; }" in response.text
    assert ".event-article-count" in response.text and "white-space: nowrap" in response.text
    assert ".event-disclosure[open] > summary { margin-bottom: 1.5rem; }" in response.text
    subject_content, articles_marker, article_content = response.text.partition(
        '<details class="articles">'
    )
    assert articles_marker
    assert '<details open class="articles">' not in response.text
    assert ".articles > summary { cursor: pointer; }" in _STYLES
    assert ".articles > summary:focus-visible" in _STYLES
    assert "Feedback on this subject" in subject_content
    assert (
        '<a href="https://example.com/analiza" target="_blank" rel="noreferrer">' in article_content
    )
    assert "Feedback on this article" in article_content


def test_two_events_render_as_separate_closed_unnamed_disclosures() -> None:
    report = _daily_report()
    section = report.sections[0]
    event = section.events[0]
    second = event.model_copy(
        update={
            "group_id": "2" * 64,
            "title_ro": "A second public debate",
        }
    )
    report = report.model_copy(
        update={
            "group_count": 2,
            "sections": (section.model_copy(update={"events": (event, second)}),),
        }
    )

    response = _read_report_response(report)
    event_details = re.findall(
        r"<details[^>]*class=\"event-disclosure event-section\"[^>]*>",
        response.text,
    )

    assert event_details == [
        '<details class="event-disclosure event-section">',
        '<details class="event-disclosure event-section">',
    ]
    assert '<details class="subject-events"><summary>2 article groups</summary>' in response.text


def test_single_report_ships_no_speculation_rules() -> None:
    reports = (
        DailyReportSummary(
            report_version_id=REPORT_VERSION,
            day=date(2026, 8, 31),
        ),
    )

    response = _read_report_response(
        _daily_report(),
        reports=reports,
        path=f"/reports/{REPORT_VERSION}",
    )

    assert '<script type="speculationrules">' not in response.text
    assert 'class="date-nav"' in response.text


def test_archived_report_event_is_closed_without_the_event_section_class() -> None:
    response = _read_report_response(_archived_report())

    assert re.search(
        r'<details class="event-disclosure">\s*<summary>\s*<h3>',
        response.text,
    )
    assert '<details open class="event-disclosure">' not in response.text
    assert '<details class="subject-events">' not in response.text
    assert "Subject 01" in response.text
    assert "Event 1" not in response.text
    assert "Feedback on this event" in response.text
    assert "Reviewed articles (1 article)" in response.text


def test_reviewed_articles_disclosure_uses_plural_count_and_preserves_order() -> None:
    report = _daily_report()
    section = report.sections[0]
    event = section.events[0]
    second = event.articles[0].model_copy(
        update={
            "article_version_id": "e" * 64,
            "title": "A second account",
        }
    )
    report = report.model_copy(
        update={
            "accepted_article_count": 2,
            "sections": (
                section.model_copy(
                    update={
                        "events": (
                            event.model_copy(update={"articles": (*event.articles, second)}),
                        )
                    }
                ),
            ),
        }
    )

    response = _read_report_response(report)

    assert re.search(
        r"<summary>\s*<h3>The public budget enters debate</h3>\s*"
        r'<span class="event-article-count">2 articles</span>\s*</summary>',
        response.text,
    )
    assert "<summary>Reviewed articles (2 articles)</summary>" in response.text
    assert response.text.index("Analiză economică a proiectului") < response.text.index(
        "A second account"
    )


def test_reader_marks_key_points_and_gives_only_them_larger_text(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.get("/")

    assert 'class="fact key-points"' in response.text
    assert ".key-points { font-size: 1.1rem;" in response.text
    assert 'class="fact"' not in response.text
    assert "Differences between sources" not in response.text


def test_singleton_with_bogus_differences_hides_differences_block() -> None:
    response = _read_report_response(_daily_report())

    assert "Differences between sources" not in response.text
    assert "Sources assess the effect of taxes differently." not in response.text


def test_two_articles_from_one_outlet_hide_differences_block() -> None:
    report = _daily_report()
    section = report.sections[0]
    event = section.events[0]
    second = event.articles[0].model_copy(
        update={"article_version_id": "e" * 64, "title": "A second account"}
    )
    report = report.model_copy(
        update={
            "sections": (
                section.model_copy(
                    update={
                        "events": (
                            event.model_copy(update={"articles": (*event.articles, second)}),
                        )
                    }
                ),
            )
        }
    )

    response = _read_report_response(report)

    assert "Differences between sources" not in response.text


def test_two_distinct_outlets_show_differences_block() -> None:
    report = _daily_report()
    section = report.sections[0]
    event = section.events[0]
    second = event.articles[0].model_copy(
        update={
            "article_version_id": "e" * 64,
            "outlet_id": "another-outlet",
            "title": "A second account",
        }
    )
    report = report.model_copy(
        update={
            "sections": (
                section.model_copy(
                    update={
                        "events": (
                            event.model_copy(update={"articles": (*event.articles, second)}),
                        )
                    }
                ),
            )
        }
    )

    response = _read_report_response(report)

    assert "Differences between sources" in response.text
    assert "Sources assess the effect of taxes differently." in response.text


def test_reader_declares_automatic_dark_mode_without_light_component_backgrounds() -> None:
    response = _read_report_response(_daily_report())

    assert '<meta name="color-scheme" content="light dark">' in response.text
    assert "@media (prefers-color-scheme: dark)" in _STYLES
    assert "--paper: #171714;" in _STYLES
    assert "background: white" not in _STYLES
    assert ".assessment { background: #f0ede4;" not in _STYLES
    assert ".site-header { border-bottom: 1px solid var(--line); background: rgba(" not in _STYLES


def test_archived_report_version_redirects_to_current_version(harness: Harness) -> None:
    archived_version = "e" * 64
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.get(f"/reports/{archived_version}", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == f"/reports/{REPORT_VERSION}"


def test_feedback_rejects_invalid_csrf_before_domain_call(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("report", csrf_token="wrong"),
        )

    assert response.status_code == 403
    assert harness.state.commands == []


@pytest.mark.parametrize(
    ("kind", "extra", "target_type"),
    [
        ("theme", {"theme_id": THEME_ID}, ThemeFeedbackTarget),
        ("group", {"group_id": GROUP_ID}, GroupFeedbackTarget),
        (
            "article",
            {"group_id": GROUP_ID, "article_version_id": ARTICLE_VERSION},
            ArticleFeedbackTarget,
        ),
    ],
)
def test_feedback_scopes_reach_typed_domain_commands(
    harness: Harness,
    kind: str,
    extra: dict[str, str],
    target_type: type,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form(kind, csrf_token=csrf_token, **extra),
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == f"/reports/{REPORT_VERSION}"
    command = harness.state.commands[-1]
    assert isinstance(command.target, target_type)
    assert command.target.report_version_id == REPORT_VERSION
    assert command.actor == "owner"
    assert command.feedback_id == UUID(FEEDBACK_ID)


def test_negative_feedback_redirects_to_saved_state(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        form = _feedback_form("report", csrf_token=csrf_token, rating="negative")
        response = client.post("/feedback", data=form)

    assert response.status_code == 200
    assert "Feedback saved: Negative" in response.text
    assert "Feedback on this report" in response.text
    assert harness.state.commands[-1].rating == "negative"


def test_note_only_feedback_reaches_the_domain_as_a_stripped_note(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form(
                "report",
                csrf_token=csrf_token,
                rating="",
                note="  The lead is too broad.  ",
            ),
            follow_redirects=False,
        )

    assert response.status_code == 303
    command = harness.state.commands[-1]
    assert command.rating is None
    assert command.note == "The lead is too broad."


def test_latest_saved_state_and_note_render_after_reload(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        client.post(
            "/feedback",
            data=_feedback_form(
                "article",
                csrf_token=csrf_token,
                group_id=GROUP_ID,
                article_version_id=ARTICLE_VERSION,
                note="The title captures the difference between sources.",
            ),
        )
        response = client.get("/")

    assert "Feedback saved: Positive" in response.text
    assert "The title captures the difference between sources." in response.text


def test_empty_analysis_details_render_no_placeholder_block() -> None:
    report = _daily_report()
    section = report.sections[0]
    event = section.events[0].model_copy(update={"disagreements_ro": (), "uncertainty_ro": None})
    report = report.model_copy(
        update={"sections": (section.model_copy(update={"events": (event,)}),)}
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        ReaderDomain(
            list_reports=lambda _limit: _report_summaries(),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: report,
            read_current_report=lambda day: _live_report(day).model_copy(
                update={"report": report.model_copy(update={"day": day})}
            ),
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )

    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/")

    assert response.text.count('class="fact key-points"') == 1
    assert 'class="fact"' not in response.text
    assert "Differences between sources" not in response.text
    assert "No relevant differences were identified" not in response.text
    assert 'class="uncertainty"' not in response.text
    assert "None" not in response.text


def test_logout_clears_session_and_protects_reader_again(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        logout = client.post(
            "/logout",
            data={"csrf_token": csrf_token},
            follow_redirects=False,
        )
        reader = client.get("/", follow_redirects=False)

    assert logout.status_code == 303
    assert logout.headers["location"] == "/login"
    assert reader.status_code == 303


def test_security_headers_are_present_on_public_and_private_responses(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        health = client.get("/healthz")
        protected = client.get("/", follow_redirects=False)

    for response in (health, protected):
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"
        assert "media-src" not in response.headers["content-security-policy"]


def test_health_does_not_touch_storage() -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("health called storage")

    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        ReaderDomain(
            list_reports=unexpected,
            resolve_current_report_version=unexpected,
            read_report=unexpected,
            read_current_report=unexpected,
            read_feedback=unexpected,
            submit_feedback=unexpected,
            read_research_flags=unexpected,
        ),
    )
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


def test_readiness_checks_storage() -> None:
    checks = []
    domain = ReaderDomain(
        list_reports=lambda _limit: (),
        resolve_current_report_version=lambda version: version,
        read_report=lambda _version: _daily_report(),
        read_current_report=_not_ready,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
        check_readiness=lambda: checks.append("ready"),
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        domain,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")
        cached = client.get("/readyz")

    assert response.status_code == 200
    assert response.text == "ok"
    assert cached.status_code == 200
    assert checks == ["ready"]


def test_readiness_reports_storage_failure() -> None:
    def unavailable() -> None:
        raise ResearchObjectUnavailable("R2 unavailable")

    domain = ReaderDomain(
        list_reports=lambda _limit: (),
        resolve_current_report_version=lambda version: version,
        read_report=lambda _version: _daily_report(),
        read_current_report=_not_ready,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
        check_readiness=unavailable,
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        domain,
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.text == "unavailable"


def test_empty_and_unavailable_report_states_are_distinct() -> None:
    settings = ReaderSettings(
        app_password="correct horse",
        session_secret=TEST_SESSION_SECRET,
    )
    empty = create_app(
        settings,
        ReaderDomain(
            list_reports=lambda _limit: (),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_not_ready,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )

    def unavailable(_limit: int):
        raise ResearchCatalogError("catalog unavailable")

    failed = create_app(
        settings,
        ReaderDomain(
            list_reports=unavailable,
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_not_ready,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(empty) as empty_client, TestClient(failed) as failed_client:
        _login(empty, empty_client)
        _login(failed, failed_client)
        empty_response = empty_client.get("/")
        failed_response = failed_client.get("/")

    assert empty_response.status_code == 200
    assert "Today's report is not ready yet" in empty_response.text
    assert failed_response.status_code == 503
    assert "The report could not be loaded" in failed_response.text
    assert '<a href="/">Retry report →</a>' in failed_response.text


def test_feedback_storage_failure_does_not_claim_the_feedback_was_saved() -> None:
    def unavailable(_command: NewsFeedbackCommand) -> NewsFeedbackEvent:
        raise ResearchObjectUnavailable("object unavailable")

    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        ReaderDomain(
            list_reports=lambda _limit: _report_summaries(),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_live_report,
            read_feedback=lambda _version: (),
            submit_feedback=unavailable,
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        csrf_token = _login(app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("report", csrf_token=csrf_token),
        )

    assert response.status_code == 503
    assert "Feedback was not saved" in response.text


def test_feedback_form_targets_its_own_control_for_htmx_swaps(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.get("/")

    assert 'hx-post="/feedback"' in response.text
    assert 'hx-target="#feedback-report"' in response.text
    assert 'hx-swap="outerHTML"' in response.text
    assert 'hx-disabled-elt="#feedback-report button"' in response.text
    assert 'action="/feedback"' in response.text
    assert 'method="post"' in response.text
    assert response.text.count("Feedback on this report") == 1
    assert response.text.count("Feedback on this subject") == 1
    assert f'value="{THEME_ID}" name="theme_id"' in response.text
    assert response.text.count("Feedback on this event") == 1
    assert response.text.count("Feedback on this article") == 1


def test_reader_loads_vendored_htmx_from_same_origin(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        page = client.get("/")
        script = client.get("/htmx.min.js")
    with TestClient(harness.app) as anonymous_client:
        anonymous = anonymous_client.get("/htmx.min.js")
        login = anonymous_client.get("/login")

    assert '<script src="/htmx.min.js" defer></script>' in page.text
    assert 'name="htmx-config"' in page.text
    assert "[45].." in page.text
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert anonymous.status_code == 200
    assert anonymous.headers["content-type"].startswith("text/javascript")
    assert len(script.text) > 10000
    assert '<script src="/htmx.min.js" defer></script>' in login.text
    assert "Feedback on this report" not in login.text


def test_unauthenticated_htmx_request_asks_htmx_to_redirect(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        response = client.get(
            "/",
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )

    assert response.status_code == 401
    assert response.headers["HX-Redirect"].startswith("/login?next=")
    assert response.text == ""


def test_require_owner_rejects_traversal_paths() -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/../../pyproject.toml",
            "query_string": b"",
            "headers": [],
        }
    )

    response = _require_owner(request)

    assert response is not None
    assert response.status_code == 404


def test_negative_htmx_feedback_swaps_the_control_with_saved_state(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("report", csrf_token=csrf_token, rating="negative"),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert "<html" not in response.text
    assert response.text.startswith('<div id="feedback-report"')
    assert "Feedback saved: Negative" in response.text
    assert "<details open>" in response.text
    assert 'hx-post="/feedback"' in response.text
    assert 'hx-target="#feedback-report"' in response.text


@pytest.mark.parametrize(
    ("kind", "extra", "control_id"),
    [
        ("report", {}, "feedback-report"),
        ("theme", {"theme_id": THEME_ID}, f"feedback-theme-{THEME_ID}"),
        ("group", {"group_id": GROUP_ID}, f"feedback-group-{GROUP_ID}"),
        (
            "article",
            {"group_id": GROUP_ID, "article_version_id": ARTICLE_VERSION},
            f"feedback-article-{GROUP_ID}-{ARTICLE_VERSION}",
        ),
    ],
)
def test_htmx_feedback_swaps_each_scope_with_saved_state(
    harness: Harness,
    kind: str,
    extra: dict[str, str],
    control_id: str,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form(kind, csrf_token=csrf_token, **extra),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert response.text.startswith(f'<div id="{control_id}"')
    assert "Feedback saved: Positive" in response.text
    assert f'hx-target="#{control_id}"' in response.text
    assert "<details open>" in response.text


def test_htmx_feedback_validation_error_swaps_into_the_control_and_keeps_note(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form(
                "report",
                csrf_token=csrf_token,
                rating="bogus",
                note="The deficit comparison needs a source.",
            ),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 400
    assert "<html" not in response.text
    assert 'id="feedback-report"' in response.text
    assert "Feedback could not be saved." in response.text
    assert 'hx-target="#feedback-report"' in response.text
    assert "The deficit comparison needs a source.</textarea>" in response.text
    assert "<details open>" in response.text
    assert harness.state.commands == []


def test_htmx_feedback_csrf_failure_renders_the_control_with_the_error(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("report", csrf_token="wrong"),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 403
    assert "<html" not in response.text
    assert 'id="feedback-report"' in response.text
    assert "The request expired. Reload the page." in response.text
    assert "<details open>" in response.text
    assert harness.state.commands == []


def test_htmx_feedback_unknown_target_kind_returns_bare_error_paragraph(
    harness: Harness,
) -> None:
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("bogus", csrf_token=csrf_token),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 400
    assert "Feedback could not be saved." in response.text
    assert 'id="feedback-' not in response.text
    assert harness.state.commands == []


def test_htmx_feedback_storage_failure_swaps_into_the_control() -> None:
    def unavailable(_command: NewsFeedbackCommand) -> NewsFeedbackEvent:
        raise ResearchObjectUnavailable("object unavailable")

    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        ReaderDomain(
            list_reports=lambda _limit: _report_summaries(),
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: _daily_report(),
            read_current_report=_live_report,
            read_feedback=lambda _version: (),
            submit_feedback=unavailable,
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        csrf_token = _login(app, client)
        response = client.post(
            "/feedback",
            data=_feedback_form("report", csrf_token=csrf_token),
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 503
    assert "<html" not in response.text
    assert 'id="feedback-report"' in response.text
    assert 'hx-target="#feedback-report"' in response.text
    assert "Feedback was not saved." in response.text
    assert "<details open>" in response.text


def _read_report_response(
    report: DailyReportDocument,
    *,
    reports: tuple[DailyReportSummary, ...] | None = None,
    path: str | None = None,
):
    report_summaries = reports if reports is not None else _report_summaries()
    route = path or f"/reports/{report_summaries[0].report_version_id}"
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        ReaderDomain(
            list_reports=lambda _limit: report_summaries,
            resolve_current_report_version=lambda version: version,
            read_report=lambda _version: report,
            read_current_report=_live_report,
            read_feedback=lambda _version: (),
            submit_feedback=lambda command: _event(command),
            read_research_flags=lambda _version: None,
        ),
    )
    with TestClient(app) as client:
        _login(app, client)
        return client.get(route)


def _speculation_rules(response_text: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="speculationrules">(.*?)</script>',
        response_text,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def _login(app: FastHTML, client: TestClient) -> str:
    response = client.post(
        "/login",
        data={"password": "correct horse", "next": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    session = decode_session_cookie(app, client.cookies[SESSION_COOKIE])
    return str(session["csrf_token"])


def _feedback_form(
    kind: str,
    *,
    csrf_token: str,
    rating: str = "positive",
    theme_id: str = "",
    group_id: str = "",
    article_version_id: str = "",
    note: str = "Clear and useful.",
) -> dict[str, str]:
    return {
        "feedback_id": FEEDBACK_ID,
        "target_kind": kind,
        "report_version_id": REPORT_VERSION,
        "theme_id": theme_id,
        "group_id": group_id,
        "article_version_id": article_version_id,
        "rating": rating,
        "note": note,
        "csrf_token": csrf_token,
    }


def _report_summaries() -> tuple[DailyReportSummary, ...]:
    return (
        DailyReportSummary(
            report_version_id=REPORT_VERSION,
            day=date(2026, 8, 31),
        ),
        DailyReportSummary(
            report_version_id=OLDER_REPORT_VERSION,
            day=date(2026, 8, 30),
        ),
    )


def _daily_report() -> DailyReport:
    event = ReportEvent(
        group_id=GROUP_ID,
        title_ro="The public budget enters debate",
        summary_ro="The government published the draft, and reactions differ between sources.",
        key_points_ro=("The estimated deficit remains the central issue.",),
        disagreements_ro=("Sources assess the effect of taxes differently.",),
        uncertainty_ro="Budget execution could change the estimate.",
        sentiment_label="mixed",
        sentiment_score=-0.1,
        sentiment_rationale_ro="The tone combines caution with moderate expectations.",
        articles=(
            ReportArticle(
                article_version_id=ARTICLE_VERSION,
                outlet_id="presa-exemplu",
                title="Analiză economică a proiectului",
                canonical_url="https://example.com/analiza",
                sentiment_label="neutral",
                sentiment_score=0.0,
            ),
        ),
    )
    return DailyReport(
        day=date(2026, 8, 31),
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=(
            DailyReportSection(
                theme_id=THEME_ID,
                title="Budget policy",
                summary="The draft budget and reactions form the subject of the day.",
                tier="main",
                semantic_rank=1,
                consequence_rationale="Budget decisions with direct national effects.",
                citations=(
                    ReportSubjectCitation(
                        article_version_id=ARTICLE_VERSION,
                        evidence_quote="Guvernul a publicat proiectul.",
                    ),
                ),
                events=(event,),
            ),
        ),
    )


def _archived_report() -> ArchivedDailyReport:
    event = _daily_report().sections[0].events[0]
    return ArchivedDailyReport(
        day=date(2026, 8, 31),
        accepted_article_count=1,
        group_count=1,
        sections=(ArchivedDailyReportSection(**event.model_dump()),),
    )


def _event(command: NewsFeedbackCommand) -> NewsFeedbackEvent:
    return NewsFeedbackEvent(
        **command.model_dump(),
        created_at=datetime(2026, 8, 31, 9, 30, tzinfo=UTC),
    )


def test_reader_has_no_youtube_surface(harness: Harness) -> None:
    with TestClient(harness.app) as client:
        _login(harness.app, client)
        queue = client.get("/youtube-reviews")
        detail = client.get(f"/youtube-reviews/{'c' * 64}")
        decision = client.post(f"/youtube-reviews/{'c' * 64}/decision")
        reports = client.get("/")
    for response in (queue, detail, decision):
        assert response.status_code == 404
    assert "youtube" not in reports.text.casefold()


def _research_flags() -> DailyResearchTriggerSet:
    return DailyResearchTriggerSet(
        day=date(2026, 8, 31),
        request_id="2" * 64,
        policy=ResearchTriggerPolicy(
            policy_id="daily-research-flags-v1",
            model="google/gemini-3.8-flash",
            prompt_digest="3" * 64,
            input_policy="schema-v3-report-sections-v1",
            strength_threshold=0.3,
            correction_policy="complete-json-once-v1",
            temperature=0,
            max_tokens=4000,
        ),
        policy_digest="4" * 64,
        report=ArtifactReference(
            artifact_id="news:daily:2026-08-31",
            version_id=REPORT_VERSION,
            content_digest="5" * 64,
            r2_key="news/reports/daily/2026-08-31/x.json",
        ),
        construction=EmptyResearchTriggerConstruction(),
        triggers=(
            SubjectResearchFlag(
                theme_id=THEME_ID,
                gap_strength=0.72,
                flagged=True,
                question="How does the projected deficit compare with 2025?",
                evidence_types=("historical_data",),
            ),
        ),
    )


def _flag_harness() -> Harness:
    state = DomainState()

    def submit(command: NewsFeedbackCommand) -> NewsFeedbackEvent:
        state.commands.append(command)
        event = NewsFeedbackEvent(
            **command.model_dump(),
            created_at=datetime(2026, 8, 31, 9, 30, tzinfo=UTC),
        )
        state.events.append(event)
        return event

    domain = ReaderDomain(
        list_reports=lambda _limit: _report_summaries(),
        resolve_current_report_version=lambda version: (
            REPORT_VERSION if version == "e" * 64 else version
        ),
        read_report=lambda _version: _daily_report(),
        read_current_report=_live_report,
        read_feedback=lambda _version: tuple(state.events),
        submit_feedback=submit,
        read_research_flags=lambda _version: _research_flags(),
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        domain,
    )
    return Harness(app=app, state=state)


def test_research_flag_vote_submits_note_only_feedback_and_swaps_saved_state() -> None:
    harness = _flag_harness()
    with TestClient(harness.app) as client:
        csrf_token = _login(harness.app, client)
        page = client.get("/")
        response = client.post(
            "/feedback",
            data={
                "feedback_id": FEEDBACK_ID,
                "target_kind": "theme",
                "report_version_id": REPORT_VERSION,
                "theme_id": THEME_ID,
                "csrf_token": csrf_token,
                "flag_vote": "1",
                "flag_note": "research-flag: accepted",
            },
            headers={"HX-Request": "true"},
        )

    assert page.status_code == 200
    assert "Research suggested" in page.text
    assert "gap strength 0.72" in page.text
    assert "How does the projected deficit compare with 2025?" in page.text
    radios = re.findall(r'<input[^>]*type="radio"[^>]*>', page.text)
    assert len(radios) == 2
    for radio, value in zip(radios, ("right", "wrong"), strict=True):
        assert 'name="flag_verdict"' in radio
        assert f'value="{value}"' in radio
        assert " required" in radio
    assert response.status_code == 200
    command = harness.state.commands[-1]
    assert command.rating is None
    assert command.note == "research-flag: accepted"
    assert command.target == ThemeFeedbackTarget(
        report_version_id=REPORT_VERSION,
        theme_id=THEME_ID,
    )
    assert "Flag feedback saved" in response.text
    assert "Feedback saved:" not in response.text


@dataclass
class StatusCalls:
    current_reads: int = 0
    freshness_reads: int = 0
    repair_requests: int = 0
    repair_reads: int = 0


def _status_harness(
    *,
    current: CurrentDailyReport | None,
    freshness: DailyReportFreshness | Exception,
    repair: RepairRun | Exception,
    reports: tuple[DailyReportSummary, ...] | None = None,
) -> tuple[Any, StatusCalls]:
    calls = StatusCalls()
    report_summaries = _report_summaries() if reports is None else reports

    def read_current(_day: date) -> CurrentDailyReport | None:
        calls.current_reads += 1
        return current

    def read_freshness(_day: date) -> DailyReportFreshness:
        calls.freshness_reads += 1
        if isinstance(freshness, Exception):
            raise freshness
        return freshness

    def request_repair(_day: date) -> RepairRun:
        calls.repair_requests += 1
        if isinstance(repair, Exception):
            raise repair
        return repair

    def read_repair(_run_id: str) -> RepairRun:
        calls.repair_reads += 1
        if isinstance(repair, Exception):
            raise repair
        return repair

    report = current.report if current is not None else _daily_report()
    domain = ReaderDomain(
        list_reports=lambda _limit: report_summaries,
        resolve_current_report_version=lambda version: version,
        read_report=lambda _version: report,
        read_current_report=read_current,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
        read_freshness=read_freshness,
        request_repair=request_repair,
        read_repair=read_repair,
    )
    return (
        create_app(
            ReaderSettings(
                app_password="correct horse",
                session_secret=TEST_SESSION_SECRET,
                cookie_secure=False,
            ),
            domain,
        ),
        calls,
    )


def _today_report(version: str = LIVE_VERSION) -> CurrentDailyReport:
    day = datetime.now(BUCHAREST).date()
    return CurrentDailyReport(
        head=CurrentDailyReportHead(
            day=day,
            version_id=version,
            run_id="7" * 64,
            content_digest="6" * 64,
            r2_key="news/reports/today.json",
            input_time=datetime(2026, 9, 15, 7, 41, tzinfo=UTC),
        ),
        report=_daily_report().model_copy(update={"day": day}),
    )


def _status_form(csrf_token: str, current: CurrentDailyReport | None) -> dict[str, str]:
    return {
        "csrf_token": csrf_token,
        "day": datetime.now(BUCHAREST).date().isoformat(),
        "viewed_version": current.head.version_id if current is not None else "",
    }


def test_initial_current_report_gets_do_not_check_freshness_or_call_dagster() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportFresh(head=current.head),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        home = client.get("/")
        today = client.get("/today")

    assert home.status_code == today.status_code == 200
    assert 'hx-post="/report-status/check"' in today.text
    assert LIVE_VERSION in today.text
    assert f'href="/reports/{REPORT_VERSION}"' in today.text
    assert calls.current_reads == 2
    assert calls.freshness_reads == 0
    assert calls.repair_requests == 0
    assert calls.repair_reads == 0


def test_fresh_status_stops_without_requesting_repair() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportFresh(head=current.head),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        response = client.post("/report-status/check", data=_status_form(csrf, current))

    assert response.status_code == 200
    assert "Inputs current through 10:41" in response.text
    assert "hx-get" not in response.text
    assert calls.repair_requests == 0


def test_stale_status_reuses_or_starts_repair_and_polls() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        response = client.post("/report-status/check", data=_status_form(csrf, current))

    assert response.status_code == 200
    assert "Showing the 10:41 input snapshot. An updated report is being built." in response.text
    assert 'href="https://example.test/run"' in response.text
    assert "Open the Dagster run" in response.text
    assert 'hx-get="/report-status?' in response.text
    assert "load delay:8s" in response.text
    assert calls.repair_requests == 1


def test_inputs_not_ready_retries_the_side_effecting_post() -> None:
    day = datetime.now(BUCHAREST).date()
    app, calls = _status_harness(
        current=None,
        freshness=ReportInputsNotReady(day=day),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        response = client.post("/report-status/check", data=_status_form(csrf, None))

    assert response.status_code == 200
    assert "Waiting for today's report inputs." in response.text
    assert 'hx-post="/report-status/check"' in response.text
    assert "hx-get=" not in response.text
    assert "load delay:8s" in response.text
    assert calls.repair_requests == 0


def test_missing_current_day_report_stays_on_waiting_page_and_requests_repair_when_ready() -> None:
    day = datetime.now(BUCHAREST).date()
    app, calls = _status_harness(
        current=None,
        freshness=ReportMissing(day=day),
        repair=RepairRun(kind="running", run_id="run", run_url="https://example.test/run"),
        reports=(
            DailyReportSummary(
                report_version_id=REPORT_VERSION,
                day=day - timedelta(days=1),
            ),
        ),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        page = client.get("/")
        status = client.post("/report-status/check", data=_status_form(csrf, None))

    assert "Today's report is not ready yet" in page.text
    assert f'href="/reports/{REPORT_VERSION}"' in page.text
    assert 'hx-post="/report-status/check"' in page.text
    assert "Today's report is being built." in status.text
    assert 'href="https://example.test/run"' in status.text
    assert "Open the Dagster run" in status.text
    assert calls.repair_requests == 1


def test_polling_observes_running_repair_without_triggering_work() -> None:
    current = _today_report()
    lifecycle = RepairRun(kind="running", run_id="run", run_url="https://example.test/run")
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=lifecycle,
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "run",
            },
        )

    assert response.status_code == 200
    assert "updated report is being built" in response.text
    assert 'href="https://example.test/run"' in response.text
    assert "Open the Dagster run" in response.text
    assert calls.repair_reads == 1
    assert calls.repair_requests == 0


def test_failed_repair_preserves_report_and_links_directly_to_dagster() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=RepairRun(kind="failed", run_id="run", run_url="https://example.test/prod/runs/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "run",
            },
        )

    assert "could not be built" in response.text
    assert 'href="https://example.test/prod/runs/run"' in response.text
    assert "Subject 01" not in response.text
    assert calls.repair_requests == 0


def test_replacement_report_stops_polling_with_open_update_link() -> None:
    viewed = _today_report()
    replacement = _today_report(NEWER_REPORT_VERSION)
    app, _calls = _status_harness(
        current=replacement,
        freshness=ReportFresh(head=replacement.head),
        repair=RepairRun(kind="succeeded", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": replacement.head.day.isoformat(),
                "viewed_version": viewed.head.version_id,
                "run_id": "run",
            },
        )

    assert "Open update" in response.text
    assert f'href="/reports/{NEWER_REPORT_VERSION}"' in response.text
    assert "hx-get" not in response.text


def test_successful_no_change_repair_stops_polling_as_up_to_date() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportFresh(head=current.head),
        repair=RepairRun(kind="succeeded", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "run",
            },
        )

    assert "Inputs current through 10:41" in response.text
    assert "Open update" not in response.text
    assert "hx-get" not in response.text
    assert calls.repair_requests == 0


def test_repair_post_rejects_invalid_csrf_before_checking_freshness() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.post("/report-status/check", data=_status_form("wrong", current))

    assert response.status_code == 403
    assert calls.freshness_reads == 0
    assert calls.repair_requests == 0


def test_missing_dagster_config_does_not_break_reader_and_fails_repair_fragment() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=DagsterRepairUnavailable("Dagster repair is not configured"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        page = client.get("/")
        status = client.post("/report-status/check", data=_status_form(csrf, current))

    assert page.status_code == 200
    assert "Subject 01" in page.text
    assert "Dagster repair is not configured" in status.text
    assert calls.repair_requests == 1


def test_waiting_status_opens_todays_report_when_fresh_head_appears() -> None:
    current = _today_report()
    app, _calls = _status_harness(
        current=None,
        freshness=ReportFresh(head=current.head),
        repair=RepairRun(kind="succeeded", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        response = client.post("/report-status/check", data=_status_form(csrf, None))

    assert "Open today's report" in response.text
    assert f'href="/reports/{current.head.version_id}"' in response.text
    assert "hx-get" not in response.text
    assert "hx-post" not in response.text


def test_stale_replacement_keeps_truthful_updating_status() -> None:
    viewed = _today_report()
    replacement = _today_report(NEWER_REPORT_VERSION)
    app, calls = _status_harness(
        current=replacement,
        freshness=ReportStale(head=replacement.head),
        repair=RepairRun(kind="running", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": replacement.head.day.isoformat(),
                "viewed_version": viewed.head.version_id,
                "run_id": "run",
            },
        )

    assert "Showing the 10:41 input snapshot. An updated report is being built." in response.text
    assert "Open update" not in response.text
    assert 'hx-get="/report-status?' in response.text
    assert calls.repair_requests == 0


def test_successful_run_that_remains_stale_retries_repair_with_post() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=RepairRun(kind="succeeded", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "run",
            },
        )

    assert "Showing the 10:41 input snapshot. An updated report is being built." in response.text
    assert 'hx-post="/report-status/check"' in response.text
    assert "load delay:8s" in response.text
    assert "Open update" not in response.text
    assert calls.repair_requests == 0


def test_second_successful_run_that_remains_stale_stops_requesting_repairs() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=RepairRun(kind="succeeded", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "run",
                "retried_after_success": "true",
            },
        )

    assert "newer inputs are still available" in response.text
    assert 'href="https://example.test/run"' in response.text
    assert "hx-get" not in response.text
    assert "hx-post" not in response.text
    assert calls.repair_requests == 0


@pytest.mark.parametrize("method", ["post", "get"])
def test_catalog_failure_keeps_report_status_retryable(method: Literal["post", "get"]) -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ResearchCatalogError("temporary"),
        repair=RepairRun(kind="running", run_id="run", run_url="https://example.test/run"),
    )
    with TestClient(app) as client:
        csrf = _login(app, client)
        if method == "post":
            response = client.post("/report-status/check", data=_status_form(csrf, current))
        else:
            response = client.get(
                "/report-status",
                params={
                    "day": current.head.day.isoformat(),
                    "viewed_version": current.head.version_id,
                    "run_id": "run",
                },
            )

    assert response.status_code == 200
    assert "temporarily unavailable" in response.text
    assert "load delay:8s" in response.text
    assert calls.repair_requests == 0


def test_transient_dagster_poll_error_keeps_observing_the_same_run() -> None:
    current = _today_report()
    app, calls = _status_harness(
        current=current,
        freshness=ReportStale(head=current.head),
        repair=DagsterRepairError("temporary"),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get(
            "/report-status",
            params={
                "day": current.head.day.isoformat(),
                "viewed_version": current.head.version_id,
                "run_id": "same-run",
            },
        )

    assert "Repair status is temporarily unavailable." in response.text
    assert "run_id=same-run" in response.text
    assert "load delay:8s" in response.text
    assert calls.repair_reads == 1
    assert calls.repair_requests == 0


def test_waiting_page_omits_prior_link_when_yesterday_is_unavailable() -> None:
    day = datetime.now(BUCHAREST).date()
    older = DailyReportSummary(
        report_version_id=OLDER_REPORT_VERSION,
        day=day - timedelta(days=2),
    )
    app, _calls = _status_harness(
        current=None,
        freshness=ReportInputsNotReady(day=day),
        repair=RepairRun(kind="queued", run_id="run", run_url="https://example.test/run"),
        reports=(older,),
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/")

    assert "Open yesterday's report" not in response.text
    assert f'href="/reports/{OLDER_REPORT_VERSION}"' not in response.text


def test_current_day_get_uses_the_loaded_report_body_once() -> None:
    current = _today_report()
    archive_reads: list[str] = []
    domain = ReaderDomain(
        list_reports=lambda _limit: (),
        resolve_current_report_version=lambda version: version,
        read_report=lambda version: archive_reads.append(version) or current.report,
        read_current_report=lambda _day: current,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            cookie_secure=False,
        ),
        domain,
    )
    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/")

    assert response.status_code == 200
    assert "Subject 01" in response.text
    assert archive_reads == []
    assert 'aria-live="polite"' in response.text


def test_anonymous_video_selection_redirect_preserves_the_full_query() -> None:
    app, _commands = _video_app(_reader_video_digest())

    with TestClient(app) as client:
        response = client.get(
            f"/reports/{REPORT_VERSION}?edition={VIDEO_EDITION}",
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/login?next=%2Freports%2F{REPORT_VERSION}%3Fedition%3D{VIDEO_EDITION}"
    )


def test_report_renders_native_video_captions_transcript_fallback_and_selector() -> None:
    app, _commands = _video_app(_reader_video_digest())

    with TestClient(app) as client:
        _login(app, client)
        response = client.get(f"/reports/{REPORT_VERSION}?edition={VIDEO_EDITION}")
        today = client.get(f"/today?edition={VIDEO_EDITION}")
        health = client.get("/healthz")

    assert response.status_code == 200
    assert (
        '<video src="https://media.example.com/video/immutable.mp4" controls '
        'preload="metadata" crossorigin="anonymous">'
    ) in response.text
    assert (
        '<track src="https://media.example.com/video/immutable.vtt" kind="captions" '
        'srclang="ro" label="Romanian" default>'
    ) in response.text
    assert response.text.count("Open video directly") == 1
    assert 'aria-label="Video transcript"' in response.text
    assert "The transcript remains visible without JavaScript." in response.text
    assert f'href="/reports/{REPORT_VERSION}?edition={VIDEO_EDITION}"' in response.text
    assert f'href="/reports/{REPORT_VERSION}?edition={OLDER_VIDEO_EDITION}"' in response.text
    assert f'href="/reports/{OLDER_REPORT_VERSION}"' in response.text
    assert 'href="/today"' in response.text
    assert "news/video-digest/private-plan.json" not in response.text
    assert f'href="/today?edition={VIDEO_EDITION}"' in today.text
    assert f'href="/today?edition={OLDER_VIDEO_EDITION}"' in today.text
    directives = health.headers["content-security-policy"].split("; ")
    assert directives.count("media-src 'self' https://media.example.com") == 1
    assert all(not directive.startswith("media-src ") for directive in directives[:-1])


def test_failed_subtitles_render_status_without_track_or_crossorigin() -> None:
    digest = _reader_video_digest(subtitle_failed=True)
    app, _commands = _video_app(digest)

    with TestClient(app) as client:
        _login(app, client)
        response = client.get(f"/reports/{REPORT_VERSION}")

    assert "Captions are unavailable for this edition." in response.text
    assert "<track" not in response.text
    assert "crossorigin" not in response.text


def test_known_video_integrity_failure_degrades_to_the_written_report() -> None:
    app, _commands = _video_app(
        _reader_video_digest(),
        video_error=ResearchObjectUnavailable("private plan unavailable"),
    )

    with TestClient(app) as client:
        _login(app, client)
        response = client.get(f"/reports/{REPORT_VERSION}")

    assert response.status_code == 200
    assert "Subject 01" in response.text
    assert "Video digest" not in response.text


@pytest.mark.parametrize(
    "origin",
    [
        "http://media.example.com",
        "https://user@media.example.com",
        "https://media.example.com/video",
        "https://media.example.com?x=1",
        "https://media.example.com#fragment",
        "https://media.example.com; media-src *",
    ],
)
def test_reader_rejects_non_origin_or_csp_injecting_media_configuration(origin: str) -> None:
    with pytest.raises(ValueError):
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
            public_media_origin=origin,
        )


def test_video_feedback_form_fields_csrf_and_submit_follow_progressive_pattern() -> None:
    app, commands = _video_app(_reader_video_digest())
    with TestClient(app) as client:
        csrf = _login(app, client)
        page = client.get(f"/reports/{REPORT_VERSION}?edition={VIDEO_EDITION}")
        edition_control = page.text.split('id="video-feedback-edition"', 1)[1].split("</div>", 1)[0]
        rejected = client.post(
            "/video-feedback",
            data={
                "feedback_id": FEEDBACK_ID,
                "edition_id": VIDEO_EDITION,
                "rating": "positive",
                "note": "",
                "csrf_token": "wrong",
            },
        )
        submitted = client.post(
            f"/video-feedback?return_to=%2Freports%2F{REPORT_VERSION}%3Fedition%3D{VIDEO_EDITION}",
            data={
                "feedback_id": FEEDBACK_ID,
                "edition_id": VIDEO_EDITION,
                "story_id": VIDEO_STORY,
                "rating": "",
                "note": "  Clear narration.  ",
                "csrf_token": csrf,
            },
            follow_redirects=False,
        )

    assert set(re.findall(r'name="([^"]+)"', edition_control)) == {
        "feedback_id",
        "edition_id",
        "csrf_token",
        "note",
        "rating",
    }
    assert 'hx-post="/video-feedback?return_to=' in page.text
    assert rejected.status_code == 403
    assert submitted.status_code == 303
    assert submitted.headers["location"] == (f"/reports/{REPORT_VERSION}?edition={VIDEO_EDITION}")
    assert len(commands) == 1
    assert commands[0].edition_id == VIDEO_EDITION
    assert commands[0].story_id == VIDEO_STORY
    assert commands[0].rating is None
    assert commands[0].note == "Clear narration."


def test_htmx_video_feedback_swaps_only_its_story_control() -> None:
    app, _commands = _video_app(_reader_video_digest())
    with TestClient(app) as client:
        csrf = _login(app, client)
        response = client.post(
            "/video-feedback?return_to=%2Ftoday",
            data={
                "feedback_id": FEEDBACK_ID,
                "edition_id": VIDEO_EDITION,
                "story_id": VIDEO_STORY,
                "rating": "negative",
                "note": "Too fast.",
                "csrf_token": csrf,
            },
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert response.text.startswith(f'<div id="video-feedback-{VIDEO_STORY}"')
    assert "Feedback saved: Negative" in response.text
    assert "<html" not in response.text


def _reader_video_digest(*, subtitle_failed: bool = False) -> ReaderVideoDigest:
    published_at = datetime(2026, 8, 31, 18, tzinfo=UTC)
    subtitle = (
        ReaderSubtitleFailed()
        if subtitle_failed
        else ReaderSubtitleAvailable.model_validate(
            {"url": "https://media.example.com/video/immutable.vtt"}
        )
    )
    selected = ReaderVideoEdition.model_validate(
        {
            "edition_id": EditionId(VIDEO_EDITION),
            "slot_name": SlotName.EVENING,
            "published_at": published_at,
            "video_url": "https://media.example.com/video/immutable.mp4",
            "subtitle": subtitle,
            "stories": (
                ReaderVideoStory(
                    story_id=StoryId(VIDEO_STORY),
                    position=0,
                    title="Budget update",
                    narration="The transcript remains visible without JavaScript.",
                    requested_duration_ms=15_000,
                ),
            ),
        }
    )
    return ReaderVideoDigest(
        editions=(
            ReaderEditionOption(
                edition_id=selected.edition_id,
                slot_name=selected.slot_name,
                published_at=selected.published_at,
            ),
            ReaderEditionOption(
                edition_id=EditionId(OLDER_VIDEO_EDITION),
                slot_name=SlotName.MIDDAY,
                published_at=published_at - timedelta(hours=6),
            ),
        ),
        selected=selected,
    )


def test_written_only_deployment_rejects_video_feedback() -> None:
    domain = ReaderDomain(
        list_reports=lambda _limit: _report_summaries(),
        resolve_current_report_version=lambda version: version,
        read_report=lambda _version: _daily_report(),
        read_current_report=_live_report,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
    )
    app = create_app(
        ReaderSettings(
            app_password="correct horse",
            session_secret=TEST_SESSION_SECRET,
        ),
        domain,
    )

    with TestClient(app) as client:
        csrf_token = _login(app, client)
        response = client.post(
            "/video-feedback",
            data={
                "feedback_id": FEEDBACK_ID,
                "edition_id": VIDEO_EDITION,
                "story_id": VIDEO_STORY,
                "rating": "positive",
                "note": "",
                "csrf_token": csrf_token,
            },
        )

    assert response.status_code == 400
    assert "Feedback could not be saved." in response.text


def _video_app(
    digest: ReaderVideoDigest,
    *,
    video_error: Exception | None = None,
) -> tuple[FastHTML, list[VideoDigestFeedbackCommand]]:
    commands: list[VideoDigestFeedbackCommand] = []

    def read_video(*_args: object) -> ReaderVideoDigest:
        if video_error is not None:
            raise video_error
        return digest

    def submit(command: VideoDigestFeedbackCommand) -> VideoDigestFeedbackEvent:
        commands.append(command)
        return VideoDigestFeedbackEvent(
            **command.model_dump(),
            created_at=datetime(2026, 8, 31, 18, 30, tzinfo=UTC),
        )

    domain = ReaderDomain(
        list_reports=lambda _limit: _report_summaries(),
        resolve_current_report_version=lambda version: version,
        read_report=lambda _version: _daily_report(),
        read_current_report=_live_report,
        read_feedback=lambda _version: (),
        submit_feedback=lambda command: _event(command),
        read_research_flags=lambda _version: None,
        read_video_digest=read_video,
        submit_video_feedback=submit,
    )
    return (
        create_app(
            ReaderSettings(
                app_password="correct horse",
                session_secret=TEST_SESSION_SECRET,
                public_media_origin="https://media.example.com",
            ),
            domain,
        ),
        commands,
    )
