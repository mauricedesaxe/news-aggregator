from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
from starlette.testclient import TestClient

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.archive_progress import ArchiveDailyReport, ArchiveDiscoveryMonth
from romanian_news.catalog.weekly_status import WeeklyStatusNotFound, WeeklyStatusSummary
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.reader.app import PRODUCTION_DOMAIN, ReaderSettings, create_app
from romanian_news.reports import (
    DailyReport,
    DailyReportDocument,
    RetrospectiveCoverage,
    RetrospectiveDailyReport,
)
from romanian_news.storage import ResearchObjectIntegrityError, ResearchObjectUnavailable
from romanian_news.weekly_status import (
    AreaAssessment,
    StatusSource,
    WeekInputDay,
    WeeklyStatusRead,
)

WEEK_START = date(2026, 9, 14)
DAILY_VERSION = "a" * 64
STATUS_VERSION = "b" * 64
THEME_ID = "c" * 64


def _read() -> WeeklyStatusRead:
    days = tuple(
        WeekInputDay(
            day=WEEK_START + timedelta(days=offset),
            report=ArtifactReference(
                artifact_id=f"news:daily:{(WEEK_START + timedelta(days=offset)).isoformat()}",
                version_id=DAILY_VERSION,
                content_digest="d" * 64,
                r2_key=f"report-{offset}.json",
            )
            if offset == 0
            else None,
        )
        for offset in range(7)
    )
    source = StatusSource(
        handle="s1",
        day=WEEK_START,
        report_version_id=DAILY_VERSION,
        locator_kind="theme",
        locator_id=THEME_ID,
        title="Budget",
        summary="The budget entered debate.",
    )
    assessments = tuple(
        AreaAssessment(
            area=area,
            judgment=None,
            what_changed=None,
            why_it_matters=None,
            source_handles=(),
            coverage="insufficient",
            coverage_note="Not enough reporting.",
        )
        for area in ("overall", "economy", "politics", "society")
    )
    return WeeklyStatusRead(
        policy="test",
        week_start=WEEK_START,
        week_end=WEEK_START + timedelta(days=6),
        days=days,
        sources=(source,),
        developments=(),
        assessments=assessments,
    )


def _app(
    discovery: tuple[ArchiveDiscoveryMonth, ...] = (),
    archive_reports: tuple[ArchiveDailyReport, ...] = (),
    report_reader: Callable[[ArtifactReference], DailyReportDocument] | None = None,
):
    domain = replace(
        PRODUCTION_DOMAIN,
        list_status=lambda _limit, _offset: (
            WeeklyStatusSummary(week_start=WEEK_START, version_id=STATUS_VERSION),
        ),
        list_archive_discovery=lambda: discovery,
        list_archive_reports=lambda _start, _end: archive_reports,
        read_status=lambda _week: (STATUS_VERSION, _read()),
        read_status_version=lambda _version: (STATUS_VERSION, _read()),
        read_report_reference=report_reader
        or (
            lambda reference: DailyReport(
                day=date.fromisoformat(reference.artifact_id.removeprefix("news:daily:")),
                accepted_article_count=0,
                theme_count=0,
                group_count=0,
                sections=(),
            )
        ),
    )
    return create_app(
        ReaderSettings(app_password="correct horse", session_secret="s" * 32),
        domain,
    )


def _retrospective_report() -> RetrospectiveDailyReport:
    return RetrospectiveDailyReport(
        day=WEEK_START,
        accepted_article_count=25,
        theme_count=0,
        group_count=0,
        sections=(),
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


def _sign_in(client: TestClient) -> None:
    assert (
        client.post(
            "/login",
            data={"password": "correct horse", "next": "/"},
            follow_redirects=False,
        ).status_code
        == 303
    )


def test_latest_status_redirects_to_completed_week() -> None:
    with TestClient(_app()) as client:
        _sign_in(client)
        response = client.get("/status", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/status/weeks/2026-09-14"


def test_status_archive_and_exact_version_render() -> None:
    with TestClient(_app()) as client:
        _sign_in(client)
        archive = client.get("/status/archive")
        week = client.get("/status/weeks/2026-09-14")
        exact = client.get(f"/status/versions/{STATUS_VERSION}")

    assert archive.status_code == 200
    assert "/status/weeks/2026-09-14" in archive.text
    assert "A missing week has no saved read." in archive.text
    assert week.status_code == 200
    assert "Based on reports from 1 of 7 days" in week.text
    assert f"/status/versions/{STATUS_VERSION}" in week.text
    assert "Permanent link to this read" in week.text
    assert "Saved weekly read" not in week.text
    assert exact.status_code == 200
    assert "What changed in Romania?" in exact.text
    assert "Saved weekly read" in exact.text
    assert (
        "This saved snapshot keeps this read available if the current week page is updated."
        in exact.text
    )
    assert 'href="/status/weeks/2026-09-14"' in exact.text
    assert "Current version for this week" in exact.text
    assert "Permanent link to this read" not in exact.text
    assert "cited highlight" not in exact.text
    assert "INSUFFICIENT" not in exact.text
    assert "source pages were captured later" not in week.text
    assert "source pages were captured later" not in exact.text


def test_historical_week_discloses_exact_daily_capture_provenance() -> None:
    captured: list[ArtifactReference] = []

    def read_report(reference: ArtifactReference) -> RetrospectiveDailyReport:
        captured.append(reference)
        return _retrospective_report()

    with TestClient(_app(report_reader=read_report)) as client:
        _sign_in(client)
        week = client.get("/status/weeks/2026-09-14")
        exact = client.get(f"/status/versions/{STATUS_VERSION}")

    for response in (week, exact):
        assert response.status_code == 200
        assert "source pages were captured later" in response.text
        assert "2026-09-27 10:00 UTC" in response.text
        assert "Open the exact saved daily reports for outlet and coverage details" in response.text
        assert f'href="/reports/exact/{DAILY_VERSION}"' in response.text
    assert captured == [_read().days[0].report, _read().days[0].report]


@pytest.mark.parametrize(
    "error", [ResearchObjectUnavailable("missing"), ResearchObjectIntegrityError("digest mismatch")]
)
def test_historical_week_fails_if_an_exact_daily_source_is_unavailable(
    error: Exception,
) -> None:
    def corrupt(_reference: ArtifactReference) -> DailyReportDocument:
        raise error

    with TestClient(_app(report_reader=corrupt)) as client:
        _sign_in(client)
        response = client.get("/status/weeks/2026-09-14")

    assert response.status_code == 503
    assert "Report unavailable" in response.text
    assert "source pages were captured later" not in response.text


def test_collection_progress_is_separate_from_weekly_archive() -> None:
    discovery = (
        ArchiveDiscoveryMonth(
            outlet_id="hotnews",
            month=date(2025, 9, 1),
            sitemap_count=7,
            url_entries=643,
            accepted_pages=42,
            rejected_pages=3,
            retryable_pages=1,
            captured_articles=5,
        ),
        ArchiveDiscoveryMonth(
            outlet_id="digi24", month=date(2025, 9, 1), sitemap_count=1, url_entries=3186
        ),
    )
    reports = (ArchiveDailyReport(day=date(2025, 9, 29), version_id=DAILY_VERSION),)
    with TestClient(_app(discovery, reports)) as client:
        _sign_in(client)
        weekly = client.get("/status/archive")
        response = client.get("/reports/backfill")

    assert weekly.status_code == 200
    assert "Historical collection" not in weekly.text
    assert f'href="/reports/{DAILY_VERSION}"' not in weekly.text
    assert "Week of 14 September 2026" in weekly.text
    assert response.status_code == 200
    assert "Historical collection" in response.text
    assert "3,186 URL entries" in response.text
    assert "42 pages with verified dates" in response.text
    assert "3 rejected" in response.text
    assert "5 articles captured" in response.text
    assert "1 daily report published" in response.text
    assert "regular reports and reports reconstructed from archived pages" in response.text
    assert f'href="/reports/{DAILY_VERSION}"' in response.text
    assert "Daily reports and weekly reads appear only when published" in response.text
    assert "URL entries may still exist" in response.text
    assert response.text.count("<summary>Monthly breakdown</summary>") == 2
    assert response.text.index('class="status-grid"') < response.text.index(
        "Daily reports in this date range"
    )


def test_old_strong_assessment_shows_cited_evidence_instead_of_rating() -> None:
    read = _read()
    full_days = tuple(
        WeekInputDay(
            day=WEEK_START + timedelta(days=offset),
            report=ArtifactReference(
                artifact_id=f"news:daily:{(WEEK_START + timedelta(days=offset)).isoformat()}",
                version_id=DAILY_VERSION,
                content_digest="d" * 64,
                r2_key=f"report-{offset}.json",
            ),
        )
        for offset in range(7)
    )
    assessment = AreaAssessment(
        area="overall",
        judgment="Budget talks continued.",
        what_changed="The budget entered debate.",
        why_it_matters="Spending is at stake.",
        source_handles=("s1",),
        coverage="strong",
        coverage_note="Only selected daily highlights were reviewed.",
    )
    legacy_read = read.model_copy(
        update={
            "days": full_days,
            "assessments": (assessment, *read.assessments[1:]),
        }
    )
    domain = replace(
        PRODUCTION_DOMAIN,
        read_status_version=lambda _version: (STATUS_VERSION, legacy_read),
        read_report_reference=lambda reference: DailyReport(
            day=date.fromisoformat(reference.artifact_id.removeprefix("news:daily:")),
            accepted_article_count=0,
            theme_count=0,
            group_count=0,
            sections=(),
        ),
    )
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        response = client.get(f"/status/versions/{STATUS_VERSION}")

    assert response.status_code == 200
    assert "1 cited highlight across 1 report day" in response.text
    assert "Only selected daily highlights were reviewed." in response.text
    assert "STRONG" not in response.text


def test_status_routes_report_catalog_and_object_failures() -> None:
    def _raise(*_args):
        raise ResearchCatalogError("catalog unavailable")

    domain = replace(
        PRODUCTION_DOMAIN,
        list_status=_raise,
        list_archive_discovery=_raise,
        read_status=_raise,
        read_status_version=_raise,
    )
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        latest = client.get("/status")
        archive = client.get("/status/archive")
        backfill = client.get("/reports/backfill")
        week = client.get("/status/weeks/2026-09-14")
        version = client.get(f"/status/versions/{STATUS_VERSION}")

    for response in (latest, archive, backfill, week, version):
        assert response.status_code == 503
        assert "Report unavailable" in response.text


def test_latest_status_reports_not_ready_before_any_week_exists() -> None:
    domain = replace(PRODUCTION_DOMAIN, list_status=lambda _limit, _offset: ())
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        response = client.get("/status")

    assert response.status_code == 503
    assert "Weekly status is not ready yet" in response.text


def test_status_weeks_rejects_unknown_and_malformed_weeks() -> None:
    domain = replace(
        PRODUCTION_DOMAIN,
        read_status=lambda _week: (_ for _ in ()).throw(WeeklyStatusNotFound("missing")),
    )
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        unknown = client.get("/status/weeks/2026-09-14")
        malformed = client.get("/status/weeks/not-a-date")

    assert unknown.status_code == 404
    assert malformed.status_code == 404


def test_status_versions_rejects_unknown_versions() -> None:
    domain = replace(
        PRODUCTION_DOMAIN,
        read_status_version=lambda _version: (_ for _ in ()).throw(WeeklyStatusNotFound("missing")),
    )
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        response = client.get(f"/status/versions/{STATUS_VERSION}")

    assert response.status_code == 404


def test_status_archive_rejects_out_of_range_pages() -> None:
    with TestClient(_app()) as client:
        _sign_in(client)
        not_a_number = client.get("/status/archive?page=soon")
        too_far = client.get("/status/archive?page=1001")

    assert not_a_number.status_code == 404
    assert too_far.status_code == 404


def test_status_archive_pages_through_completed_weeks() -> None:
    summaries = tuple(
        WeeklyStatusSummary(
            week_start=WEEK_START - timedelta(weeks=offset), version_id=STATUS_VERSION
        )
        for offset in range(21)
    )
    domain = replace(
        PRODUCTION_DOMAIN, list_status=lambda limit, offset: summaries[offset : offset + limit]
    )
    app = create_app(ReaderSettings(app_password="correct horse", session_secret="s" * 32), domain)
    with TestClient(app) as client:
        _sign_in(client)
        first = client.get("/status/archive")
        second = client.get("/status/archive?page=2")

    assert first.status_code == 200
    assert len(_week_hrefs(first.text)) == 20
    assert 'href="/status/archive?page=2"' in first.text
    assert "Older" in first.text
    assert second.status_code == 200
    assert len(_week_hrefs(second.text)) == 1
    assert 'href="/status/archive?page=1"' in second.text
    assert "Newer" in second.text


def _week_hrefs(page: str) -> list[str]:
    return re.findall(r'href="/status/weeks/[0-9]{4}-[0-9]{2}-[0-9]{2}"', page)
