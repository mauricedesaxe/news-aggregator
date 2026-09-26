from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from starlette.testclient import TestClient

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.weekly_status import WeeklyStatusSummary
from romanian_news.reader.app import PRODUCTION_DOMAIN, ReaderSettings, create_app
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


def _app():
    domain = replace(
        PRODUCTION_DOMAIN,
        list_status=lambda _limit, _offset: (
            WeeklyStatusSummary(week_start=WEEK_START, version_id=STATUS_VERSION),
        ),
        read_status=lambda _week: (STATUS_VERSION, _read()),
        read_status_version=lambda _version: (STATUS_VERSION, _read()),
    )
    return create_app(
        ReaderSettings(app_password="correct horse", session_secret="s" * 32),
        domain,
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
    assert week.status_code == 200
    assert "Based on reports from 1 of 7 days" in week.text
    assert f"/status/versions/{STATUS_VERSION}" in week.text
    assert exact.status_code == 200
    assert "What changed in Romania?" in exact.text
