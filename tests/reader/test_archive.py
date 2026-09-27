from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

from starlette.testclient import TestClient

from romanian_news.feedback import DailyReportSummary
from romanian_news.reader.app import PRODUCTION_DOMAIN, ReaderSettings, create_app
from tests.reader.test_app import _daily_report

REPORT_VERSION = "a" * 64
OLD_VERSION = "b" * 64


def _client_domain(*, reports: tuple[DailyReportSummary, ...]):
    return replace(
        PRODUCTION_DOMAIN,
        list_report_archive=lambda limit, offset: reports[offset : offset + limit],
        read_report=lambda _version: _daily_report(),
    )


def _sign_in(client: TestClient) -> None:
    response = client.post(
        "/login",
        data={"password": "correct horse", "next": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _app(reports: tuple[DailyReportSummary, ...]):
    return create_app(
        ReaderSettings(app_password="correct horse", session_secret="s" * 32),
        _client_domain(reports=reports),
    )


def test_archive_pages_through_all_daily_reports() -> None:
    reports = tuple(
        DailyReportSummary(
            report_version_id=f"{number:064x}",
            day=date(2026, 8, 31),
        )
        for number in range(35)
    )
    with TestClient(_app(reports)) as client:
        _sign_in(client)
        first = client.get("/reports")
        second = client.get("/reports?page=2")

    assert first.status_code == 200
    assert len(re.findall(r'href="/reports/[0-9a-f]{64}"', first.text)) == 30
    assert 'href="/reports?page=2"' in first.text
    assert second.status_code == 200
    assert len(re.findall(r'href="/reports/[0-9a-f]{64}"', second.text)) == 5
    assert 'href="/reports?page=1"' in second.text


def test_report_archive_shows_year_break_for_historical_report() -> None:
    reports = (
        DailyReportSummary(report_version_id=REPORT_VERSION, day=date(2026, 9, 27)),
        DailyReportSummary(report_version_id=OLD_VERSION, day=date(2025, 9, 29)),
    )
    with TestClient(_app(reports)) as client:
        _sign_in(client)
        response = client.get("/reports")

    assert response.status_code == 200
    assert response.text.index("2026</li>") < response.text.index("27 September 2026")
    assert response.text.index("2025</li>") < response.text.index("29 September 2025")
    assert response.text.index("27 September 2026") < response.text.index("29 September 2025")
    assert 'href="/reports/backfill"' in response.text


def test_exact_report_keeps_requested_version() -> None:
    seen: list[str] = []
    domain = replace(
        _client_domain(reports=()),
        resolve_current_report_version=lambda _version: REPORT_VERSION,
        read_report=lambda version: seen.append(version) or _daily_report(),
    )
    app = create_app(
        ReaderSettings(app_password="correct horse", session_secret="s" * 32),
        domain,
    )
    with TestClient(app) as client:
        _sign_in(client)
        response = client.get(f"/reports/exact/{OLD_VERSION}", follow_redirects=False)

    assert response.status_code == 200
    assert seen == [OLD_VERSION]
    assert "Saved version" in response.text
    assert f'id="source-{"f" * 64}"' in response.text
