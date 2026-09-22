from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import date

import psycopg
from fasthtml.common import FastHTML
from starlette.testclient import TestClient

from romanian_news.reader.app import (
    PRODUCTION_DOMAIN,
    SESSION_COOKIE,
    ReaderSettings,
    create_app,
    decode_session_cookie,
)
from romanian_news.storage import ResearchObjectIntegrityError
from tests.daily_report_catalog import REPORT_VERSION, daily_report, seed_daily_report
from tests.postgres_catalog import PostgresCatalog

FEEDBACK_ID = "00000000-0000-4000-8000-000000000001"
TEST_SESSION_SECRET = "s" * 32
TEST_DAY = date(2026, 9, 20)


def test_feedback_round_trip_through_the_real_domain(monkeypatch, postgres_catalog) -> None:
    monkeypatch.setattr("romanian_news.reader.app._bucharest_today", lambda: TEST_DAY)
    _wire_real_domain(monkeypatch, postgres_catalog)
    app = create_app(_settings(), PRODUCTION_DOMAIN)
    with TestClient(app) as client:
        csrf_token = _login(app, client)
        page = client.get("/")

        assert page.status_code == 200
        assert "Subject 01" in page.text
        assert "Feedback saved" not in page.text

        submitted = client.post(
            "/feedback",
            data=_feedback_form(csrf_token),
            follow_redirects=False,
        )

        assert submitted.status_code == 303
        assert submitted.headers["location"] == f"/reports/{REPORT_VERSION}"
        rows = postgres_catalog.execute(
            "SELECT feedback_id, report_version_id, target_kind, rating, note, actor"
            " FROM news_feedback"
        ).fetchall()
        assert rows == [
            {
                "feedback_id": FEEDBACK_ID,
                "report_version_id": REPORT_VERSION,
                "target_kind": "report",
                "rating": "positive",
                "note": "Clear and useful.",
                "actor": "owner",
            }
        ]

        retried = client.post(
            "/feedback",
            data=_feedback_form(csrf_token),
            follow_redirects=False,
        )

        assert retried.status_code == 303
        assert (
            postgres_catalog.execute("SELECT count(*) AS count FROM news_feedback").fetchone()[
                "count"
            ]
            == 1
        )

        reloaded = client.get("/")

        assert "Feedback saved: Positive" in reloaded.text
        assert "Clear and useful." in reloaded.text


def test_catalog_failure_maps_to_service_unavailable_over_http(monkeypatch) -> None:
    def unavailable_catalog(*_args: object, **_kwargs: object) -> object:
        raise psycopg.errors.UndefinedTable('relation "artifacts" does not exist')

    monkeypatch.setattr(
        "romanian_news.catalog_transport.NEWS_POSTGRES_DSN",
        "postgresql://catalog.test/news",
    )
    monkeypatch.setattr(psycopg, "connect", unavailable_catalog)
    app = create_app(_settings(), PRODUCTION_DOMAIN)
    with TestClient(app) as client:
        _login(app, client)
        response = client.get("/")

    assert response.status_code == 503
    assert "The report could not be loaded" in response.text


def test_safe_next_rejects_open_redirects() -> None:
    app = create_app(_settings())
    next_values = ("//evil.example", "/\\evil.example", "/today")

    with TestClient(app) as client:
        responses = [
            client.post(
                "/login",
                data={"password": "correct horse", "next": next_value},
                follow_redirects=False,
            )
            for next_value in next_values
        ]

    assert [response.status_code for response in responses] == [303, 303, 303]
    assert [response.headers["location"] for response in responses] == ["/", "/", "/today"]


def _settings() -> ReaderSettings:
    return ReaderSettings(
        app_password="correct horse",
        session_secret=TEST_SESSION_SECRET,
        cookie_secure=False,
    )


def _r2_reader(payload: bytes) -> Callable[[str, str], bytes]:
    digest = hashlib.sha256(payload).hexdigest()

    def read(key: str, expected_digest: str) -> bytes:
        if expected_digest != digest:
            raise ResearchObjectIntegrityError(f"R2 verification failed for {key}")
        return payload

    return read


def _wire_real_domain(monkeypatch, catalog: PostgresCatalog) -> None:
    payload = seed_daily_report(catalog, daily_report(TEST_DAY))
    read = _r2_reader(payload)
    monkeypatch.setattr("romanian_news.feedback.read_verified_r2_object", read)
    # read_current_daily_report imports the storage reader inside its function body.
    monkeypatch.setattr("romanian_news.storage.read_verified_r2_object", read)


def _login(app: FastHTML, client: TestClient) -> str:
    response = client.post(
        "/login",
        data={"password": "correct horse", "next": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    session = decode_session_cookie(app, client.cookies[SESSION_COOKIE])
    return str(session["csrf_token"])


def _feedback_form(csrf_token: str) -> dict[str, str]:
    return {
        "feedback_id": FEEDBACK_ID,
        "target_kind": "report",
        "report_version_id": REPORT_VERSION,
        "rating": "positive",
        "note": "Clear and useful.",
        "csrf_token": csrf_token,
    }
