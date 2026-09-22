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
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.storage import ResearchObjectIntegrityError
from tests.postgres_catalog import PostgresCatalog

REPORT_VERSION = "a" * 64
RUN_ID = "7" * 64
INPUT_VERSION = "3" * 64
INPUT_DIGEST = "9" * 64
FILE_ID = "1" * 64
GROUP_ID = "c" * 64
ARTICLE_VERSION = "d" * 64
THEME_ID = "f" * 64
FEEDBACK_ID = "00000000-0000-4000-8000-000000000001"
TEST_SESSION_SECRET = "s" * 32
CAPTURED_AT = "2026-09-20T06:00:00+00:00"
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


def _daily_report(day: date) -> DailyReport:
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
        day=day,
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


def _seed_catalog(catalog: PostgresCatalog, report: DailyReport) -> bytes:
    day = report.day
    payload = report.model_dump_json().encode()
    digest = hashlib.sha256(payload).hexdigest()
    r2_key = f"news/reports/daily/{day.isoformat()}/{digest}.json"
    catalog.execute(
        "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            RUN_ID,
            "news.report_daily",
            "python",
            "reader-e2e",
            f'{{"day": "{day.isoformat()}"}}',
            "chartly",
            "completed",
            f"news.report_daily:{RUN_ID}",
            None,
            CAPTURED_AT,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state,"
        " visibility, current_version_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            f"news:daily:{day.isoformat()}",
            "news_daily_report",
            "Daily report",
            "derived",
            "current",
            "private",
            None,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state,"
        " visibility, current_version_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            f"news:themes:{day.isoformat()}",
            "news_daily_themes",
            "Daily themes",
            "derived",
            "current",
            "private",
            None,
            CAPTURED_AT,
        ),
    )
    catalog.execute(
        "INSERT INTO artifact_versions VALUES (%s, %s, %s, %s, %s, %s)",
        (REPORT_VERSION, f"news:daily:{day.isoformat()}", 3, digest, None, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions VALUES (%s, %s, %s, %s, %s, %s)",
        (INPUT_VERSION, f"news:themes:{day.isoformat()}", 1, INPUT_DIGEST, None, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_files VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (FILE_ID, REPORT_VERSION, r2_key, "application/json", digest, len(payload), None, None),
    )
    catalog.execute(
        "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (RUN_ID, 0, INPUT_VERSION, "themes", None, INPUT_DIGEST, "whole_file", None),
    )
    catalog.execute(
        "INSERT INTO run_outputs VALUES (%s, %s, %s, %s)",
        (RUN_ID, 0, REPORT_VERSION, "output"),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s, current_run_id = %s WHERE id = %s",
        (REPORT_VERSION, RUN_ID, f"news:daily:{day.isoformat()}"),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (INPUT_VERSION, f"news:themes:{day.isoformat()}"),
    )
    return payload


def _r2_reader(payload: bytes) -> Callable[[str, str], bytes]:
    digest = hashlib.sha256(payload).hexdigest()

    def read(key: str, expected_digest: str) -> bytes:
        if expected_digest != digest:
            raise ResearchObjectIntegrityError(f"R2 verification failed for {key}")
        return payload

    return read


def _wire_real_domain(monkeypatch, catalog: PostgresCatalog) -> None:
    payload = _seed_catalog(catalog, _daily_report(TEST_DAY))
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
