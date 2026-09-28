from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, cast

import psycopg
from botocore.exceptions import ClientError
from fasthtml.common import FastHTML
from starlette.testclient import TestClient

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.weekly_status import publish_weekly_status, read_weekly_status
from romanian_news.reader.app import (
    PRODUCTION_DOMAIN,
    SESSION_COOKIE,
    ReaderSettings,
    create_app,
    decode_session_cookie,
)
from romanian_news.storage import ResearchObjectIntegrityError
from romanian_news.weekly_status import (
    AreaAssessment,
    Development,
    WeekInput,
    WeekInputDay,
)
from romanian_news.weekly_status_generation import GeneratedStatus, build_weekly_status
from tests.daily_report_catalog import REPORT_VERSION, THEME_ID, daily_report, seed_daily_report
from tests.postgres_catalog import PostgresCatalog

FEEDBACK_ID = "00000000-0000-4000-8000-000000000001"
TEST_SESSION_SECRET = "s" * 32
TEST_DAY = date(2026, 9, 20)
WEEK_START = date(2026, 9, 14)
CAPTURED_AT = "2026-09-21T06:00:00+00:00"


def test_feedback_round_trip_through_the_real_domain(monkeypatch, postgres_catalog) -> None:
    monkeypatch.setattr("romanian_news.reader.clock.bucharest_today", lambda: TEST_DAY)
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


class _R2Library:
    """In-memory stand-in for the R2 S3 client at the storage network boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject")
        return {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs: object) -> None:
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


def _seed_week_of_daily_reports(
    catalog: PostgresCatalog, library: _R2Library, week_start: date
) -> dict[date, ArtifactReference]:
    references: dict[date, ArtifactReference] = {}
    for offset in range(7):
        day = week_start + timedelta(days=offset)
        payload = daily_report(day).model_dump_json().encode()
        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"news:daily:{day.isoformat()}"
        version_id = hashlib.sha256(day.isoformat().encode()).hexdigest()
        r2_key = f"news/reports/daily/{day.isoformat()}/{digest}.json"
        library.objects[r2_key] = payload
        catalog.execute(
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, created_at) VALUES (%s, 'news_daily_report', %s, 'derived', "
            "'current', 'private', %s) ON CONFLICT DO NOTHING",
            (artifact_id, f"Daily report {day.isoformat()}", CAPTURED_AT),
        )
        catalog.execute(
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
            "created_at) VALUES (%s, %s, 3, %s, %s) ON CONFLICT DO NOTHING",
            (version_id, artifact_id, digest, CAPTURED_AT),
        )
        catalog.execute(
            "INSERT INTO artifact_files (id, artifact_version_id, r2_key, media_type, "
            "content_digest, byte_size, row_count, schema_fingerprint) "
            "VALUES (%s, %s, %s, 'application/json', %s, %s, NULL, NULL)",
            (
                hashlib.sha256(r2_key.encode()).hexdigest(),
                version_id,
                r2_key,
                digest,
                len(payload),
            ),
        )
        catalog.execute(
            "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
            (version_id, artifact_id),
        )
        references[day] = ArtifactReference(
            artifact_id=artifact_id,
            version_id=version_id,
            content_digest=digest,
            r2_key=r2_key,
        )
    return references


def _week_input(references: dict[date, ArtifactReference], week_start: date) -> WeekInput:
    return WeekInput(
        week_start=week_start,
        days=tuple(
            WeekInputDay(
                day=week_start + timedelta(days=offset),
                report=references[week_start + timedelta(days=offset)],
            )
            for offset in range(7)
        ),
    )


def _draft(sources) -> GeneratedStatus:
    handles = tuple(source.handle for source in sources)
    return GeneratedStatus(
        developments=(
            Development(
                title="Budget debate",
                what_changed="The budget discussion continued during the week.",
                source_handles=handles[:1],
            ),
        ),
        assessments=(
            AreaAssessment(
                area="overall",
                judgment="The budget debate dominated the selected reports.",
                what_changed="The draft entered public debate.",
                why_it_matters="The budget could affect national spending.",
                source_handles=handles[:1],
                coverage="limited",
                coverage_note="All seven daily reports were captured.",
            ),
            *(
                AreaAssessment(
                    area=area,
                    judgment=None,
                    what_changed=None,
                    why_it_matters=None,
                    source_handles=(),
                    coverage="insufficient",
                    coverage_note="Not enough selected reporting.",
                )
                for area in ("economy", "politics", "society")
            ),
        ),
    )


def test_weekly_status_pages_render_from_the_real_catalog(monkeypatch, postgres_catalog) -> None:
    library = _R2Library()
    monkeypatch.setattr("romanian_news.storage._r2_client", lambda: cast(Any, library))
    references = _seed_week_of_daily_reports(postgres_catalog, library, WEEK_START)
    inputs = _week_input(references, WEEK_START)
    reports = {
        slot.report.version_id: daily_report(slot.day) for slot in inputs.days if slot.report
    }
    output = build_weekly_status(
        inputs,
        reports,
        compose=lambda _inputs, sources: _draft(sources),
        verify=lambda *_args: None,
    )
    publication = publish_weekly_status(output, "reader-e2e")
    version_id, read = read_weekly_status(WEEK_START)

    app = create_app(_settings(), PRODUCTION_DOMAIN)
    with TestClient(app) as client:
        _login(app, client)

        latest = client.get("/status", follow_redirects=False)
        assert latest.status_code == 303
        assert latest.headers["location"] == f"/status/weeks/{WEEK_START.isoformat()}"

        week_page = client.get(f"/status/weeks/{WEEK_START.isoformat()}")
        assert week_page.status_code == 200
        assert "Budget debate" in week_page.text
        assert "Romania overall" in week_page.text
        source_link = f"/reports/exact/{read.sources[0].report_version_id}#source-{THEME_ID}"
        assert source_link in week_page.text

        version_page = client.get(f"/status/versions/{version_id}")
        assert version_page.status_code == 200
        assert "Budget debate" in version_page.text

        archive_page = client.get("/status/archive")
        assert archive_page.status_code == 200
        assert f"/status/weeks/{WEEK_START.isoformat()}" in archive_page.text

        source_page = client.get(source_link)
        assert source_page.status_code == 200
        assert f'id="source-{THEME_ID}"' in source_page.text
        assert version_id == publication.version_id
