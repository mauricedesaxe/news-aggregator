from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import dagster as dg
import pytest

from romanian_news.archive import capture_batch
from romanian_news.worker import archive_capture
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog
from tests.worker.conftest import FakeR2Client

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)

DAY = date(2025, 9, 19)
PUBLISHED_AT = datetime(2025, 9, 19, 10, tzinfo=UTC)
ARTICLE_HTML = (
    "<html><head>"
    '<meta property="og:title" content="Anunț despre un proiect public">'
    '<meta property="article:published_time" content="2025-09-19T13:00:00+03:00">'
    "</head><body><article class='article-single-content'>"
    "<h1>Anunț despre un proiect public</h1>"
    f"<p>{'Instituția a publicat detalii despre proiectul local. ' * 6}</p>"
    "</article></body></html>"
).encode()


def _seed_accepted_page_checks(
    catalog: PostgresCatalog, pages: tuple[tuple[str, datetime], ...]
) -> None:
    observation_id = "a" * 64
    fetched_at = datetime(2026, 9, 27, tzinfo=UTC)
    catalog.batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    "hotnews",
                    "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=19",
                    "b" * 64,
                    fetched_at,
                    len(pages),
                ],
            ),
            *[
                (
                    "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s)",
                    [observation_id, url, url],
                )
                for url, _published_at in pages
            ],
        ]
    )
    catalog.batch(
        [
            *[
                (
                    "INSERT INTO news_archive_page_checks "
                    "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
                    "page_sha256, title, published_at, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        f"{index:064x}",
                        observation_id,
                        "hotnews",
                        url,
                        url,
                        fetched_at,
                        "e" * 64,
                        "Story",
                        published_at,
                        "accepted",
                    ],
                )
                for index, (url, published_at) in enumerate(pages)
            ],
        ]
    )


def _run_capture_job() -> dg.ExecuteInProcessResult:
    return archive_capture.archive_article_capture_batch.execute_in_process(
        run_config={
            "ops": {
                "archive_article_capture": {
                    "config": {
                        "outlet": "hotnews",
                        "start": DAY.isoformat(),
                        "end": DAY.isoformat(),
                        "limit": 1,
                    }
                }
            }
        },
        raise_on_error=False,
    )


def test_capture_job_fails_and_persists_nothing_for_a_robots_disallowed_page(
    postgres_catalog: PostgresCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://hotnews.ro/secret-story"
    _seed_accepted_page_checks(postgres_catalog, ((url, PUBLISHED_AT),))

    def serve_robots_only(_session: object, fetched_url: str) -> tuple[str, bytes]:
        if fetched_url == "https://hotnews.ro/robots.txt":
            return fetched_url, b"User-agent: *\nDisallow: /secret-story\n"
        raise AssertionError(f"robots-disallowed page must not be fetched: {fetched_url}")

    monkeypatch.setattr(capture_batch, "_fetch", serve_robots_only)

    result = _run_capture_job()

    assert not result.success
    captures = postgres_catalog.execute("SELECT id FROM news_archive_article_captures").fetchall()
    assert captures == []
    articles = postgres_catalog.execute(
        "SELECT artifact_version_id FROM news_article_versions"
    ).fetchall()
    assert articles == []


def test_capture_job_publishes_one_archive_article_end_to_end(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://hotnews.ro/proiect-public"
    _seed_accepted_page_checks(postgres_catalog, ((url, PUBLISHED_AT),))

    def serve_robots_and_article(_session: object, fetched_url: str) -> tuple[str, bytes]:
        if fetched_url == "https://hotnews.ro/robots.txt":
            return fetched_url, b"User-agent: *\nAllow: /\n"
        if fetched_url == url:
            return url, ARTICLE_HTML
        raise AssertionError(f"unexpected fetch: {fetched_url}")

    monkeypatch.setattr(capture_batch, "_fetch", serve_robots_and_article)

    result = _run_capture_job()

    assert result.success
    capture_rows = postgres_catalog.execute(
        "SELECT discovered_url, capture_artifact_version_id, published_at "
        "FROM news_archive_article_captures"
    ).fetchall()
    assert len(capture_rows) == 1
    capture = capture_rows[0]
    assert capture["discovered_url"] == url
    assert capture["published_at"] == PUBLISHED_AT
    capture_files = postgres_catalog.execute(
        "SELECT r2_key, content_digest FROM artifact_files " "WHERE artifact_version_id = %s",
        (capture["capture_artifact_version_id"],),
    ).fetchall()
    assert len(capture_files) == 1
    assert (
        hashlib.sha256(fake_r2.objects[capture_files[0]["r2_key"]]).hexdigest()
        == (capture_files[0]["content_digest"])
    )
    article_rows = postgres_catalog.execute(
        "SELECT metadata.bucharest_day, metadata.archive_capture_id, file.r2_key, "
        "file.content_digest FROM news_article_versions metadata "
        "JOIN artifact_files file ON file.artifact_version_id = metadata.artifact_version_id"
    ).fetchall()
    assert len(article_rows) == 1
    article = article_rows[0]
    assert article["bucharest_day"] == DAY
    assert article["archive_capture_id"] is not None
    stored_article = fake_r2.objects[article["r2_key"]]
    assert hashlib.sha256(stored_article).hexdigest() == article["content_digest"]
    run_rows = postgres_catalog.execute(
        "SELECT status FROM runs WHERE operation_key = 'news.normalize_archive_article'"
    ).fetchall()
    assert [row["status"] for row in run_rows] == ["completed"]
