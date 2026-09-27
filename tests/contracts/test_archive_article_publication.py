from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, date, datetime

import pytest

from romanian_news.archive.articles import capture_archive_article
from romanian_news.catalog.archive_articles import publish_archive_article
from romanian_news.catalog.daily import read_daily_article_references
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch, catalog_query
from romanian_news.feeds.registry import feed_registry
from romanian_news.storage import ImmutableR2Publication


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_archive_article_publication_keeps_real_source_and_replays(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    discovered_url = "https://hotnews.ro/proiect-public-1"
    fetched_at = datetime(2026, 9, 27, 10, tzinfo=UTC)
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES " "(%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    "hotnews",
                    "https://hotnews.ro/sitemap-2025-09-19.xml",
                    "b" * 64,
                    fetched_at,
                    1,
                ],
            ),
            (
                "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s, %s)",
                [observation_id, discovered_url, discovered_url, None],
            ),
        ]
    )
    feed = next(value for value in feed_registry().feeds if value.id == "hotnews")
    html = (
        "<html><head>"
        '<meta property="og:title" content="Anunț despre un proiect public">'
        '<meta property="article:published_time" content="2025-09-19T10:00:00+03:00">'
        "</head><body><article class='article-single-content'>"
        "<h1>Anunț despre un proiect public</h1>"
        f"<p>{'Instituția a publicat detalii despre proiectul local. ' * 6}</p>"
        "</article></body></html>"
    ).encode()
    capture = capture_archive_article(
        observation_id=observation_id,
        discovered_url=discovered_url,
        final_url=discovered_url,
        html=html,
        fetched_at=fetched_at,
        feed=feed,
    )
    uploaded: list[list[str]] = []

    def fake_upload(objects: Iterable[tuple[str, bytes]]) -> ImmutableR2Publication:
        keys = [key for key, _content in objects]
        uploaded.append(keys)
        return ImmutableR2Publication(bucket="test", uploaded_objects=len(keys), reused_objects=0)

    monkeypatch.setattr(
        "romanian_news.catalog.archive_articles.publish_immutable_r2_objects", fake_upload
    )

    first = publish_archive_article(capture, "test-ref")
    second = publish_archive_article(capture, "test-ref")

    assert first.published_articles == 1
    assert second.unchanged_articles == 1
    assert [len(keys) for keys in uploaded] == [2, 1]
    rows = catalog_query(
        "SELECT version.feed_snapshot_version_id, version.archive_capture_id, "
        "version.published_at, version.captured_at, input.role, input.artifact_version_id "
        "FROM news_article_versions version "
        "JOIN artifact_versions artifact ON artifact.id = version.artifact_version_id "
        "JOIN run_inputs input ON input.run_id = artifact.produced_by_run_id"
    )
    assert len(rows) == 1
    assert rows[0]["feed_snapshot_version_id"] is None
    assert rows[0]["archive_capture_id"] == capture.capture_id
    assert datetime.fromisoformat(str(rows[0]["published_at"])).year == 2025
    assert datetime.fromisoformat(str(rows[0]["captured_at"])).year == 2026
    assert rows[0]["role"] == "archive_capture"
    captures = catalog_query(
        "SELECT capture_artifact_version_id FROM news_archive_article_captures"
    )
    assert rows[0]["artifact_version_id"] == captures[0]["capture_artifact_version_id"]
    assert len(catalog_query("SELECT id FROM news_archive_article_captures")) == 1
    assert len(catalog_query("SELECT id FROM runs")) == 1
    references = read_daily_article_references(date(2025, 9, 19))
    assert len(references) == 1
    assert references[0].artifact_id == f"news:article:{capture.article.article_id}"
