from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pytest

from romanian_news.archive.capture_batch import pending_archive_articles
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_archive_capture_selects_verified_unpublished_pages(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    url = "https://hotnews.ro/story-1"
    publication = datetime(2025, 9, 19, 10, tzinfo=UTC)
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    "hotnews",
                    "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=19",
                    "b" * 64,
                    datetime(2026, 9, 27, tzinfo=UTC),
                    1,
                ],
            ),
            (
                "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s)",
                [observation_id, url, url],
            ),
            (
                "INSERT INTO news_archive_page_checks "
                "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
                "page_sha256, title, published_at, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    "d" * 64,
                    observation_id,
                    "hotnews",
                    url,
                    url,
                    datetime(2026, 9, 27, tzinfo=UTC),
                    "e" * 64,
                    "Story",
                    publication,
                    "accepted",
                ],
            ),
        ]
    )

    selected = pending_archive_articles("hotnews", date(2025, 9, 19), date(2025, 9, 19), 10)

    assert len(selected) == 1
    assert selected[0].canonical_url == url
    assert selected[0].published_at == publication
    assert pending_archive_articles("hotnews", date(2025, 9, 20), date(2025, 9, 20), 10) == ()
