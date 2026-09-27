from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pytest

from romanian_news.catalog.archive_progress import (
    ArchiveDiscoveryMonth,
    list_archive_discovery_months,
)
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_status_archive_reads_verified_page_counts_through_catalog_transport(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    fetched_at = datetime(2026, 9, 27, 10, tzinfo=UTC)
    sitemap_url = "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=19"
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES " "(%s, %s, %s, %s, %s, %s)",
                [observation_id, "hotnews", sitemap_url, "b" * 64, fetched_at, 1],
            ),
            (
                "INSERT INTO news_archive_page_checks "
                "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
                "page_sha256, title, published_at, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    "c" * 64,
                    observation_id,
                    "hotnews",
                    "https://hotnews.ro/story-1",
                    "https://hotnews.ro/story-1",
                    fetched_at,
                    "d" * 64,
                    "Story",
                    datetime(2025, 9, 19, 10, tzinfo=UTC),
                    "accepted",
                ],
            ),
        ]
    )

    months = list_archive_discovery_months()

    assert months == (ArchiveDiscoveryMonth("hotnews", date(2025, 9, 1), 1, 1, 1),)
