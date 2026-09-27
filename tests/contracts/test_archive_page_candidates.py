from __future__ import annotations

import os
from datetime import UTC, date, datetime

import pytest

from romanian_news.archive.discovery import sitemap_targets
from romanian_news.archive.page_checks import pending_page_candidates
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_pending_page_candidates_uses_latest_sitemap_observation(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    day = date(2025, 9, 19)
    sitemap_url = sitemap_targets("hotnews", day, day)[0]
    old_url = "https://hotnews.ro/old-story-1"
    current_url = "https://hotnews.ro/current-story-1"
    observations = (("a" * 64, "b" * 64, old_url), ("c" * 64, "d" * 64, current_url))
    statements: list[tuple[str, list[object]]] = []
    for hour, (observation_id, digest, url) in enumerate(observations):
        statements.extend(
            [
                (
                    "INSERT INTO news_archive_sitemap_observations VALUES "
                    "(%s, %s, %s, %s, %s, %s)",
                    [
                        observation_id,
                        "hotnews",
                        sitemap_url,
                        digest,
                        datetime(2026, 9, 27, hour, tzinfo=UTC),
                        1,
                    ],
                ),
                (
                    "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s, %s)",
                    [observation_id, url, url, None],
                ),
            ]
        )
    catalog_batch(statements)

    candidates = pending_page_candidates("hotnews", day, day, 50)

    assert len(candidates) == 1
    assert candidates[0].observation_id == "c" * 64
    assert candidates[0].canonical_url == current_url
