from datetime import UTC, date, datetime

import pytest

from romanian_news.archive.discovery import parse_sitemap, sitemap_targets


def test_sitemap_targets_keep_monthly_and_daily_sources_distinct() -> None:
    start = date(2025, 9, 30)
    end = date(2025, 10, 1)

    assert sitemap_targets("hotnews", start, end) == (
        "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=30",
        "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=10&dd=01",
    )
    assert sitemap_targets("digi24", start, end) == (
        "https://www.digi24.ro/sitemaps/sitemap-articles-2025-09.xml",
        "https://www.digi24.ro/sitemaps/sitemap-articles-2025-10.xml",
    )
    with pytest.raises(ValueError, match="1 to 31 days"):
        sitemap_targets("hotnews", start, date(2025, 11, 1))


def test_sitemap_observation_deduplicates_urls_without_treating_lastmod_as_publication() -> None:
    content = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://hotnews.ro/story-1?utm_source=sample</loc>
        <lastmod>2025-09-19T10:00:00+03:00</lastmod></url>
      <url><loc>https://hotnews.ro/story-1</loc>
        <lastmod>2025-09-20T10:00:00+03:00</lastmod></url>
    </urlset>"""
    fetched_at = datetime(2026, 9, 27, tzinfo=UTC)
    sitemap_url = "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=19"

    first = parse_sitemap("hotnews", sitemap_url, content, fetched_at)
    replay = parse_sitemap("hotnews", sitemap_url, content, fetched_at)

    assert first.id == replay.id
    assert len(first.entries) == 1
    assert first.entries[0].canonical_url == "https://hotnews.ro/story-1"
    assert first.entries[0].lastmod_hint == "2025-09-19T10:00:00+03:00"
    assert first.fetched_at == fetched_at
