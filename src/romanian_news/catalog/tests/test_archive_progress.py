from datetime import date

from romanian_news.catalog import archive_progress


def test_discovery_inventory_groups_latest_sitemaps_by_month(monkeypatch) -> None:
    monkeypatch.setattr(
        archive_progress,
        "catalog_query",
        lambda _query, _parameters: [
            {
                "outlet_id": "hotnews",
                "sitemap_url": "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=19",
                "entry_count": 90,
            },
            {
                "outlet_id": "hotnews",
                "sitemap_url": "https://hotnews.ro/sitemap.xml?yyyy=2025&mm=09&dd=20",
                "entry_count": 69,
            },
            {
                "outlet_id": "digi24",
                "sitemap_url": "https://www.digi24.ro/sitemaps/sitemap-articles-2025-09.xml",
                "entry_count": 3186,
            },
        ],
    )

    rows = archive_progress.list_archive_discovery_months()

    assert archive_progress.ArchiveDiscoveryMonth("hotnews", date(2025, 9, 1), 2, 159) in rows
    assert archive_progress.ArchiveDiscoveryMonth("digi24", date(2025, 9, 1), 1, 3186) in rows
