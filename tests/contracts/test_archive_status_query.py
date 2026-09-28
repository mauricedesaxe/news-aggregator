from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import pytest

from romanian_news.archive.discovery import sitemap_targets
from romanian_news.catalog.archive_progress import (
    ArchiveDailyReport,
    ArchiveDiscoveryMonth,
    list_archive_daily_reports,
    list_archive_discovery_months,
)
from romanian_news.catalog.archive_report_coverage import read_retrospective_coverage
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch

DAY = date(2025, 9, 19)


def _observation_statement(
    observation_id: str,
    outlet_id: str,
    sitemap_url: str,
    content_sha256: str,
    fetched_at: datetime,
    entry_count: int,
) -> tuple[str, list[object]]:
    return (
        "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
        [observation_id, outlet_id, sitemap_url, content_sha256, fetched_at, entry_count],
    )


def _entry_statement(observation_id: str, canonical_url: str) -> tuple[str, list[object]]:
    return (
        "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s, %s)",
        [observation_id, canonical_url, canonical_url, None],
    )


def _page_check_statement(
    check_id: str,
    observation_id: str,
    outlet_id: str,
    canonical_url: str,
    fetched_at: datetime,
    status: str,
    *,
    published_at: datetime | None = None,
    page_sha256: str | None = None,
    title: str | None = None,
    rejection: str | None = None,
) -> tuple[str, list[object]]:
    return (
        "INSERT INTO news_archive_page_checks "
        "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
        "page_sha256, title, published_at, status, rejection) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        [
            check_id,
            observation_id,
            outlet_id,
            canonical_url,
            canonical_url,
            fetched_at,
            page_sha256,
            title,
            published_at,
            status,
            rejection,
        ],
    )


def _capture_statements(
    capture_id: str,
    version_id: str,
    artifact_id: str,
    observation_id: str,
    discovered_url: str,
    fetched_at: datetime,
    published_at: datetime,
) -> list[tuple[str, list[object]]]:
    created_at = datetime(2026, 9, 27, 9, tzinfo=UTC)
    return [
        (
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, current_version_id, created_at) "
            "VALUES (%s, 'news_article', 'Article', 'source', 'current', 'private', NULL, %s)",
            [artifact_id, created_at],
        ),
        (
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, "
            "content_digest, created_at) VALUES (%s, %s, 1, %s, %s)",
            [version_id, artifact_id, version_id, created_at],
        ),
        (
            "INSERT INTO news_archive_article_captures (id, observation_id, discovered_url, "
            "final_url, capture_artifact_version_id, page_sha256, fetched_at, published_at, "
            "publication_evidence) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                capture_id,
                observation_id,
                discovered_url,
                discovered_url,
                version_id,
                "e" * 64,
                fetched_at,
                published_at,
                "meta article:published_time",
            ],
        ),
    ]


def _daily_report_statements(
    day: date, version_id: str, *, current: bool = True
) -> list[tuple[str, list[object]]]:
    artifact_id = f"news:daily:{day.isoformat()}"
    created_at = datetime(2026, 9, 27, 10, tzinfo=UTC)
    statements: list[tuple[str, list[object]]] = [
        (
            "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
            "visibility, current_version_id, created_at) "
            "VALUES (%s, 'news_daily_report', 'Daily report', 'derived', 'current', "
            "'private', NULL, %s)",
            [artifact_id, created_at],
        ),
        (
            "INSERT INTO artifact_versions (id, artifact_id, schema_version, "
            "content_digest, created_at) VALUES (%s, %s, 3, %s, %s)",
            [version_id, artifact_id, version_id, created_at],
        ),
    ]
    if current:
        statements.append(
            (
                "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
                [version_id, artifact_id],
            )
        )
    return statements


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


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_discovery_inventory_reports_full_month_counters(postgres_news_schema: str) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    superseded_observation_id = "b" * 64
    sitemap_url = sitemap_targets("hotnews", DAY, DAY)[0]
    story_urls = [f"https://hotnews.ro/story-{index}" for index in range(1, 5)]
    statements = [
        _observation_statement(
            observation_id,
            "hotnews",
            sitemap_url,
            "8" * 64,
            datetime(2026, 9, 27, 11, tzinfo=UTC),
            40,
        ),
        _observation_statement(
            superseded_observation_id,
            "hotnews",
            sitemap_url,
            "9" * 64,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            90,
        ),
    ]
    statements.extend(_entry_statement(observation_id, url) for url in story_urls)
    statements.extend(
        [
            _page_check_statement(
                "c" * 64,
                observation_id,
                "hotnews",
                story_urls[0],
                datetime(2026, 9, 27, 11, 30, tzinfo=UTC),
                "accepted",
                published_at=datetime(2025, 9, 19, 10, tzinfo=UTC),
                page_sha256="d" * 64,
                title="Story",
            ),
            _page_check_statement(
                "1" * 64,
                observation_id,
                "hotnews",
                story_urls[0],
                datetime(2026, 9, 27, 11, tzinfo=UTC),
                "rejected",
                rejection="no publication date",
            ),
            _page_check_statement(
                "2" * 64,
                observation_id,
                "hotnews",
                story_urls[1],
                datetime(2026, 9, 27, 11, tzinfo=UTC),
                "rejected",
                rejection="paywall",
            ),
            _page_check_statement(
                "3" * 64,
                observation_id,
                "hotnews",
                story_urls[2],
                datetime.now(UTC) - timedelta(days=2),
                "retryable",
            ),
            _page_check_statement(
                "4" * 64,
                observation_id,
                "hotnews",
                story_urls[3],
                datetime.now(UTC) - timedelta(hours=2),
                "retryable",
            ),
        ]
    )
    statements.extend(
        _capture_statements(
            "5" * 64,
            "6" * 64,
            "news:article:captured",
            observation_id,
            story_urls[0],
            datetime(2026, 9, 27, 12, tzinfo=UTC),
            datetime(2025, 9, 19, 10, tzinfo=UTC),
        )
    )
    catalog_batch(statements)

    months = list_archive_discovery_months()

    assert months == (ArchiveDiscoveryMonth("hotnews", date(2025, 9, 1), 1, 40, 1, 1, 2, 1),)


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_discovery_inventory_parses_digi24_months_from_the_sitemap_url_path(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    url = "https://www.digi24.ro/stiri/story-1"
    catalog_batch(
        [
            _observation_statement(
                observation_id,
                "digi24",
                sitemap_targets("digi24", DAY, DAY)[0],
                "b" * 64,
                datetime(2026, 9, 27, 10, tzinfo=UTC),
                25,
            ),
            _entry_statement(observation_id, url),
            _page_check_statement(
                "c" * 64,
                observation_id,
                "digi24",
                url,
                datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
                "accepted",
                published_at=datetime(2025, 9, 19, 8, tzinfo=UTC),
                page_sha256="d" * 64,
                title="Stire",
            ),
            *_capture_statements(
                "5" * 64,
                "6" * 64,
                "news:article:digi24",
                observation_id,
                url,
                datetime(2026, 9, 27, 11, tzinfo=UTC),
                datetime(2025, 9, 19, 8, tzinfo=UTC),
            ),
        ]
    )

    months = list_archive_discovery_months()

    assert months == (ArchiveDiscoveryMonth("digi24", date(2025, 9, 1), 1, 25, 1, 0, 0, 1),)


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_retrospective_coverage_aggregates_one_bucharest_publication_day(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    hotnews_observation_id = "a" * 64
    digi24_observation_id = "b" * 64
    hotnews_sitemap = sitemap_targets("hotnews", DAY, DAY)[0]
    digi24_sitemap = sitemap_targets("digi24", DAY, DAY)[0]
    inside_one = "https://hotnews.ro/story-1"
    inside_two = "https://hotnews.ro/story-3"
    outside = "https://hotnews.ro/story-2"
    digi24_url = "https://www.digi24.ro/stiri/story-1"
    statements = [
        _observation_statement(
            hotnews_observation_id,
            "hotnews",
            hotnews_sitemap,
            "8" * 64,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            3,
        ),
        _observation_statement(
            digi24_observation_id,
            "digi24",
            digi24_sitemap,
            "9" * 64,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            1,
        ),
        _entry_statement(hotnews_observation_id, inside_one),
        _entry_statement(hotnews_observation_id, inside_two),
        _entry_statement(hotnews_observation_id, outside),
        _entry_statement(digi24_observation_id, digi24_url),
        _page_check_statement(
            "c" * 64,
            hotnews_observation_id,
            "hotnews",
            inside_one,
            datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
            "accepted",
            published_at=datetime(2025, 9, 19, 8, tzinfo=UTC),
            page_sha256="d" * 64,
            title="Story",
        ),
        _page_check_statement(
            "1" * 64,
            hotnews_observation_id,
            "hotnews",
            inside_two,
            datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
            "accepted",
            published_at=datetime(2025, 9, 19, 12, tzinfo=UTC),
            page_sha256="d" * 64,
            title="Story",
        ),
        _page_check_statement(
            "2" * 64,
            hotnews_observation_id,
            "hotnews",
            outside,
            datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
            "accepted",
            published_at=datetime(2025, 9, 19, 21, tzinfo=UTC),
            page_sha256="d" * 64,
            title="Story",
        ),
        _page_check_statement(
            "3" * 64,
            digi24_observation_id,
            "digi24",
            digi24_url,
            datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
            "accepted",
            published_at=datetime(2025, 9, 19, 9, tzinfo=UTC),
            page_sha256="d" * 64,
            title="Stire",
        ),
        *_capture_statements(
            "4" * 64,
            "5" * 64,
            "news:article:hotnews-one",
            hotnews_observation_id,
            inside_one,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            datetime(2025, 9, 19, 8, tzinfo=UTC),
        ),
        *_capture_statements(
            "6" * 64,
            "7" * 64,
            "news:article:hotnews-two",
            hotnews_observation_id,
            inside_two,
            datetime(2026, 9, 27, 11, tzinfo=UTC),
            datetime(2025, 9, 19, 12, tzinfo=UTC),
        ),
        *_capture_statements(
            "8" * 64,
            "9" * 64,
            "news:article:digi24-one",
            digi24_observation_id,
            digi24_url,
            datetime(2026, 9, 27, 9, 30, tzinfo=UTC),
            datetime(2025, 9, 19, 9, tzinfo=UTC),
        ),
    ]
    catalog_batch(statements)

    coverage = read_retrospective_coverage(DAY)

    assert coverage is not None
    assert coverage.verified_page_count == coverage.discovered_url_count == 3
    assert coverage.captured_article_count == 3
    assert coverage.included_outlets == ("digi24", "hotnews")
    assert coverage.capture_started_at == datetime(2026, 9, 27, 9, 30, tzinfo=UTC)
    assert coverage.capture_ended_at == datetime(2026, 9, 27, 11, tzinfo=UTC)


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_retrospective_coverage_returns_none_without_captures(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    url = "https://hotnews.ro/story-1"
    catalog_batch(
        [
            _observation_statement(
                observation_id,
                "hotnews",
                sitemap_targets("hotnews", DAY, DAY)[0],
                "b" * 64,
                datetime(2026, 9, 27, 10, tzinfo=UTC),
                1,
            ),
            _entry_statement(observation_id, url),
            _page_check_statement(
                "c" * 64,
                observation_id,
                "hotnews",
                url,
                datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
                "accepted",
                published_at=datetime(2025, 9, 19, 8, tzinfo=UTC),
                page_sha256="d" * 64,
                title="Story",
            ),
        ]
    )

    assert read_retrospective_coverage(DAY) is None


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_retrospective_coverage_rejects_more_captures_than_verified_pages(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    observation_id = "a" * 64
    verified = "https://hotnews.ro/story-1"
    uncaptured = "https://hotnews.ro/story-3"
    statements = [
        _observation_statement(
            observation_id,
            "hotnews",
            sitemap_targets("hotnews", DAY, DAY)[0],
            "b" * 64,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            2,
        ),
        _entry_statement(observation_id, verified),
        _entry_statement(observation_id, uncaptured),
        _page_check_statement(
            "c" * 64,
            observation_id,
            "hotnews",
            verified,
            datetime(2026, 9, 27, 10, 30, tzinfo=UTC),
            "accepted",
            published_at=datetime(2025, 9, 19, 8, tzinfo=UTC),
            page_sha256="d" * 64,
            title="Story",
        ),
        *_capture_statements(
            "4" * 64,
            "5" * 64,
            "news:article:one",
            observation_id,
            verified,
            datetime(2026, 9, 27, 10, tzinfo=UTC),
            datetime(2025, 9, 19, 8, tzinfo=UTC),
        ),
        *_capture_statements(
            "6" * 64,
            "7" * 64,
            "news:article:two",
            observation_id,
            uncaptured,
            datetime(2026, 9, 27, 11, tzinfo=UTC),
            datetime(2025, 9, 19, 12, tzinfo=UTC),
        ),
    ]
    catalog_batch(statements)

    with pytest.raises(ValueError, match="Archive captures exceed verified pages"):
        read_retrospective_coverage(DAY)


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_archive_daily_reports_order_days_descending_and_honor_the_artifact_id_range(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    statements: list[tuple[str, list[object]]] = []
    for day, version_id, current in (
        (date(2025, 9, 17), "1" * 64, True),
        (date(2025, 9, 18), "2" * 64, True),
        (DAY, "3" * 64, True),
        (date(2025, 9, 20), "4" * 64, True),
        (date(2025, 9, 21), "5" * 64, False),
        (date(2025, 9, 22), "6" * 64, True),
    ):
        statements.extend(_daily_report_statements(day, version_id, current=current))
    catalog_batch(statements)

    assert list_archive_daily_reports(date(2025, 9, 18), date(2025, 9, 20)) == (
        ArchiveDailyReport(date(2025, 9, 20), "4" * 64),
        ArchiveDailyReport(date(2025, 9, 19), "3" * 64),
        ArchiveDailyReport(date(2025, 9, 18), "2" * 64),
    )
    assert list_archive_daily_reports(DAY, DAY) == (ArchiveDailyReport(DAY, "3" * 64),)
