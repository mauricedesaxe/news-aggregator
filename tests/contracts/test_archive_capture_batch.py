from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import pytest

from romanian_news.archive import capture_batch
from romanian_news.archive.capture_batch import (
    capture_archive_batch,
    next_capture_window,
    pending_archive_articles,
)
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch, catalog_query


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


def _seed_accepted_page_checks(pages: tuple[tuple[str, datetime], ...]) -> str:
    observation_id = "a" * 64
    fetched_at = datetime(2026, 9, 27, tzinfo=UTC)
    catalog_batch(
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
        ],
    )
    catalog_batch(
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
        ],
    )
    return observation_id


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_capture_batch_skips_robots_disallowed_pages_without_fetching(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/secret-story"
    _seed_accepted_page_checks(((url, datetime(2025, 9, 19, 10, tzinfo=UTC)),))

    def serve_robots_only(_session: object, fetched_url: str) -> tuple[str, bytes]:
        if fetched_url == "https://hotnews.ro/robots.txt":
            return fetched_url, b"User-agent: *\nDisallow: /secret-story\n"
        raise AssertionError(f"robots-disallowed page must not be fetched: {fetched_url}")

    monkeypatch.setattr(capture_batch, "_fetch", serve_robots_only)

    result = capture_archive_batch(
        "hotnews", date(2025, 9, 19), date(2025, 9, 19), limit=5, implementation_ref="git:test"
    )

    assert result.selected == 1
    assert result.published == 0
    assert result.unchanged == 0
    assert result.failed == ((url, "robots_disallowed"),)
    assert catalog_query("SELECT id FROM news_archive_article_captures") == []


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_capture_batch_rejects_a_publisher_date_changed_after_verification(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/drift-story"
    _seed_accepted_page_checks(((url, datetime(2025, 9, 19, 10, tzinfo=UTC)),))
    html = (
        "<html><head>"
        '<meta property="og:title" content="Anunț despre un proiect public">'
        '<meta property="article:published_time" content="2025-09-20T10:00:00+03:00">'
        "</head><body><article class='article-single-content'>"
        "<h1>Anunț despre un proiect public</h1>"
        f"<p>{'Instituția a publicat detalii despre proiectul local. ' * 6}</p>"
        "</article></body></html>"
    ).encode()

    def serve_robots_and_article(_session: object, fetched_url: str) -> tuple[str, bytes]:
        if fetched_url == "https://hotnews.ro/robots.txt":
            return fetched_url, b"User-agent: *\nAllow: /\n"
        if fetched_url == url:
            return url, html
        raise AssertionError(f"unexpected fetch: {fetched_url}")

    monkeypatch.setattr(capture_batch, "_fetch", serve_robots_and_article)

    result = capture_archive_batch(
        "hotnews", date(2025, 9, 19), date(2025, 9, 19), limit=5, implementation_ref="git:test"
    )

    assert result.selected == 1
    assert result.published == 0
    assert result.unchanged == 0
    assert result.failed == (
        (url, "ValueError: Publisher publication date changed after verification"),
    )
    assert catalog_query("SELECT id FROM news_archive_article_captures") == []


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_captured_pages_leave_pending_and_capture_windows(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/captured-story"
    publication = datetime(2025, 9, 19, 10, tzinfo=UTC)
    observation_id = _seed_accepted_page_checks(((url, publication),))
    month = (date(2025, 9, 1), date(2025, 9, 30))
    assert next_capture_window("hotnews", *month) == month

    catalog_batch(
        [
            (
                "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
                "visibility, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    "2" * 64,
                    "news:article",
                    "Story",
                    "publisher",
                    "current",
                    "public",
                    datetime(2026, 9, 27, tzinfo=UTC),
                ],
            ),
            (
                "INSERT INTO artifact_versions (id, artifact_id, schema_version, "
                "content_digest, produced_by_run_id, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ["3" * 64, "2" * 64, 1, "4" * 64, None, datetime(2026, 9, 27, tzinfo=UTC)],
            ),
            (
                "INSERT INTO news_archive_article_captures (id, observation_id, "
                "discovered_url, final_url, capture_artifact_version_id, page_sha256, "
                "fetched_at, published_at, modified_at, publication_evidence) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    "5" * 64,
                    observation_id,
                    url,
                    url,
                    "3" * 64,
                    "6" * 64,
                    datetime(2026, 9, 27, 12, tzinfo=UTC),
                    publication,
                    None,
                    "article:published_time",
                ],
            ),
        ]
    )

    assert pending_archive_articles("hotnews", date(2025, 9, 19), date(2025, 9, 19), 10) == ()
    assert next_capture_window("hotnews", *month) is None


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_pending_archive_articles_batches_earliest_publications_first(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    _seed_accepted_page_checks(
        (
            ("https://hotnews.ro/story-late", datetime(2025, 9, 19, 10, tzinfo=UTC)),
            ("https://hotnews.ro/story-early", datetime(2025, 9, 19, 8, tzinfo=UTC)),
            ("https://hotnews.ro/story-middle", datetime(2025, 9, 19, 9, tzinfo=UTC)),
        )
    )

    selected = pending_archive_articles("hotnews", date(2025, 9, 19), date(2025, 9, 19), 2)

    assert tuple(page.canonical_url for page in selected) == (
        "https://hotnews.ro/story-early",
        "https://hotnews.ro/story-middle",
    )
    assert selected[0].published_at < selected[1].published_at


@pytest.mark.parametrize(
    ("day", "start", "end", "expected_hours"),
    (
        (
            date(2025, 3, 30),
            datetime(2025, 3, 29, 22, tzinfo=UTC),
            datetime(2025, 3, 30, 21, tzinfo=UTC),
            23,
        ),
        (
            date(2025, 10, 26),
            datetime(2025, 10, 25, 21, tzinfo=UTC),
            datetime(2025, 10, 26, 22, tzinfo=UTC),
            25,
        ),
        (
            date(2025, 9, 27),
            datetime(2025, 9, 26, 21, tzinfo=UTC),
            datetime(2025, 9, 27, 21, tzinfo=UTC),
            24,
        ),
    ),
)
@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_pending_archive_articles_uses_bucharest_day_boundaries(
    postgres_news_schema: str,
    day: date,
    start: datetime,
    end: datetime,
    expected_hours: int,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    assert (end - start) == timedelta(hours=expected_hours)
    _seed_accepted_page_checks(
        (
            ("https://hotnews.ro/before-window", start - timedelta(microseconds=1)),
            ("https://hotnews.ro/at-window-start", start),
            ("https://hotnews.ro/before-window-end", end - timedelta(microseconds=1)),
            ("https://hotnews.ro/at-window-end", end),
        )
    )

    selected = pending_archive_articles("hotnews", day, day, 10)

    assert tuple(page.canonical_url for page in selected) == (
        "https://hotnews.ro/at-window-start",
        "https://hotnews.ro/before-window-end",
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2025, 10, 1), date(2025, 9, 1)),
        (date(2025, 9, 1), date(2025, 10, 5)),
    ],
)
def test_archive_capture_window_must_be_a_bounded_month(start: date, end: date) -> None:
    with pytest.raises(ValueError, match="at most 31 days"):
        pending_archive_articles("hotnews", start, end, 1)


@pytest.mark.parametrize("limit", [0, 51])
def test_archive_capture_limit_must_stay_within_the_politeness_bound(limit: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 50"):
        pending_archive_articles("hotnews", date(2025, 9, 1), date(2025, 9, 30), limit)


def test_archive_capture_requires_at_least_one_second_between_requests() -> None:
    with pytest.raises(ValueError, match="at least one second"):
        capture_archive_batch(
            "hotnews",
            date(2025, 9, 19),
            date(2025, 9, 19),
            limit=5,
            implementation_ref="git:test",
            delay_seconds=0.5,
        )


def test_archive_capture_rejects_outlets_outside_the_archive_campaign() -> None:
    with pytest.raises(ValueError, match="Unknown archive outlet"):
        capture_archive_batch(
            "gsp",
            date(2025, 9, 19),
            date(2025, 9, 19),
            limit=5,
            implementation_ref="git:test",
        )
