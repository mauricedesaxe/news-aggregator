from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime

import pytest

from romanian_news.archive import page_checks
from romanian_news.archive.backfill import advance_page_checks
from romanian_news.archive.discovery import sitemap_targets
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import catalog_batch, catalog_query
from scripts import next_historical_article_capture as capture_cli
from scripts.next_historical_article_capture import next_capture_config

ARCHIVE_PAGE = b"""<html><head>
  <meta property="og:title" content="An archived story">
  <meta property="article:published_time" content="2025-10-05T10:00:00+03:00">
</head></html>"""


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _seed_sitemap_entry(outlet_id: str, day: date, url: str) -> None:
    sitemap_url = sitemap_targets(outlet_id, day, day)[0]
    observation_id = _sha(f"{outlet_id}-observation-{day}")
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    outlet_id,
                    sitemap_url,
                    _sha(f"{outlet_id}-content-{day}"),
                    datetime(2026, 9, 27, tzinfo=UTC),
                    1,
                ],
            ),
            (
                "INSERT INTO news_archive_sitemap_entries VALUES (%s, %s, %s, %s)",
                [observation_id, url, url, None],
            ),
        ]
    )


def _serve_pages(monkeypatch: pytest.MonkeyPatch, pages: dict[str, bytes]) -> None:
    def fetch(_session: object, url: str) -> tuple[str, bytes]:
        if url.endswith("/robots.txt"):
            return url, b"User-agent: *\nAllow: /\n"
        return url, pages[url]

    monkeypatch.setattr(page_checks, "_fetch", fetch)


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_advance_page_checks_verifies_the_oldest_month_with_pending_pages(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/october-story-1"
    _seed_sitemap_entry("hotnews", date(2025, 10, 5), url)
    _serve_pages(monkeypatch, {url: ARCHIVE_PAGE})

    result = advance_page_checks("hotnews", date(2025, 9, 27), date(2025, 10, 31))

    assert result == {
        "outlet": "hotnews",
        "start": "2025-10-01",
        "end": "2025-10-31",
        "checked": 1,
        "statuses": {"accepted": 1},
    }
    rows = catalog_query(
        "SELECT status, title FROM news_archive_page_checks WHERE canonical_url = %s", [url]
    )
    assert [(row["status"], row["title"]) for row in rows] == [("accepted", "An archived story")]

    assert advance_page_checks("hotnews", date(2025, 9, 27), date(2025, 10, 31)) == {
        "outlet": "hotnews",
        "checked": 0,
        "complete": True,
    }


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_capture_config_targets_the_oldest_month_with_verified_articles(
    postgres_news_schema: str,
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/october-story-2"
    observation_id = _sha("hotnews-capture-observation")
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    "hotnews",
                    sitemap_targets("hotnews", date(2025, 10, 5), date(2025, 10, 5))[0],
                    _sha("hotnews-capture-content"),
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
                    _sha("hotnews-capture-check"),
                    observation_id,
                    "hotnews",
                    url,
                    url,
                    datetime(2026, 9, 27, tzinfo=UTC),
                    _sha("hotnews-capture-page"),
                    "Story",
                    datetime(2025, 10, 5, 10, tzinfo=UTC),
                    "accepted",
                ],
            ),
        ]
    )

    assert next_capture_config("hotnews", date(2025, 9, 27), date(2025, 10, 31)) == {
        "ops": {
            "archive_article_capture": {
                "config": {
                    "outlet": "hotnews",
                    "start": "2025-10-01",
                    "end": "2025-10-31",
                    "limit": 50,
                }
            }
        }
    }
    assert next_capture_config("hotnews", date(2025, 9, 27), date(2025, 9, 30)) is None


@pytest.mark.skipif(
    os.getenv("NEWS_TEST_POSTGRES_DSN") is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)
def test_capture_cli_reports_the_next_window_and_rejects_unknown_outlets(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    del postgres_news_schema
    ensure_news_catalog_schema()
    url = "https://hotnews.ro/october-story-3"
    observation_id = _sha("hotnews-capture-cli-observation")
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    observation_id,
                    "hotnews",
                    sitemap_targets("hotnews", date(2025, 10, 5), date(2025, 10, 5))[0],
                    _sha("hotnews-capture-cli-content"),
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
                    _sha("hotnews-capture-cli-check"),
                    observation_id,
                    "hotnews",
                    url,
                    url,
                    datetime(2026, 9, 27, tzinfo=UTC),
                    _sha("hotnews-capture-cli-page"),
                    "Story",
                    datetime(2025, 10, 5, 10, tzinfo=UTC),
                    "accepted",
                ],
            ),
        ]
    )

    monkeypatch.setattr("sys.argv", ["next_historical_article_capture", "--outlet", "hotnews"])
    capture_cli.main()
    assert json.loads(capsys.readouterr().out) == {
        "ops": {
            "archive_article_capture": {
                "config": {
                    "outlet": "hotnews",
                    "start": "2025-10-01",
                    "end": "2025-10-31",
                    "limit": 50,
                }
            }
        }
    }

    monkeypatch.setattr("sys.argv", ["next_historical_article_capture", "--outlet", "digi24"])
    capture_cli.main()
    assert capsys.readouterr().out == "complete\n"

    monkeypatch.setattr("sys.argv", ["next_historical_article_capture", "--outlet", "unknown"])
    with pytest.raises(SystemExit) as exited:
        capture_cli.main()
    assert exited.value.code == 2
