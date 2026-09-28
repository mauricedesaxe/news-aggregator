"""Publish a bounded batch of verified historical article pages."""

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from romanian_news.archive.articles import capture_archive_article
from romanian_news.archive.discovery import USER_AGENT
from romanian_news.archive.page_checks import _fetch
from romanian_news.archive.windows import month_windows
from romanian_news.articles.extraction import normalize_article_url
from romanian_news.catalog.archive_articles import publish_archive_article
from romanian_news.catalog_transport import catalog_query
from romanian_news.daily import bucharest_day_window
from romanian_news.feeds.registry import feed_registry


@dataclass(frozen=True)
class AcceptedArchivePage:
    observation_id: str
    canonical_url: str
    published_at: datetime


@dataclass(frozen=True)
class ArchiveCaptureBatchResult:
    selected: int
    published: int
    unchanged: int
    failed: tuple[tuple[str, str], ...]


def next_capture_window(outlet_id: str, start: date, end: date) -> tuple[date, date] | None:
    for month_start, month_end in month_windows(start, end):
        if pending_archive_articles(outlet_id, month_start, month_end, 1):
            return month_start, month_end
    return None


def pending_archive_articles(
    outlet_id: str, start: date, end: date, limit: int
) -> tuple[AcceptedArchivePage, ...]:
    if start > end or (end - start).days > 31:
        raise ValueError("Archive article window must span at most 31 days")
    if not 1 <= limit <= 50:
        raise ValueError("Archive article limit must be between 1 and 50")
    window_start = bucharest_day_window(start)[0]
    window_end = bucharest_day_window(end)[1]
    rows = catalog_query(
        """
        SELECT accepted.observation_id, accepted.canonical_url, accepted.published_at
        FROM (
            SELECT DISTINCT ON (outlet_id, canonical_url)
                   observation_id, canonical_url, published_at
            FROM news_archive_page_checks
            WHERE outlet_id = %s AND status = 'accepted'
              AND published_at >= %s AND published_at < %s
            ORDER BY outlet_id, canonical_url, fetched_at DESC, id DESC
        ) accepted
        WHERE NOT EXISTS (
            SELECT 1 FROM news_archive_article_captures capture
            WHERE capture.observation_id = accepted.observation_id
              AND capture.discovered_url = accepted.canonical_url
        )
        ORDER BY accepted.published_at, accepted.canonical_url
        LIMIT %s
        """,
        [outlet_id, window_start, window_end, limit],
    )
    return tuple(
        AcceptedArchivePage(
            observation_id=str(row["observation_id"]),
            canonical_url=str(row["canonical_url"]),
            published_at=datetime.fromisoformat(str(row["published_at"])),
        )
        for row in rows
    )


def capture_archive_batch(
    outlet_id: str,
    start: date,
    end: date,
    *,
    limit: int,
    implementation_ref: str,
    delay_seconds: float = 1.0,
) -> ArchiveCaptureBatchResult:
    if delay_seconds < 1.0:
        raise ValueError("Archive capture requires at least one second between requests")
    feed = next((item for item in feed_registry().feeds if item.id == outlet_id), None)
    if feed is None or outlet_id not in {"hotnews", "digi24"}:
        raise ValueError(f"Unknown archive outlet: {outlet_id}")
    candidates = pending_archive_articles(outlet_id, start, end, limit)
    if not candidates:
        return ArchiveCaptureBatchResult(0, 0, 0, ())
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    parts = urlsplit(str(feed.url))
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    _, robots_content = _fetch(session, robots_url)
    robots = RobotFileParser()
    robots.set_url(robots_url)
    robots.parse(robots_content.decode("utf-8", errors="replace").splitlines())
    published = unchanged = 0
    failed: list[tuple[str, str]] = []
    for index, candidate in enumerate(candidates):
        if index:
            time.sleep(delay_seconds)
        if not robots.can_fetch(USER_AGENT, candidate.canonical_url):
            failed.append((candidate.canonical_url, "robots_disallowed"))
            continue
        try:
            final_url, html = _fetch(session, candidate.canonical_url)
            normalize_article_url(final_url, feed.article_hosts)
            capture = capture_archive_article(
                observation_id=candidate.observation_id,
                discovered_url=candidate.canonical_url,
                final_url=final_url,
                html=html,
                fetched_at=datetime.now(UTC),
                feed=feed,
            )
            if capture.article.published_at != candidate.published_at:
                raise ValueError("Publisher publication date changed after verification")
            result = publish_archive_article(capture, implementation_ref)
            published += result.published_articles
            unchanged += result.unchanged_articles
        except (requests.RequestException, ValueError) as error:
            failed.append((candidate.canonical_url, type(error).__name__ + ": " + str(error)))
    return ArchiveCaptureBatchResult(len(candidates), published, unchanged, tuple(failed))
