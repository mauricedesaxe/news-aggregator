"""Bounded, metadata-only checks of discovered publisher article pages."""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
from lxml.etree import ParserError

from romanian_news.archive.discovery import (
    MAX_RESPONSE_BYTES,
    TIMEOUT_SECONDS,
    USER_AGENT,
    sitemap_targets,
)
from romanian_news.archive.page_metadata import ArchivePageMetadata, extract_archive_page_metadata
from romanian_news.articles.extraction import normalize_article_url
from romanian_news.catalog_transport import catalog_batch, catalog_query
from romanian_news.feeds.registry import feed_registry


@dataclass(frozen=True)
class ArchivePageCandidate:
    observation_id: str
    outlet_id: str
    canonical_url: str


@dataclass(frozen=True)
class ArchivePageCheck:
    id: str
    candidate: ArchivePageCandidate
    final_url: str | None
    fetched_at: datetime
    page_sha256: str | None
    metadata: ArchivePageMetadata | None
    status: str
    rejection: str | None


def pending_page_candidates(
    outlet_id: str, start: date, end: date, limit: int
) -> tuple[ArchivePageCandidate, ...]:
    if not 1 <= limit <= 100:
        raise ValueError("Page check limit must be between 1 and 100")
    targets = sitemap_targets(outlet_id, start, end)
    rows = catalog_query(
        """
        WITH latest AS (
            SELECT DISTINCT ON (sitemap_url) id, outlet_id, sitemap_url
            FROM news_archive_sitemap_observations
            WHERE outlet_id = %s AND sitemap_url = ANY(%s::text[])
            ORDER BY sitemap_url, fetched_at DESC, id DESC
        ), candidates AS (
            SELECT DISTINCT ON (entry.canonical_url)
                   latest.id AS observation_id, latest.outlet_id, entry.canonical_url
            FROM latest
            JOIN news_archive_sitemap_entries entry ON entry.observation_id = latest.id
            ORDER BY entry.canonical_url, latest.id
        )
        SELECT candidate.observation_id, candidate.outlet_id, candidate.canonical_url
        FROM candidates candidate
        WHERE NOT EXISTS (
            SELECT 1 FROM news_archive_page_checks checked
            WHERE checked.outlet_id = candidate.outlet_id
              AND checked.canonical_url = candidate.canonical_url
              AND checked.status IN ('accepted', 'rejected')
        )
          AND NOT EXISTS (
            SELECT 1 FROM news_archive_page_checks recent
            WHERE recent.outlet_id = candidate.outlet_id
              AND recent.canonical_url = candidate.canonical_url
              AND recent.status = 'retryable'
              AND recent.fetched_at > now() - interval '1 day'
        )
        ORDER BY candidate.canonical_url
        LIMIT %s
        """,
        [outlet_id, list(targets), limit],
    )
    return tuple(ArchivePageCandidate(**row) for row in rows)


def _fetch(session: requests.Session, url: str) -> tuple[str, bytes]:
    with session.get(url, timeout=TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > MAX_RESPONSE_BYTES:
                raise ValueError("Page response exceeds size limit")
        return response.url, bytes(content)


def check_page(
    candidate: ArchivePageCandidate,
    session: requests.Session,
    robots: RobotFileParser,
    allowed_hosts: tuple[str, ...],
) -> ArchivePageCheck:
    fetched_at = datetime.now(UTC)
    final_url = None
    digest = None
    metadata = None
    status = "retryable"
    rejection = None
    if not robots.can_fetch(USER_AGENT, candidate.canonical_url):
        status, rejection = "rejected", "robots_disallowed"
    else:
        try:
            final_url, content = _fetch(session, candidate.canonical_url)
            digest = hashlib.sha256(content).hexdigest()
            normalized_url = normalize_article_url(final_url, allowed_hosts)
            if urlsplit(normalized_url).path == "/":
                status, rejection = "rejected", "redirected_to_homepage"
            else:
                metadata = extract_archive_page_metadata(content)
                status = "accepted" if metadata.rejection is None else "rejected"
                rejection = metadata.rejection
        except requests.RequestException as error:
            rejection = type(error).__name__
        except (ValueError, ParserError) as error:
            status, rejection = "rejected", type(error).__name__
    identity = hashlib.sha256(
        "\0".join(
            (
                candidate.observation_id,
                candidate.canonical_url,
                digest or fetched_at.isoformat(),
                status,
            )
        ).encode()
    ).hexdigest()
    return ArchivePageCheck(
        id=identity,
        candidate=candidate,
        final_url=final_url,
        fetched_at=fetched_at,
        page_sha256=digest,
        metadata=metadata,
        status=status,
        rejection=rejection,
    )


def record_page_check(check: ArchivePageCheck) -> None:
    metadata = check.metadata
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_page_checks "
                "(id, observation_id, outlet_id, canonical_url, final_url, fetched_at, "
                "page_sha256, title, published_at, modified_at, publication_evidence, "
                "rejection, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                [
                    check.id,
                    check.candidate.observation_id,
                    check.candidate.outlet_id,
                    check.candidate.canonical_url,
                    check.final_url,
                    check.fetched_at.isoformat(),
                    check.page_sha256,
                    metadata.title if metadata else None,
                    metadata.published_at.isoformat()
                    if metadata and metadata.published_at
                    else None,
                    metadata.modified_at.isoformat() if metadata and metadata.modified_at else None,
                    metadata.publication_evidence if metadata else None,
                    check.rejection,
                    check.status,
                ],
            )
        ],
        retry_transient_errors=True,
    )


def check_archive_pages(
    outlet_id: str, start: date, end: date, *, limit: int = 50, delay_seconds: float = 1.0
) -> tuple[ArchivePageCheck, ...]:
    if delay_seconds < 1.0:
        raise ValueError("Page checks require at least one second between requests")
    feed = next((item for item in feed_registry().feeds if item.id == outlet_id), None)
    if feed is None:
        raise ValueError(f"Unknown archive outlet: {outlet_id}")
    candidates = pending_page_candidates(outlet_id, start, end, limit)
    if not candidates:
        return ()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    parts = urlsplit(str(feed.url))
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    _, robots_content = _fetch(session, robots_url)
    robots = RobotFileParser()
    robots.set_url(robots_url)
    robots.parse(robots_content.decode("utf-8", errors="replace").splitlines())
    checks = []
    for index, candidate in enumerate(candidates):
        if index:
            time.sleep(delay_seconds)
        check = check_page(candidate, session, robots, feed.article_hosts)
        record_page_check(check)
        checks.append(check)
    return tuple(checks)
