from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import requests

from romanian_news.articles.extraction import normalize_article_url
from romanian_news.catalog_transport import catalog_batch
from romanian_news.feeds.registry import feed_registry

USER_AGENT = (
    "Romanian news archive research/1.0 (+https://github.com/mauricedesaxe/news-aggregator)"
)
MAX_RESPONSE_BYTES = 5_000_000
TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ArchiveSitemapEntry:
    canonical_url: str
    source_url: str
    lastmod_hint: str | None


@dataclass(frozen=True)
class ArchiveSitemapObservation:
    id: str
    outlet_id: str
    sitemap_url: str
    content_sha256: str
    fetched_at: datetime
    entries: tuple[ArchiveSitemapEntry, ...]


def sitemap_targets(outlet_id: str, start: date, end: date) -> tuple[str, ...]:
    if end < start or (end - start).days > 30:
        raise ValueError("Discovery range must contain 1 to 31 days")
    if outlet_id == "hotnews":
        days = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
        return tuple(
            f"https://hotnews.ro/sitemap.xml?yyyy={day.year}&mm={day.month:02d}&dd={day.day:02d}"
            for day in days
        )
    if outlet_id == "digi24":
        months = {
            (day.year, day.month)
            for day in (start + timedelta(days=offset) for offset in range((end - start).days + 1))
        }
        return tuple(
            f"https://www.digi24.ro/sitemaps/sitemap-articles-{year}-{month:02d}.xml"
            for year, month in sorted(months)
        )
    raise ValueError(f"Unsupported archive outlet: {outlet_id}")


def parse_sitemap(
    outlet_id: str, sitemap_url: str, content: bytes, fetched_at: datetime
) -> ArchiveSitemapObservation:
    feed = next((item for item in feed_registry().feeds if item.id == outlet_id), None)
    if feed is None:
        raise ValueError(f"Unknown archive outlet: {outlet_id}")
    root = ElementTree.fromstring(content)
    if root.tag.rsplit("}", 1)[-1] != "urlset":
        raise ValueError("Expected an article URL sitemap")
    entries: dict[str, ArchiveSitemapEntry] = {}
    for item in root:
        source_url = item.findtext("{*}loc")
        if not source_url:
            continue
        canonical_url = normalize_article_url(source_url, feed.article_hosts)
        entries.setdefault(
            canonical_url,
            ArchiveSitemapEntry(
                canonical_url=canonical_url,
                source_url=source_url.strip(),
                lastmod_hint=item.findtext("{*}lastmod"),
            ),
        )
    digest = hashlib.sha256(content).hexdigest()
    identity = hashlib.sha256(f"{outlet_id}\0{sitemap_url}\0{digest}".encode()).hexdigest()
    return ArchiveSitemapObservation(
        id=identity,
        outlet_id=outlet_id,
        sitemap_url=sitemap_url,
        content_sha256=digest,
        fetched_at=fetched_at.astimezone(UTC),
        entries=tuple(entries.values()),
    )


def record_sitemap(observation: ArchiveSitemapObservation) -> None:
    entries = observation.entries
    catalog_batch(
        [
            (
                "INSERT INTO news_archive_sitemap_observations "
                "(id, outlet_id, sitemap_url, content_sha256, fetched_at, entry_count) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    observation.id,
                    observation.outlet_id,
                    observation.sitemap_url,
                    observation.content_sha256,
                    observation.fetched_at.isoformat(),
                    len(entries),
                ],
            ),
            (
                "INSERT INTO news_archive_sitemap_entries "
                "(observation_id, canonical_url, source_url, lastmod_hint) "
                "SELECT %s, entry.canonical_url, entry.source_url, entry.lastmod_hint "
                "FROM unnest(%s::text[], %s::text[], %s::text[]) "
                "AS entry(canonical_url, source_url, lastmod_hint) "
                "ON CONFLICT DO NOTHING",
                [
                    observation.id,
                    [entry.canonical_url for entry in entries],
                    [entry.source_url for entry in entries],
                    [entry.lastmod_hint for entry in entries],
                ],
            ),
        ],
        retry_transient_errors=True,
    )


def discover_sitemaps(
    outlet_id: str, start: date, end: date, *, delay_seconds: float = 1.0
) -> tuple[ArchiveSitemapObservation, ...]:
    if delay_seconds < 0:
        raise ValueError("Delay must be non-negative")
    targets = sitemap_targets(outlet_id, start, end)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots_by_host: dict[str, RobotFileParser] = {}
    observations = []
    for index, sitemap_url in enumerate(targets):
        if index:
            time.sleep(delay_seconds)
        parts = urlsplit(sitemap_url)
        host = parts.hostname or ""
        if host not in robots_by_host:
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            robots_content = _fetch(session, robots_url)
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(robots_content.decode("utf-8", errors="replace").splitlines())
            robots_by_host[host] = parser
        if not robots_by_host[host].can_fetch(USER_AGENT, sitemap_url):
            raise ValueError(f"Robots policy disallows sitemap: {sitemap_url}")
        content = _fetch(session, sitemap_url)
        observation = parse_sitemap(outlet_id, sitemap_url, content, datetime.now(UTC))
        record_sitemap(observation)
        observations.append(observation)
    return tuple(observations)


def _fetch(session: requests.Session, url: str) -> bytes:
    with session.get(url, timeout=TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > MAX_RESPONSE_BYTES:
                raise ValueError(f"Response exceeds {MAX_RESPONSE_BYTES} bytes: {url}")
        return bytes(content)
