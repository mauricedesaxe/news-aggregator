"""Measure historical sitemap and article extraction without publishing data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import requests
from lxml import html as lxml_html

from romanian_news import BUCHAREST
from romanian_news.archive.page_metadata import extract_archive_page_metadata
from romanian_news.articles.extraction import extract_archive_article
from romanian_news.feeds.models import FeedSpec
from romanian_news.feeds.registry import feed_registry

USER_AGENT = (
    "Romanian news archive research/1.0 (+https://github.com/mauricedesaxe/news-aggregator)"
)
TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 5_000_000


def _get(session: requests.Session, url: str) -> tuple[str, bytes]:
    with session.get(url, timeout=TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(65536):
            content.extend(chunk)
            if len(content) > MAX_RESPONSE_BYTES:
                raise ValueError(f"Response exceeds {MAX_RESPONSE_BYTES} bytes: {url}")
        return response.url, bytes(content)


def _robots(session: requests.Session, feed: FeedSpec) -> RobotFileParser:
    parts = urlsplit(str(feed.url))
    url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    _final, content = _get(session, url)
    parser = RobotFileParser()
    parser.set_url(url)
    parser.parse(content.decode("utf-8", errors="replace").splitlines())
    return parser


def _sitemap_urls(content: bytes) -> list[tuple[str, str | None]]:
    root = ElementTree.fromstring(content)
    if root.tag.rsplit("}", 1)[-1] != "urlset":
        raise ValueError("Expected an article URL sitemap")
    values = []
    for item in root:
        loc = item.find("{*}loc")
        modified = item.find("{*}lastmod")
        if loc is not None and loc.text:
            values.append(
                (
                    loc.text.strip(),
                    modified.text.strip() if modified is not None and modified.text else None,
                )
            )
    return values


def _page_section(content: bytes, url: str) -> str:
    root = lxml_html.fromstring(content)
    for key in ("article:section", "parsely-section"):
        value = root.xpath(f"string(//meta[@property='{key}']/@content)").strip()
        if value:
            return value
    path = urlsplit(url).path.strip("/").split("/")
    return path[1] if len(path) > 1 and path[0] == "stiri" else "unclassified"


def _sample(values: list[str], count: int) -> list[str]:
    values = sorted(set(values))
    selected = min(count, len(values))
    return [values[(2 * i + 1) * len(values) // (2 * selected)] for i in range(selected)]


def _day_urls(
    session: requests.Session,
    feed: FeedSpec,
    day: date,
    robots: RobotFileParser,
    sitemap_cache: dict[str, tuple[list[tuple[str, str | None]], int]],
) -> tuple[list[str], int]:
    if feed.id == "hotnews":
        url = f"https://hotnews.ro/sitemap.xml?yyyy={day.year}&mm={day.month:02d}&dd={day.day:02d}"
    elif feed.id == "digi24":
        url = f"https://www.digi24.ro/sitemaps/sitemap-articles-{day.year}-{day.month:02d}.xml"
    else:
        raise ValueError(f"Unsupported archive source: {feed.id}")
    if not robots.can_fetch(USER_AGENT, url):
        raise ValueError(f"Robots policy disallows {url}")
    if url not in sitemap_cache:
        _final, content = _get(session, url)
        sitemap_cache[url] = (_sitemap_urls(content), len(content))
    entries, sitemap_bytes = sitemap_cache[url]
    if feed.id == "digi24":
        entries = [
            (article, modified)
            for article, modified in entries
            if modified and modified[:10] == day.isoformat()
        ]
    return [article for article, _modified in entries], sitemap_bytes


def run(start: date, days: int, sample_per_day: int, delay_seconds: float) -> dict[str, object]:
    feeds = {feed.id: feed for feed in feed_registry().feeds}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, object]] = []
    sitemap_cache: dict[str, tuple[list[tuple[str, str | None]], int]] = {}
    for feed_id in ("hotnews", "digi24"):
        feed = feeds[feed_id]
        robots = _robots(session, feed)
        for offset in range(days):
            day = start + timedelta(days=offset)
            discovered, sitemap_bytes = _day_urls(session, feed, day, robots, sitemap_cache)
            row: dict[str, object] = {
                "outlet": feed_id,
                "day": day.isoformat(),
                "sitemap_candidates": len(discovered),
                "sitemap_bytes": sitemap_bytes,
                "sampled": 0,
                "valid_day": 0,
                "extracted": 0,
                "failures": [],
            }
            for url in _sample(discovered, sample_per_day):
                if not robots.can_fetch(USER_AGENT, url):
                    row["failures"].append({"url": url, "reason": "robots_disallowed"})
                    continue
                time.sleep(delay_seconds)
                row["sampled"] += 1
                try:
                    final_url, content = _get(session, url)
                    final_host = urlsplit(final_url).hostname or ""
                    if final_host.removeprefix("www.") not in feed.article_hosts:
                        raise ValueError("Redirected outside the outlet")
                    metadata = extract_archive_page_metadata(content)
                    if metadata.rejection or metadata.published_at is None:
                        raise ValueError(f"Page date rejected: {metadata.rejection}")
                    published_at = metadata.published_at
                    if published_at.astimezone(BUCHAREST).date() != day:
                        raise ValueError(f"Published on {published_at.date()}, not requested day")
                    row["valid_day"] += 1
                    article = extract_archive_article(content, final_url, feed)
                    if len(article.body) < 200:
                        raise ValueError(f"Extracted body too short: {len(article.body)} chars")
                    row["extracted"] += 1
                    row.setdefault("examples", []).append(
                        {
                            "url": final_url,
                            "published_at": published_at.isoformat(),
                            "updated_at": metadata.modified_at.isoformat()
                            if metadata.modified_at
                            else None,
                            "precision": "exact",
                            "body_chars": len(article.body),
                            "section": _page_section(content, final_url),
                        }
                    )
                except (requests.RequestException, ValueError, ElementTree.ParseError) as error:
                    row["failures"].append({"url": url, "reason": str(error)[:160]})
            rows.append(row)
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for key in ("sitemap_candidates", "sampled", "valid_day", "extracted"):
            totals[str(row["outlet"])][key] += int(row[key])
    sections: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for example in row.get("examples", []):
            sections[str(row["outlet"])][str(example["section"])] += 1
    return {
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(days=days - 1)).isoformat(),
        "sample_per_day": sample_per_day,
        "totals": {outlet: dict(values) for outlet, values in totals.items()},
        "sample_sections": {outlet: dict(values) for outlet, values in sections.items()},
        "days": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 9, 15))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--sample-per-day", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.days <= 7 or not 1 <= args.sample_per_day <= 5 or args.delay_seconds < 0:
        parser.error("days must be 1..7, sample-per-day 1..5, and delay non-negative")
    json.dump(
        run(args.start, args.days, args.sample_per_day, args.delay_seconds),
        sys.stdout,
        indent=2,
        ensure_ascii=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
