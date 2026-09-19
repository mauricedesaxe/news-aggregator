from __future__ import annotations

import calendar
import time
from datetime import UTC, datetime
from typing import Any

import feedparser

from romanian_news.feeds.models import FeedEntry, FeedSpec


def parse_feed(feed: FeedSpec, content: bytes) -> tuple[tuple[FeedEntry, ...], int]:
    """Parse valid dated feed entries and report malformed entries separately."""
    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"Feed XML is invalid: {parsed.bozo_exception}")
    entries = []
    rejected = 0
    for value in parsed.entries:
        try:
            entries.append(_parse_entry(feed, value))
        except (KeyError, TypeError, ValueError):
            rejected += 1
    return tuple(entries), rejected


def _parse_entry(feed: FeedSpec, value: Any) -> FeedEntry:
    published = value.get("published_parsed") or value.get("updated_parsed")
    if published is None:
        raise ValueError("Feed entry has no publication time")
    updated = value.get("updated_parsed")
    url = value.get("link")
    title = value.get("title")
    if not url or not title:
        raise ValueError("Feed entry has no URL or title")
    content_values = value.get("content") or ()
    feed_content = content_values[0].get("value") if content_values else None
    return FeedEntry(
        feed_id=feed.id,
        source_id=str(value.get("id") or value.get("guid") or url),
        url=url,
        title=str(title),
        summary=str(value.get("summary") or value.get("description") or ""),
        feed_content=str(feed_content) if feed_content else None,
        published_at=_feed_datetime(published),
        source_updated_at=_feed_datetime(updated) if updated else None,
        author=str(value.get("author")) if value.get("author") else None,
    )


def _feed_datetime(value: time.struct_time) -> datetime:
    return datetime.fromtimestamp(calendar.timegm(value), tz=UTC)
