"""Resolve original publication evidence from a publisher article page."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from lxml import html as lxml_html
from lxml.html import HtmlElement

from romanian_news import BUCHAREST


@dataclass(frozen=True)
class ArchivePageMetadata:
    title: str | None
    published_at: datetime | None
    modified_at: datetime | None
    publication_evidence: str | None
    rejection: str | None


def _parse_timestamp(value: str) -> datetime | None:
    if "T" not in value and " " not in value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=BUCHAREST)).astimezone(UTC)


def _article_objects(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for child in value for item in _article_objects(child)]
    if not isinstance(value, dict):
        return []
    kind = value.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    found = (
        [value] if any(item in {"Article", "NewsArticle", "BlogPosting"} for item in kinds) else []
    )
    for key in ("@graph", "mainEntity"):
        found.extend(_article_objects(value.get(key)))
    return found


def _meta_dates(root: HtmlElement) -> tuple[list[tuple[str, datetime]], list[datetime]]:
    published: list[tuple[str, datetime]] = []
    modified: list[datetime] = []
    for meta in root.xpath("//meta[@content]"):
        key = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").lower()
        raw = meta.get("content") or ""
        parsed = _parse_timestamp(raw)
        if parsed is None:
            continue
        if key in {"article:published_time", "datepublished", "pubdate", "publish_date"}:
            published.append((key, parsed))
        elif key in {"article:modified_time", "datemodified", "lastmod"}:
            modified.append(parsed)
    return published, modified


def _jsonld_dates(root: HtmlElement) -> tuple[list[tuple[str, datetime]], list[datetime]]:
    published: list[tuple[str, datetime]] = []
    modified: list[datetime] = []
    for node in root.xpath("//script[@type='application/ld+json']/text()"):
        try:
            value = json.loads(node)
        except json.JSONDecodeError:
            continue
        for item in _article_objects(value):
            publication = item.get("datePublished")
            if isinstance(publication, str) and (parsed := _parse_timestamp(publication)):
                published.append(("schema:datePublished", parsed))
            modification = item.get("dateModified")
            if isinstance(modification, str) and (parsed := _parse_timestamp(modification)):
                modified.append(parsed)
    return published, modified


def extract_archive_page_metadata(content: bytes) -> ArchivePageMetadata:
    root = lxml_html.fromstring(content)
    title = root.xpath("string(//meta[@property='og:title']/@content)").strip()
    if not title:
        title = root.xpath("string(//h1[1])").strip()
    meta_published, meta_modified = _meta_dates(root)
    json_published, json_modified = _jsonld_dates(root)
    published = meta_published + json_published
    modified = meta_modified + json_modified
    days = {timestamp.astimezone(BUCHAREST).date() for _, timestamp in published}
    rejection = None
    if not published:
        rejection = "missing_publication_date"
    elif len(days) > 1:
        rejection = "conflicting_publication_dates"
    elif not title:
        rejection = "missing_title"
    return ArchivePageMetadata(
        title=title or None,
        published_at=published[0][1] if published and rejection is None else None,
        modified_at=max(modified, default=None),
        publication_evidence=published[0][0] if published and rejection is None else None,
        rejection=rejection,
    )
