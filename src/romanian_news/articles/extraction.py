import json
import re
from datetime import UTC
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import HttpUrl

from romanian_news import BUCHAREST, Sha256
from romanian_news.archive.page_metadata import extract_archive_page_metadata
from romanian_news.articles.models import ExtractedArticle
from romanian_news.feeds.models import FeedEntry, FeedSpec
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256

_TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "gjlxzggqhj",
        "mc_cid",
        "mc_eid",
        "utm_campaign",
        "utm_content",
        "utm_medium",
        "utm_source",
        "utm_term",
    }
)


def normalize_article_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    """Normalize a publisher URL without merging distinct publisher resources."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Article URL must be HTTP or HTTPS: {url}")
    host = parts.hostname.lower().removeprefix("www.")
    normalized_hosts = {value.lower().removeprefix("www.") for value in allowed_hosts}
    if host not in normalized_hosts:
        raise ValueError(f"Article URL host is not registered: {host}")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMETERS
        )
    )
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, query, ""))


def article_id(outlet_id: str, canonical_url: str) -> Sha256:
    """Identify one publisher article across feed and tracking URL variants."""
    return _sha256(f"{outlet_id}\0{canonical_url}".encode())


def extract_article(
    entry: FeedEntry,
    feed: FeedSpec,
    *,
    html: bytes | None,
    final_url: str | None = None,
) -> ExtractedArticle:
    """Parse one feed entry and optional page capture into stable article content."""
    import trafilatura

    canonical_url = normalize_article_url(final_url or str(entry.url), feed.article_hosts)
    body = ""
    body_source = "feed_summary"
    extracted_title = entry.title
    extracted_author = entry.author
    if html:
        parsed, body_source = _extract_page(html, feed)
        if parsed:
            body = parsed.get("text") or ""
            extracted_title = parsed.get("title") or extracted_title
            extracted_author = parsed.get("author") or extracted_author
            candidate_url = parsed.get("url")
            if candidate_url:
                canonical_url = normalize_article_url(candidate_url, feed.article_hosts)
    if not body and entry.feed_content:
        body = (
            trafilatura.extract(
                entry.feed_content,
                include_comments=False,
                include_tables=True,
            )
            or ""
        )
        body_source = "feed_content"
    if not body:
        body = _plain_text(entry.summary)
        body_source = "feed_summary"
    body = _normalize_text(body)
    title = _normalize_text(extracted_title)
    if not body:
        body = title
        body_source = "title"
    if len(title) + len(body) < 80:
        raise ValueError(f"Article content is too short: {entry.url}")
    canonical_author = _normalize_text(extracted_author) if extracted_author else None
    material = _canonical_json(
        {
            "body": body,
            "title": title,
            "version": "article-material-v1",
        }
    )
    extraction = _canonical_json(
        {
            "article_xpath": feed.article_xpath,
            "body": body,
            "body_source": body_source,
            "extractor": "trafilatura-2.2.0",
            "title": title,
        }
    )
    return ExtractedArticle(
        article_id=article_id(feed.outlet_id, canonical_url),
        outlet_id=feed.outlet_id,
        canonical_url=HttpUrl(canonical_url),
        title=title,
        body=body,
        author=canonical_author,
        published_at=entry.published_at.astimezone(UTC),
        source_updated_at=(
            entry.source_updated_at.astimezone(UTC) if entry.source_updated_at is not None else None
        ),
        bucharest_day=entry.published_at.astimezone(BUCHAREST).date(),
        material_digest=_sha256(material),
        extraction_digest=_sha256(extraction),
    )


def extract_archive_article(
    html: bytes,
    final_url: str,
    feed: FeedSpec,
) -> ExtractedArticle:
    """Extract page content using the publisher's verified original date."""
    metadata = extract_archive_page_metadata(html)
    if metadata.rejection or metadata.published_at is None or not metadata.title:
        raise ValueError("Archive article requires a verified publication date and title")
    canonical_url = normalize_article_url(final_url, feed.article_hosts)
    parsed, body_source = _extract_page(html, feed)
    if (not parsed or not parsed.get("text")) and feed.article_xpath:
        parsed, body_source = _extract_page(html, feed.model_copy(update={"article_xpath": None}))
    if not parsed or not parsed.get("text"):
        raise ValueError(f"Archive page has no extractable article text: {final_url}")
    candidate_url = parsed.get("url")
    if candidate_url:
        canonical_url = normalize_article_url(candidate_url, feed.article_hosts)
    title = _normalize_text(metadata.title)
    body = _normalize_text(parsed["text"])
    if len(title) + len(body) < 80:
        raise ValueError(f"Archive article content is too short: {final_url}")
    author = _normalize_text(parsed["author"]) if parsed.get("author") else None
    material = _canonical_json({"body": body, "title": title, "version": "article-material-v1"})
    extraction = _canonical_json(
        {
            "article_xpath": feed.article_xpath,
            "body": body,
            "body_source": body_source,
            "extractor": "trafilatura-2.2.0",
            "title": title,
        }
    )
    published_at = metadata.published_at.astimezone(UTC)
    return ExtractedArticle(
        article_id=article_id(feed.outlet_id, canonical_url),
        outlet_id=feed.outlet_id,
        canonical_url=HttpUrl(canonical_url),
        title=title,
        body=body,
        author=author,
        published_at=published_at,
        source_updated_at=metadata.modified_at.astimezone(UTC) if metadata.modified_at else None,
        bucharest_day=published_at.astimezone(BUCHAREST).date(),
        material_digest=_sha256(material),
        extraction_digest=_sha256(extraction),
    )


def _extract_page(html: bytes, feed: FeedSpec) -> tuple[dict[str, str] | None, str]:
    import trafilatura

    extraction_html: bytes | None = html
    body_source = "page"
    if feed.article_xpath:
        from lxml import html as lxml_html

        root = lxml_html.fromstring(trafilatura.utils.decode_file(html))
        matches = root.xpath(feed.article_xpath)
        serialized = lxml_html.tostring(matches[0], encoding="utf-8") if matches else None
        extraction_html = serialized.encode("utf-8") if isinstance(serialized, str) else serialized
        body_source = "page_selector"
    if not extraction_html:
        return None, body_source
    raw = trafilatura.extract(
        extraction_html,
        output_format="json",
        include_comments=False,
        include_tables=True,
        with_metadata=True,
    )
    return (json.loads(raw) if raw else None), body_source


def _plain_text(value: str) -> str:
    import trafilatura

    return trafilatura.extract(value) or re.sub(r"<[^>]+>", " ", value)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00ad", "").replace("\u00a0", " ")).strip()
