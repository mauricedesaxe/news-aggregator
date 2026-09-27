"""Build a dated article capture from one publisher page without retaining its HTML."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field, HttpUrl, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.archive.page_metadata import extract_archive_page_metadata
from romanian_news.articles.extraction import extract_archive_article, normalize_article_url
from romanian_news.articles.models import ExtractedArticle
from romanian_news.feeds.models import FeedSpec
from romanian_news.identity import canonical_json, sha256


class ArchiveArticleCapture(NewsModel):
    capture_id: Sha256
    observation_id: Sha256
    discovered_url: HttpUrl
    final_url: HttpUrl
    page_sha256: Sha256
    fetched_at: datetime
    publication_evidence: Annotated[str, Field(min_length=1)]
    article: ExtractedArticle

    @model_validator(mode="after")
    def validate_capture(self) -> ArchiveArticleCapture:
        if self.fetched_at.tzinfo is None:
            raise ValueError("Archive fetch time must include a UTC offset")
        if self.fetched_at < self.article.published_at:
            raise ValueError("Archive fetch predates article publication")
        return self


def capture_archive_article(
    *,
    observation_id: Sha256,
    discovered_url: str,
    final_url: str,
    html: bytes,
    fetched_at: datetime,
    feed: FeedSpec,
) -> ArchiveArticleCapture:
    """Extract article content and provenance from the same fetched bytes."""
    if fetched_at.tzinfo is None:
        raise ValueError("Archive fetch time must include a UTC offset")
    metadata = extract_archive_page_metadata(html)
    if metadata.rejection or metadata.publication_evidence is None:
        raise ValueError(f"Archive page date rejected: {metadata.rejection}")
    article = extract_archive_article(html, final_url, feed)
    normalized_discovered = normalize_article_url(discovered_url, feed.article_hosts)
    normalized_final = normalize_article_url(final_url, feed.article_hosts)
    digest = hashlib.sha256(html).hexdigest()
    timestamp = fetched_at.astimezone(UTC).isoformat()
    identity = sha256(
        canonical_json(
            {
                "observation_id": observation_id,
                "discovered_url": normalized_discovered,
                "page_sha256": digest,
                "fetched_at": timestamp,
            }
        )
    )
    return ArchiveArticleCapture(
        capture_id=identity,
        observation_id=observation_id,
        discovered_url=HttpUrl(normalized_discovered),
        final_url=HttpUrl(normalized_final),
        page_sha256=digest,
        fetched_at=fetched_at.astimezone(UTC),
        publication_evidence=metadata.publication_evidence,
        article=article,
    )
