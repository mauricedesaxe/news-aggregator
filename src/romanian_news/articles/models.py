from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, HttpUrl

from romanian_news import NewsModel, OutletId, Sha256
from romanian_news.feeds.models import CatalogedFeedEntry, CatalogedFeedEntryReference


class ExtractedArticle(NewsModel):
    article_id: Sha256
    outlet_id: OutletId
    canonical_url: HttpUrl
    title: Annotated[str, Field(min_length=1)]
    body: Annotated[str, Field(min_length=1)]
    author: str | None
    published_at: datetime
    source_updated_at: datetime | None
    bucharest_day: date
    material_digest: Sha256
    extraction_digest: Sha256


class ArticleWorkLane(StrEnum):
    LIVE_UNSEEN = "live_unseen"
    HISTORICAL_UNSEEN = "historical_unseen"
    SOURCE_UPDATE = "source_update"
    REVALIDATION = "revalidation"


class ArticleWorkItem(NewsModel):
    source: CatalogedFeedEntryReference
    lane: ArticleWorkLane
    last_captured_at: datetime | None = None


class MaterializedArticleWorkItem(NewsModel):
    source: CatalogedFeedEntry
    lane: ArticleWorkLane
    last_captured_at: datetime | None = None


class ArticleWorkPlan(NewsModel):
    selected: tuple[ArticleWorkItem, ...]
    remaining_entries: Annotated[int, Field(ge=0)]
    remaining_days: tuple[date, ...]
    complete_entries: Annotated[int, Field(ge=0)]
    deferred_event_ids: tuple[Sha256, ...] = ()
    quarantined_event_ids: tuple[Sha256, ...] = ()
    source_covered_days: tuple[date, ...] = ()


class ArticleWorkStatus(NewsModel):
    retryable_entries: Annotated[int, Field(ge=0)]
    deferred_event_ids: tuple[Sha256, ...]
    quarantined_event_ids: tuple[Sha256, ...]
    source_covered_days: tuple[date, ...]


class ArticleFailureKind(StrEnum):
    DETERMINISTIC = "deterministic"
    INFRASTRUCTURE = "infrastructure"


class ArticleAcquisitionFailure(NewsModel):
    event_id: Sha256
    kind: ArticleFailureKind
    fingerprint: Sha256
    message: Annotated[str, Field(min_length=1)]


class ArticleCapture(NewsModel):
    article: ExtractedArticle
    source: CatalogedFeedEntry
    captured_at: datetime
    page_url: str | None
    page_content: bytes | None
    page_content_digest: Sha256 | None
    latency_ms: Annotated[int, Field(ge=0)] | None
    retrieval_error: str | None


class ArticleAcquisitionResult(NewsModel):
    captures: tuple[ArticleCapture, ...]
    skipped_entries: Annotated[int, Field(ge=0)]
    remaining_entries: Annotated[int, Field(ge=0)] = 0
    errors: tuple[str, ...] = ()
    remaining_days: tuple[date, ...] = ()
    source_covered_days: tuple[date, ...] = ()


class ArticleBatchCapture(NewsModel):
    capture: ArticleCapture


class ArticleBatchSkip(NewsModel):
    event_id: Sha256


ArticleBatchItemResult = ArticleBatchCapture | ArticleBatchSkip | ArticleAcquisitionFailure
