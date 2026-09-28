"""Read-only exact-input and spend preview for one historical report day."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from romanian_news.analysis.embeddings import read_pending_embedding_references
from romanian_news.analysis.groups.pending import read_pending_group_analysis_references
from romanian_news.analysis.relevance import read_pending_relevance_references
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.archive.campaign import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.archive_model_spend import ArchiveSpend, read_archive_spend
from romanian_news.catalog.archive_report_coverage import read_retrospective_coverage
from romanian_news.config import ARCHIVE_DAY_SPEND_LIMIT_USD
from romanian_news.daily import read_daily_article_references


@dataclass(frozen=True)
class ReplayPreflight:
    day: date
    outlets: tuple[str, ...]
    verified_pages: int
    captured_articles: int
    article_version_ids: tuple[str, ...]
    pending_relevance: int
    pending_embeddings: int
    pending_summaries: int
    pending_sentiment: int
    estimated_remaining_usd: Decimal
    estimate_basis: str
    spend: ArchiveSpend
    limit_usd: Decimal
    available_usd: Decimal
    source_ready: bool
    paid_work_admissible: bool


def read_replay_preflight(day: date) -> ReplayPreflight:
    if not ARCHIVE_START <= day <= ARCHIVE_END:
        raise ValueError("Replay day is outside the one-year archive")
    coverage = read_retrospective_coverage(day)
    articles = read_daily_article_references(day).values
    relevance = read_pending_relevance_references(
        day=day, request_id_for_article=production_relevance_v3_request_id
    )
    embeddings = read_pending_embedding_references(day=day)
    groups = read_pending_group_analysis_references({day})
    pending_summaries = sum(value.summary_needed for value in groups)
    pending_sentiment = sum(value.sentiment_needed for value in groups)
    pending_calls = len(relevance) + len(embeddings) + pending_summaries + pending_sentiment
    spend = read_archive_spend(day)
    return ReplayPreflight(
        day=day,
        outlets=coverage.included_outlets if coverage else (),
        verified_pages=coverage.verified_page_count if coverage else 0,
        captured_articles=coverage.captured_article_count if coverage else 0,
        article_version_ids=tuple(value.version_id for value in articles),
        pending_relevance=len(relevance),
        pending_embeddings=len(embeddings),
        pending_summaries=pending_summaries,
        pending_sentiment=pending_sentiment,
        estimated_remaining_usd=Decimal(pending_calls) * Decimal("0.0041"),
        estimate_basis="2026 report-linked relevance call proxy; excludes provider errors and future stages",
        spend=spend,
        limit_usd=ARCHIVE_DAY_SPEND_LIMIT_USD,
        available_usd=max(
            Decimal(0), ARCHIVE_DAY_SPEND_LIMIT_USD - spend.spent_usd - spend.held_usd
        ),
        source_ready=bool(coverage and len(coverage.included_outlets) >= 2 and articles),
        paid_work_admissible=spend.spent_usd + spend.held_usd < ARCHIVE_DAY_SPEND_LIMIT_USD,
    )
