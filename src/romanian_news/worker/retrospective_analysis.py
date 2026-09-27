"""Run one bounded historical analysis pilot and publish its sourced report."""

from datetime import date

import dagster as dg

from romanian_news.archive.backfill import ARCHIVE_END, ARCHIVE_START
from romanian_news.catalog.archive_report_coverage import read_retrospective_coverage
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.daily import read_daily_article_references
from romanian_news.retrospective_report import publish_retrospective_daily_report
from romanian_news.worker.operations import (
    materialize_clusters,
    materialize_daily_themes,
    materialize_embeddings,
    materialize_group_sentiment,
    materialize_group_summaries,
    materialize_relevance,
    materialize_subject_assessments,
)

_PILOT_ARTICLE_LIMIT = 60


class RetrospectiveAnalysisConfig(dg.Config):
    day: str


def analyze_retrospective_day(day: date, implementation_ref: str) -> str:
    if not ARCHIVE_START <= day <= ARCHIVE_END:
        raise ValueError("Retrospective analysis day is outside the one-year archive")
    coverage = read_retrospective_coverage(day)
    if coverage is None or len(coverage.included_outlets) < 2:
        raise ValueError("Retrospective pilot needs captured articles from two outlets")
    article_count = len(read_daily_article_references(day).values)
    if not 1 <= article_count <= _PILOT_ARTICLE_LIMIT:
        raise ValueError(
            f"Retrospective pilot requires 1-{_PILOT_ARTICLE_LIMIT} published articles; "
            f"found {article_count}"
        )
    if article_count > coverage.captured_article_count:
        raise ValueError("Published articles exceed recorded archive captures")
    for stage in (
        materialize_relevance,
        materialize_embeddings,
        materialize_clusters,
        materialize_group_summaries,
        materialize_group_sentiment,
        materialize_daily_themes,
        materialize_subject_assessments,
    ):
        stage(day, implementation_ref)
    return publish_retrospective_daily_report(day, implementation_ref).version_id


@dg.op
def retrospective_analysis(
    context: dg.OpExecutionContext, config: RetrospectiveAnalysisConfig
) -> None:
    day = date.fromisoformat(config.day)
    version_id = analyze_retrospective_day(day, IMPLEMENTATION_REF)
    context.log.info("Retrospective report for %s: version=%s", day, version_id)


@dg.job(tags={"dagster/max_runtime": "7200", "dagster/max_retries": "0"})
def retrospective_analysis_pilot() -> None:
    retrospective_analysis()
