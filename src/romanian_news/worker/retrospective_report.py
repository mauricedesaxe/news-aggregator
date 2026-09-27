"""Manual production job for publishing one ready historical daily report."""

from datetime import date

import dagster as dg

from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.retrospective_report import publish_retrospective_daily_report


class RetrospectiveReportConfig(dg.Config):
    day: str


@dg.op
def retrospective_daily_report(
    context: dg.OpExecutionContext, config: RetrospectiveReportConfig
) -> None:
    day = date.fromisoformat(config.day)
    publication = publish_retrospective_daily_report(day, IMPLEMENTATION_REF)
    context.log.info(
        "Retrospective report for %s: status=%s version=%s",
        day,
        publication.status,
        publication.version_id,
    )


@dg.job(tags={"dagster/max_runtime": "1800", "dagster/max_retries": "0"})
def retrospective_daily_report_job() -> None:
    retrospective_daily_report()
