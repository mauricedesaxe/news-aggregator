"""Manual production job for bounded historical article capture."""

from datetime import date

import dagster as dg

from romanian_news.archive.capture_batch import capture_archive_batch
from romanian_news.config import IMPLEMENTATION_REF


class ArchiveCaptureConfig(dg.Config):
    outlet: str
    start: str
    end: str
    limit: int = 10


@dg.op
def archive_article_capture(context: dg.OpExecutionContext, config: ArchiveCaptureConfig) -> None:
    result = capture_archive_batch(
        config.outlet,
        date.fromisoformat(config.start),
        date.fromisoformat(config.end),
        limit=config.limit,
        implementation_ref=IMPLEMENTATION_REF,
    )
    context.log.info(
        "Archive article capture: selected=%s published=%s unchanged=%s failed=%s",
        result.selected,
        result.published,
        result.unchanged,
        len(result.failed),
    )
    for url, reason in result.failed:
        context.log.warning("Archive article skipped: %s (%s)", url, reason)
    if result.failed:
        raise RuntimeError(f"Archive capture failed for {len(result.failed)} selected pages")


@dg.job(tags={"dagster/max_runtime": "1800", "dagster/max_retries": "0"})
def archive_article_capture_batch() -> None:
    archive_article_capture()
