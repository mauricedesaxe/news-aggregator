from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from romanian_news import Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.archive_progress import (
    ArchiveDailyReport,
    ArchiveDiscoveryMonth,
    list_archive_daily_reports,
    list_archive_discovery_months,
)
from romanian_news.catalog.research_triggers import read_daily_research_trigger_set
from romanian_news.catalog.weekly_status import (
    WeeklyStatusSummary,
    list_weekly_status,
    read_weekly_status,
    read_weekly_status_version,
)
from romanian_news.catalog_transport import catalog_query
from romanian_news.current_report import (
    CurrentDailyReport,
    DailyReportFreshness,
    read_current_daily_report,
    read_daily_report_freshness,
)
from romanian_news.feedback import (
    DailyReportSummary,
    NewsFeedbackCommand,
    NewsFeedbackEvent,
    list_daily_reports,
    read_daily_report_version,
    read_latest_news_feedback,
    resolve_current_daily_report_version,
    submit_news_feedback,
)
from romanian_news.reader.dagster_repair import (
    RepairRun,
    read_daily_report_repair,
    request_daily_report_repair,
)
from romanian_news.reports import (
    DailyReportDocument,
    parse_daily_report,
)
from romanian_news.research_triggers import DailyResearchTriggerSet
from romanian_news.storage import (
    check_r2_access,
    read_verified_r2_object,
)
from romanian_news.video_digest.models import EditionId
from romanian_news.video_digest.reader import (
    ReaderVideoDigest,
    read_reader_video_digest,
)
from romanian_news.video_digest_feedback import (
    VideoDigestFeedbackCommand,
    VideoDigestFeedbackEvent,
    submit_video_digest_feedback,
)
from romanian_news.weekly_status import WeeklyStatusRead


def _read_report_reference(reference: ArtifactReference) -> DailyReportDocument:
    return _read_report_content(reference.r2_key, reference.content_digest)


@lru_cache(maxsize=64)
def _read_report_content(r2_key: str, content_digest: Sha256) -> DailyReportDocument:
    return parse_daily_report(read_verified_r2_object(r2_key, content_digest))


@dataclass(frozen=True)
class ReaderDomain:
    list_reports: Callable[[int], tuple[DailyReportSummary, ...]]
    resolve_current_report_version: Callable[[Sha256], Sha256]
    read_report: Callable[[Sha256], DailyReportDocument]
    read_current_report: Callable[[date], CurrentDailyReport | None]
    read_feedback: Callable[[Sha256], tuple[NewsFeedbackEvent, ...]]
    submit_feedback: Callable[[NewsFeedbackCommand], NewsFeedbackEvent]
    read_research_flags: Callable[[Sha256], DailyResearchTriggerSet | None]
    read_freshness: Callable[[date], DailyReportFreshness] = read_daily_report_freshness
    request_repair: Callable[[date], RepairRun] = request_daily_report_repair
    read_repair: Callable[[str], RepairRun] = read_daily_report_repair
    check_readiness: Callable[[], None] = lambda: _check_storage_readiness()
    read_video_digest: Callable[[date, Sha256, EditionId | None, str], ReaderVideoDigest | None] = (
        lambda _day, _report, _edition, _origin: None
    )
    list_report_archive: Callable[[int, int], tuple[DailyReportSummary, ...]] = (
        lambda _limit, _offset: ()
    )
    list_status: Callable[[int, int], tuple[WeeklyStatusSummary, ...]] = lambda _limit, _offset: ()
    list_archive_discovery: Callable[[], tuple[ArchiveDiscoveryMonth, ...]] = lambda: ()
    list_archive_reports: Callable[[date, date], tuple[ArchiveDailyReport, ...]] = (
        lambda _start, _end: ()
    )
    read_status: Callable[[date], tuple[Sha256, WeeklyStatusRead]] | None = None
    read_status_version: Callable[[Sha256], tuple[Sha256, WeeklyStatusRead]] | None = None
    read_report_reference: Callable[[ArtifactReference], DailyReportDocument] = (
        _read_report_reference
    )
    submit_video_feedback: Callable[[VideoDigestFeedbackCommand], VideoDigestFeedbackEvent] = (
        lambda _command: _written_only_video_feedback()
    )


def _written_only_video_feedback() -> VideoDigestFeedbackEvent:
    raise ValueError("Video feedback is unavailable in written-only mode")


PRODUCTION_DOMAIN = ReaderDomain(
    list_reports=list_daily_reports,
    list_report_archive=list_daily_reports,
    list_status=list_weekly_status,
    list_archive_discovery=list_archive_discovery_months,
    list_archive_reports=list_archive_daily_reports,
    read_status=read_weekly_status,
    read_status_version=read_weekly_status_version,
    resolve_current_report_version=resolve_current_daily_report_version,
    read_report=read_daily_report_version,
    read_current_report=read_current_daily_report,
    read_freshness=read_daily_report_freshness,
    request_repair=request_daily_report_repair,
    read_repair=read_daily_report_repair,
    read_feedback=read_latest_news_feedback,
    submit_feedback=submit_news_feedback,
    read_research_flags=read_daily_research_trigger_set,
    read_video_digest=read_reader_video_digest,
    submit_video_feedback=submit_video_digest_feedback,
)


def _check_storage_readiness() -> None:
    catalog_query("SELECT 1 AS ready")
    check_r2_access()
