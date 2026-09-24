from romanian_news.catalog.report_inputs import read_current_daily_report_record
from romanian_news.reports import DailyReport
from romanian_news.storage import read_verified_r2_object
from romanian_news.video_digest.models import (
    EditionIdentity,
    ScheduledSlot,
    SlotSkipReason,
    edition_id,
)
from romanian_news.video_digest.orchestration import (
    SourceReady,
    SourceUnavailable,
    VideoDigestRunRequest,
)
from romanian_news.video_digest.preflight import (
    PRODUCTION_POLICY,
    PlanningReport,
    publish_policy,
)


def resolve_video_digest_run_request(
    slot: ScheduledSlot, *, owner_token: str
) -> VideoDigestRunRequest:
    """Check the current report once when a Dagster slot run starts."""
    record = read_current_daily_report_record(slot.bucharest_day)
    if record is None:
        source = SourceUnavailable(reason=SlotSkipReason.SOURCE_MISSING)
    else:
        content = read_verified_r2_object(record.r2_key, record.content_digest)
        report = DailyReport.model_validate_json(content, strict=True)
        if report.day != slot.bucharest_day:
            raise ValueError("Video digest report day does not match the scheduled slot")
        PlanningReport(version_id=record.version_id, report=report)
        if not any(section.tier == "main" for section in report.sections):
            source = SourceUnavailable(reason=SlotSkipReason.SOURCE_EMPTY)
        else:
            policy_version = publish_policy(PRODUCTION_POLICY)
            source = SourceReady(
                edition=EditionIdentity(
                    edition_id=edition_id(record.version_id, policy_version),
                    daily_report_version_id=record.version_id,
                    policy_bundle_version_id=policy_version,
                )
            )
    return VideoDigestRunRequest(slot=slot, owner_token=owner_token, source=source)
