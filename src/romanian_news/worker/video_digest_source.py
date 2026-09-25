from datetime import UTC, datetime

from romanian_news.catalog.video_digest import schedule_slot
from romanian_news.catalog.video_digest_selection import (
    capture_slot_report_source,
    has_unselected_earlier_edition,
    read_earlier_slot_selections,
    read_slot_subject_selection,
    record_slot_subject_selection,
)
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
    _call_model,
    _openrouter_completion,
    _provider_request,
    publish_policy,
)
from romanian_news.video_digest.selection import (
    MaterialChangeJudgment,
    material_change_request_id,
    select_slot_subjects,
)


def resolve_video_digest_run_request(
    slot: ScheduledSlot, *, owner_token: str
) -> VideoDigestRunRequest:
    """Freeze one report head and one ordered subject decision per scheduled slot."""
    schedule_slot(slot, recorded_at=datetime.now(UTC))
    record = capture_slot_report_source(slot)
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
            selection = read_slot_subject_selection(slot.slot_id)
            if selection is None:
                if has_unselected_earlier_edition(slot):
                    return VideoDigestRunRequest(
                        slot=slot,
                        owner_token=owner_token,
                        source=SourceUnavailable(reason=SlotSkipReason.UNCHANGED),
                    )
                selection = select_slot_subjects(
                    report,
                    record.version_id,
                    read_earlier_slot_selections(slot),
                    _adjudicate_material_change,
                )
                selection = record_slot_subject_selection(slot, selection)
            if selection.report_version_id != record.version_id:
                raise ValueError("Stored selection differs from frozen report version")
            selection.validate_report(report)
            if not selection.selected_sections:
                source = SourceUnavailable(reason=SlotSkipReason.UNCHANGED)
                return VideoDigestRunRequest(slot=slot, owner_token=owner_token, source=source)
            policy_version = publish_policy(PRODUCTION_POLICY)
            source = SourceReady(
                edition=EditionIdentity(
                    edition_id=edition_id(record.version_id, policy_version, selection.digest),
                    daily_report_version_id=record.version_id,
                    policy_bundle_version_id=policy_version,
                    selection_digest=selection.digest,
                )
            )
    return VideoDigestRunRequest(slot=slot, owner_token=owner_token, source=source)


def _adjudicate_material_change(current, previous) -> MaterialChangeJudgment:
    request_id = material_change_request_id(current, previous)
    request = _provider_request(
        PRODUCTION_POLICY.definition.policy.verification_model,
        "Decide if this report section has a new material fact or consequence compared with "
        "the earlier covered sections. A material decision requires one exact evidence_quote "
        "and article_version_id from the current citations that supports the new fact. "
        "If the cited evidence was already covered, or you cannot establish novelty, "
        "return unchanged or uncertain. Do not infer facts from a changed title alone.",
        {
            "current": current.model_dump(mode="json"),
            "previous": [item.model_dump(mode="json") for item in previous],
        },
        MaterialChangeJudgment.model_json_schema(),
        "romanian_news_video_digest_subject_change",
    )
    parsed, _response, _error = _call_model(
        "video_digest_subject_change",
        request_id,
        request,
        0,
        _openrouter_completion,
        MaterialChangeJudgment,
    )
    return parsed or MaterialChangeJudgment(
        disposition="uncertain",
        citation_article_version_id=None,
        evidence_quote=None,
        new_fact_or_consequence=None,
    )
