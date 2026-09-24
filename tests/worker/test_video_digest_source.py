from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from romanian_news import BUCHAREST
from romanian_news.catalog.artifacts import artifact_file, canonical_json, sha256
from romanian_news.catalog.report_inputs import CurrentDailyReportRecord
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    DailyReportV2,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.video_digest.models import (
    ScheduledSlot,
    SlotName,
    SlotSkipReason,
    edition_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import SourceReady, SourceUnavailable
from romanian_news.video_digest.preflight import PRODUCTION_POLICY
from romanian_news.worker import video_digest_source as source_module

NOW = datetime(2026, 9, 25, 5, tzinfo=UTC)
SLOT = ScheduledSlot(
    slot_id=scheduled_slot_id(SlotName.MORNING, NOW),
    name=SlotName.MORNING,
    scheduled_at=NOW,
    bucharest_day=NOW.astimezone(BUCHAREST).date(),
)


def _report(*, day: date = SLOT.bucharest_day, main: bool = True) -> DailyReport:
    article_id = "1" * 64
    sections = (
        DailyReportSection(
            theme_id="2" * 64,
            title="News subject",
            summary="A Romanian consequence",
            events=(
                ReportEvent(
                    group_id="3" * 64,
                    title_ro="Titlu",
                    summary_ro="Rezumat",
                    key_points_ro=(),
                    disagreements_ro=(),
                    sentiment_label="neutral",
                    sentiment_score=0,
                    sentiment_rationale_ro="Neutru",
                    articles=(
                        ReportArticle(
                            article_version_id=article_id,
                            outlet_id="example",
                            title="Source article",
                            canonical_url="https://example.com/article",
                            sentiment_label="neutral",
                            sentiment_score=0,
                        ),
                    ),
                ),
            ),
            tier="main" if main else "worth_knowing",
            semantic_rank=1,
            consequence_rationale="Material consequence",
            citations=(
                ReportSubjectCitation(
                    article_version_id=article_id,
                    evidence_quote="Source evidence",
                ),
            ),
        ),
    )
    return DailyReport(
        day=day,
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=sections,
    )


def _record(report: DailyReport | DailyReportV2) -> tuple[CurrentDailyReportRecord, bytes]:
    content = canonical_json(report.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"news:daily:{report.day.isoformat()}",
        artifact_kind="news_daily_report",
        title=f"Romanian news report for {report.day.isoformat()}",
        content=content,
        r2_key=f"news/reports/daily/{report.day.isoformat()}/{sha256(content)}.json",
        media_type="application/json",
    )
    return (
        CurrentDailyReportRecord(
            day=report.day,
            version_id=file.version_id,
            run_id="4" * 64,
            content_digest=file.content_digest,
            r2_key=file.r2_key,
            input_time=NOW,
        ),
        content,
    )


def _install_report(
    monkeypatch: pytest.MonkeyPatch,
    report: DailyReport | DailyReportV2,
    *,
    version_id: str | None = None,
) -> None:
    record, content = _record(report)
    if version_id is not None:
        record = record.model_copy(update={"version_id": version_id})
    monkeypatch.setattr(source_module, "read_current_daily_report_record", lambda _day: record)
    monkeypatch.setattr(
        source_module,
        "read_verified_r2_object",
        lambda key, digest: content
        if (key, digest) == (record.r2_key, record.content_digest)
        else b"invalid",
    )


def test_missing_report_skips_without_touching_storage_or_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(source_module, "read_current_daily_report_record", lambda _day: None)
    monkeypatch.setattr(
        source_module,
        "read_verified_r2_object",
        lambda *_args: pytest.fail("missing source read storage"),
    )
    monkeypatch.setattr(
        source_module,
        "publish_policy",
        lambda *_args: pytest.fail("missing source published policy"),
    )

    request = source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")

    assert request.slot == SLOT
    assert request.owner_token == "run-id"
    assert request.source == SourceUnavailable(reason=SlotSkipReason.SOURCE_MISSING)


def test_report_without_main_subjects_skips_without_policy_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_report(monkeypatch, _report(main=False))
    monkeypatch.setattr(
        source_module,
        "publish_policy",
        lambda *_args: pytest.fail("empty source published policy"),
    )

    request = source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")

    assert request.source == SourceUnavailable(reason=SlotSkipReason.SOURCE_EMPTY)


def test_main_subject_uses_exact_report_and_registered_planning_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    record, _content = _record(report)
    _install_report(monkeypatch, report)
    published = []

    def publish(policy: object) -> str:
        published.append(policy)
        return PRODUCTION_POLICY.artifact.version_id

    monkeypatch.setattr(source_module, "publish_policy", publish)

    request = source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")

    assert isinstance(request.source, SourceReady)
    assert request.source.edition.daily_report_version_id == record.version_id
    assert request.source.edition.policy_bundle_version_id == PRODUCTION_POLICY.artifact.version_id
    assert request.source.edition.edition_id == edition_id(
        record.version_id, PRODUCTION_POLICY.artifact.version_id
    )
    assert published == [PRODUCTION_POLICY]


def test_older_report_schema_fails_visibly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_report(
        monkeypatch,
        DailyReportV2(
            day=SLOT.bucharest_day,
            accepted_article_count=0,
            theme_count=0,
            group_count=0,
            sections=(),
        ),
    )

    with pytest.raises(ValidationError):
        source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")


def test_report_version_mismatch_fails_visibly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_report(monkeypatch, _report(), version_id="f" * 64)

    with pytest.raises(ValidationError, match="Planning report version"):
        source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")


def test_wrong_report_day_fails_visibly(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_report(monkeypatch, _report(day=date(2026, 9, 24)))

    with pytest.raises(ValueError, match="report day"):
        source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")
