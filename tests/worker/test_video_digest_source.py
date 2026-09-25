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
from romanian_news.video_digest.selection import (
    MaterialChangeJudgment,
    SlotSubjectSelection,
    select_slot_subjects,
)
from romanian_news.worker import video_digest_source as source_module

NOW = datetime(2026, 9, 25, 5, tzinfo=UTC)
SLOT = ScheduledSlot(
    slot_id=scheduled_slot_id(SlotName.MORNING, NOW),
    name=SlotName.MORNING,
    scheduled_at=NOW,
    bucharest_day=NOW.astimezone(BUCHAREST).date(),
)


@pytest.fixture(autouse=True)
def _selection_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    selections: dict[str, SlotSubjectSelection] = {}
    monkeypatch.setattr(source_module, "schedule_slot", lambda *_args, **_kwargs: SLOT)
    monkeypatch.setattr(source_module, "has_unselected_earlier_edition", lambda _slot: False)
    monkeypatch.setattr(source_module, "read_earlier_slot_selections", lambda _slot: ())
    monkeypatch.setattr(
        source_module, "read_slot_subject_selection", lambda slot_id: selections.get(slot_id)
    )

    def record(slot: ScheduledSlot, selection: SlotSubjectSelection) -> SlotSubjectSelection:
        return selections.setdefault(slot.slot_id, selection)

    monkeypatch.setattr(source_module, "record_slot_subject_selection", record)


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
    monkeypatch.setattr(source_module, "capture_slot_report_source", lambda _slot: record)
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
    monkeypatch.setattr(source_module, "capture_slot_report_source", lambda _slot: None)
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
        record.version_id,
        PRODUCTION_POLICY.artifact.version_id,
        request.source.edition.selection_digest,
    )
    assert published == [PRODUCTION_POLICY]


def test_retry_reuses_stored_selection_without_rechecking_prior_editions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_report(monkeypatch, _report())
    monkeypatch.setattr(
        source_module, "publish_policy", lambda _policy: PRODUCTION_POLICY.artifact.version_id
    )
    first = source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")
    monkeypatch.setattr(
        source_module,
        "read_earlier_slot_selections",
        lambda _slot: pytest.fail("retry recalculated prior coverage"),
    )
    second = source_module.resolve_video_digest_run_request(SLOT, owner_token="retry-id")
    assert isinstance(first.source, SourceReady)
    assert isinstance(second.source, SourceReady)
    assert second.source.edition == first.source.edition


def test_unchanged_later_slot_skips_before_policy_or_paid_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    _install_report(monkeypatch, report)
    first = select_slot_subjects(report, "a" * 64, (), lambda *_args: pytest.fail())
    monkeypatch.setattr(
        source_module, "read_earlier_slot_selections", lambda _slot: (("b" * 64, first),)
    )
    monkeypatch.setattr(
        source_module,
        "publish_policy",
        lambda _policy: pytest.fail("unchanged slot published policy"),
    )
    monkeypatch.setattr(
        source_module,
        "_adjudicate_material_change",
        lambda *_args: pytest.fail("exact unchanged subject called model"),
    )
    request = source_module.resolve_video_digest_run_request(SLOT, owner_token="run-id")
    assert request.source == SourceUnavailable(reason=SlotSkipReason.UNCHANGED)


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


def test_later_slot_skips_exact_unchanged_subject_without_adjudication() -> None:
    report = _report()
    first = select_slot_subjects(
        report,
        "a" * 64,
        (),
        lambda *_args: pytest.fail("first slot called adjudicator"),
    )
    later = select_slot_subjects(
        report,
        "b" * 64,
        (("c" * 64, first),),
        lambda *_args: pytest.fail("unchanged subject called adjudicator"),
    )
    assert later.selected_subject_ids == ()
    assert later.decisions[0].reason == "exact_match"


def test_late_new_subject_is_selected_in_report_order() -> None:
    first_report = _report()
    first = select_slot_subjects(
        first_report,
        "a" * 64,
        (),
        lambda *_args: pytest.fail("first slot called adjudicator"),
    )
    old = first_report.sections[0]
    new_event = old.events[0].model_copy(
        update={
            "group_id": "6" * 64,
            "articles": (
                old.events[0]
                .articles[0]
                .model_copy(
                    update={
                        "article_version_id": "7" * 64,
                        "canonical_url": "https://example.com/new",
                    }
                ),
            ),
        }
    )
    new = old.model_copy(
        update={
            "theme_id": "8" * 64,
            "title": "Late subject",
            "events": (new_event,),
            "citations": (
                ReportSubjectCitation(article_version_id="7" * 64, evidence_quote="Late evidence"),
            ),
        }
    )
    later_report = first_report.model_copy(update={"sections": (old, new)})
    later = select_slot_subjects(
        later_report,
        "b" * 64,
        (("c" * 64, first),),
        lambda *_args: pytest.fail("new subject called adjudicator"),
    )
    assert later.selected_subject_ids == (new.theme_id,)
    assert tuple(item.reason for item in later.decisions) == ("exact_match", "new_subject")


def test_related_revision_requires_new_current_citation_evidence() -> None:
    report = _report()
    first = select_slot_subjects(report, "a" * 64, (), lambda *_args: pytest.fail())
    old = report.sections[0]
    revised = old.model_copy(
        update={
            "summary": "A new consequence",
            "citations": (
                *old.citations,
                ReportSubjectCitation(
                    article_version_id="9" * 64, evidence_quote="New sourced consequence"
                ),
            ),
        }
    )
    revised_report = report.model_copy(update={"sections": (revised,)})
    prior = (("c" * 64, first),)
    selected = select_slot_subjects(
        revised_report,
        "b" * 64,
        prior,
        lambda *_args: MaterialChangeJudgment(
            disposition="material",
            citation_article_version_id="9" * 64,
            evidence_quote="New sourced consequence",
            new_fact_or_consequence="A new consequence",
        ),
    )
    assert selected.selected_subject_ids == (old.theme_id,)
    assert selected.decisions[0].reason == "material_change"

    unsupported = select_slot_subjects(
        revised_report,
        "b" * 64,
        prior,
        lambda *_args: MaterialChangeJudgment(
            disposition="material",
            citation_article_version_id="1" * 64,
            evidence_quote="Source evidence",
            new_fact_or_consequence="A new consequence",
        ),
    )
    assert unsupported.selected_subject_ids == ()
    assert unsupported.decisions[0].reason == "no_new_fact"


def test_same_subject_with_new_theme_id_and_uncertain_judgment_fails_closed() -> None:
    report = _report()
    first = select_slot_subjects(report, "a" * 64, (), lambda *_args: pytest.fail())
    duplicate = report.sections[0].model_copy(update={"theme_id": "d" * 64})
    later = select_slot_subjects(
        report.model_copy(update={"sections": (duplicate,)}),
        "b" * 64,
        (("c" * 64, first),),
        lambda *_args: MaterialChangeJudgment(
            disposition="uncertain",
            citation_article_version_id=None,
            evidence_quote=None,
            new_fact_or_consequence=None,
        ),
    )
    assert later.selected_subject_ids == ()


def test_duplicate_title_with_new_structural_ids_is_not_generated_without_new_fact() -> None:
    report = _report()
    first = select_slot_subjects(report, "a" * 64, (), lambda *_args: pytest.fail())
    old = report.sections[0]
    duplicate_article = (
        old.events[0]
        .articles[0]
        .model_copy(
            update={
                "article_version_id": "e" * 64,
                "canonical_url": "https://example.com/duplicate",
            }
        )
    )
    duplicate_event = old.events[0].model_copy(
        update={"group_id": "f" * 64, "articles": (duplicate_article,)}
    )
    duplicate = old.model_copy(
        update={
            "theme_id": "d" * 64,
            "events": (duplicate_event,),
            "citations": (
                ReportSubjectCitation(
                    article_version_id="e" * 64, evidence_quote="Same reported fact"
                ),
            ),
        }
    )
    later = select_slot_subjects(
        report.model_copy(update={"sections": (duplicate,)}),
        "b" * 64,
        (("c" * 64, first),),
        lambda *_args: MaterialChangeJudgment(
            disposition="unchanged",
            citation_article_version_id=None,
            evidence_quote=None,
            new_fact_or_consequence=None,
        ),
    )
    assert later.selected_subject_ids == ()
    assert later.decisions[0].reason == "no_new_fact"
