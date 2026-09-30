from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from romanian_news import current_report as current_report_module
from romanian_news import reports as reports_module
from romanian_news.articles.models import ArticleWorkStatus
from romanian_news.artifacts import ArtifactReference
from romanian_news.current_report import (
    CurrentDailyReport,
    CurrentDailyReportHead,
    ReportFresh,
    ReportInputsNotReady,
    ReportMissing,
    ReportStale,
    build_and_publish_current_daily_report,
    read_current_daily_report,
    read_daily_report_coverage_status,
    read_daily_report_freshness,
)
from romanian_news.reports import DailyReport, DailyReportInput

DAY = date(2026, 9, 14)
THEMES_VERSION = "a" * 64
ASSESSMENTS_VERSION = "b" * 64
CLUSTER_VERSION = f"{DAY.day:064x}"
REQUEST_ID = "7" * 64
REPORT_VERSION = "9" * 64
MORNING = datetime(2026, 9, 14, 4, 2, tzinfo=UTC)
LATER = datetime(2026, 9, 14, 4, 30, tzinfo=UTC)


def _reference(version_id: str, artifact_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        version_id=version_id,
        r2_key=f"news/test/{artifact_id}/{version_id}.json",
        content_digest=version_id,
    )


def _empty_report() -> DailyReport:
    return DailyReport(
        day=DAY,
        accepted_article_count=0,
        theme_count=0,
        group_count=0,
        sections=(),
    )


@dataclass
class Pipeline:
    reads: int = 0
    publishes: int = 0
    published_ref: str | None = None
    current: DailyReportInput = DailyReportInput(
        day=DAY,
        themes=_reference(THEMES_VERSION, "news:themes:2026-09-14"),
        assessments=_reference(ASSESSMENTS_VERSION, "news:subject-assessments:2026-09-14"),
        cluster_set=_reference(CLUSTER_VERSION, "news:clusters:2026-09-14"),
        summaries=(),
        sentiments=(),
    )


@pytest.fixture
def pipeline(monkeypatch):
    """Fake the full assembly path: input discovery, build, publish, timestamps."""
    state = Pipeline()

    def read_input(day: date) -> DailyReportInput:
        state.reads += 1
        return state.current

    def build(value: DailyReportInput, *, coverage_status: str) -> SimpleNamespace:
        assert coverage_status == "provisional"
        return SimpleNamespace(
            request_id=REQUEST_ID,
            report=_empty_report(),
            content_digest="6" * 64,
        )

    def publish(output: object, implementation_ref: str) -> SimpleNamespace:
        state.publishes += 1
        state.published_ref = implementation_ref
        return SimpleNamespace(
            status="published",
            run_id="1" * 64,
            version_id=REPORT_VERSION,
            uploaded_objects=1,
            reused_objects=0,
        )

    def current_references(artifact_ids: tuple[str, ...]) -> tuple[ArtifactReference, ...]:
        value = state.current
        by_id = {
            reference.artifact_id: reference
            for reference in (
                value.themes,
                value.assessments,
                value.cluster_set,
                *value.summaries,
                *value.sentiments,
            )
        }
        return tuple(by_id[artifact_id] for artifact_id in sorted(by_id))

    monkeypatch.setattr(reports_module, "read_daily_report_input", read_input)
    monkeypatch.setattr(current_report_module, "build_daily_report_from_input", build)
    monkeypatch.setattr(
        current_report_module, "read_daily_report_coverage_status", lambda *_args: "provisional"
    )
    monkeypatch.setattr(current_report_module, "publish_daily_report", publish)
    monkeypatch.setattr(
        "romanian_news.catalog.artifacts.current_artifact_references", current_references
    )
    monkeypatch.setattr(
        "romanian_news.catalog.artifacts.current_artifact_version_times",
        lambda _ids: (MORNING, LATER),
    )
    reports_module.clear_daily_report_input_cache()
    yield state
    reports_module.clear_daily_report_input_cache()


def test_current_report_builds_publishes_and_stamps_latest_input_time(pipeline) -> None:
    result = build_and_publish_current_daily_report(DAY, "git:test")

    assert isinstance(result, CurrentDailyReport)
    assert result.head.day == DAY
    assert result.head.version_id == REPORT_VERSION
    assert result.head.input_time == LATER
    assert pipeline.reads == 1
    assert pipeline.publishes == 1
    assert pipeline.published_ref == "git:test"


def test_current_report_reuses_cached_input_while_versions_hold(pipeline) -> None:
    build_and_publish_current_daily_report(DAY, "git:test")
    build_and_publish_current_daily_report(DAY, "git:test")

    assert pipeline.reads == 1
    assert pipeline.publishes == 2


def test_current_report_rereads_after_an_input_version_advances(pipeline) -> None:
    build_and_publish_current_daily_report(DAY, "git:test")
    pipeline.current = pipeline.current.model_copy(
        update={
            "themes": pipeline.current.themes.model_copy(update={"version_id": "c" * 64}),
        }
    )

    build_and_publish_current_daily_report(DAY, "git:test")

    assert pipeline.reads == 2


def test_coverage_stays_provisional_until_the_day_closes_and_work_finishes(monkeypatch) -> None:
    status = ArticleWorkStatus(
        retryable_entries=0,
        deferred_event_ids=(),
        quarantined_event_ids=(),
        source_covered_days=(DAY,),
    )
    monkeypatch.setattr(
        "romanian_news.articles.acquisition.read_article_work_status",
        lambda *_args, **_kwargs: status,
    )
    assert (
        read_daily_report_coverage_status(
            DAY, "git:test", now=datetime(2026, 9, 14, 12, tzinfo=UTC)
        )
        == "provisional"
    )
    after = datetime(2026, 9, 15, 12, tzinfo=UTC)
    assert read_daily_report_coverage_status(DAY, "git:test", now=after) == "complete"

    for incomplete in (
        status.model_copy(update={"retryable_entries": 1}),
        status.model_copy(update={"deferred_event_ids": ("1" * 64,)}),
        status.model_copy(update={"quarantined_event_ids": ("2" * 64,)}),
        status.model_copy(update={"source_covered_days": ()}),
    ):
        monkeypatch.setattr(
            "romanian_news.articles.acquisition.read_article_work_status",
            lambda *_args, _value=incomplete, **_kwargs: _value,
        )
        assert read_daily_report_coverage_status(DAY, "git:test", now=after) == "provisional"


def test_current_report_read_path_never_builds_or_publishes(monkeypatch) -> None:
    from romanian_news.catalog.report_inputs import CurrentDailyReportRecord

    record = CurrentDailyReportRecord(
        day=DAY,
        version_id=REPORT_VERSION,
        run_id="1" * 64,
        content_digest="6" * 64,
        r2_key="news/reports/current.json",
        input_time=LATER,
    )
    content = _empty_report().model_dump_json().encode()
    monkeypatch.setattr(
        "romanian_news.catalog.report_inputs.read_current_daily_report_record",
        lambda _day: record,
    )
    monkeypatch.setattr(
        "romanian_news.storage.read_verified_r2_object",
        lambda key, digest: (
            content if (key, digest) == (record.r2_key, record.content_digest) else b""
        ),
    )
    monkeypatch.setattr(
        current_report_module,
        "build_daily_report_from_input",
        lambda _value: pytest.fail("read path must not build"),
    )
    monkeypatch.setattr(
        current_report_module,
        "publish_daily_report",
        lambda *_args: pytest.fail("read path must not publish"),
    )

    result = read_current_daily_report(DAY)

    assert result == CurrentDailyReport(
        head=CurrentDailyReportHead(
            day=DAY,
            version_id=REPORT_VERSION,
            run_id="1" * 64,
            content_digest="6" * 64,
            r2_key="news/reports/current.json",
            input_time=LATER,
        ),
        report=_empty_report(),
    )


def test_freshness_compares_inputs_for_the_captured_report_run(monkeypatch) -> None:
    from romanian_news.catalog import report_inputs
    from romanian_news.catalog.report_inputs import DailyReportInputVersions

    head = CurrentDailyReportHead(
        day=DAY,
        version_id=REPORT_VERSION,
        run_id="1" * 64,
        content_digest="6" * 64,
        r2_key="news/reports/current.json",
        input_time=LATER,
    )
    versions = DailyReportInputVersions(
        day=DAY,
        themes=THEMES_VERSION,
        assessments=ASSESSMENTS_VERSION,
        cluster_set=CLUSTER_VERSION,
        summaries=(),
        sentiments=(),
    )
    captured_run_ids: list[str] = []
    monkeypatch.setattr(current_report_module, "read_current_daily_report_head", lambda _day: head)
    monkeypatch.setattr(
        report_inputs, "read_current_daily_report_input_versions", lambda _day: versions
    )
    monkeypatch.setattr(
        "romanian_news.storage.read_verified_r2_object",
        lambda *_args: pytest.fail("freshness checks must not read R2"),
    )

    def recorded(_day: date, run_id: str) -> DailyReportInputVersions:
        captured_run_ids.append(run_id)
        return versions

    monkeypatch.setattr(report_inputs, "read_daily_report_run_input_versions", recorded)
    assert isinstance(read_daily_report_freshness(DAY), ReportFresh)
    assert captured_run_ids == [head.run_id]

    monkeypatch.setattr(
        report_inputs,
        "read_current_daily_report_input_versions",
        lambda _day: versions.model_copy(update={"themes": "c" * 64}),
    )
    assert isinstance(read_daily_report_freshness(DAY), ReportStale)


def test_freshness_distinguishes_inputs_not_ready_and_missing_report(monkeypatch) -> None:
    from romanian_news.catalog import report_inputs
    from romanian_news.catalog.report_inputs import DailyReportInputVersions

    monkeypatch.setattr(current_report_module, "read_current_daily_report_head", lambda _day: None)
    monkeypatch.setattr(
        report_inputs, "read_current_daily_report_input_versions", lambda _day: None
    )
    assert isinstance(read_daily_report_freshness(DAY), ReportInputsNotReady)

    versions = DailyReportInputVersions(
        day=DAY,
        themes=THEMES_VERSION,
        assessments=ASSESSMENTS_VERSION,
        cluster_set=CLUSTER_VERSION,
        summaries=(),
        sentiments=(),
    )
    monkeypatch.setattr(
        report_inputs, "read_current_daily_report_input_versions", lambda _day: versions
    )
    assert isinstance(read_daily_report_freshness(DAY), ReportMissing)
