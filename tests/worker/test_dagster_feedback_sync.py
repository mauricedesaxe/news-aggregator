from types import SimpleNamespace

import dagster as dg
import pytest
from dagster._core.definitions.metadata.metadata_value import IntMetadataValue
from dagster._core.events import StepOutputData

from romanian_news.worker import feedback_sync


def test_feedback_sync_applies_schema_and_exposes_result_counts(monkeypatch) -> None:
    result = _result(failed=0, unresolved=2)
    monkeypatch.setattr(feedback_sync, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        feedback_sync,
        "load_news_evaluation_release",
        lambda content: SimpleNamespace(manifest_reference=SimpleNamespace(version_id="a" * 64)),
    )
    monkeypatch.setattr(
        feedback_sync,
        "sync_news_feedback",
        lambda version_id: result,
    )

    execution = feedback_sync.news_feedback_sync.execute_in_process()

    assert execution.success
    assert execution.output_for_node("news_feedback_sync_op") == result
    output_event = next(
        event for event in execution.all_node_events if event.event_type_value == "STEP_OUTPUT"
    )
    event_data = output_event.event_specific_data
    assert isinstance(event_data, StepOutputData)
    metadata = event_data.metadata
    assert {name: value.value for name, value in metadata.items()} == result.model_dump()


def test_feedback_sync_fails_when_any_attempt_failed(monkeypatch) -> None:
    result = _result(failed=1, unresolved=3)
    monkeypatch.setattr(feedback_sync, "ensure_news_catalog_schema", lambda: None)
    monkeypatch.setattr(
        feedback_sync,
        "load_news_evaluation_release",
        lambda _content: SimpleNamespace(manifest_reference=SimpleNamespace(version_id="a" * 64)),
    )
    monkeypatch.setattr(feedback_sync, "sync_news_feedback", lambda _version_id: result)

    execution = feedback_sync.news_feedback_sync.execute_in_process(raise_on_error=False)

    assert execution.success is False
    failure = execution.failure_data_for_node("news_feedback_sync_op")
    assert failure is not None
    error = failure.error
    assert error is not None
    assert error.cls_name == "Failure"
    assert "1 feedback sync attempts failed" in error.message
    user_failure = failure.user_failure_data
    assert user_failure is not None
    unresolved = user_failure.metadata["unresolved"]
    assert isinstance(unresolved, IntMetadataValue)
    assert unresolved.value == 3


def test_feedback_sync_schedule_runs_every_fifteen_minutes() -> None:
    assert feedback_sync.quarter_hourly_news_feedback_sync.cron_schedule == "*/15 * * * *"
    assert feedback_sync.quarter_hourly_news_feedback_sync.default_status.name == "RUNNING"
    assert feedback_sync.quarter_hourly_news_feedback_sync.job_name == "news_feedback_sync"
    assert dg.DagsterRunStatus.QUEUED in feedback_sync.NONTERMINAL_RUN_STATUSES


def test_feedback_sync_schedule_requests_run_without_existing_run() -> None:
    with dg.DagsterInstance.local_temp() as instance:
        with dg.build_schedule_context(instance=instance) as context:
            evaluation = feedback_sync.quarter_hourly_news_feedback_sync.evaluate_tick(context)

    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    assert evaluation.skip_message is None


@pytest.mark.parametrize("status", feedback_sync.NONTERMINAL_RUN_STATUSES[1:])
def test_feedback_sync_schedule_skips_while_run_is_nonterminal(
    status: dg.DagsterRunStatus,
) -> None:
    with dg.DagsterInstance.local_temp() as instance:
        _ = instance.create_run_for_job(
            feedback_sync.news_feedback_sync,
            status=status,
        )
        with dg.build_schedule_context(instance=instance) as context:
            evaluation = feedback_sync.quarter_hourly_news_feedback_sync.evaluate_tick(context)

    assert evaluation.run_requests == []
    assert evaluation.skip_message == (
        "Skipping because news_feedback_sync already has a queued or active run."
    )


def test_feedback_sync_schedule_ignores_successful_run() -> None:
    with dg.DagsterInstance.local_temp() as instance:
        _ = instance.create_run_for_job(
            feedback_sync.news_feedback_sync,
            status=dg.DagsterRunStatus.SUCCESS,
        )
        with dg.build_schedule_context(instance=instance) as context:
            evaluation = feedback_sync.quarter_hourly_news_feedback_sync.evaluate_tick(context)

    assert evaluation.run_requests is not None
    assert len(evaluation.run_requests) == 1
    assert evaluation.skip_message is None


def _result(*, failed: int, unresolved: int):
    return feedback_sync.NewsFeedbackSyncResult(
        selected_feedback=4,
        matched_attempts=5,
        completed=4,
        failed=failed,
        unresolved=unresolved,
    )
