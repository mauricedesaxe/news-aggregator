from datetime import datetime
from zoneinfo import ZoneInfo

import dagster as dg
import pytest

from romanian_news.video_digest.models import ScheduledSlot, SlotFailureReason, SlotSkipReason
from romanian_news.video_digest.orchestration import (
    IncidentAlert,
    NoAlert,
    RunDeferred,
    RunFailed,
)
from romanian_news.worker import video_digest


def test_schedule_contract_and_deterministic_slot_mapping() -> None:
    schedule = video_digest.scheduled_video_digest
    assert schedule.cron_schedule == "0 8,13,20 * * *"
    assert schedule.execution_timezone == "Europe/Bucharest"
    assert schedule.default_status is dg.DefaultScheduleStatus.STOPPED
    assert schedule.job_name == "video_digest"
    assert len(video_digest.video_digest_job.nodes) == 1
    assert "dagster/max_retries" not in video_digest.video_digest_job.tags
    assert "dagster/retry_on_asset_or_op_failure" not in video_digest.video_digest_job.tags
    assert video_digest.orchestrate_video_digest.retry_policy is not None
    assert video_digest.orchestrate_video_digest.retry_policy.max_retries == 95
    assert video_digest.orchestrate_video_digest.retry_policy.delay == 60

    for hour, expected in ((8, "morning"), (13, "midday"), (20, "evening")):
        slot = video_digest.scheduled_video_digest_slot(
            datetime(2026, 9, 21, hour, tzinfo=ZoneInfo("Europe/Bucharest"))
        )
        assert slot.name.value == expected


def test_schedule_records_slot_and_uses_slot_run_key(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled_at = datetime(2026, 9, 21, 8, tzinfo=ZoneInfo("Europe/Bucharest"))
    recorded: list[ScheduledSlot] = []

    def record_slot(slot: ScheduledSlot, *, recorded_at: datetime) -> None:
        assert recorded_at == scheduled_at
        recorded.append(slot)

    monkeypatch.setattr(video_digest, "schedule_slot", record_slot)

    with dg.DagsterInstance.local_temp() as instance:
        with dg.build_schedule_context(
            instance=instance, scheduled_execution_time=scheduled_at
        ) as context:
            evaluation = video_digest.scheduled_video_digest.evaluate_tick(context)

    slot = recorded[0]
    assert evaluation.run_requests is not None
    assert evaluation.run_requests[0].run_key == f"video-digest:{slot.slot_id}"
    assert {
        key: evaluation.run_requests[0].tags[key]
        for key in (
            "news/video_digest_slot_id",
            "news/video_digest_slot_name",
            "news/scheduled_at",
        )
    } == {
        "news/video_digest_slot_id": slot.slot_id,
        "news/video_digest_slot_name": "morning",
        "news/scheduled_at": scheduled_at.isoformat(),
    }


def test_schedule_durably_skips_colliding_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled_at = datetime(2026, 9, 21, 13, tzinfo=ZoneInfo("Europe/Bucharest"))
    skipped: list[tuple[str, SlotSkipReason]] = []

    def record_slot(slot: ScheduledSlot, *, recorded_at: datetime) -> ScheduledSlot:
        assert recorded_at == scheduled_at
        return slot

    def record_skip(slot_id: str, reason: SlotSkipReason, *, recorded_at: datetime) -> None:
        assert recorded_at == scheduled_at
        skipped.append((slot_id, reason))

    monkeypatch.setattr(video_digest, "schedule_slot", record_slot)
    monkeypatch.setattr(video_digest, "skip_slot", record_skip)

    with dg.DagsterInstance.local_temp() as instance:
        _ = instance.create_run_for_job(
            video_digest.video_digest_job,
            status=dg.DagsterRunStatus.STARTED,
        )
        with dg.build_schedule_context(
            instance=instance, scheduled_execution_time=scheduled_at
        ) as context:
            evaluation = video_digest.scheduled_video_digest.evaluate_tick(context)

    assert evaluation.run_requests == []
    assert evaluation.skip_message == "A video digest run is already queued or active."
    assert skipped[0][1] is SlotSkipReason.OVERLAPPING_RUN


def test_action_waiting_requests_a_dagster_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    slot = video_digest.scheduled_video_digest_slot(
        datetime(2026, 9, 21, 8, tzinfo=ZoneInfo("Europe/Bucharest"))
    )

    class Runtime:
        def run(self, slot_id: str, *, owner_token: str) -> tuple[RunDeferred, NoAlert]:
            assert slot_id == slot.slot_id
            assert owner_token == "run-1"
            return RunDeferred(reason="provider is processing", retry_after_seconds=45), NoAlert()

    class Run:
        tags = {"news/video_digest_slot_id": slot.slot_id}

    class Context:
        run = Run()
        run_id = "run-1"

        def add_output_metadata(self, metadata: dict[str, object]) -> None:
            assert metadata == {
                "alert": "none",
                "outcome": "deferred",
                "slot_id": slot.slot_id,
            }

    monkeypatch.setattr(video_digest, "runtime_factory", Runtime)
    compute_fn = video_digest.orchestrate_video_digest.compute_fn
    decorated_fn = getattr(compute_fn, "decorated_fn", None)
    assert callable(decorated_fn)

    with pytest.raises(dg.RetryRequested) as raised:
        decorated_fn(Context())

    assert raised.value.max_retries == 95
    assert raised.value.seconds_to_wait == 60


@pytest.mark.parametrize("tags", [{}, {"news/video_digest_slot_id": "not-a-slot-id"}])
def test_missing_or_malformed_slot_tag_fails_without_retry(tags: dict[str, str]) -> None:
    class Run:
        def __init__(self, run_tags: dict[str, str]) -> None:
            self.tags = run_tags

    class Context:
        run = Run(tags)
        run_id = "run-1"

    compute_fn = video_digest.orchestrate_video_digest.compute_fn
    decorated_fn = getattr(compute_fn, "decorated_fn", None)
    assert callable(decorated_fn)

    with pytest.raises(dg.Failure) as raised:
        decorated_fn(Context())

    assert raised.value.allow_retries is False
    assert raised.value.description == (
        "Video digest run requires a valid news/video_digest_slot_id tag"
    )


def test_terminal_failure_fails_the_dagster_step_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot = video_digest.scheduled_video_digest_slot(
        datetime(2026, 9, 21, 8, tzinfo=ZoneInfo("Europe/Bucharest"))
    )
    outcome = RunFailed(reason=SlotFailureReason.DEADLINE)
    alert = IncidentAlert(
        alert_id="a" * 64,
        category="deadline",
        slot_id=slot.slot_id,
    )

    class Runtime:
        def run(self, slot_id: str, *, owner_token: str) -> tuple[RunFailed, IncidentAlert]:
            assert slot_id == slot.slot_id
            assert owner_token == "run-1"
            return outcome, alert

    class Run:
        tags = {"news/video_digest_slot_id": slot.slot_id}

    class Context:
        run = Run()
        run_id = "run-1"

        def add_output_metadata(self, metadata: dict[str, object]) -> None:
            assert metadata == {
                "alert": "incident",
                "outcome": "failed",
                "slot_id": slot.slot_id,
            }

    monkeypatch.setattr(video_digest, "runtime_factory", Runtime)
    compute_fn = video_digest.orchestrate_video_digest.compute_fn
    decorated_fn = getattr(compute_fn, "decorated_fn", None)
    assert callable(decorated_fn)

    with pytest.raises(dg.Failure) as raised:
        decorated_fn(Context())

    assert raised.value.allow_retries is False
    assert raised.value.description == "Video digest slot failed: deadline"
