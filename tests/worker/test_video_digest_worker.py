from datetime import datetime
from zoneinfo import ZoneInfo

import dagster as dg
import pytest

from romanian_news.video_digest.models import SlotFailureReason
from romanian_news.video_digest.orchestration import (
    IncidentAlert,
    NoAlert,
    RunDeferred,
    RunFailed,
)
from romanian_news.worker import video_digest
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog

requires_postgres = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


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

    assert video_digest.runtime_factory is not None

    for hour, expected in ((8, "morning"), (13, "midday"), (20, "evening")):
        slot = video_digest.scheduled_video_digest_slot(
            datetime(2026, 9, 21, hour, tzinfo=ZoneInfo("Europe/Bucharest"))
        )
        assert slot.name.value == expected


def test_runtime_factory_wires_the_h3_port(monkeypatch: pytest.MonkeyPatch) -> None:
    generation = object()
    runtime = object()

    def make_runtime(port: object) -> object:
        assert port is generation
        return runtime

    monkeypatch.setattr(video_digest, "ProductionH3GenerationPort", lambda: generation)
    monkeypatch.setattr(video_digest, "ProductionVideoDigestRuntime", make_runtime)
    assert video_digest.runtime_factory is not None
    assert video_digest.runtime_factory() is runtime


@pytest.mark.parametrize(
    ("hour", "minute", "message"),
    ((15, 0, "outside a configured slot"), (8, 30, "on the hour")),
)
def test_schedule_slot_rejects_off_slot_times(hour: int, minute: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        video_digest.scheduled_video_digest_slot(
            datetime(2026, 9, 21, hour, minute, tzinfo=ZoneInfo("Europe/Bucharest"))
        )


@requires_postgres
def test_schedule_records_slot_and_uses_slot_run_key(
    postgres_catalog: PostgresCatalog,
) -> None:
    scheduled_at = datetime(2026, 9, 21, 8, tzinfo=ZoneInfo("Europe/Bucharest"))

    with dg.DagsterInstance.local_temp() as instance:
        with dg.build_schedule_context(
            instance=instance, scheduled_execution_time=scheduled_at
        ) as context:
            evaluation = video_digest.scheduled_video_digest.evaluate_tick(context)

    assert evaluation.run_requests is not None
    run_request = evaluation.run_requests[0]
    stored = postgres_catalog.execute(
        "SELECT name, stage FROM video_digest_slots WHERE slot_id = %s",
        (run_request.tags["news/video_digest_slot_id"],),
    ).fetchone()
    assert stored == {"name": "morning", "stage": "scheduled"}
    assert run_request.run_key == (f"video-digest:{run_request.tags['news/video_digest_slot_id']}")
    assert run_request.tags["news/video_digest_slot_name"] == "morning"
    assert run_request.tags["news/scheduled_at"] == scheduled_at.isoformat()


@requires_postgres
def test_schedule_durably_skips_colliding_tick(postgres_catalog: PostgresCatalog) -> None:
    scheduled_at = datetime(2026, 9, 21, 13, tzinfo=ZoneInfo("Europe/Bucharest"))

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
    skipped = postgres_catalog.execute(
        "SELECT slot_id, stage, skip_reason FROM video_digest_slots WHERE stage = 'skipped'"
    ).fetchall()
    assert [(row["slot_id"], row["stage"], row["skip_reason"]) for row in skipped] == [
        (skipped[0]["slot_id"], "skipped", "overlapping_run")
    ]


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


def test_unconfigured_runtime_fails_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    slot = video_digest.scheduled_video_digest_slot(
        datetime(2026, 9, 21, 8, tzinfo=ZoneInfo("Europe/Bucharest"))
    )

    class Run:
        tags = {"news/video_digest_slot_id": slot.slot_id}

    class Context:
        run = Run()
        run_id = "run-1"

    monkeypatch.setattr(video_digest, "runtime_factory", None)
    compute_fn = video_digest.orchestrate_video_digest.compute_fn
    decorated_fn = getattr(compute_fn, "decorated_fn", None)
    assert callable(decorated_fn)

    with pytest.raises(dg.Failure) as raised:
        decorated_fn(Context())

    assert raised.value.allow_retries is False
    assert raised.value.description == "Video digest production adapters are not configured"


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
