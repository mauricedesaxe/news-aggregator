import re
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

import dagster as dg
from dagster import OpExecutionContext

from romanian_news import BUCHAREST
from romanian_news.catalog.video_digest import schedule_slot, skip_slot
from romanian_news.video_digest.models import (
    ScheduledSlot,
    SlotId,
    SlotName,
    SlotSkipReason,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    AlertDisposition,
    RunDeferred,
    RunFailed,
    VideoDigestRunOutcome,
)
from romanian_news.worker.assets import BUCHAREST_TIMEZONE

VIDEO_DIGEST_CRON = "0 8,13,20 * * *"
VIDEO_DIGEST_JOB_NAME = "video_digest"
VIDEO_DIGEST_RUN_KEY_PREFIX = "video-digest"
VIDEO_DIGEST_MAX_RETRIES = 95
VIDEO_DIGEST_MIN_RETRY_SECONDS = 60


class VideoDigestRuntime(Protocol):
    def run(
        self, slot_id: SlotId, *, owner_token: str
    ) -> tuple[VideoDigestRunOutcome, AlertDisposition]: ...


runtime_factory: Callable[[], VideoDigestRuntime] | None = None


def scheduled_video_digest_slot(scheduled_at: datetime) -> ScheduledSlot:
    local = scheduled_at.astimezone(BUCHAREST)
    names = {8: SlotName.MORNING, 13: SlotName.MIDDAY, 20: SlotName.EVENING}
    try:
        name = names[local.hour]
    except KeyError as error:
        raise ValueError("Video digest schedule time is outside a configured slot") from error
    if local.minute != 0:
        raise ValueError("Video digest schedule time must be on the hour")
    return ScheduledSlot(
        slot_id=scheduled_slot_id(name, scheduled_at),
        name=name,
        scheduled_at=scheduled_at,
        bucharest_day=local.date(),
    )


@dg.op(
    retry_policy=dg.RetryPolicy(
        max_retries=VIDEO_DIGEST_MAX_RETRIES,
        delay=VIDEO_DIGEST_MIN_RETRY_SECONDS,
    )
)
def orchestrate_video_digest(context: OpExecutionContext) -> None:
    slot_value = context.run.tags.get("news/video_digest_slot_id")
    if slot_value is None or re.fullmatch(r"[0-9a-f]{64}", slot_value) is None:
        raise dg.Failure(
            "Video digest run requires a valid news/video_digest_slot_id tag",
            allow_retries=False,
        )
    slot_id = SlotId(slot_value)
    if runtime_factory is None:
        raise dg.Failure(
            "Video digest production adapters are not configured",
            allow_retries=False,
        )
    outcome, alert = runtime_factory().run(slot_id, owner_token=context.run_id)
    context.add_output_metadata({"alert": alert.kind, "outcome": outcome.kind, "slot_id": slot_id})
    if isinstance(outcome, RunDeferred):
        raise dg.RetryRequested(
            max_retries=VIDEO_DIGEST_MAX_RETRIES,
            seconds_to_wait=max(
                VIDEO_DIGEST_MIN_RETRY_SECONDS,
                outcome.retry_after_seconds,
            ),
        )
    if isinstance(outcome, RunFailed):
        raise dg.Failure(
            f"Video digest slot failed: {outcome.reason.value}",
            allow_retries=False,
        )


@dg.job(
    name=VIDEO_DIGEST_JOB_NAME,
    executor_def=dg.multiprocess_executor.configured({"max_concurrent": 1}),
)
def video_digest_job() -> None:
    orchestrate_video_digest()


@dg.schedule(
    job=video_digest_job,
    cron_schedule=VIDEO_DIGEST_CRON,
    execution_timezone=BUCHAREST_TIMEZONE,
    default_status=dg.DefaultScheduleStatus.STOPPED,
)
def scheduled_video_digest(
    context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    if context.scheduled_execution_time is None:
        raise ValueError("Schedule execution time is required")
    slot = scheduled_video_digest_slot(context.scheduled_execution_time)
    schedule_slot(slot, recorded_at=context.scheduled_execution_time)
    active = context.instance.get_runs(
        filters=dg.RunsFilter(
            job_name=video_digest_job.name,
            statuses=(
                dg.DagsterRunStatus.QUEUED,
                dg.DagsterRunStatus.NOT_STARTED,
                dg.DagsterRunStatus.MANAGED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.STARTED,
                dg.DagsterRunStatus.CANCELING,
            ),
        ),
        limit=1,
    )
    if active:
        skip_slot(
            slot.slot_id,
            SlotSkipReason.OVERLAPPING_RUN,
            recorded_at=context.scheduled_execution_time,
        )
        return dg.SkipReason("A video digest run is already queued or active.")
    return dg.RunRequest(
        run_key=f"{VIDEO_DIGEST_RUN_KEY_PREFIX}:{slot.slot_id}",
        tags={
            "news/video_digest_slot_id": slot.slot_id,
            "news/video_digest_slot_name": slot.name.value,
            "news/scheduled_at": slot.scheduled_at.isoformat(),
        },
    )
