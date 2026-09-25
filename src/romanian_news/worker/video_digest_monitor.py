from datetime import UTC, datetime

import dagster as dg

from romanian_news.catalog.video_digest_monitor import read_due_video_incidents
from romanian_news.video_digest.incidents import send_video_incident


@dg.op
def deliver_video_digest_incidents(context: dg.OpExecutionContext) -> None:
    incidents = read_due_video_incidents(datetime.now(UTC))
    delivered = sum(send_video_incident(incident) for incident in incidents)
    context.add_output_metadata({"incident_candidates": len(incidents), "delivered": delivered})


@dg.job(name="video_digest_incident_monitor")
def video_digest_incident_monitor_job() -> None:
    deliver_video_digest_incidents()


@dg.schedule(
    job=video_digest_incident_monitor_job,
    cron_schedule="*/5 * * * *",
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.STOPPED,
)
def scheduled_video_digest_incident_monitor(
    _context: dg.ScheduleEvaluationContext,
) -> dg.RunRequest:
    return dg.RunRequest()
