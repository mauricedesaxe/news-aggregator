"""Small, stable payloads for video incidents sent to Better Stack."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import requests

from romanian_news.config import BETTERSTACK_VIDEO_INCIDENT_WEBHOOK_URL
from romanian_news.video_digest.orchestration import IncidentAlert

_logger = logging.getLogger(__name__)

VideoIncidentCategory = Literal[
    "terminal_failure",
    "deadline",
    "publication_late",
    "fal_queue_stalled",
    "budget_80_percent",
    "no_success_24_hours",
    "publication_conflict",
]


@dataclass(frozen=True)
class VideoIncident:
    alert_id: str
    category: VideoIncidentCategory
    scope_id: str


def send_video_incident(
    incident: VideoIncident,
    *,
    webhook_url: str | None = None,
    session: requests.Session | None = None,
) -> bool:
    url = webhook_url or BETTERSTACK_VIDEO_INCIDENT_WEBHOOK_URL
    if not url:
        _logger.warning("Video incident webhook is not configured")
        return False
    payload = {
        "incident": {
            "id": incident.alert_id,
            "status": "alert",
            "title": f"Video digest: {incident.category.replace('_', ' ')}",
            "description": f"{incident.category} for {incident.scope_id}",
            "metadata": {
                "service": "video_digest",
                "category": incident.category,
                "scope_id": incident.scope_id,
            },
        }
    }
    try:
        response = (session or requests).post(url, json=payload, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        _logger.warning("Video incident delivery failed for %s", incident.alert_id)
        return False
    return True


def send_slot_failure_alert(alert: IncidentAlert) -> bool:
    return send_video_incident(
        VideoIncident(
            alert_id=alert.alert_id,
            category=alert.category,
            scope_id=alert.slot_id,
        )
    )
