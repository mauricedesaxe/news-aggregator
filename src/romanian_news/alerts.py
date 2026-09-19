"""Better Stack heartbeat pings for the news pipelines."""

import logging
import threading
from typing import Literal

import requests

from romanian_news.config import (
    BETTERSTACK_MORNING_REPORT_HEARTBEAT_URL,
    BETTERSTACK_REPORT_HEARTBEAT_URL,
    BETTERSTACK_RESEARCH_TRIGGER_HEARTBEAT_URL,
)

_logger = logging.getLogger(__name__)
_heartbeat_threads: list[threading.Thread] = []


def ping_heartbeat(kind: Literal["report", "research_trigger", "morning_report"]) -> None:
    """Record one successful pipeline pass; silence alerts on pipeline failure.

    Alerting is a side channel, so a failed ping logs and moves on instead of
    failing the pipeline work that earned it. The daemon thread keeps slow or
    dead alert endpoints from delaying the pipeline that pings them.
    """
    url = {
        "report": BETTERSTACK_REPORT_HEARTBEAT_URL,
        "research_trigger": BETTERSTACK_RESEARCH_TRIGGER_HEARTBEAT_URL,
        "morning_report": BETTERSTACK_MORNING_REPORT_HEARTBEAT_URL,
    }[kind]
    if not url:
        _logger.warning("No heartbeat URL configured for %s; the check will not page", kind)
        return

    def _ping() -> None:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            _logger.warning("Heartbeat ping failed for %s", kind)

    thread = threading.Thread(target=_ping, daemon=True)
    _heartbeat_threads.append(thread)
    thread.start()
