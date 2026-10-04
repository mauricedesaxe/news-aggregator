import json
import threading
from datetime import UTC, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import dagster as dg
import pytest

from romanian_news.catalog import video_digest as catalog
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.identity import canonical_json, sha256
from romanian_news.video_digest import incidents as incidents_module
from romanian_news.video_digest.models import GenerationAdmission, GenerationBudgetLimits
from romanian_news.worker import video_digest_monitor
from romanian_news.worker.definitions import defs
from tests.contracts.postgres_video_digest_generation import _GenerationPipeline
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog

requires_postgres = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


class _WebhookHandler(BaseHTTPRequestHandler):
    delivered: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        _WebhookHandler.delivered.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _local_webhook(monkeypatch: pytest.MonkeyPatch) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), _WebhookHandler)
    _WebhookHandler.delivered = []
    monkeypatch.setattr(
        incidents_module,
        "BETTERSTACK_VIDEO_INCIDENT_WEBHOOK_URL",
        f"http://127.0.0.1:{server.server_port}/video-incident",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@requires_postgres
def test_monitor_op_delivers_durable_incidents_through_the_real_webhook(
    postgres_catalog: PostgresCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_news_catalog_schema()
    failed = _GenerationPipeline(seed=61)
    catalog.fail_slot_deadline(failed.lease, recorded_at=datetime.now(UTC))
    budget = _GenerationPipeline(seed=62, scheduled_at=datetime.now(UTC))
    budget.checkpoint_plan()
    budget.verify_story(0)
    budget.start_request(
        0,
        admission=GenerationAdmission(
            generation_policy_artifact_version_id=budget.edition.policy_bundle_version_id,
            reserved_usd=Decimal("3.25632"),
            limits=GenerationBudgetLimits(
                story_usd=Decimal("7"),
                edition_usd=Decimal("7"),
                bucharest_day_usd=Decimal("4"),
                calendar_month_usd=Decimal("1000"),
            ),
        ),
    )
    server = _local_webhook(monkeypatch)
    try:
        with dg.build_op_context() as context:
            video_digest_monitor.deliver_video_digest_incidents(context)
    finally:
        server.shutdown()
        server.server_close()

    by_category = {
        payload["incident"]["metadata"]["category"]: payload["incident"]
        for payload in _WebhookHandler.delivered
    }
    assert set(by_category) == {"deadline", "budget_80_percent"}
    deadline = by_category["deadline"]
    assert deadline["status"] == "alert"
    assert deadline["metadata"]["service"] == "video_digest"
    assert deadline["metadata"]["scope_id"] == failed.slot.slot_id
    assert deadline["id"] == sha256(
        canonical_json({"category": "deadline", "slot_id": failed.slot.slot_id})
    )
    day_key = budget.slot.bucharest_day.isoformat()
    budget_incident = by_category["budget_80_percent"]
    scope = f"bucharest_day:{day_key}"
    assert budget_incident["metadata"]["scope_id"] == scope
    assert budget_incident["id"] == sha256(
        canonical_json({"category": "budget_80_percent", "slot_id": scope})
    )
    assert postgres_catalog.execute(
        "SELECT stage, failure_reason FROM video_digest_slots WHERE slot_id = %s",
        (failed.slot.slot_id,),
    ).fetchone() == {"stage": "failed", "failure_reason": "deadline"}
    assert postgres_catalog.execute(
        """SELECT SUM(reserved_usd) >= MIN(limit_usd) * 0.8 AS over_budget
           FROM video_digest_generation_reservations
           WHERE scope_kind = 'bucharest_day' AND scope_key = %s""",
        (day_key,),
    ).fetchone() == {"over_budget": True}


def test_video_incident_monitor_is_registered_but_stopped() -> None:
    schedule = video_digest_monitor.scheduled_video_digest_incident_monitor
    assert schedule.cron_schedule == "*/5 * * * *"
    assert schedule.default_status is dg.DefaultScheduleStatus.STOPPED
    assert defs.get_schedule_def(schedule.name) is not None
