from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github/workflows"

HISTORICAL_SCHEDULES = (
    "scheduled_historical_weekly_status",
    "scheduled_archive_page_backfill",
    "scheduled_archive_article_capture",
    "scheduled_retrospective_analysis",
)
HISTORICAL_JOBS = (
    "retrospective_analysis_pilot",
    "weekly_status_refresh",
    "archive_article_capture_batch",
    "archive_page_backfill_job",
)


def _embedded_script(workflow: str, step_name: str) -> str:
    workflow_text = (WORKFLOWS / workflow).read_text()
    step = workflow_text.split(f"- name: {step_name}", maxsplit=1)[1]
    return textwrap.dedent(step.split("<<'PY'", maxsplit=1)[1].split("\n          PY")[0])


def _run_script(workflow: str, step_name: str, url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-c", _embedded_script(workflow, step_name)),
        env=os.environ
        | {
            "DAGSTER_CLOUD_API_TOKEN": "test-token",
            "DAGSTER_CLOUD_GRAPHQL_URL": url,
            "DAGSTER_CLOUD_LOCATION": "test-location",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


class _GraphQLHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        service = cast(_GraphQLHTTPServer, self.server).service
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        variables = body.get("variables") or {}
        token = self.headers.get("Dagster-Cloud-Api-Token")
        encoded = json.dumps(service.respond(body["query"], variables, token)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _GraphQLHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: GraphQLService) -> None:
        super().__init__(("127.0.0.1", 0), _GraphQLHandler)
        self.service = service


Respond = Callable[[str, dict[str, Any]], dict[str, Any]]


class GraphQLService:
    """Real local HTTP endpoint standing in for the Dagster Cloud GraphQL API."""

    def __init__(self, respond: Respond) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []
        self._respond = respond
        self._httpd = _GraphQLHTTPServer(self)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def respond(self, query: str, variables: dict[str, Any], token: str | None) -> dict[str, Any]:
        self.calls.append((query, variables, token))
        return self._respond(query, variables)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> GraphQLService:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._httpd.shutdown()
        self._thread.join()
        self._httpd.server_close()


class StopBackfillService:
    """Dagster Cloud state machine for the stop-historical-backfill script."""

    def __init__(self, schedule_status: dict[str, str], runs: list[dict[str, str]]) -> None:
        self.schedule_status = schedule_status
        self.runs = runs
        self.stop_calls: list[dict[str, Any]] = []
        self.terminated_run_ids: list[str] = []
        self._canceling_seen: set[str] = set()

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if "query Repositories" in query:
            return {"data": self._repositories()}
        if "query Schedules" in query:
            return {"data": self._schedules()}
        if "mutation StopSchedule" in query:
            self.stop_calls.append(variables)
            name = variables["origin"].removeprefix("origin:")
            self.schedule_status[name] = "STOPPED"
            return {"data": {"stopRunningSchedule": {"__typename": "StopScheduleSuccess"}}}
        if "mutation TerminateRun" in query:
            self.terminated_run_ids.append(variables["runId"])
            self.runs = [run for run in self.runs if run["runId"] != variables["runId"]]
            return {"data": {"terminateRun": {"__typename": "TerminateRunSuccess"}}}
        if "query Runs" in query:
            return {"data": self._runs(variables["filter"]["pipelineName"])}
        raise AssertionError(f"Unexpected GraphQL operation: {query.splitlines()[0]}")

    def _repositories(self) -> dict[str, Any]:
        return {
            "repositoriesOrError": {
                "__typename": "RepositoryConnection",
                "nodes": [
                    {"name": "other_news", "location": {"name": "other-location"}},
                    {"name": "romanian_news", "location": {"name": "test-location"}},
                ],
            }
        }

    def _schedules(self) -> dict[str, Any]:
        return {
            "schedulesOrError": {
                "__typename": "Schedules",
                "results": [
                    {
                        "id": f"selector:{name}",
                        "name": name,
                        "scheduleState": {"id": f"origin:{name}", "status": status},
                    }
                    for name, status in sorted(self.schedule_status.items())
                ],
            }
        }

    def _runs(self, job: str) -> dict[str, Any]:
        results = []
        for run in self.runs:
            if run["job"] != job:
                continue
            if run["status"] == "CANCELING":
                if run["runId"] in self._canceling_seen:
                    continue
                self._canceling_seen.add(run["runId"])
            results.append({"runId": run["runId"], "status": run["status"]})
        return {"runsOrError": {"__typename": "Runs", "results": results}}


def test_stop_workflow_stops_historical_schedules_and_terminates_runs() -> None:
    service = StopBackfillService(
        schedule_status={
            "hourly_registered_feed_poll": "RUNNING",
            "scheduled_weekly_status": "RUNNING",
            "scheduled_historical_weekly_status": "STOPPED",
            "scheduled_archive_page_backfill": "RUNNING",
            "scheduled_archive_article_capture": "STOPPED",
            "scheduled_retrospective_analysis": "RUNNING",
        },
        runs=[
            {"runId": "run-active", "job": "retrospective_analysis_pilot", "status": "STARTED"},
            {"runId": "run-canceling", "job": "archive_page_backfill_job", "status": "CANCELING"},
        ],
    )
    with GraphQLService(service) as graphql:
        result = _run_script(
            "stop-historical-backfill.yml", "Stop historical schedules", graphql.url
        )

    assert result.returncode == 0, result.stderr
    assert service.stop_calls == [
        {
            "origin": "origin:scheduled_archive_page_backfill",
            "selector": "selector:scheduled_archive_page_backfill",
        },
        {
            "origin": "origin:scheduled_retrospective_analysis",
            "selector": "selector:scheduled_retrospective_analysis",
        },
    ]
    assert service.terminated_run_ids == ["run-active"]
    assert service.schedule_status == {
        "hourly_registered_feed_poll": "RUNNING",
        "scheduled_weekly_status": "RUNNING",
        **{name: "STOPPED" for name in HISTORICAL_SCHEDULES},
    }
    assert "scheduled_historical_weekly_status: STOPPED" in result.stdout
    assert "hourly_registered_feed_poll: RUNNING" in result.stdout
    assert "scheduled_weekly_status: RUNNING" in result.stdout
    assert "Termination requested: retrospective_analysis_pilot run-active" in result.stdout
    assert "No historical runs active" in result.stdout
    assert {token for _, _, token in graphql.calls} == {"test-token"}


def test_stop_workflow_refuses_to_run_when_a_historical_schedule_is_missing() -> None:
    service = StopBackfillService(
        schedule_status={
            name: "STOPPED"
            for name in HISTORICAL_SCHEDULES
            if name != "scheduled_retrospective_analysis"
        },
        runs=[],
    )
    with GraphQLService(service) as graphql:
        result = _run_script(
            "stop-historical-backfill.yml", "Stop historical schedules", graphql.url
        )

    assert result.returncode != 0
    assert "Historical schedules missing" in result.stderr
    assert service.stop_calls == []


def test_status_workflow_reports_run_durations_and_pending_runs() -> None:
    completed: dict[str, str | float | None] = {
        "runId": "run-completed",
        "status": "SUCCESS",
        "startTime": 1695900000.0,
        "endTime": 1695900012.0,
    }
    catalog: dict[str, list[dict[str, str | float | None]]] = {
        job: [dict(completed)] for job in HISTORICAL_JOBS if job != "weekly_status_refresh"
    }
    catalog["weekly_status_refresh"] = [
        dict(completed),
        {"runId": "run-pending", "status": "QUEUED", "startTime": None, "endTime": None},
    ]

    def respond(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        assert "RecentRuns" in query
        job = variables["filter"]["pipelineName"]
        return {"data": {"runsOrError": {"__typename": "Runs", "results": catalog[job]}}}

    with GraphQLService(respond) as graphql:
        result = _run_script(
            "historical-backfill-status.yml", "List recent Dagster runs", graphql.url
        )

    assert result.returncode == 0, result.stderr
    assert {variables["filter"]["pipelineName"] for _, variables, _ in graphql.calls} == set(
        HISTORICAL_JOBS
    )
    assert {token for _, _, token in graphql.calls} == {"test-token"}
    for job in HISTORICAL_JOBS:
        assert f"{job}:" in result.stdout
    started = datetime.fromtimestamp(1695900000, UTC).isoformat()
    ended = datetime.fromtimestamp(1695900012, UTC).isoformat()
    assert f"run-completed SUCCESS start={started} end={ended} seconds=12.0" in result.stdout
    assert "run-pending QUEUED start=pending end=pending seconds=None" in result.stdout


def test_status_workflow_fails_when_the_run_query_errors() -> None:
    def respond(_query: str, _variables: dict[str, Any]) -> dict[str, Any]:
        return {"errors": [{"message": "boom"}]}

    with GraphQLService(respond) as graphql:
        result = _run_script(
            "historical-backfill-status.yml", "List recent Dagster runs", graphql.url
        )

    assert result.returncode != 0
    assert "Dagster run query failed" in result.stderr
