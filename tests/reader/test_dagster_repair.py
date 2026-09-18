from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from romanian_news.reader import dagster_repair

DAY = date(2026, 9, 15)
RUN_ID = "repair-run"


class Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def _payload(data: dict[str, object]) -> dict[str, object]:
    return {"data": data}


def _repository() -> dict[str, object]:
    return _payload(
        {
            "repositoriesOrError": {
                "__typename": "RepositoryConnection",
                "nodes": [{"name": "__repository__", "location": {"name": "romanian-news"}}],
            }
        }
    )


def _partition() -> dict[str, object]:
    return _payload(
        {
            "pipelineOrError": {
                "__typename": "Pipeline",
                "partition": {
                    "name": DAY.isoformat(),
                    "jobName": "daily_report_repair",
                    "runConfigOrError": {
                        "__typename": "PartitionRunConfig",
                        "yaml": "execution: {}",
                    },
                    "tagsOrError": {
                        "__typename": "PartitionTags",
                        "results": [
                            {"key": "dagster/partition", "value": DAY.isoformat()},
                            {
                                "key": "dagster/partition_set",
                                "value": "daily_report_repair_partition_set",
                            },
                        ],
                    },
                },
            }
        }
    )


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dagster_repair, "DAGSTER_CLOUD_GRAPHQL_URL", "https://example.test/prod/graphql"
    )
    monkeypatch.setattr(dagster_repair, "DAGSTER_CLOUD_API_TOKEN", "secret")


def test_repeated_repair_requests_reuse_the_active_partition_run(monkeypatch) -> None:
    _configure(monkeypatch)
    calls: list[dict[str, Any]] = []

    def post(_url: str, **kwargs: Any) -> Response:
        body = kwargs["json"]
        calls.append(body)
        query = body["query"]
        if "RepairRepositories" in query:
            return Response(_repository())
        if "RepairPartition" in query:
            return Response(_partition())
        if "ActiveDailyReportRepairs" in query:
            return Response(
                _payload(
                    {
                        "runsOrError": {
                            "__typename": "Runs",
                            "results": [
                                {"__typename": "Run", "runId": RUN_ID, "status": "STARTED"}
                            ],
                        }
                    }
                )
            )
        raise AssertionError("launchRun must not be called when a matching run is active")

    monkeypatch.setattr(dagster_repair.requests, "post", post)

    first = dagster_repair.request_daily_report_repair(DAY)
    assert len(calls) == 1
    second = dagster_repair.request_daily_report_repair(DAY)

    assert first.kind == second.kind == "running"
    assert first.run_id == second.run_id == RUN_ID
    assert len(calls) == 2
    assert not any("LaunchDailyReportRepair" in call["query"] for call in calls)


def test_repair_request_launches_with_resolved_partition_config_and_tags(monkeypatch) -> None:
    _configure(monkeypatch)
    calls: list[dict[str, Any]] = []

    def post(_url: str, **kwargs: Any) -> Response:
        body = kwargs["json"]
        calls.append(body)
        query = body["query"]
        if "RepairRepositories" in query:
            return Response(_repository())
        if "RepairPartition" in query:
            return Response(_partition())
        if "ActiveDailyReportRepairs" in query:
            return Response(_payload({"runsOrError": {"__typename": "Runs", "results": []}}))
        if "LaunchDailyReportRepair" in query:
            return Response(
                _payload(
                    {
                        "launchRun": {
                            "__typename": "LaunchRunSuccess",
                            "run": {"__typename": "Run", "runId": RUN_ID, "status": "QUEUED"},
                        }
                    }
                )
            )
        raise AssertionError(query)

    monkeypatch.setattr(dagster_repair.requests, "post", post)

    result = dagster_repair.request_daily_report_repair(DAY)

    assert result.kind == "queued"
    launch = calls[-1]["variables"]["executionParams"]
    assert launch["selector"] == {
        "repositoryLocationName": "romanian-news",
        "repositoryName": "__repository__",
        "jobName": "daily_report_repair",
    }
    assert launch["runConfigData"] == "execution: {}"
    assert launch["executionMetadata"]["tags"][0] == {
        "key": "dagster/partition",
        "value": DAY.isoformat(),
    }


@pytest.mark.parametrize(
    ("status", "kind"),
    [("QUEUED", "queued"), ("STARTED", "running"), ("SUCCESS", "succeeded"), ("FAILURE", "failed")],
)
def test_run_status_maps_to_the_small_repair_lifecycle(monkeypatch, status: str, kind: str) -> None:
    _configure(monkeypatch)
    monkeypatch.setattr(
        dagster_repair.requests,
        "post",
        lambda *_args, **_kwargs: Response(
            _payload(
                {
                    "runOrError": {
                        "__typename": "Run",
                        "runId": RUN_ID,
                        "status": status,
                    }
                }
            )
        ),
    )

    assert dagster_repair.read_daily_report_repair(RUN_ID).kind == kind


def test_missing_dagster_token_fails_only_the_repair_call(monkeypatch) -> None:
    monkeypatch.setattr(dagster_repair, "DAGSTER_CLOUD_API_TOKEN", None)

    with pytest.raises(dagster_repair.DagsterRepairUnavailable, match="not configured"):
        dagster_repair.request_daily_report_repair(DAY)


def test_graphql_failure_typename_is_not_parsed_as_success(monkeypatch) -> None:
    _configure(monkeypatch)

    def post(_url: str, **kwargs: Any) -> Response:
        if "ActiveDailyReportRepairs" in kwargs["json"]["query"]:
            return Response(_payload({"runsOrError": {"__typename": "Runs", "results": []}}))
        return Response(
            _payload(
                {
                    "repositoriesOrError": {
                        "__typename": "PythonError",
                        "message": "location failed",
                    }
                }
            )
        )

    monkeypatch.setattr(dagster_repair.requests, "post", post)

    with pytest.raises(dagster_repair.DagsterRepairError, match="location failed"):
        dagster_repair.request_daily_report_repair(DAY)
