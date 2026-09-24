from __future__ import annotations

from datetime import date
from typing import Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.config import DAGSTER_CLOUD_API_TOKEN, DAGSTER_CLOUD_GRAPHQL_URL
from romanian_news.reader.repair_requests import (
    RepairReservation,
    bind_repair_run,
    mark_repair_launch_started,
    release_rejected_repair,
    release_unlaunched_repair,
    replace_terminal_repair,
    reserve_repair,
)

LOCATION_NAME = "romanian-news"
JOB_NAME = "daily_report_repair"
REPAIR_REQUEST_TAG = "news/daily_report_repair_request"
_ACTIVE_STATUSES = ("QUEUED", "NOT_STARTED", "MANAGED", "STARTING", "STARTED", "CANCELING")


class DagsterRepairError(RuntimeError):
    """Dagster could not accept or report a daily report repair."""


class DagsterRepairUnavailable(DagsterRepairError):
    """Dagster repair configuration is unavailable to the reader."""


class DagsterRepairRejected(DagsterRepairError):
    """Dagster definitively rejected a repair run before creating it."""


class RepairRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["queued", "running", "succeeded", "failed"]
    run_id: str
    run_url: str


class _GraphQLError(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: dict[str, object] | None = None
    errors: tuple[_GraphQLError, ...] = ()


class _Location(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class _RepositoryNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    location: _Location


class _RepositoryConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["RepositoryConnection"] = Field(
        default="RepositoryConnection", alias="__typename"
    )
    nodes: tuple[_RepositoryNode, ...]


class _Failure(BaseModel):
    model_config = ConfigDict(extra="allow")
    typename: str = Field(alias="__typename")
    message: str | None = None


class _RepositoriesData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repositoriesOrError: dict[str, object]


class _Tag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str


class _PartitionTags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["PartitionTags"] = Field(default="PartitionTags", alias="__typename")
    results: tuple[_Tag, ...]


class _PartitionRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["PartitionRunConfig"] = Field(
        default="PartitionRunConfig", alias="__typename"
    )
    yaml: str


class _Partition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    job_name: str = Field(alias="jobName")
    run_config: dict[str, object] = Field(alias="runConfigOrError")
    tags: dict[str, object] = Field(alias="tagsOrError")


class _Pipeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["Pipeline"] = Field(default="Pipeline", alias="__typename")
    partition: _Partition | None


class _PartitionData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipelineOrError: dict[str, object]


class _Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["Run"] = Field(default="Run", alias="__typename")
    run_id: str = Field(alias="runId")
    status: str


class _Runs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["Runs"] = Field(default="Runs", alias="__typename")
    results: tuple[_Run, ...]


class _RunsData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runsOrError: dict[str, object]


class _LaunchSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")
    typename: Literal["LaunchRunSuccess"] = Field(default="LaunchRunSuccess", alias="__typename")
    run: _Run


class _LaunchData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    launchRun: dict[str, object]


class _RunData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runOrError: dict[str, object]


_DataT = TypeVar("_DataT", bound=BaseModel)

_REPOSITORIES_QUERY = """
query DailyReportRepairRepositories {
  repositoriesOrError {
    __typename
    ... on RepositoryConnection { nodes { name location { name } } }
    ... on PythonError { message }
  }
}
"""
_PARTITION_QUERY = """
query DailyReportRepairPartition($selector: PipelineSelector!, $partitionName: String!) {
  pipelineOrError(params: $selector) {
    __typename
    ... on Pipeline {
      partition(partitionName: $partitionName) {
        name
        jobName
        runConfigOrError {
          __typename
          ... on PartitionRunConfig { yaml }
          ... on PythonError { message }
        }
        tagsOrError {
          __typename
          ... on PartitionTags { results { key value } }
          ... on PythonError { message }
        }
      }
    }
    ... on PythonError { message }
  }
}
"""
_RUNS_QUERY = """
query ActiveDailyReportRepairs($filter: RunsFilter!) {
  runsOrError(filter: $filter, limit: 1) {
    __typename
    ... on Runs { results { __typename runId status } }
    ... on PythonError { message }
  }
}
"""
_LAUNCH_MUTATION = """
mutation LaunchDailyReportRepair($executionParams: ExecutionParams!) {
  launchRun(executionParams: $executionParams) {
    __typename
    ... on LaunchRunSuccess { run { __typename runId status } }
    ... on RunConfigValidationInvalid { errors { message } }
    ... on PythonError { message }
  }
}
"""
_RUN_QUERY = """
query DailyReportRepairRun($runId: ID!) {
  runOrError(runId: $runId) {
    __typename
    ... on Run { __typename runId status }
    ... on RunNotFoundError { runId }
    ... on PythonError { message }
  }
}
"""


def request_daily_report_repair(day: date) -> RepairRun:
    """Reuse an active repair for the day or launch one with Dagster partition data."""
    active = _active_run(day)
    if active is not None:
        return _repair_lifecycle(active)
    try:
        reservation = reserve_repair(day)
        if reservation.run_id is not None:
            prior = read_daily_report_repair(reservation.run_id)
            if prior.kind in ("queued", "running"):
                return prior
            reservation = replace_terminal_repair(reservation)
        if not reservation.owns_launch:
            return _reconcile_reserved_repair(reservation)
        return _launch_reserved_repair(reservation)
    except DagsterRepairError:
        raise
    except (ResearchCatalogError, RuntimeError) as error:
        raise DagsterRepairError("Daily report repair reservation failed") from error


def _launch_reserved_repair(reservation: RepairReservation) -> RepairRun:
    day = reservation.day
    try:
        repository = _repository_name()
        selector = {
            "repositoryLocationName": LOCATION_NAME,
            "repositoryName": repository,
            "pipelineName": JOB_NAME,
        }
        partition_data = _graphql(
            _PARTITION_QUERY,
            {"selector": selector, "partitionName": day.isoformat()},
            _PartitionData,
        )
        pipeline = _parse_variant(partition_data.pipelineOrError, _Pipeline, "pipeline")
        if pipeline.partition is None:
            raise DagsterRepairError(f"Dagster partition is unavailable: {day.isoformat()}")
        partition = pipeline.partition
        run_config = _parse_variant(partition.run_config, _PartitionRunConfig, "partition config")
        tags = _parse_variant(partition.tags, _PartitionTags, "partition tags")
    except DagsterRepairError:
        release_unlaunched_repair(reservation)
        raise
    if not mark_repair_launch_started(reservation):
        return _reconcile_reserved_repair(reservation)
    try:
        launch_data = _graphql(
            _LAUNCH_MUTATION,
            {
                "executionParams": {
                    "selector": {
                        "repositoryLocationName": LOCATION_NAME,
                        "repositoryName": repository,
                        "jobName": partition.job_name,
                    },
                    "runConfigData": run_config.yaml,
                    "executionMetadata": {
                        "tags": [
                            *(tag.model_dump() for tag in tags.results),
                            {"key": REPAIR_REQUEST_TAG, "value": str(reservation.request_id)},
                        ],
                    },
                }
            },
            _LaunchData,
        )
        launch = _parse_variant(launch_data.launchRun, _LaunchSuccess, "run launch")
    except DagsterRepairRejected:
        release_rejected_repair(reservation)
        raise
    except DagsterRepairError:
        return _reconcile_reserved_repair(reservation)
    bind_repair_run(reservation, launch.run.run_id)
    return _repair_lifecycle(launch.run)


def _reconcile_reserved_repair(reservation: RepairReservation) -> RepairRun:
    data = _graphql(
        _RUNS_QUERY,
        {
            "filter": {
                "pipelineName": JOB_NAME,
                "tags": [
                    {"key": REPAIR_REQUEST_TAG, "value": str(reservation.request_id)},
                    {"key": "dagster/partition", "value": reservation.day.isoformat()},
                ],
            }
        },
        _RunsData,
    )
    runs = _parse_variant(data.runsOrError, _Runs, "runs")
    if not runs.results:
        raise DagsterRepairError("Daily report repair launch outcome is pending")
    run = runs.results[0]
    bind_repair_run(reservation, run.run_id)
    return _repair_lifecycle(run)


def read_daily_report_repair(run_id: str) -> RepairRun:
    """Read one Dagster run and map its raw status to the reader lifecycle."""
    data = _graphql(_RUN_QUERY, {"runId": run_id}, _RunData)
    run = _parse_variant(data.runOrError, _Run, "repair run")
    return _repair_lifecycle(run)


def _repository_name() -> str:
    data = _graphql(_REPOSITORIES_QUERY, {}, _RepositoriesData)
    connection = _parse_variant(data.repositoriesOrError, _RepositoryConnection, "repositories")
    matches = tuple(node for node in connection.nodes if node.location.name == LOCATION_NAME)
    if len(matches) != 1:
        raise DagsterRepairError(f"Expected one repository in location {LOCATION_NAME}")
    return matches[0].name


def _active_run(day: date) -> _Run | None:
    data = _graphql(
        _RUNS_QUERY,
        {
            "filter": {
                "pipelineName": JOB_NAME,
                "statuses": list(_ACTIVE_STATUSES),
                "tags": [{"key": "dagster/partition", "value": day.isoformat()}],
            }
        },
        _RunsData,
    )
    runs = _parse_variant(data.runsOrError, _Runs, "runs")
    return runs.results[0] if runs.results else None


def _repair_lifecycle(run: _Run) -> RepairRun:
    run_url = _run_url(run.run_id)
    if run.status in ("QUEUED", "NOT_STARTED", "MANAGED", "STARTING"):
        return RepairRun(kind="queued", run_id=run.run_id, run_url=run_url)
    if run.status in ("STARTED", "CANCELING"):
        return RepairRun(kind="running", run_id=run.run_id, run_url=run_url)
    if run.status == "SUCCESS":
        return RepairRun(kind="succeeded", run_id=run.run_id, run_url=run_url)
    if run.status in ("FAILURE", "CANCELED"):
        return RepairRun(kind="failed", run_id=run.run_id, run_url=run_url)
    raise DagsterRepairError(f"Unknown Dagster run status: {run.status}")


def _run_url(run_id: str) -> str:
    url, _ = _configuration()
    parts = urlsplit(url)
    path = parts.path.removesuffix("/graphql")
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/runs/{run_id}", "", ""))


def _graphql(query: str, variables: dict[str, object], model: type[_DataT]) -> _DataT:
    url, token = _configuration()
    try:
        response = requests.post(
            url,
            headers={"Dagster-Cloud-Api-Token": token},
            json={"query": query, "variables": variables},
            timeout=20,
        )
        response.raise_for_status()
        envelope = _Envelope.model_validate(response.json())
    except (requests.RequestException, ValueError, ValidationError) as error:
        raise DagsterRepairError("Dagster GraphQL request failed") from error
    if envelope.errors:
        raise DagsterRepairError(envelope.errors[0].message)
    if envelope.data is None:
        raise DagsterRepairError("Dagster GraphQL response has no data")
    try:
        return model.model_validate(envelope.data)
    except ValidationError as error:
        raise DagsterRepairError("Dagster GraphQL response is invalid") from error


def _parse_variant(value: dict[str, object], model: type[_DataT], label: str) -> _DataT:
    typename = value.get("__typename")
    if typename == model.model_fields["typename"].default:
        try:
            return model.model_validate(value)
        except ValidationError as error:
            raise DagsterRepairError(f"Dagster {label} response is invalid") from error
    if typename == "RunConfigValidationInvalid":
        raise DagsterRepairRejected("Dagster rejected the repair run config")
    failure = TypeAdapter(_Failure).validate_python(value)
    raise DagsterRepairError(failure.message or f"Dagster {label} failed with {failure.typename}")


def _configuration() -> tuple[str, str]:
    if not DAGSTER_CLOUD_GRAPHQL_URL or not DAGSTER_CLOUD_API_TOKEN:
        raise DagsterRepairUnavailable("Dagster repair is not configured")
    return DAGSTER_CLOUD_GRAPHQL_URL, DAGSTER_CLOUD_API_TOKEN
