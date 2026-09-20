#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple, cast

import psycopg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/jev-research-v1/report-versions-v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/jev-research-v1/news-report-model-cost-v1.json"
CATALOG_MIGRATION = ROOT / "src/romanian_news/catalog/migrations/0001_initial.sql"
MODEL_OPERATIONS = {
    "news.relevance",
    "news.relevance.v3",
    "news.embed",
    "news.summarize_group",
    "news.score_group_sentiment",
    "news.construct_daily_themes",
    "news.assess_daily_subjects",
}

ROOT_QUERY = """
WITH requested(report_version_id) AS (
    SELECT jsonb_array_elements_text(%s::jsonb)
)
SELECT
    requested.report_version_id,
    artifact.kind AS report_kind,
    version.produced_by_run_id AS report_run_id,
    run.operation_key AS report_operation,
    run.status AS report_run_status,
    run.completed_at,
    EXISTS (
        SELECT 1 FROM run_outputs output
        WHERE output.run_id = version.produced_by_run_id
          AND output.artifact_version_id = version.id
    ) AS registered_as_run_output
FROM requested
LEFT JOIN artifact_versions version ON version.id = requested.report_version_id
LEFT JOIN artifacts artifact ON artifact.id = version.artifact_id
LEFT JOIN runs run ON run.id = version.produced_by_run_id
ORDER BY requested.report_version_id COLLATE "C"
"""

ACCOUNTING_QUERY = """
WITH RECURSIVE
requested(report_version_id) AS (
    SELECT jsonb_array_elements_text(%s::jsonb)
),
lineage(report_version_id, artifact_version_id) AS (
    SELECT report_version_id, report_version_id FROM requested
    UNION
    SELECT lineage.report_version_id, input.artifact_version_id
    FROM lineage
    JOIN artifact_versions parent ON parent.id = lineage.artifact_version_id
    JOIN run_inputs input ON input.run_id = parent.produced_by_run_id
    WHERE input.role <> 'prior_output'
)
SELECT
    lineage.report_version_id,
    lineage.artifact_version_id,
    artifact.kind AS artifact_kind,
    producer.operation_key AS producer_operation,
    model_call.operation_key AS model_stage,
    model_call.model,
    model_call.input_tokens,
    model_call.output_tokens,
    model_call.cost_usd::text AS cost_usd,
    model_call.latency_ms,
    model_call.response_count
FROM lineage
JOIN artifact_versions version ON version.id = lineage.artifact_version_id
JOIN artifacts artifact ON artifact.id = version.artifact_id
LEFT JOIN runs producer ON producer.id = version.produced_by_run_id
LEFT JOIN news_model_calls model_call
    ON model_call.artifact_version_id = lineage.artifact_version_id
ORDER BY lineage.report_version_id COLLATE "C", lineage.artifact_version_id COLLATE "C"
"""
ROOT_COLUMNS = (
    "report_version_id",
    "report_kind",
    "report_run_id",
    "report_operation",
    "report_run_status",
    "completed_at",
    "registered_as_run_output",
)
ACCOUNTING_COLUMNS = (
    "report_version_id",
    "artifact_version_id",
    "artifact_kind",
    "producer_operation",
    "model_stage",
    "model",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
    "response_count",
)


class Arguments(NamedTuple):
    input: Path
    output: Path


def aggregate(
    reports: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows_by_report: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    references_by_artifact: dict[str, set[str]] = defaultdict(set)
    model_rows_by_artifact: dict[str, Mapping[str, object]] = {}
    for row in rows:
        report_id = cast(str, row["report_version_id"])
        artifact_id = cast(str, row["artifact_version_id"])
        rows_by_report[report_id].append(row)
        if row.get("model") is not None:
            references_by_artifact[artifact_id].add(report_id)
            previous = model_rows_by_artifact.setdefault(artifact_id, row)
            if _accounting_identity(previous) != _accounting_identity(row):
                raise ValueError(f"Accounting changed across references to {artifact_id}")

    report_results: list[dict[str, object]] = []
    for report in reports:
        report_id = cast(str, report["report_version_id"])
        report_rows = rows_by_report.get(report_id)
        if not report_rows:
            raise ValueError(f"No lineage returned for report {report_id}")
        if len({cast(str, row["artifact_version_id"]) for row in report_rows}) != len(report_rows):
            raise ValueError(f"Duplicate lineage node for report {report_id}")
        model_rows = [row for row in report_rows if row.get("model") is not None]
        gaps = [
            {
                "artifact_version_id": row["artifact_version_id"],
                "artifact_kind": row["artifact_kind"],
                "producer_operation": row["producer_operation"],
                "code": "missing_model_call",
            }
            for row in report_rows
            if row.get("model") is None and row.get("producer_operation") in MODEL_OPERATIONS
        ]
        report_results.append(
            {
                "day": report["day"],
                "report_version_id": report_id,
                "lineage_node_count": len(report_rows),
                "model_output_count": len(model_rows),
                **_totals(model_rows),
                "breakdown": _breakdown(model_rows),
                "accounting_gaps": gaps,
            }
        )

    unique_rows = list(model_rows_by_artifact.values())
    attributed_cost = sum(
        (Decimal(cast(str, item["cost_usd"])) for item in report_results), Decimal(0)
    )
    unique_cost = sum((Decimal(cast(str, row["cost_usd"])) for row in unique_rows), Decimal(0))
    shared_ids = {
        artifact_id
        for artifact_id, report_ids in references_by_artifact.items()
        if len(report_ids) > 1
    }
    shared_cost = sum(
        (Decimal(cast(str, model_rows_by_artifact[item]["cost_usd"])) for item in shared_ids),
        Decimal(0),
    )
    return {
        "reports": report_results,
        "portfolio": {
            "report_count": len(report_results),
            "attributed_cost_usd": _cost(attributed_cost),
            "unique_physical_cost_usd": _cost(unique_cost),
            "shared_cost_usd": _cost(shared_cost),
            "unique_model_output_count": len(unique_rows),
            "shared_model_output_count": len(shared_ids),
            "maximum_report_reference_count": max(
                (len(items) for items in references_by_artifact.values()), default=0
            ),
        },
    }


def _accounting_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        row.get(key)
        for key in (
            "model_stage",
            "model",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "latency_ms",
            "response_count",
        )
    )


def _totals(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    input_tokens = sum(cast(int, row["input_tokens"]) for row in rows)
    output_tokens = sum(cast(int, row["output_tokens"]) for row in rows)
    return {
        "response_count": sum(cast(int, row["response_count"]) for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": sum(cast(int, row["latency_ms"]) for row in rows),
        "cost_usd": _cost(sum((Decimal(cast(str, row["cost_usd"])) for row in rows), Decimal(0))),
    }


def _breakdown(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(cast(str, row["model_stage"]), cast(str, row["model"]))].append(row)
    return [
        {
            "stage": stage,
            "model": model,
            "model_output_count": len(items),
            **_totals(items),
        }
        for (stage, model), items in sorted(grouped.items())
    ]


def snapshot_cost(input_path: Path, output_path: Path) -> dict[str, object]:
    input_bytes = input_path.read_bytes()
    input_value = cast(dict[str, object], json.loads(input_bytes))
    reports = cast(list[dict[str, object]], input_value["reports"])
    report_ids = [cast(str, report["report_version_id"]) for report in reports]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("Report version IDs must be unique")
    dsn = os.getenv("NEWS_POSTGRES_DSN")
    if not dsn:
        raise ValueError("NEWS_POSTGRES_DSN is required")
    payload = json.dumps(report_ids, separators=(",", ":"))
    with psycopg.connect(dsn) as connection:
        _ = connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        root_values = cast(
            list[tuple[object, ...]],
            connection.execute(ROOT_QUERY, (payload,)).fetchall(),
        )
        roots: list[dict[str, object]] = [
            dict(zip(ROOT_COLUMNS, row, strict=True)) for row in root_values
        ]
        _validate_roots(report_ids, roots)
        accounting_values = cast(
            list[tuple[object, ...]],
            connection.execute(ACCOUNTING_QUERY, (payload,)).fetchall(),
        )
        rows: list[dict[str, object]] = [
            dict(zip(ACCOUNTING_COLUMNS, row, strict=True)) for row in accounting_values
        ]
    aggregated = aggregate(reports, rows)
    result = {
        "schema_version": "news-report-model-cost/v1",
        "accounting_basis": {
            "cost_source": "news_model_calls.cost_usd",
            "lineage": "transitive run inputs excluding prior_output",
            "per_report_deduplication_key": ["report_version_id", "artifact_version_id"],
            "portfolio_unique_key": "artifact_version_id",
            "excluded_operational_spend": (
                "rejected articles, abandoned runs, and attempts without a report-lineage artifact"
            ),
        },
        "input_manifest_sha256": _sha256(input_bytes),
        "query_sha256": _sha256(ACCOUNTING_QUERY.encode()),
        "generator_sha256": _sha256(Path(__file__).read_bytes()),
        "catalog_migration_sha256": _sha256(CATALOG_MIGRATION.read_bytes()),
        **aggregated,
        "contains_report_or_article_text": False,
        "contains_provider_output": False,
        "contains_credentials": False,
    }
    _ = output_path.write_text(
        json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def _validate_roots(report_ids: Sequence[str], roots: Sequence[Mapping[str, object]]) -> None:
    by_id = {cast(str, row["report_version_id"]): row for row in roots}
    if set(by_id) != set(report_ids):
        raise ValueError("The catalog did not return the exact requested report roots")
    for report_id in report_ids:
        row = by_id[report_id]
        if not (
            row.get("report_kind") == "news_daily_report"
            and row.get("report_run_id") is not None
            and row.get("report_operation") == "news.publish_daily"
            and row.get("report_run_status") in {"completed", "no_change"}
            and row.get("completed_at") is not None
            and row.get("registered_as_run_output") is True
        ):
            raise ValueError(f"Invalid immutable report root {report_id}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _cost(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000001")), "f")


def _arguments() -> Arguments:
    parser = argparse.ArgumentParser(description="Snapshot model cost for exact News reports")
    _ = parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parsed = vars(parser.parse_args())
    return Arguments(
        input=cast(Path, parsed["input"]),
        output=cast(Path, parsed["output"]),
    )


def main() -> None:
    arguments = _arguments()
    _ = snapshot_cost(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
