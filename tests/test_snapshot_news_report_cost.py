# pyright: reportAny=false
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from scripts import snapshot_news_report_cost as snapshot

ROOT = Path(__file__).parents[1]


def test_aggregation_separates_attributed_and_unique_cost() -> None:
    reports = (
        {"day": "2026-09-19", "report_version_id": "a" * 64},
        {"day": "2026-09-20", "report_version_id": "b" * 64},
    )
    rows = (
        _row("a" * 64, "c" * 64, "news.relevance", "model-a", 100, 10, "0.10"),
        _row("a" * 64, "d" * 64, "news.embed", "model-b", 50, 0, "0.01"),
        _row("b" * 64, "c" * 64, "news.relevance", "model-a", 100, 10, "0.10"),
        _row("b" * 64, "e" * 64, "news.summarize_group", "model-a", 80, 20, "0.20"),
    )

    result = cast(dict[str, Any], snapshot.aggregate(reports, rows))

    assert [item["cost_usd"] for item in result["reports"]] == [
        "0.110000000",
        "0.300000000",
    ]
    assert result["portfolio"] == {
        "report_count": 2,
        "attributed_cost_usd": "0.410000000",
        "unique_physical_cost_usd": "0.310000000",
        "shared_cost_usd": "0.100000000",
        "unique_model_output_count": 3,
        "shared_model_output_count": 1,
        "maximum_report_reference_count": 2,
    }


def test_aggregation_reports_model_accounting_gaps() -> None:
    reports = ({"day": "2026-09-20", "report_version_id": "a" * 64},)
    rows = (
        {
            **_row("a" * 64, "c" * 64, None, None, None, None, None),
            "producer_operation": "news.relevance",
            "artifact_kind": "news_relevance",
        },
    )

    result = cast(dict[str, Any], snapshot.aggregate(reports, rows))

    assert result["reports"][0]["accounting_gaps"] == [
        {
            "artifact_version_id": "c" * 64,
            "artifact_kind": "news_relevance",
            "producer_operation": "news.relevance",
            "code": "missing_model_call",
        }
    ]


def _row(
    report_version_id: str,
    artifact_version_id: str,
    model_stage: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: str | None,
) -> dict[str, object]:
    return {
        "report_version_id": report_version_id,
        "artifact_version_id": artifact_version_id,
        "artifact_kind": "news_relevance",
        "producer_operation": model_stage,
        "model_stage": model_stage,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_ms": 100 if model is not None else None,
        "response_count": 1 if model is not None else None,
    }
