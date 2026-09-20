# pyright: reportAny=false
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

from romanian_news.analysis.groups.models import GroupSummary

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/inventory_jev_production_context.py"
SPEC = importlib.util.spec_from_file_location("inventory_jev_production_context", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


def _request(day: str, characters: int, suffix: str) -> dict[str, object]:
    return {
        "concern": "grouping",
        "day": day,
        "report_version_id": "a" * 64,
        "selection_ids": [f"article:{suffix}"],
        "source_version_ids": ["b" * 64],
        "question_id": "same_group",
        "question_digest": "c" * 64,
        "state_digest": suffix * 64,
        "state_characters": characters,
    }


def test_calibration_uses_maximum_observed_ratio_and_ceil() -> None:
    calibrations = inventory.derive_calibrations(
        {
            "concerns": [
                {
                    "concern": "grouping",
                    "requests": [
                        {"input_tokens": 10, "state_characters": 4},
                        {"input_tokens": 8, "state_characters": 2},
                    ],
                }
            ]
        }
    )

    grouping = calibrations["grouping"]
    assert grouping is not None
    assert grouping["max_observed_input_tokens"] == 8
    assert grouping["max_observed_state_characters"] == 2
    assert grouping["max_observed_input_tokens_per_state_character"] == 4.0
    assert inventory.estimate_input_tokens(3, grouping) == 12
    assert calibrations["relevance"] is None
    assert inventory.estimate_input_tokens(3, calibrations["relevance"]) is None


def test_aggregation_reports_percentiles_days_and_both_threshold_kinds() -> None:
    calibration = {
        "method": "test",
        "source_request_count": 1,
        "max_observed_input_tokens": 1,
        "max_observed_state_characters": 2,
        "max_observed_input_tokens_per_state_character": 0.5,
    }
    requests = [
        _request("2026-09-01", 50, "1"),
        _request("2026-09-01", 75, "2"),
        _request("2026-09-02", 100, "3"),
    ]

    result = inventory.summarize_concern(
        "grouping",
        requests,
        calibration,
        approximate_character_limit=100,
        token_limit=50,
    )

    assert result["state_character_statistics"] == {
        "count": 3,
        "p50": 75,
        "p95": 100,
        "p99": 100,
        "max": 100,
        "thresholds": [
            {"percent": 50, "value": 50, "count_at_or_above": 3, "share_at_or_above": 1.0},
            {
                "percent": 75,
                "value": 75,
                "count_at_or_above": 2,
                "share_at_or_above": 0.666667,
            },
            {
                "percent": 90,
                "value": 90,
                "count_at_or_above": 1,
                "share_at_or_above": 0.333333,
            },
            {
                "percent": 100,
                "value": 100,
                "count_at_or_above": 1,
                "share_at_or_above": 0.333333,
            },
        ],
    }
    assert result["estimated_input_token_statistics"]["max"] == 50
    assert result["estimated_input_token_statistics"]["thresholds"][-1]["count_at_or_above"] == 1
    assert [item["day"] for item in result["daily_statistics"]] == [
        "2026-09-01",
        "2026-09-02",
    ]


def test_report_writing_is_deterministic_and_contains_no_state_text(tmp_path: Path) -> None:
    request = _request("2026-09-01", 50, "1")
    concern = inventory.summarize_concern(
        "grouping",
        [request],
        None,
        approximate_character_limit=100,
        token_limit=50,
    )
    report = {
        "schema_version": "jev-production-context-inventory/v1",
        "concerns": [concern],
        "evidence_gaps": [{"concern": "all", "code": "test-gap", "detail": "Known gap."}],
    }
    json_path = tmp_path / "inventory.json"
    markdown_path = tmp_path / "inventory.md"
    raw_path = tmp_path / "inventory-requests.json.gz"

    inventory.write_reports(report, json_path, markdown_path, raw_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes(), raw_path.read_bytes())
    inventory.write_reports(report, json_path, markdown_path, raw_path)

    assert (json_path.read_bytes(), markdown_path.read_bytes(), raw_path.read_bytes()) == first
    parsed = json.loads(json_path.read_text())
    assert "requests" not in parsed["concerns"][0]
    raw = json.loads(gzip.decompress(raw_path.read_bytes()))
    stored_request = raw["concerns"][0]["requests"][0]
    assert "state" not in stored_request
    assert stored_request["estimated_input_tokens"] is None
    assert "accepted cluster article pairs" in markdown_path.read_text()
    assert "Known gap." in markdown_path.read_text()


def test_summary_parser_accepts_json_arrays_at_the_boundary() -> None:
    summary = GroupSummary(
        title_ro="Titlu suficient",
        summary_ro="Rezumat suficient.",
        key_points_ro=("Punct suficient.",),
        disagreements_ro=(),
        cited_article_version_ids=("d" * 64,),
    )
    content = json.dumps(
        {"group_id": "e" * 64, "summary": summary.model_dump(mode="json")}
    ).encode()

    group_id, parsed = inventory._read_summary(content)

    assert group_id == "e" * 64
    assert parsed == summary
