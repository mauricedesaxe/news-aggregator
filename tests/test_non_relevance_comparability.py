from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

MAPPING_PATH = (
    Path(__file__).parents[1] / "src" / "romanian_news" / "non_relevance_comparability.json"
)


def test_mapping_exactly_covers_the_pinned_non_relevance_inventory() -> None:
    document = cast(dict[str, Any], json.loads(MAPPING_PATH.read_bytes()))
    concerns = cast(list[dict[str, Any]], document["concerns"])

    assert {row["concern"]: row["case_count"] for row in concerns} == {
        "grouping": 23,
        "ranking": 28,
        "tier": 25,
        "confidence": 4,
        "daily_theme": 6,
        "summary_format": 3,
        "reader_presentation": 2,
    }
    assert sum(cast(int, row["case_count"]) for row in concerns) == 91
    assert sum(cast(int, row["calls_per_model_trial"]) for row in concerns) == 132
    assert all(
        cast(str, row["follow_up"]).startswith("news-aggregator-")
        for row in concerns
        if row["classification"] == "reducible_to_frozen_binary_questions"
    )
    assert all(
        row["calls_per_model_trial"] == 0
        for row in concerns
        if row["classification"] == "not_a_model_comparison"
    )
