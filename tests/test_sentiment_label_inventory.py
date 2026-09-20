from __future__ import annotations

import json
from pathlib import Path


def test_sentiment_inventory_forbids_model_outputs_as_ground_truth() -> None:
    path = Path(__file__).parents[1] / "src" / "romanian_news" / "sentiment_label_inventory.json"
    inventory = json.loads(path.read_text())

    assert inventory["human_article_sentiment_labels"] == 0
    assert inventory["human_group_sentiment_labels"] == 0
    assert inventory["human_control_count"] == 0
    assert inventory["expected_classes"] == ["negative", "neutral", "positive", "mixed"]
    assert inventory["observed_human_class_counts"] == {}
    assert inventory["valid_for_model_quality_comparison"] is False
    assert inventory["disposition"] == "not_evaluable"
    assert "model-generated" in inventory["evidence"]["persisted_sentiment_semantics"]


def test_sentiment_result_reports_every_candidate_as_unscored() -> None:
    path = Path(__file__).parents[1] / "src" / "romanian_news" / "sentiment_evaluation_result.json"
    result = json.loads(path.read_text())

    assert result["status"] == "not_evaluable"
    assert [model["model"] for model in result["models"]] == [
        "jev-1.13.0",
        "google/gemini-2.5-flash",
        "google/gemini-3.8-flash",
        "openai/gpt-4.1-mini",
    ]
    assert all(model["trials"] == 0 and model["metrics"] is None for model in result["models"])
