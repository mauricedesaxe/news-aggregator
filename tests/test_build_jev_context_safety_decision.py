# pyright: reportAny=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts import build_jev_context_safety_decision as builder

ROOT = Path(__file__).parents[1]


def test_decision_encodes_all_role_effects_and_fallbacks() -> None:
    report = cast(dict[str, Any], builder.build_decision())
    roles = {item["role"]: item for item in report["roles"]}

    assert {role: item["status"] for role, item in roles.items()} == {
        "relevance": "conditional",
        "grouping": "safe",
        "ranking": "conditional",
        "tier": "unsafe",
        "confidence": "safe",
        "daily_theme": "safe",
    }
    assert roles["tier"]["fallback"]["preflight"] == "always_use_incumbent_with_full_state"
    assert all(not item["fallback"]["clip_or_chunk_state"] for item in roles.values())
    assert all(not item["fallback"]["retry_unchanged_jev_request"] for item in roles.values())


def test_decision_preserves_evidence_strength_and_batching_scope() -> None:
    report = cast(dict[str, Any], builder.build_decision())

    assert report["policy"]["formal_tokenizer_bound"] is False
    assert report["tier_batching"]["quality_equivalence_evaluated"] is False
    assert report["tier_batching"]["state_size_reduced"] is False


def test_decision_output_equals_the_registered_artifact() -> None:
    assert builder.build_decision() == json.loads(
        (ROOT / "artifacts/jev-aspect-v1/jev-context-safety-decision-v1.json").read_bytes()
    )


def test_reports_are_deterministic_and_exclude_sensitive_text(tmp_path: Path) -> None:
    report = cast(dict[str, Any], builder.build_decision())
    json_path = tmp_path / "decision.json"
    markdown_path = tmp_path / "decision.md"

    builder.write_reports(report, json_path, markdown_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    builder.write_reports(report, json_path, markdown_path)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first
    parsed = json.loads(json_path.read_text())
    assert parsed["article_or_state_text_included"] is False
    assert parsed["full_provider_output_included"] is False
    text = json_path.read_text() + markdown_path.read_text()
    assert '"state"' not in text
    assert "response_body" not in text
    assert "formal tokenizer bound" in markdown_path.read_text()
