# pyright: reportAny=false
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/build_jev_context_safety_decision.py"
SPEC = importlib.util.spec_from_file_location("build_jev_context_safety_decision", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_decision_encodes_all_role_effects_and_fallbacks() -> None:
    report = builder.build_decision()
    roles = {item["role"]: item for item in report["roles"]}

    assert tuple(roles) == builder.ROLE_ORDER
    assert {role: item["status"] for role, item in roles.items()} == {
        "relevance": "conditional",
        "grouping": "safe",
        "ranking": "conditional",
        "tier": "unsafe",
        "confidence": "safe",
        "daily_theme": "safe",
    }
    assert roles["ranking"]["production"]["affected_question_requests"] == 190
    assert roles["ranking"]["production"]["maximum_naive_tail_clip_characters"] == 70_493
    assert roles["tier"]["production"]["affected_question_requests"] == 394
    assert roles["tier"]["production"]["affected_provider_requests"] == 197
    assert roles["tier"]["production"]["provider_requests"] == 228
    assert roles["tier"]["fallback"]["preflight"] == "always_use_incumbent_with_full_state"
    assert all(not item["fallback"]["clip_or_chunk_state"] for item in roles.values())
    assert all(not item["fallback"]["retry_unchanged_jev_request"] for item in roles.values())


def test_decision_preserves_evidence_strength_and_batching_scope() -> None:
    report = builder.build_decision()
    roles = {item["role"]: item for item in report["roles"]}

    assert report["policy"]["formal_tokenizer_bound"] is False
    assert roles["relevance"]["minimum_estimated_production_token_headroom"] is None
    assert roles["ranking"]["minimum_estimated_production_token_headroom"] == -28_927
    assert roles["ranking"]["reviewed_routing_effect"] == {
        "affected_requests": 0,
        "accuracy_delta_percentage_points": "0.000000",
        "affected_production_quality": "not_estimable",
    }
    assert report["tier_batching"]["separate_input_tokens"] == 11_482
    assert report["tier_batching"]["batched_input_tokens"] == 5_841
    assert report["tier_batching"]["quality_equivalence_evaluated"] is False
    assert report["tier_batching"]["state_size_reduced"] is False


def test_reports_are_deterministic_and_exclude_sensitive_text(tmp_path: Path) -> None:
    report = builder.build_decision()
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
