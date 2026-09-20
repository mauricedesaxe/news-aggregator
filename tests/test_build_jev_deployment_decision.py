# pyright: reportAny=false
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/build_jev_deployment_decision.py"
SPEC = importlib.util.spec_from_file_location("build_jev_deployment_decision", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_decision_covers_every_evaluated_and_unsupported_prompt_class() -> None:
    decision = builder.build_decision()
    dispositions = {(item["product"], item["aspect"]): item for item in decision["dispositions"]}

    assert len(dispositions) == 10
    assert dispositions[("news", "relevance")]["disposition"] == ("guarded_shadow_candidate")
    assert dispositions[("news", "relevance")]["implementation_ticket"] == ("news-aggregator-vt6.8")
    assert dispositions[("job_finder", "relevance")]["disposition"] == ("retain_incumbent")
    assert dispositions[("news", "tier")]["disposition"] == "retain_incumbent"
    assert dispositions[("news", "confidence")]["disposition"] == ("require_more_evidence")
    assert dispositions[("cross_product", "structured_extraction")]["implementation_ticket"] is None
    assert dispositions[("cross_product", "prose_generation")]["disposition"] == (
        "retain_incumbent"
    )


def test_accepted_rollout_has_quality_telemetry_fallback_and_rollback() -> None:
    rollout = builder.build_decision()["accepted_rollout"]

    assert rollout["mode"] == "shadow_only_incumbent_authoritative"
    assert "len(state) <= 70000" in rollout["preflight"]
    assert "unchanged full state" in rollout["fallback"]
    assert rollout["clip_or_chunk_state"] is False
    assert rollout["retry_unchanged_rejected_request"] is False
    assert "paired Jev and incumbent verdicts" in rollout["telemetry"]
    assert "recall 1.0" in rollout["promotion_gate"]
    assert "configuration-only" in rollout["rollback"]


def test_decision_is_deterministic_and_contains_no_sensitive_text(tmp_path: Path) -> None:
    decision = builder.build_decision()
    json_path = tmp_path / "decision.json"
    markdown_path = tmp_path / "decision.md"

    builder.write_reports(decision, json_path, markdown_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    builder.write_reports(decision, json_path, markdown_path)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first
    parsed = json.loads(json_path.read_text())
    assert parsed["changes_production_behavior"] is False
    assert parsed["contains_report_or_article_text"] is False
    assert parsed["contains_provider_output"] is False
    assert parsed["contains_credentials"] is False
    text = json_path.read_text() + markdown_path.read_text()
    assert "provider_request_id" not in text
    assert '"state"' not in text
    assert "NEWS_POSTGRES_DSN" not in text


def test_registered_decision_hashes_match_files() -> None:
    manifest = json.loads((ROOT / "artifacts/jev-aspect-v1/manifest.json").read_text())
    registration = next(
        item for item in manifest["artifacts"] if item["concern"] == "deployment_decision"
    )

    assert (
        registration["sha256"]
        == hashlib.sha256(
            (ROOT / "artifacts/jev-research-v1/jev-deployment-decision-v1.json").read_bytes()
        ).hexdigest()
    )
    assert (
        registration["report_markdown_sha256"]
        == hashlib.sha256(
            (ROOT / "artifacts/jev-research-v1/jev-deployment-decision-v1.md").read_bytes()
        ).hexdigest()
    )
    assert registration["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
