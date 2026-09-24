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


def test_relevance_rollout_preserves_issue_9_safety_contract() -> None:
    decision = builder.build_decision()
    relevance = next(
        item
        for item in decision["dispositions"]
        if item["product"] == "news" and item["aspect"] == "relevance"
    )

    assert relevance["disposition"] == "guarded_shadow_candidate"
    assert relevance["implementation_ticket"] == "news-aggregator-vt6.8"
    assert decision["accepted_rollout"] == {
        "implementation_ticket": "news-aggregator-vt6.8",
        "mode": "shadow_only_incumbent_authoritative",
        "preflight": "send to Jev only when len(state) <= 70000",
        "fallback": "use incumbent with unchanged full state on guard breach or provider rejection",
        "clip_or_chunk_state": False,
        "retry_unchanged_rejected_request": False,
        "telemetry": [
            "paired Jev and incumbent verdicts",
            "provider failures",
            "immutable input identities",
            "guard and fallback reason",
            "latency and provider usage",
            "estimated Jev cost",
        ],
        "promotion_gate": (
            "A separately reviewed immutable sample preserves every positive control, "
            "has recall 1.0 and precision >= 0.80, with no unresolved accounting gaps."
        ),
        "rollback": "configuration-only return to the incumbent, exercised before promotion",
    }


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
