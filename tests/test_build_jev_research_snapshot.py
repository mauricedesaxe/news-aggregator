# pyright: reportAny=false
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/build_jev_research_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_jev_research_snapshot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_snapshot_preserves_cross_product_gate_decisions() -> None:
    snapshot = builder.build_snapshot()
    products = {item["product"]: item for item in snapshot["product_relevance"]}
    news_jev = next(
        item for item in products["news"]["targets"] if item["target_id"] == "typesafe-jev"
    )
    job_jev = next(
        item for item in products["job_finder"]["targets"] if item["target_id"] == "jev-1.13.0"
    )

    assert snapshot["overall"]["production_migration_approved"] is False
    assert news_jev["deployment_eligible"] is True
    assert job_jev["deployment_eligible"] is False


def test_snapshot_preserves_aspect_context_and_sentiment_limits() -> None:
    snapshot = builder.build_snapshot()
    aspects = {item["concern"]: item for item in snapshot["news_aspects"]}
    roles = {item["role"]: item for item in snapshot["context_policy"]["roles"]}

    assert aspects["tier"]["context_status"] == "unsafe"
    assert snapshot["sentiment"]["status"] == "not_evaluable"
    assert snapshot["context_policy"]["state_character_guard"] == 70_000
    assert snapshot["context_policy"]["formal_tokenizer_bound"] is False
    assert {key: value["status"] for key, value in roles.items()} == {
        "relevance": "conditional",
        "grouping": "safe",
        "ranking": "conditional",
        "tier": "unsafe",
        "confidence": "safe",
        "daily_theme": "safe",
    }
    models = {item["model"]: item for item in snapshot["model_catalog"]}
    assert models["jev-1.13.0"]["documented_request_token_budget"] == 64_000


def test_snapshot_output_equals_the_registered_artifact() -> None:
    assert builder.build_snapshot() == json.loads(
        (ROOT / "artifacts/jev-research-v1/jev-research-snapshot-v1.json").read_bytes()
    )


def test_snapshot_preserves_source_provenance() -> None:
    sources = builder.build_snapshot()["sources"]
    paths = [item["path"] for item in sources]

    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in sources)
    assert "artifacts/jev-research-v1/imports/manifest.json" in paths
    assert "artifacts/jev-research-v1/imports/job-current-policy-v1-raw.json.gz" in paths
    assert "artifacts/jev-research-v1/imports/news-relevance-v11-raw.json.gz" in paths


def test_reports_are_deterministic_and_exclude_sensitive_text(tmp_path: Path) -> None:
    snapshot = builder.build_snapshot()
    json_path = tmp_path / "snapshot.json"
    markdown_path = tmp_path / "snapshot.md"

    builder.write_reports(snapshot, json_path, markdown_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    builder.write_reports(snapshot, json_path, markdown_path)

    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first
    parsed = json.loads(json_path.read_text())
    assert parsed["contains_report_or_article_text"] is False
    assert parsed["contains_provider_output"] is False
    assert parsed["contains_credentials"] is False
    text = json_path.read_text() + markdown_path.read_text()
    assert "provider_request_id" not in text
    assert "response_body" not in text
    assert '"state"' not in text
    assert "NEWS_POSTGRES_DSN" not in text


def test_registered_snapshot_hashes_match_files() -> None:
    manifest = json.loads((ROOT / "artifacts/jev-aspect-v1/manifest.json").read_text())
    registration = next(
        item for item in manifest["artifacts"] if item["concern"] == "research_snapshot"
    )

    assert (
        registration["sha256"]
        == hashlib.sha256(
            (ROOT / "artifacts/jev-research-v1/jev-research-snapshot-v1.json").read_bytes()
        ).hexdigest()
    )
    assert (
        registration["report_markdown_sha256"]
        == hashlib.sha256(
            (ROOT / "artifacts/jev-research-v1/jev-research-snapshot-v1.md").read_bytes()
        ).hexdigest()
    )
    assert registration["generator_sha256"] == hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
