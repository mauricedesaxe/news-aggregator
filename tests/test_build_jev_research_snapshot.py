# pyright: reportAny=false
from __future__ import annotations

import hashlib
import importlib.util
import json
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
    assert [item["correct"] for item in news_jev["trials"]] == [146, 147, 146]
    assert [item["recall"] for item in news_jev["trials"]] == [1.0, 1.0, 1.0]
    assert job_jev["coverage_pass"] is True
    assert job_jev["direct_gate_pass"] is False
    assert job_jev["deployment_eligible"] is False
    assert products["job_finder"]["atomic_architecture_evaluated"] is False


def test_snapshot_preserves_aspect_context_and_sentiment_limits() -> None:
    snapshot = builder.build_snapshot()
    aspects = {item["concern"]: item for item in snapshot["news_aspects"]}
    roles = {item["role"]: item for item in snapshot["context_policy"]["roles"]}

    assert aspects["grouping"]["evidence_status"] == "derived_task_only"
    assert aspects["ranking"]["deployment_conclusion"] == "not_established"
    assert aspects["tier"]["context_status"] == "unsafe"
    assert [
        item["exact_matches"]
        for item in aspects["tier"]["composed_metrics_by_target"]["typesafe-jev"]
    ] == [
        13,
        13,
        13,
    ]
    assert aspects["confidence"]["evidence_status"] == "descriptive_only"
    assert aspects["daily_theme"]["jev_constraint_results"][0] == {
        "must_link_passed": 6,
        "must_link_total": 13,
        "must_separate_passed": 14,
        "must_separate_total": 14,
    }
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


def test_snapshot_preserves_comparison_dimensions_and_task_boundaries() -> None:
    snapshot = builder.build_snapshot()
    products = {item["product"]: item for item in snapshot["product_relevance"]}
    aspects = {item["concern"]: item for item in snapshot["news_aspects"]}
    models = {item["model"]: item for item in snapshot["model_catalog"]}
    comparability = {
        (item["product"], item["concern"]): item for item in snapshot["task_comparability"]
    }

    news_jev = next(
        item for item in products["news"]["targets"] if item["target_id"] == "typesafe-jev"
    )
    assert news_jev["trials"][0]["precision_95_exact_ci"]["lower"] > 0
    assert news_jev["stability"]["case_count"] == 174
    assert len(products["news"]["paired_comparisons"]) == 9
    assert products["news"]["pareto"]["eligible_targets_frontier"] == ["typesafe-jev"]

    job_jev = next(
        item for item in products["job_finder"]["targets"] if item["target_id"] == "jev-1.13.0"
    )
    direct = next(item for item in job_jev["suites"] if item["suite"] == "direct")
    assert float(direct["trials"][0]["false_positive_interval_95"]["upper"]) > 0
    assert direct["trials"][0]["operational_failure_count"] == 0
    assert direct["projected_cost_per_1000_usd"] == "0.3338214380165289256198347107"

    grouping_jev = next(
        item for item in aspects["grouping"]["targets"] if item["target_id"] == "typesafe-jev"
    )
    assert grouping_jev["projected_cost_per_1000_scored_units_usd"] == "0.116694261"
    assert grouping_jev["stability"]["flip_count"] == 1
    assert len(aspects["grouping"]["paired_comparisons"]) == 9
    assert models["jev-1.13.0"]["documented_request_token_budget"] == 64_000
    assert models["openai/gpt-4.1-mini"]["comparative_roles"] == []
    assert comparability[("news", "ranking")]["relationship"] == ("derived_binary_decomposition")
    assert comparability[("job_finder", "relevance")]["complete_pipeline_output_tested"] is True


def test_snapshot_preserves_cost_and_source_provenance() -> None:
    snapshot = builder.build_snapshot()
    cost = snapshot["current_news_cost_baseline"]
    sources = snapshot["sources"]

    assert cost["portfolio"]["report_count"] == 7
    assert cost["portfolio"]["unique_physical_cost_usd"] == "2.032867430"
    assert cost["accounting_gap_count"] == 0
    paths = [item["path"] for item in sources]
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all(len(item["sha256"]) == 64 for item in sources)
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
