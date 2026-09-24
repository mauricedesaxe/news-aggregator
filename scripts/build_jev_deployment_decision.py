#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/jev-research-v1/jev-research-snapshot-v1.json"
OUTPUT_JSON = ROOT / "artifacts/jev-research-v1/jev-deployment-decision-v1.json"
OUTPUT_MARKDOWN = ROOT / "artifacts/jev-research-v1/jev-deployment-decision-v1.md"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_snapshot(root: Path) -> dict[str, object]:
    value = cast(
        object,
        json.loads((root / "artifacts/jev-research-v1/jev-research-snapshot-v1.json").read_bytes()),
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a research snapshot object")
    snapshot = cast(dict[str, object], value)
    if snapshot.get("schema_version") != "jev-research-snapshot/v1":
        raise ValueError("Research snapshot schema changed")
    return snapshot


def _objects(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Expected object list for {label}")
    items = cast(list[object], value)
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Expected object list for {label}")
    return cast(list[dict[str, object]], value)


def _verify_decision_inputs(snapshot: dict[str, object]) -> None:
    products = {
        cast(str, item["product"]): item
        for item in _objects(snapshot.get("product_relevance"), "product relevance")
    }
    news_targets = _objects(products["news"].get("targets"), "News relevance targets")
    news_jev = next(item for item in news_targets if item.get("target_id") == "typesafe-jev")
    job_targets = _objects(products["job_finder"].get("targets"), "Job targets")
    job_jev = next(item for item in job_targets if item.get("target_id") == "jev-1.13.0")
    aspects = {
        cast(str, item["concern"]): item
        for item in _objects(snapshot.get("news_aspects"), "News aspects")
    }
    context = cast(dict[str, object], snapshot["context_policy"])
    roles = {
        cast(str, item["role"]): item for item in _objects(context.get("roles"), "context roles")
    }
    if not (
        news_jev.get("deployment_eligible") is True
        and job_jev.get("deployment_eligible") is False
        and roles["relevance"].get("status") == "conditional"
        and roles["tier"].get("status") == "unsafe"
        and all(
            aspects[concern].get("deployment_conclusion") == "not_established"
            for concern in ("grouping", "ranking", "tier", "confidence", "daily_theme")
        )
        and cast(dict[str, object], snapshot["sentiment"]).get("status") == "not_evaluable"
    ):
        raise ValueError("Research evidence no longer supports the registered dispositions")


def build_decision(root: Path = ROOT) -> dict[str, object]:
    snapshot = _load_snapshot(root)
    _verify_decision_inputs(snapshot)
    source_path = root / "artifacts/jev-research-v1/jev-research-snapshot-v1.json"
    return {
        "schema_version": "jev-deployment-decision/v1",
        "issue_id": "news-gh9",
        "external_ref": "gh-9",
        "decision_scope": "prompt-class deployment disposition; no production behavior change",
        "default": "retain_incumbent_when_evidence_is_not_directly_comparable",
        "dispositions": [
            {
                "product": "news",
                "aspect": "relevance",
                "prompt_class": "direct_binary_classification",
                "disposition": "guarded_shadow_candidate",
                "implementation_ticket": "news-gh9",
                "implementation_ticket_external_ref": "gh-9",
                "basis": (
                    "Jev alone passed the registered News relevance quality gate; "
                    "production relevance states were below the local guard, but reviewed "
                    "context headroom and affected-production quality remain unavailable."
                ),
            },
            {
                "product": "job_finder",
                "aspect": "relevance",
                "prompt_class": "multi_criterion_binary_classification",
                "disposition": "retain_incumbent",
                "implementation_ticket": None,
                "basis": "Every tested target failed the registered direct-suite gate; the proposed atomic architecture is untested.",
            },
            {
                "product": "news",
                "aspect": "grouping",
                "prompt_class": "pairwise_relation_judgment",
                "disposition": "research_only",
                "implementation_ticket": None,
                "basis": "Pair judgments do not establish a complete partition.",
            },
            {
                "product": "news",
                "aspect": "ranking",
                "prompt_class": "pairwise_relation_judgment",
                "disposition": "research_only",
                "implementation_ticket": None,
                "basis": "Pair precedence does not establish a coherent global order, and 190 production requests exceed the guard.",
            },
            {
                "product": "news",
                "aspect": "tier",
                "prompt_class": "deterministic_composition_of_binary_questions",
                "disposition": "retain_incumbent",
                "implementation_ticket": None,
                "basis": "Composed exact accuracy is weak, complete assessment generation is untested, and production context is unsafe.",
            },
            {
                "product": "news",
                "aspect": "confidence",
                "prompt_class": "direct_binary_classification",
                "disposition": "require_more_evidence",
                "implementation_ticket": None,
                "basis": "Only four immutable labeled cases exist.",
            },
            {
                "product": "news",
                "aspect": "daily_theme",
                "prompt_class": "pairwise_relation_judgment",
                "disposition": "research_only",
                "implementation_ticket": None,
                "basis": "Pair judgments do not establish transitivity, partition validity, or report usefulness.",
            },
            {
                "product": "news",
                "aspect": "sentiment",
                "prompt_class": "four_class_classification",
                "disposition": "retain_incumbent",
                "implementation_ticket": None,
                "basis": "No immutable human-reviewed article or group sentiment labels exist.",
            },
            {
                "product": "cross_product",
                "aspect": "structured_extraction",
                "prompt_class": "structured_extraction",
                "disposition": "retain_incumbent",
                "implementation_ticket": None,
                "basis": "No comparable labeled structured-extraction evaluation was completed.",
            },
            {
                "product": "cross_product",
                "aspect": "prose_generation",
                "prompt_class": "prose_generation",
                "disposition": "retain_incumbent",
                "implementation_ticket": None,
                "basis": "No comparable human-reviewed prose quality evaluation was completed.",
            },
        ],
        "accepted_rollout": {
            "implementation_ticket": "news-gh9",
            "implementation_ticket_external_ref": "gh-9",
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
        },
        "source": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": _sha256(source_path.read_bytes()),
        },
        "contains_report_or_article_text": False,
        "contains_provider_output": False,
        "contains_credentials": False,
        "changes_production_behavior": False,
    }


def _markdown(decision: dict[str, object]) -> str:
    lines = [
        "# Jev deployment decision",
        "",
        "This artifact records prompt-class dispositions. It does not change production behavior.",
        "",
        "| Product | Aspect | Prompt class | Disposition | Implementation | Basis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in _objects(decision["dispositions"], "dispositions"):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["product"]),
                    str(item["aspect"]),
                    str(item["prompt_class"]),
                    str(item["disposition"]),
                    "N/A"
                    if item["implementation_ticket"] is None
                    else str(item["implementation_ticket"]),
                    str(item["basis"]),
                )
            )
            + " |"
        )
    rollout = cast(dict[str, object], decision["accepted_rollout"])
    lines.extend(
        (
            "",
            "## Guarded rollout",
            "",
            (
                f'- Ticket: `{rollout["implementation_ticket"]}` '
                f'(`{rollout["implementation_ticket_external_ref"]}`)'
            ),
            f'- Mode: `{rollout["mode"]}`',
            f'- Preflight: {rollout["preflight"]}',
            f'- Fallback: {rollout["fallback"]}',
            f'- Promotion gate: {rollout["promotion_gate"]}',
            f'- Rollback: {rollout["rollback"]}',
            "- Evidence is never clipped or chunked, and an unchanged rejected request is never retried.",
            "",
            "## Source",
            "",
            f'- `{cast(dict[str, object], decision["source"])["path"]}`: `{cast(dict[str, object], decision["source"])["sha256"]}`',
            "",
        )
    )
    return "\n".join(lines)


def write_reports(
    decision: dict[str, object],
    json_path: Path = OUTPUT_JSON,
    markdown_path: Path = OUTPUT_MARKDOWN,
) -> None:
    _ = json_path.write_text(
        json.dumps(decision, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    _ = markdown_path.write_text(_markdown(decision))


def main() -> None:
    write_reports(build_decision())


if __name__ == "__main__":
    main()
