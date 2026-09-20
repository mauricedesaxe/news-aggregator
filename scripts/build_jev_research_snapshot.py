#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
ASPECT_ARTIFACTS = ROOT / "artifacts/jev-aspect-v1"
RESEARCH_ARTIFACTS = ROOT / "artifacts/jev-research-v1"
IMPORTS = RESEARCH_ARTIFACTS / "imports"
OUTPUT_JSON = RESEARCH_ARTIFACTS / "jev-research-snapshot-v1.json"
OUTPUT_MARKDOWN = RESEARCH_ARTIFACTS / "jev-research-snapshot-v1.md"
TARGET_ORDER = (
    "typesafe-jev",
    "openrouter-gemini-2.5-flash",
    "openrouter-gemini-3.8-flash",
)
ASPECT_ORDER = ("grouping", "ranking", "tier", "confidence", "daily_theme")
ASPECT_FILES = {
    "grouping": "news-grouping-live-v1-report.json",
    "ranking": "news-ranking-live-v1-report.json",
    "tier": "news-tier-live-v1-report.json",
    "confidence": "news-confidence-live-v1-descriptive.json",
    "daily_theme": "news-daily-theme-live-v1-report.json",
}
SOURCE_PATHS = (
    IMPORTS / "manifest.json",
    IMPORTS / "job-current-policy-v1-preregistration.json.gz",
    IMPORTS / "job-current-policy-v1-raw.json.gz",
    IMPORTS / "job-current-policy-v1-report.json.gz",
    IMPORTS / "job-current-policy-v1-report.md.gz",
    IMPORTS / "news-relevance-v11-raw.json.gz",
    IMPORTS / "news-relevance-v11-report.json.gz",
    IMPORTS / "news-relevance-v11-report.md.gz",
    ROOT / "src/romanian_news/binary_relevance_preregistration.json",
    ROOT / "src/romanian_news/jev_aspect_evaluation_protocol.json",
    ASPECT_ARTIFACTS / "news-grouping-live-v1-report.json",
    ASPECT_ARTIFACTS / "news-ranking-live-v1-report.json",
    ASPECT_ARTIFACTS / "news-tier-live-v1-report.json",
    ASPECT_ARTIFACTS / "news-tier-live-v1-composed.json",
    ASPECT_ARTIFACTS / "news-confidence-live-v1-descriptive.json",
    ASPECT_ARTIFACTS / "news-daily-theme-live-v1-report.json",
    ASPECT_ARTIFACTS / "news-daily-theme-live-v1-constraints.json",
    ROOT / "src/romanian_news/sentiment_evaluation_result.json",
    ASPECT_ARTIFACTS / "jev-context-safety-decision-v1.json",
    RESEARCH_ARTIFACTS / "news-report-model-cost-v1.json",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    value = cast(object, json.loads(content))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return cast(dict[str, object], value)


def _objects(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"Expected object list for {label}")
    items = cast(list[object], value)
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Expected object list for {label}")
    return cast(list[dict[str, object]], value)


def _one(
    values: list[dict[str, object]], key: str, expected: object, label: str
) -> dict[str, object]:
    matches = [item for item in values if item.get(key) == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected one {label} with {key}={expected}")
    return matches[0]


def _money(value: object) -> str:
    return format(Decimal(str(value)).quantize(Decimal("0.000000001")), "f")


def _quality_trials(
    trials: list[dict[str, object]], target_id: str, scored_units: int
) -> list[dict[str, object]]:
    selected = [item for item in trials if item.get("target_id") == target_id]
    if len(selected) != 3:
        raise ValueError(f"Expected three {target_id} trials")
    return [
        {
            "trial_ref": item["trial_ref"],
            "correct": item["correct"],
            "scored_units": scored_units,
            "accuracy": item["accuracy"],
            "accuracy_95_exact_interval": item["accuracy_95_exact_interval"],
            "completed": item["completed"],
            "failed": item["failed"],
            "attempts": item["attempt_count"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "cost_usd": _money(item["cost_usd"]),
            "p50_latency_ms": item["p50_latency_ms"],
            "p95_latency_ms": item["p95_latency_ms"],
        }
        for item in selected
    ]


def _news_relevance(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema_version") != "binary-relevance-analysis/v1":
        raise ValueError("News relevance report schema changed")
    identity = cast(dict[str, object], report["identity"])
    if identity.get("case_count") != 174:
        raise ValueError("News relevance case count changed")
    gates = cast(dict[str, object], report["gates"])
    gate_rows = _objects(gates.get("targets"), "news relevance gates")
    aggregates = _objects(report.get("target_aggregates"), "news relevance aggregates")
    trials = _objects(report.get("trials"), "news relevance trials")
    stability = _objects(report.get("stability"), "news relevance stability")
    targets: list[dict[str, object]] = []
    for target_id in TARGET_ORDER:
        gate = _one(gate_rows, "target_id", target_id, "news relevance gate")
        aggregate = _one(aggregates, "target_id", target_id, "news relevance aggregate")
        target_trials = [item for item in trials if item.get("target_id") == target_id]
        if len(target_trials) != 3:
            raise ValueError(f"Expected three relevance trials for {target_id}")
        targets.append(
            {
                "target_id": target_id,
                "coverage_pass": gate["coverage_pass"],
                "quality_pass": gate["quality_pass"],
                "deployment_eligible": gate["deployment_eligible"],
                "trials": [
                    {
                        "trial_ref": item["trial_ref"],
                        "tp": item["tp"],
                        "tn": item["tn"],
                        "fp": item["fp"],
                        "fn": item["fn"],
                        "correct": cast(int, item["tp"]) + cast(int, item["tn"]),
                        "case_count": 174,
                        "precision": item["precision"],
                        "precision_95_exact_ci": item["precision_95_exact_ci"],
                        "recall": item["recall"],
                        "recall_95_exact_ci": item["recall_95_exact_ci"],
                        "accuracy": item["accuracy"],
                        "completed": item["completed_cases"],
                        "failed": item["failed_cases"],
                        "requests": item["request_count"],
                        "input_tokens": item["input_tokens"],
                        "output_tokens": item["output_tokens"],
                        "cost_usd": _money(item["total_cost_usd"]),
                        "p50_latency_ms": item["p50_article_latency_ms"],
                        "p95_latency_ms": item["p95_article_latency_ms"],
                    }
                    for item in target_trials
                ],
                "total_requests": aggregate["total_request_count"],
                "total_input_tokens": aggregate["total_input_tokens"],
                "total_output_tokens": aggregate["total_output_tokens"],
                "total_cost_usd": _money(aggregate["total_cost_usd"]),
                "median_trial_p95_latency_ms": aggregate["median_trial_p95_article_latency_ms"],
                "projected_cost_per_1000_articles_usd": _money(
                    aggregate["projected_cost_per_1000_articles_usd"]
                ),
                "stability": _compact_news_relevance_stability(
                    _one(stability, "target_id", target_id, "news relevance stability")
                ),
                "cost_basis": _one(
                    _objects(identity.get("targets"), "news relevance target identities"),
                    "target_id",
                    target_id,
                    "news relevance target identity",
                )["cost_basis"],
            }
        )
    return {
        "product": "news",
        "concern": "relevance",
        "manifest_version": identity["declared_manifest_version"],
        "source_artifact_id": identity["source_artifact_id"],
        "case_count": identity["case_count"],
        "trial_count": 3,
        "repeated_trials_are_independent": False,
        "paired_comparisons": [
            {
                key: item[key]
                for key in (
                    "trial_ref",
                    "first_target_id",
                    "second_target_id",
                    "both_correct",
                    "both_wrong",
                    "first_only_correct",
                    "second_only_correct",
                    "mcnemar_two_sided_exact_p_value",
                )
            }
            for item in _objects(report.get("paired_comparisons"), "news relevance pairs")
        ],
        "pareto": report["pareto"],
        "recommended_use": (
            "Jev is eligible for guarded relevance use under the registered quality gate; "
            "retain full-state incumbent fallback above the context guard or on rejection."
        ),
        "targets": targets,
    }


def _compact_news_relevance_stability(item: dict[str, object]) -> dict[str, object]:
    return {
        "case_count": item["case_count"],
        "flip_count": item["flip_count"],
        "unanimous_case_count": item["unanimous_case_count"],
        "unanimous_verdict_rate": item["unanimous_verdict_rate"],
    }


def _job_relevance(report: dict[str, object], preregistration_bytes: bytes) -> dict[str, object]:
    if report.get("schema_version") != "binary-corpus-analysis/v1":
        raise ValueError("Job relevance report schema changed")
    preregistration = cast(dict[str, object], report["preregistration"])
    if preregistration.get("sha256") != _sha256(preregistration_bytes):
        raise ValueError("Job preregistration digest does not match the imported bytes")
    source = cast(dict[str, object], report["source"])
    identity = cast(dict[str, object], source["benchmark_identity"])
    corpora = _objects(identity.get("corpora"), "Job corpora")
    if {cast(str, item["suite"]): item["case_count"] for item in corpora} != {
        "direct": 121,
        "ats": 25,
    }:
        raise ValueError("Job corpus shape changed")
    gates = _objects(report.get("gates"), "Job gates")
    trials = _objects(report.get("trials"), "Job trials")
    aggregates = _objects(report.get("aggregates"), "Job aggregates")
    stability = _objects(report.get("stability"), "Job stability")
    targets: list[dict[str, object]] = []
    for target in _objects(identity.get("targets"), "Job target identities"):
        target_name = cast(str, target["name"])
        gate = _one(gates, "target", target_name, "Job gate")
        suites: list[dict[str, object]] = []
        for suite in ("direct", "ats"):
            suite_trials = [
                item
                for item in trials
                if item.get("target") == target_name and item.get("suite") == suite
            ]
            if len(suite_trials) != 3:
                raise ValueError(f"Expected three {target_name}/{suite} trials")
            aggregate = next(
                (
                    item
                    for item in aggregates
                    if item.get("target") == target_name and item.get("suite") == suite
                ),
                None,
            )
            if aggregate is None:
                raise ValueError(f"Missing {target_name}/{suite} aggregate")
            suites.append(
                {
                    "suite": suite,
                    "case_count_per_trial": suite_trials[0]["case_count"],
                    "trials": [
                        {
                            "trial_ref": item["trial_ref"],
                            "case_count": item["case_count"],
                            "correct": cast(int, item["case_count"])
                            - cast(int, item["false_positive_count"])
                            - cast(int, item["false_negative_count"]),
                            "false_positive_count": item["false_positive_count"],
                            "false_positive_rate": item["false_positive_rate"],
                            "false_positive_interval_95": item["false_positive_interval_95"],
                            "false_negative_count": item["false_negative_count"],
                            "false_negative_rate": item["false_negative_rate"],
                            "false_negative_interval_95": item["false_negative_interval_95"],
                            "operational_failure_count": item["operational_failure_count"],
                            "criterion_evaluations": item["criterion_count"],
                            "requests": item["request_count"],
                            "input_tokens": item["input_tokens"],
                            "output_tokens": item["output_tokens"],
                            "cost_usd": _money(item["cost_usd"]),
                            "p50_case_latency_ms": item["p50_case_latency_ms"],
                            "p95_case_latency_ms": item["p95_case_latency_ms"],
                        }
                        for item in suite_trials
                    ],
                    "criterion_evaluations": sum(
                        cast(int, item["criterion_count"]) for item in suite_trials
                    ),
                    "requests": aggregate["request_count"],
                    "input_tokens": aggregate["input_tokens"],
                    "output_tokens": aggregate["output_tokens"],
                    "cost_usd": _money(aggregate["cost_usd"]),
                    "projected_cost_per_1000_usd": aggregate["projected_cost_per_1000_usd"],
                    "p50_case_latency_ms": aggregate["p50_case_latency_ms"],
                    "p95_case_latency_ms": aggregate["p95_case_latency_ms"],
                    "stability": _compact_job_stability(
                        next(
                            item
                            for item in stability
                            if item.get("target") == target_name and item.get("suite") == suite
                        )
                    ),
                }
            )
        targets.append(
            {
                "target_id": target_name,
                "coverage_pass": gate["coverage_pass"],
                "direct_gate_pass": gate["direct_pass"],
                "ats_gate_pass": gate["ats_pass"],
                "deployment_eligible": gate["deployment_eligible"],
                "cost_basis": target["cost_basis"],
                "suites": suites,
            }
        )
    return {
        "product": "job_finder",
        "concern": "relevance",
        "benchmark_identity_digest": source["benchmark_identity_digest"],
        "case_entries": 146,
        "distinct_urls": 143,
        "trial_count": identity["trial_count"],
        "repeated_trials_are_independent": False,
        "current_policy_only": True,
        "atomic_architecture_evaluated": False,
        "paired_comparisons": [
            {
                key: item[key]
                for key in (
                    "suite",
                    "trial",
                    "trial_ref_a",
                    "trial_ref_b",
                    "target_a",
                    "target_b",
                    "both_correct_count",
                    "both_wrong_count",
                    "a_correct_b_wrong_count",
                    "a_wrong_b_correct_count",
                    "excluded_operational_count",
                    "mcnemar_exact_two_sided_p_value",
                )
            }
            for item in _objects(report.get("paired_correctness"), "Job paired correctness")
        ],
        "pareto": report["pareto"],
        "recommended_use": (
            "Retain the current Job relevance implementation; no target passed the "
            "registered direct-suite gate, and the proposed atomic architecture is untested."
        ),
        "targets": targets,
    }


def _compact_job_stability(item: dict[str, object]) -> dict[str, object]:
    return {
        "case_count": item["case_count"],
        "flip_count": item["flip_count"],
        "unanimous_verdict_count": item["unanimous_verdict_count"],
        "unanimous_verdict_rate": item["unanimous_verdict_rate"],
    }


def _news_aspects(root: Path, context: dict[str, object]) -> list[dict[str, object]]:
    artifacts = root / "artifacts/jev-aspect-v1"
    context_roles = {
        cast(str, item["role"]): item for item in _objects(context.get("roles"), "context roles")
    }
    results: list[dict[str, object]] = []
    for concern in ASPECT_ORDER:
        report = _load_json(artifacts / ASPECT_FILES[concern])
        if report.get("benchmark") != concern:
            raise ValueError(f"{concern} report identity changed")
        case_count = cast(int, report["case_count"])
        scored_units = cast(int, report["judgment_count_per_run"])
        trials = _objects(report.get("trials"), f"{concern} trials")
        aggregates = _objects(report.get("target_aggregates"), f"{concern} aggregates")
        stability = _objects(report.get("stability"), f"{concern} stability")
        targets: list[dict[str, object]] = []
        for target_id in TARGET_ORDER:
            aggregate = _one(aggregates, "target_id", target_id, f"{concern} aggregate")
            targets.append(
                {
                    "target_id": target_id,
                    "trials": _quality_trials(trials, target_id, scored_units),
                    "total_attempts": aggregate["total_attempt_count"],
                    "total_input_tokens": aggregate["total_input_tokens"],
                    "total_output_tokens": aggregate["total_output_tokens"],
                    "total_cost_usd": _money(aggregate["total_cost_usd"]),
                    "median_trial_p50_latency_ms": aggregate["median_trial_p50_latency_ms"],
                    "median_trial_p95_latency_ms": aggregate["median_trial_p95_latency_ms"],
                    "projected_cost_per_1000_scored_units_usd": _money(
                        Decimal(str(aggregate["total_cost_usd"]))
                        / Decimal(3 * scored_units)
                        * Decimal(1000)
                    ),
                    "stability": {
                        key: _one(stability, "target_id", target_id, f"{concern} stability")[key]
                        for key in (
                            "judgment_count",
                            "flip_count",
                            "unanimous_count",
                            "unanimous_rate",
                            "unavailable_count",
                        )
                    },
                }
            )
        context_role = context_roles[concern]
        result: dict[str, object] = {
            "concern": concern,
            "case_count": case_count,
            "scored_units_per_trial": scored_units,
            "trial_count": 3,
            "evidence_status": (
                "descriptive_only" if concern == "confidence" else "derived_task_only"
            ),
            "context_status": context_role["status"],
            "deployment_conclusion": "not_established",
            "limitation": report["limitation"],
            "paired_comparisons": [
                {
                    key: item[key]
                    for key in (
                        "trial_ref",
                        "first_target_id",
                        "second_target_id",
                        "paired_count",
                        "both_correct",
                        "both_wrong",
                        "first_only_correct",
                        "second_only_correct",
                        "unavailable_count",
                        "mcnemar_two_sided_exact_p_value",
                    )
                }
                for item in _objects(report.get("paired_comparisons"), f"{concern} pairs")
            ],
            "pareto": report["pareto"],
            "recommended_use": _aspect_recommendation(concern),
            "targets": targets,
        }
        if concern == "tier":
            composed = _load_json(artifacts / "news-tier-live-v1-composed.json")
            runs = _objects(composed.get("runs"), "composed tier runs")
            result["composed_metrics_by_target"] = {
                target_id: [
                    {
                        "trial_ref": item["trial_ref"],
                        **cast(dict[str, object], item["metrics"]),
                    }
                    for item in runs
                    if cast(dict[str, object], item["target"]).get("target_id") == target_id
                ]
                for target_id in TARGET_ORDER
            }
        if concern == "daily_theme":
            constraints_path = artifacts / "news-daily-theme-live-v1-constraints.json"
            constraint_value = cast(object, json.loads(constraints_path.read_bytes()))
            constraints = _objects(constraint_value, "daily theme constraints")
            result["jev_constraint_results"] = [
                {
                    "must_link_passed": item["must_link_passed"],
                    "must_link_total": item["must_link_total"],
                    "must_separate_passed": item["must_separate_passed"],
                    "must_separate_total": item["must_separate_total"],
                }
                for item in constraints
                if item.get("target_id") == "typesafe-jev"
            ]
        results.append(result)
    return results


def _aspect_recommendation(concern: str) -> str:
    return {
        "grouping": (
            "Use only as guarded pairwise evidence or in shadow evaluation; complete "
            "partition replacement is not established."
        ),
        "ranking": (
            "Use only as guarded pairwise evidence with incumbent fallback; coherent "
            "global ordering and over-guard quality are not established."
        ),
        "tier": "Retain the incumbent; composed quality is weak and production context is unsafe.",
        "confidence": "Retain the incumbent until more than four immutable labeled cases exist.",
        "daily_theme": (
            "Do not replace complete theme generation from pair judgments; transitivity, "
            "partition validity, and report usefulness are untested."
        ),
    }[concern]


def _context_summary(context: dict[str, object]) -> dict[str, object]:
    policy = cast(dict[str, object], context["policy"])
    roles = _objects(context.get("roles"), "context roles")
    return {
        "state_character_guard": policy["state_character_guard"],
        "documented_state_plus_longest_question_tokens": policy[
            "documented_state_plus_longest_question_tokens"
        ],
        "formal_tokenizer_bound": policy["formal_tokenizer_bound"],
        "fallback": "unchanged full state to incumbent; no clipping or unchanged retry",
        "roles": [
            {
                "role": item["role"],
                "status": item["status"],
                "production": item["production"],
                "minimum_observed_reviewed_token_headroom": item[
                    "minimum_observed_reviewed_token_headroom"
                ],
                "minimum_estimated_production_token_headroom": item[
                    "minimum_estimated_production_token_headroom"
                ],
                "affected_production_quality": cast(
                    dict[str, object], item["reviewed_routing_effect"]
                )["affected_production_quality"],
            }
            for item in roles
        ],
        "tier_batching": context["tier_batching"],
    }


def _research_backlog() -> list[dict[str, object]]:
    return [
        {
            "experiment": "job_atomic_policy",
            "measurement": "Run frozen atomic questions plus deterministic composition on disjoint labeled direct and ATS cases.",
            "completion_criterion": "A preregistered target passes every direct and ATS error gate in every trial.",
        },
        {
            "experiment": "news_sentiment_labels",
            "measurement": "Collect blinded, adjudicated four-class labels tied to immutable article and group versions.",
            "completion_criterion": "A frozen disjoint test set has enough labels to report per-class recall and exact intervals.",
        },
        {
            "experiment": "news_complete_structures",
            "measurement": "Evaluate grouping partitions, coherent global rankings, and complete tier assessments rather than pairwise surrogates.",
            "completion_criterion": "Registered end-to-end structure metrics pass on disjoint immutable reports.",
        },
        {
            "experiment": "over_guard_routing_quality",
            "measurement": "Label production-shaped ranking and tier cases above the 70,000-character guard and compare fallback outcomes.",
            "completion_criterion": "Quality deltas and exact intervals exist for the 190 ranking and 197 tier provider-request strata represented by the inventory.",
        },
        {
            "experiment": "tier_batch_quality",
            "measurement": "Compare separate and batched tier answers on a preregistered multi-case sample.",
            "completion_criterion": "Equivalence bounds are reported; the existing one-sample 49.1291% input reduction remains transport evidence only.",
        },
        {
            "experiment": "confidence_and_theme_coverage",
            "measurement": "Expand confidence labels and evaluate daily-theme transitivity, partition validity, and report usefulness.",
            "completion_criterion": "Confidence exceeds four cases and daily themes pass registered end-to-end usefulness metrics.",
        },
    ]


def _model_catalog(
    news_report: dict[str, object], job_report: dict[str, object], context: dict[str, object]
) -> list[dict[str, object]]:
    news_identity = cast(dict[str, object], news_report["identity"])
    news_targets = _objects(news_identity.get("targets"), "news model identities")
    job_source = cast(dict[str, object], job_report["source"])
    job_identity = cast(dict[str, object], job_source["benchmark_identity"])
    job_targets = _objects(job_identity.get("targets"), "Job model identities")
    policy = cast(dict[str, object], context["policy"])
    models: list[dict[str, object]] = []
    for target_id in TARGET_ORDER:
        news_target = _one(news_targets, "target_id", target_id, "news model identity")
        requested_model = cast(str, news_target["requested_model"])
        job_target = _one(job_targets, "requested_model", requested_model, "Job model identity")
        model: dict[str, object] = {
            "model": requested_model,
            "news_target_id": target_id,
            "news_provider": news_target["provider"],
            "job_provider": job_target["provider"],
            "execution_policy_id": news_target["execution_policy_id"],
            "news_cost_basis": news_target["cost_basis"],
            "job_cost_basis": job_target["cost_basis"],
            "comparative_roles": ["job_relevance", "news_relevance", *ASPECT_ORDER],
        }
        if target_id == "typesafe-jev":
            model["documented_request_token_budget"] = 64_000
            model["documented_state_plus_longest_question_tokens"] = policy[
                "documented_state_plus_longest_question_tokens"
            ]
            model["local_state_character_guard"] = policy["state_character_guard"]
            model["formal_tokenizer_bound"] = False
        models.append(model)
    models.append(
        {
            "model": "openai/gpt-4.1-mini",
            "news_target_id": None,
            "news_provider": "openai",
            "job_provider": None,
            "execution_policy_id": None,
            "news_cost_basis": "production model-call accounting",
            "job_cost_basis": None,
            "comparative_roles": [],
            "evidence_status": "production cost baseline and unlabeled sentiment incumbent only",
        }
    )
    return models


def _task_comparability() -> list[dict[str, object]]:
    return [
        {
            "product": "job_finder",
            "concern": "relevance",
            "relationship": "current_six_criterion_policy",
            "directly_tested": True,
            "complete_pipeline_output_tested": True,
            "limitation": "The proposed atomic architecture was not evaluated.",
        },
        {
            "product": "news",
            "concern": "relevance",
            "relationship": "shared_monolithic_prompt_model_isolation",
            "directly_tested": True,
            "complete_pipeline_output_tested": False,
            "limitation": "The benchmark isolates one shared relevance question; trials reuse 174 cases.",
        },
        *[
            {
                "product": "news",
                "concern": concern,
                "relationship": "derived_binary_decomposition",
                "directly_tested": False,
                "complete_pipeline_output_tested": False,
                "limitation": {
                    "grouping": "Pair labels do not establish a complete partition.",
                    "ranking": "Pair precedence does not establish a coherent global order.",
                    "tier": "Two binary questions do not evaluate complete assessment generation.",
                    "confidence": "Only four evidence-sufficiency labels exist.",
                    "daily_theme": "Pair labels do not establish transitivity, partition validity, or report usefulness.",
                }[concern],
            }
            for concern in ASPECT_ORDER
        ],
        {
            "product": "news",
            "concern": "sentiment",
            "relationship": "not_evaluable",
            "directly_tested": False,
            "complete_pipeline_output_tested": False,
            "limitation": "No immutable human-reviewed article or group sentiment labels exist.",
        },
    ]


def build_snapshot(root: Path = ROOT) -> dict[str, object]:
    imports = root / "artifacts/jev-research-v1/imports"
    import_manifest = _load_json(imports / "manifest.json")
    if import_manifest.get("schema_version") != "jev-research-import-manifest/v1":
        raise ValueError("Research import manifest schema changed")
    for imported in _objects(import_manifest.get("artifacts"), "research imports"):
        imported_path = imports / cast(str, imported["path"])
        compressed = imported_path.read_bytes()
        if _sha256(compressed) != imported["sha256"]:
            raise ValueError(f"Compressed import digest changed for {imported_path.name}")
        if _sha256(gzip.decompress(compressed)) != imported["decompressed_sha256"]:
            raise ValueError(f"Decompressed import digest changed for {imported_path.name}")
    job_preregistration_path = imports / "job-current-policy-v1-preregistration.json.gz"
    job_preregistration_bytes = gzip.decompress(job_preregistration_path.read_bytes())
    job_preregistration = cast(
        dict[str, object], cast(object, json.loads(job_preregistration_bytes))
    )
    if job_preregistration.get("schema_version") != ("job-model-prompt-matrix-preregistration/v1"):
        raise ValueError("Job preregistration schema changed")
    news_preregistration = _load_json(
        root / "src/romanian_news/binary_relevance_preregistration.json"
    )
    if news_preregistration.get("schema_version") != ("news-binary-relevance-preregistration/v1"):
        raise ValueError("News relevance preregistration schema changed")
    context = _load_json(root / "artifacts/jev-aspect-v1/jev-context-safety-decision-v1.json")
    cost = _load_json(root / "artifacts/jev-research-v1/news-report-model-cost-v1.json")
    cost_reports = _objects(cost.get("reports"), "production cost reports")
    accounting_gap_count = sum(
        len(_objects(item.get("accounting_gaps"), "production cost accounting gaps"))
        for item in cost_reports
    )
    sentiment = _load_json(root / "src/romanian_news/sentiment_evaluation_result.json")
    if sentiment.get("status") != "not_evaluable":
        raise ValueError("Sentiment evidence status changed")
    source_paths = tuple(
        root / path.relative_to(ROOT) if root != ROOT else path for path in SOURCE_PATHS
    )
    news_report = _load_json(imports / "news-relevance-v11-report.json.gz")
    job_report = _load_json(imports / "job-current-policy-v1-report.json.gz")
    return {
        "schema_version": "jev-research-snapshot/v1",
        "issue_id": "news-aggregator-vt6.7",
        "decision_scope": "cross-product research synthesis; no blanket production migration approval",
        "overall": {
            "production_migration_approved": False,
            "reason_codes": [
                "job_relevance_gate_failed",
                "news_tier_context_unsafe",
                "news_sentiment_not_evaluable",
                "derived_aspects_do_not_validate_complete_pipeline_outputs",
            ],
        },
        "model_catalog": _model_catalog(news_report, job_report, context),
        "task_comparability": _task_comparability(),
        "product_relevance": [
            _news_relevance(news_report),
            _job_relevance(
                job_report,
                job_preregistration_bytes,
            ),
        ],
        "news_aspects": _news_aspects(root, context),
        "sentiment": {
            "status": sentiment["status"],
            "reason": sentiment["reason"],
            "human_article_or_group_labels": 0,
            "production_recommendation": sentiment["production_recommendation"],
        },
        "context_policy": _context_summary(context),
        "current_news_cost_baseline": {
            "accounting_basis": cost["accounting_basis"],
            "portfolio": cost["portfolio"],
            "accounting_gap_count": accounting_gap_count,
            "reports": cost_reports,
        },
        "research_backlog": _research_backlog(),
        "sources": [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path.read_bytes())}
            for path in source_paths
        ],
        "contains_report_or_article_text": False,
        "contains_provider_output": False,
        "contains_credentials": False,
    }


def _markdown(snapshot: dict[str, object]) -> str:
    relevance = _objects(snapshot["product_relevance"], "product relevance")
    lines = [
        "# Jev research snapshot",
        "",
        "This snapshot preserves the completed cross-product evidence. It does not approve a blanket production migration.",
        "",
        "## Model identity and architecture",
        "",
        "| Model | News provider | Job provider | News cost basis | Job cost basis | Comparative roles |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in _objects(snapshot["model_catalog"], "model catalog"):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(model["model"]),
                    _display(model["news_provider"]),
                    _display(model["job_provider"]),
                    _display(model["news_cost_basis"]),
                    _display(model["job_cost_basis"]),
                    ", ".join(cast(list[str], model["comparative_roles"])) or "none",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Jev documents a 64,000-token request budget and a 32,000-token state-plus-longest-question budget. The local 70,000-character guard is empirical and is not a tokenizer bound.",
            "",
            "## Corpus and task comparability",
            "",
            "| Product | Concern | Relationship | Directly tested | Complete output tested | Limitation |",
            "| --- | --- | --- | --- | --- | --- |",
        )
    )
    for item in _objects(snapshot["task_comparability"], "task comparability"):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["product"]),
                    str(item["concern"]),
                    str(item["relationship"]),
                    _yes_no(item["directly_tested"]),
                    _yes_no(item["complete_pipeline_output_tested"]),
                    str(item["limitation"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Product relevance",
            "",
            "| Product | Jev status | Cases | Trials | Gate result |",
            "| --- | --- | ---: | ---: | --- |",
        )
    )
    for product in relevance:
        target_key = "target_id"
        target_value = "typesafe-jev" if product["product"] == "news" else "jev-1.13.0"
        target = _one(
            _objects(product["targets"], "relevance targets"),
            target_key,
            target_value,
            "Jev target",
        )
        lines.append(
            "".join(
                (
                    f'| {product["product"]} | ',
                    f'{"eligible" if target["deployment_eligible"] else "not eligible"} | ',
                    f'{product.get("case_count", product.get("case_entries"))} | ',
                    f'{product["trial_count"]} | ',
                    f'{"pass" if target["deployment_eligible"] else "fail"} |',
                )
            )
        )
    lines.extend(
        (
            "",
            "### News relevance trials",
            "",
            "| Model | Correct by trial | Precision by trial (95% exact CI) | Recall by trial (95% exact CI) | Requests | Tokens in/out | Cost | Median trial p95 | Eligible Pareto |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    news = _one(relevance, "product", "news", "News relevance")
    news_pareto = cast(dict[str, object], news["pareto"])
    eligible_frontier = cast(list[str], news_pareto["eligible_targets_frontier"])
    for target in _objects(news["targets"], "News relevance targets"):
        trials = _objects(target["trials"], "News relevance trials")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(target["target_id"]),
                    ", ".join(f'{item["correct"]}/174' for item in trials),
                    ", ".join(
                        f'{_percent(item["precision"])} {_interval(item["precision_95_exact_ci"])}'
                        for item in trials
                    ),
                    ", ".join(
                        f'{_percent(item["recall"])} {_interval(item["recall_95_exact_ci"])}'
                        for item in trials
                    ),
                    str(target["total_requests"]),
                    f'{target["total_input_tokens"]}/{target["total_output_tokens"]}',
                    f'${target["total_cost_usd"]}',
                    _number(target["median_trial_p95_latency_ms"]),
                    _yes_no(target["target_id"] in eligible_frontier),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### Job relevance trials",
            "",
            "| Model | Suite | Correct by trial | FP by trial (95% exact CI) | FN by trial (95% exact CI) | Requests | Tokens in/out | Cost | Aggregate p95 | Gate |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    job = _one(relevance, "product", "job_finder", "Job relevance")
    for target in _objects(job["targets"], "Job relevance targets"):
        for suite in _objects(target["suites"], "Job relevance suites"):
            trials = _objects(suite["trials"], "Job relevance trials")
            gate_key = "direct_gate_pass" if suite["suite"] == "direct" else "ats_gate_pass"
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(target["target_id"]),
                        str(suite["suite"]),
                        ", ".join(f'{item["correct"]}/{item["case_count"]}' for item in trials),
                        ", ".join(
                            f'{item["false_positive_count"]} {_interval(item["false_positive_interval_95"])}'
                            for item in trials
                        ),
                        ", ".join(
                            f'{item["false_negative_count"]} {_interval(item["false_negative_interval_95"])}'
                            for item in trials
                        ),
                        str(suite["requests"]),
                        f'{suite["input_tokens"]}/{suite["output_tokens"]}',
                        f'${suite["cost_usd"]}',
                        _number(suite["p95_case_latency_ms"]),
                        "pass" if target[gate_key] else "fail",
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## News aspects",
            "",
            "| Concern | Evidence | Context | Deployment conclusion | Limitation |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for aspect in _objects(snapshot["news_aspects"], "news aspects"):
        lines.append(
            "".join(
                (
                    f'| {aspect["concern"]} | {aspect["evidence_status"]} | ',
                    f'{aspect["context_status"]} | {aspect["deployment_conclusion"]} | ',
                    f'{aspect["limitation"]} |',
                )
            )
        )
    sentiment = cast(dict[str, object], snapshot["sentiment"])
    lines.append(
        f'| sentiment | {sentiment["status"]} | N/A | retain incumbent | {sentiment["reason"]} |'
    )
    lines.extend(
        (
            "",
            "### Derived aspect model evidence",
            "",
            "| Concern | Model | Correct by trial (95% exact CI) | Failures | Tokens in/out | Cost | Cost/1,000 scored units | Median trial p50/p95 | Stability | Pareto |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        )
    )
    for aspect in _objects(snapshot["news_aspects"], "news aspects"):
        pareto = cast(dict[str, object], aspect["pareto"])
        frontier = cast(list[str], pareto["frontier"])
        for target in _objects(aspect["targets"], "aspect targets"):
            trials = _objects(target["trials"], "aspect trials")
            stability = cast(dict[str, object], target["stability"])
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(aspect["concern"]),
                        str(target["target_id"]),
                        ", ".join(
                            f'{item["correct"]}/{item["scored_units"]} {_interval(item["accuracy_95_exact_interval"])}'
                            for item in trials
                        ),
                        str(sum(cast(int, item["failed"]) for item in trials)),
                        f'{target["total_input_tokens"]}/{target["total_output_tokens"]}',
                        f'${target["total_cost_usd"]}',
                        f'${target["projected_cost_per_1000_scored_units_usd"]}',
                        f'{_number(target["median_trial_p50_latency_ms"])}/{_number(target["median_trial_p95_latency_ms"])}',
                        f'{stability["unanimous_count"]}/{stability["judgment_count"]}',
                        _yes_no(target["target_id"] in frontier),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "### Paired significance and recommendations",
            "",
            "| Concern | Paired comparisons | p < 0.05 | Minimum p | Recommended use |",
            "| --- | ---: | ---: | ---: | --- |",
        )
    )
    recommendation_rows = [news, job, *_objects(snapshot["news_aspects"], "news aspects")]
    for item in recommendation_rows:
        comparisons = _objects(item["paired_comparisons"], "paired comparisons")
        p_key = (
            "mcnemar_exact_two_sided_p_value"
            if item.get("product") == "job_finder"
            else "mcnemar_two_sided_exact_p_value"
        )
        p_values = [float(cast(float | str, row[p_key])) for row in comparisons]
        lines.append(
            "".join(
                (
                    f'| {item.get("product", "news")}/{item["concern"]} | ',
                    f"{len(comparisons)} | {sum(value < 0.05 for value in p_values)} | ",
                    f'{min(p_values):.9f} | {item["recommended_use"]} |',
                )
            )
        )
    lines.append(f'| news/sentiment | 0 | 0 | N/A | {sentiment["production_recommendation"]} |')
    lines.extend(
        (
            "",
            "## Context routing",
            "",
            "The operational guard is `len(state) <= 70,000`. It is not a formal tokenizer bound. Over-guard or provider-rejected requests retain the full state and use the incumbent; evidence is never clipped or chunked.",
            "",
            "| Role | Status | Affected question requests | Affected provider requests |",
            "| --- | --- | ---: | ---: |",
        )
    )
    context = cast(dict[str, object], snapshot["context_policy"])
    for role in _objects(context["roles"], "context roles"):
        production = cast(dict[str, object], role["production"])
        lines.append(
            "".join(
                (
                    f'| {role["role"]} | {role["status"]} | ',
                    f'{production["affected_question_requests"]}/{production["question_requests"]} | ',
                    f'{production["affected_provider_requests"]}/{production["provider_requests"]} |',
                )
            )
        )
    cost = cast(dict[str, object], snapshot["current_news_cost_baseline"])
    portfolio = cast(dict[str, object], cost["portfolio"])
    lines.extend(
        (
            "",
            "## Current News cost baseline",
            "",
            f'- Exact immutable reports: {portfolio["report_count"]}',
            f'- Attributed cost: ${portfolio["attributed_cost_usd"]}',
            f'- Unique physical cost: ${portfolio["unique_physical_cost_usd"]}',
            f'- Accounting gaps: {cost["accounting_gap_count"]}',
            "",
            "## Research backlog",
            "",
        )
    )
    for item in _objects(snapshot["research_backlog"], "research backlog"):
        lines.append(
            f'- `{item["experiment"]}`: {item["measurement"]} Completion: {item["completion_criterion"]}'
        )
    lines.extend(("", "## Sources", ""))
    for source in _objects(snapshot["sources"], "sources"):
        lines.append(f'- `{source["path"]}`: `{source["sha256"]}`')
    return "\n".join(lines) + "\n"


def _display(value: object) -> str:
    return "N/A" if value is None else str(value)


def _yes_no(value: object) -> str:
    return "yes" if value is True else "no"


def _percent(value: object) -> str:
    return f"{float(cast(float | str, value)) * 100:.2f}%"


def _number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _interval(value: object) -> str:
    interval = cast(dict[str, object], value)
    return f'[{_percent(interval["lower"])}, {_percent(interval["upper"])}]'


def write_reports(
    snapshot: dict[str, object],
    json_path: Path = OUTPUT_JSON,
    markdown_path: Path = OUTPUT_MARKDOWN,
) -> None:
    _ = json_path.write_text(
        json.dumps(snapshot, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    _ = markdown_path.write_text(_markdown(snapshot))


def main() -> None:
    write_reports(build_snapshot())


if __name__ == "__main__":
    main()
