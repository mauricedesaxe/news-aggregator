#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations, pairwise
from pathlib import Path
from statistics import median
from typing import Any, cast

from romanian_news.binary_relevance_evaluation import (
    BinaryRelevanceEvaluationResult,
    BinaryRelevanceRunResult,
)

EXPECTED_TARGETS = 3
EXPECTED_TRIALS = 3
EXPECTED_CASES = 174
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
PREREGISTRATION_PATH = (
    Path(__file__).parents[1] / "src" / "romanian_news" / "binary_relevance_preregistration.json"
)


@dataclass(frozen=True)
class Arguments:
    input_path: Path
    json_output: Path
    markdown_output: Path
    preregistration: Path


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(
        description="Analyze a completed three-target binary relevance evaluation."
    )
    _ = parser.add_argument("--input", required=True, type=Path)
    _ = parser.add_argument("--json-output", required=True, type=Path)
    _ = parser.add_argument("--markdown-output", required=True, type=Path)
    _ = parser.add_argument(
        "--preregistration",
        type=Path,
        default=PREREGISTRATION_PATH,
    )
    parsed = parser.parse_args(argv)
    return Arguments(
        input_path=cast(Path, parsed.input),
        json_output=cast(Path, parsed.json_output),
        markdown_output=cast(Path, parsed.markdown_output),
        preregistration=cast(Path, parsed.preregistration),
    )


def exact_clopper_pearson(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Clopper-Pearson counts must satisfy 0 <= successes <= total")
    lower = 0.0 if successes == 0 else _inverse_binomial_survival(ALPHA / 2, successes, total)
    upper = (
        1.0
        if successes == total
        else _inverse_binomial_survival(1 - ALPHA / 2, successes + 1, total)
    )
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "successes": successes,
        "total": total,
        "lower": lower,
        "upper": upper,
    }


def exact_mcnemar_p_value(first_only_correct: int, second_only_correct: int) -> float:
    if first_only_correct < 0 or second_only_correct < 0:
        raise ValueError("McNemar disagreement counts cannot be negative")
    disagreements = first_only_correct + second_only_correct
    if disagreements == 0:
        return 1.0
    smaller = min(first_only_correct, second_only_correct)
    tail = sum(math.comb(disagreements, index) for index in range(smaller + 1))
    return min(1.0, 2 * tail / (2**disagreements))


def analyze(
    source_path: Path,
    preregistration_path: Path = PREREGISTRATION_PATH,
) -> dict[str, Any]:
    source_content = source_path.read_bytes()
    result = BinaryRelevanceEvaluationResult.model_validate_json(source_content, strict=True)
    preregistration_content = preregistration_path.read_bytes()
    preregistration = _object(json.loads(preregistration_content), "preregistration")
    _require_complete_registered_evaluation(result, preregistration)

    runs = {(run.target.target_id, run.trial_ref): run for run in result.runs}
    trials = [_trial_analysis(run) for run in result.runs]
    target_aggregates = [
        _target_aggregate(
            target.target_id,
            [runs[(target.target_id, trial_ref)] for trial_ref in result.trial_refs],
        )
        for target in result.targets
    ]
    paired = _paired_comparisons(result, runs)
    stability = _stability_analysis(result, runs)
    gates = _gate_analysis(preregistration, target_aggregates)
    pareto = _pareto_analysis(target_aggregates, gates)

    return {
        "schema_version": "binary-relevance-analysis/v1",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": hashlib.sha256(source_content).hexdigest(),
        },
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": hashlib.sha256(preregistration_content).hexdigest(),
            "schema_version": preregistration["schema_version"],
        },
        "identity": {
            "source_artifact_id": result.source_artifact_id,
            "declared_manifest_version": result.declared_manifest_version,
            "registered_source_content_digest": _object(
                preregistration["held_out_source"], "held_out_source"
            )["source_content_digest"],
            "question_id": result.question_id,
            "question_digest": result.question_digest,
            "targets": [target.model_dump(mode="json") for target in result.targets],
            "trial_refs": list(result.trial_refs),
            "case_count": len(result.case_ids),
        },
        "aggregation": {
            "method": "descriptive repeated-run summaries",
            "independence_assumption": False,
            "note": "Trials share the same 174 cases and are not treated as independent samples.",
        },
        "trials": trials,
        "target_aggregates": target_aggregates,
        "paired_comparisons": paired,
        "stability": stability,
        "gates": gates,
        "pareto": pareto,
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = _object(report["source"], "source")
    identity = _object(report["identity"], "identity")
    lines = [
        "# Binary relevance evaluation analysis",
        "",
        f"- Source: `{source['path']}`",
        f"- Source SHA-256: `{source['sha256']}`",
        f"- Source artifact: `{identity['source_artifact_id']}`",
        f"- Question: `{identity['question_id']}` (`{identity['question_digest']}`)",
        f"- Shape: 3 targets x 3 trials x {identity['case_count']} cases",
        "- Aggregation: descriptive only; repeated trials use the same cases and are not independent.",
        "",
        "## Registered gates",
        "",
        "| Target | Coverage | Quality | Deployment eligible | Failures |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    gates = _object(report["gates"], "gates")
    for target in _list(gates["targets"], "gate targets"):
        row = _object(target, "gate target")
        failures = "; ".join(cast(list[str], row["failures"])) or "none"
        lines.append(
            f"| `{row['target_id']}` | {_mark(row['coverage_pass'])} | "
            f"{_mark(row['quality_pass'])} | {_mark(row['deployment_eligible'])} | "
            f"{failures} |"
        )

    lines.extend(
        [
            "",
            "## Trial results",
            "",
            "| Target | Trial | TP | TN | FP | FN | Precision (95% exact CI) | Recall (95% exact CI) | Controls (95% exact CI) | Requests | Tokens | Cost USD | Cost/article | Cost/1000 | Attempt p50/p95 ms | Article p50/p95 ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for trial in _list(report["trials"], "trials"):
        row = _object(trial, "trial")
        lines.append(
            f"| `{row['target_id']}` | `{row['trial_ref']}` | {row['tp']} | {row['tn']} | "
            f"{row['fp']} | {row['fn']} | {_metric_ci(row, 'precision')} | "
            f"{_metric_ci(row, 'recall')} | {_metric_ci(row, 'positive_control_preservation')} | "
            f"{row['request_count']} | {row['total_tokens']} | {row['total_cost_usd']:.9f} | "
            f"{row['cost_per_article_usd']:.9f} | {row['projected_cost_per_1000_articles_usd']:.6f} | "
            f"{row['p50_attempt_latency_ms']:.1f}/{row['p95_attempt_latency_ms']:.1f} | "
            f"{row['p50_article_latency_ms']:.1f}/{row['p95_article_latency_ms']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Target aggregates",
            "",
            "These are descriptive summaries across repeated runs. No pooled confidence interval is reported.",
            "",
            "| Target | Errors by trial | Mean cost/article USD | Projected cost/1000 USD | Median trial article p95 ms | Total requests | Total tokens | Total cost USD |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for aggregate in _list(report["target_aggregates"], "target aggregates"):
        row = _object(aggregate, "target aggregate")
        errors = "/".join(str(value) for value in cast(list[int], row["quality_errors_by_trial"]))
        lines.append(
            f"| `{row['target_id']}` | {errors} | {row['mean_cost_per_article_usd']:.9f} | "
            f"{row['projected_cost_per_1000_articles_usd']:.6f} | "
            f"{row['median_trial_p95_article_latency_ms']:.1f} | {row['total_request_count']} | "
            f"{row['total_tokens']} | {row['total_cost_usd']:.9f} |"
        )

    lines.extend(["", "## Paired correctness", ""])
    for comparison in _list(report["paired_comparisons"], "paired comparisons"):
        row = _object(comparison, "paired comparison")
        lines.append(
            f"- `{row['trial_ref']}` `{row['first_target_id']}` vs `{row['second_target_id']}`: "
            f"both correct {row['both_correct']}, first-only correct {row['first_only_correct']}, "
            f"second-only correct {row['second_only_correct']}, both wrong {row['both_wrong']}; "
            f"exact McNemar p={row['mcnemar_two_sided_exact_p_value']:.6g}."
        )

    lines.extend(["", "## Cross-trial stability", ""])
    for item in _list(report["stability"], "stability"):
        row = _object(item, "stability target")
        lines.extend(
            [
                f"### `{row['target_id']}`",
                "",
                f"- Unanimous verdict rate: {row['unanimous_verdict_rate']:.6f} "
                f"({row['unanimous_case_count']}/{row['case_count']}).",
                f"- Adjacent-trial verdict flips: {row['flip_count']}.",
                f"- Stable false positives: {_ids(row['stable_false_positive_ids'])}.",
                f"- Stable false negatives: {_ids(row['stable_false_negative_ids'])}.",
                f"- Unstable IDs: {_ids(row['unstable_ids'])}.",
                "",
            ]
        )

    pareto = _object(report["pareto"], "pareto")
    lines.extend(
        [
            "## Pareto analysis",
            "",
            "Dimensions are minimized: worst-trial quality errors, mean cost/article, and median trial article p95 latency.",
            "",
            f"- All-target frontier: {_ids(pareto['all_targets_frontier'])}.",
            f"- Deployment-eligible frontier: {_ids(pareto['eligible_targets_frontier'])}.",
            "",
            "| Target | Worst trial errors | Mean cost/article USD | Median trial p95 ms | Gate eligible | Non-dominated among all |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in _list(pareto["targets"], "pareto targets"):
        row = _object(item, "pareto target")
        lines.append(
            f"| `{row['target_id']}` | {row['worst_trial_quality_errors']} | "
            f"{row['mean_cost_per_article_usd']:.9f} | "
            f"{row['median_trial_p95_article_latency_ms']:.1f} | "
            f"{_mark(row['deployment_eligible'])} | {_mark(row['non_dominated_among_all_targets'])} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
) -> None:
    if json_output.resolve() == markdown_output.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    json_content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    _atomic_write(json_output, json_content)
    _atomic_write(markdown_output, render_markdown(report).encode())


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = analyze(arguments.input_path, arguments.preregistration)
    write_reports(report, arguments.json_output, arguments.markdown_output)
    return 0


def _require_complete_registered_evaluation(
    result: BinaryRelevanceEvaluationResult,
    preregistration: dict[str, Any],
) -> None:
    held_out = _object(preregistration["held_out_source"], "held_out_source")
    policy = _object(preregistration["monolithic_policy"], "monolithic_policy")
    registered_models = {
        _object(item, "registered model")["target"]: _object(item, "registered model")
        for item in _list(preregistration["models"], "models")
    }
    policy_digests = _object(policy["execution_policy_digests"], "execution policy digests")
    if (
        result.source_artifact_id != held_out["source_artifact_id"]
        or result.declared_manifest_version != held_out["declared_manifest_version"]
        or result.question_id != policy["question_id"]
        or result.question_digest != policy["question_digest"]
    ):
        raise ValueError("Evaluation source or question identity does not match preregistration")
    if len(result.targets) != EXPECTED_TARGETS or len(result.trial_refs) != EXPECTED_TRIALS:
        raise ValueError("Evaluation must contain exactly 3 targets and 3 trials")
    if len(result.case_ids) != EXPECTED_CASES or held_out["relevance_case_count"] != EXPECTED_CASES:
        raise ValueError("Evaluation and preregistration must contain exactly 174 cases")
    if set(registered_models) != {target.target_id for target in result.targets}:
        raise ValueError("Evaluation targets do not exactly match preregistration")
    for target in result.targets:
        registered = registered_models[target.target_id]
        if (
            target.provider != registered["provider"]
            or target.requested_model != registered["requested_model"]
            or target.execution_policy_digest != policy_digests[target.target_id]
        ):
            raise ValueError(f"Target identity does not match preregistration: {target.target_id}")
    for run in result.runs:
        if len(run.cases) != EXPECTED_CASES:
            raise ValueError(
                f"Run does not contain 174 cases: {run.target.target_id} {run.trial_ref}"
            )
        if run.metrics.completed_cases != EXPECTED_CASES or run.metrics.failed_cases != 0:
            raise ValueError(
                f"Run is not complete with zero failures: {run.target.target_id} {run.trial_ref}"
            )
        if any(case.status != "completed" for case in run.cases):
            raise ValueError(f"Run contains a failed case: {run.target.target_id} {run.trial_ref}")
        if any(case.actual_model != run.target.requested_model for case in run.cases):
            raise ValueError(
                f"Run actual model differs from requested model: {run.target.target_id}"
            )


def _trial_analysis(run: BinaryRelevanceRunResult) -> dict[str, Any]:
    tp = sum(case.expected and case.verdict is True for case in run.cases)
    tn = sum(not case.expected and case.verdict is False for case in run.cases)
    fp = sum(not case.expected and case.verdict is True for case in run.cases)
    fn = sum(case.expected and case.verdict is False for case in run.cases)
    controls = [case for case in run.cases if case.control and case.expected]
    preserved_controls = sum(case.verdict is True for case in controls)
    attempts = [attempt for case in run.cases for attempt in case.attempts]
    attempt_latencies = [attempt.latency_ms for attempt in attempts]
    article_latencies = [case.wall_latency_ms for case in run.cases]
    cost = sum((case.cost_usd for case in run.cases), Decimal(0))
    precision_total = tp + fp
    return {
        "target_id": run.target.target_id,
        "trial_ref": run.trial_ref,
        "completed_cases": len(run.cases),
        "failed_cases": 0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _ratio(tp + tn, len(run.cases)),
        "precision": _ratio(tp, precision_total),
        "precision_95_exact_ci": exact_clopper_pearson(tp, precision_total),
        "recall": _ratio(tp, tp + fn),
        "recall_95_exact_ci": exact_clopper_pearson(tp, tp + fn),
        "positive_control_preservation": _ratio(preserved_controls, len(controls)),
        "positive_control_preservation_95_exact_ci": exact_clopper_pearson(
            preserved_controls, len(controls)
        ),
        "request_count": len(attempts),
        "input_tokens": sum(case.input_tokens for case in run.cases),
        "output_tokens": sum(case.output_tokens for case in run.cases),
        "total_tokens": sum(case.input_tokens + case.output_tokens for case in run.cases),
        "total_cost_usd": float(cost),
        "cost_per_article_usd": float(cost / len(run.cases)),
        "projected_cost_per_1000_articles_usd": float(cost / len(run.cases) * 1000),
        "p50_attempt_latency_ms": _percentile(attempt_latencies, 0.50),
        "p95_attempt_latency_ms": _percentile(attempt_latencies, 0.95),
        "p50_article_latency_ms": _percentile(article_latencies, 0.50),
        "p95_article_latency_ms": _percentile(article_latencies, 0.95),
        "false_positive_ids": [
            case.case_id for case in run.cases if not case.expected and case.verdict is True
        ],
        "false_negative_ids": [
            case.case_id for case in run.cases if case.expected and case.verdict is False
        ],
    }


def _target_aggregate(
    target_id: str,
    runs: list[BinaryRelevanceRunResult],
) -> dict[str, Any]:
    trial_rows = [_trial_analysis(run) for run in runs]
    costs = [cast(float, row["cost_per_article_usd"]) for row in trial_rows]
    p95_articles = [cast(float, row["p95_article_latency_ms"]) for row in trial_rows]
    quality_errors = [cast(int, row["fp"]) + cast(int, row["fn"]) for row in trial_rows]
    mean_cost = sum(costs) / len(costs)
    return {
        "target_id": target_id,
        "trial_count": len(runs),
        "case_count_per_trial": len(runs[0].cases),
        "quality_errors_by_trial": quality_errors,
        "worst_trial_quality_errors": max(quality_errors),
        "precision_by_trial": [row["precision"] for row in trial_rows],
        "recall_by_trial": [row["recall"] for row in trial_rows],
        "positive_control_preservation_by_trial": [
            row["positive_control_preservation"] for row in trial_rows
        ],
        "mean_cost_per_article_usd": mean_cost,
        "projected_cost_per_1000_articles_usd": mean_cost * 1000,
        "p95_article_latency_ms_by_trial": p95_articles,
        "median_trial_p95_article_latency_ms": median(p95_articles),
        "total_request_count": sum(cast(int, row["request_count"]) for row in trial_rows),
        "total_input_tokens": sum(cast(int, row["input_tokens"]) for row in trial_rows),
        "total_output_tokens": sum(cast(int, row["output_tokens"]) for row in trial_rows),
        "total_tokens": sum(cast(int, row["total_tokens"]) for row in trial_rows),
        "total_cost_usd": sum(cast(float, row["total_cost_usd"]) for row in trial_rows),
        "inference": "none; repeated trials are not independent",
    }


def _paired_comparisons(
    result: BinaryRelevanceEvaluationResult,
    runs: dict[tuple[str, str], BinaryRelevanceRunResult],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    target_ids = [target.target_id for target in result.targets]
    for trial_ref in result.trial_refs:
        for first_target, second_target in combinations(target_ids, 2):
            first = runs[(first_target, trial_ref)]
            second = runs[(second_target, trial_ref)]
            paired = list(zip(first.cases, second.cases, strict=True))
            both_correct = sum(a.passed and b.passed for a, b in paired)
            first_only = sum(a.passed and not b.passed for a, b in paired)
            second_only = sum(not a.passed and b.passed for a, b in paired)
            both_wrong = sum(not a.passed and not b.passed for a, b in paired)
            comparisons.append(
                {
                    "trial_ref": trial_ref,
                    "first_target_id": first_target,
                    "second_target_id": second_target,
                    "both_correct": both_correct,
                    "first_only_correct": first_only,
                    "second_only_correct": second_only,
                    "both_wrong": both_wrong,
                    "disagreements": first_only + second_only,
                    "mcnemar_two_sided_exact_p_value": exact_mcnemar_p_value(
                        first_only, second_only
                    ),
                }
            )
    return comparisons


def _stability_analysis(
    result: BinaryRelevanceEvaluationResult,
    runs: dict[tuple[str, str], BinaryRelevanceRunResult],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for target in result.targets:
        target_runs = [runs[(target.target_id, trial_ref)] for trial_ref in result.trial_refs]
        case_rows: list[dict[str, Any]] = []
        for case_index, case_id in enumerate(result.case_ids):
            cases = [run.cases[case_index] for run in target_runs]
            verdicts = [cast(bool, case.verdict) for case in cases]
            probabilities = [cast(Decimal, case.probability) for case in cases]
            unanimous = len(set(verdicts)) == 1
            flips = sum(first != second for first, second in pairwise(verdicts))
            expected = cases[0].expected
            stable_outcome = "correct"
            if unanimous and verdicts[0] and not expected:
                stable_outcome = "false_positive"
            elif unanimous and not verdicts[0] and expected:
                stable_outcome = "false_negative"
            elif not unanimous:
                stable_outcome = "unstable"
            case_rows.append(
                {
                    "case_id": case_id,
                    "expected": expected,
                    "verdicts": verdicts,
                    "probabilities": [str(value) for value in probabilities],
                    "probability_min": str(min(probabilities)),
                    "probability_max": str(max(probabilities)),
                    "probability_range": str(max(probabilities) - min(probabilities)),
                    "unanimous": unanimous,
                    "flip_count": flips,
                    "classification": stable_outcome,
                }
            )
        unanimous_count = sum(cast(bool, row["unanimous"]) for row in case_rows)
        analyses.append(
            {
                "target_id": target.target_id,
                "case_count": len(case_rows),
                "unanimous_case_count": unanimous_count,
                "unanimous_verdict_rate": unanimous_count / len(case_rows),
                "flip_count": sum(cast(int, row["flip_count"]) for row in case_rows),
                "stable_false_positive_ids": [
                    row["case_id"] for row in case_rows if row["classification"] == "false_positive"
                ],
                "stable_false_negative_ids": [
                    row["case_id"] for row in case_rows if row["classification"] == "false_negative"
                ],
                "unstable_ids": [
                    row["case_id"] for row in case_rows if row["classification"] == "unstable"
                ],
                "cases": case_rows,
            }
        )
    return analyses


def _gate_analysis(
    preregistration: dict[str, Any],
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    registered = _object(preregistration["gates"], "gates")
    targets: list[dict[str, Any]] = []
    for aggregate in aggregates:
        target_id = cast(str, aggregate["target_id"])
        coverage_pass = (
            aggregate["trial_count"] == EXPECTED_TRIALS
            and aggregate["case_count_per_trial"] == EXPECTED_CASES
        )
        failures: list[str] = []
        precision_values = cast(list[float], aggregate["precision_by_trial"])
        recall_values = cast(list[float], aggregate["recall_by_trial"])
        control_values = cast(list[float], aggregate["positive_control_preservation_by_trial"])
        for index, (precision, recall, controls) in enumerate(
            zip(precision_values, recall_values, control_values, strict=True), start=1
        ):
            if controls != 1.0:
                failures.append(f"trial {index}: positive controls {controls:.6f} != 1.0")
            if recall != 1.0:
                failures.append(f"trial {index}: recall {recall:.6f} != 1.0")
            if precision < 0.80:
                failures.append(f"trial {index}: precision {precision:.6f} < 0.80")
        quality_pass = not failures
        targets.append(
            {
                "target_id": target_id,
                "coverage_pass": coverage_pass,
                "quality_pass": quality_pass,
                "deployment_eligible": coverage_pass and quality_pass,
                "failures": failures,
            }
        )
    return {
        "registered": registered,
        "targets": targets,
        "all_coverage_pass": all(cast(bool, item["coverage_pass"]) for item in targets),
        "all_quality_pass": all(cast(bool, item["quality_pass"]) for item in targets),
    }


def _pareto_analysis(
    aggregates: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    eligibility = {
        cast(str, item["target_id"]): cast(bool, item["deployment_eligible"])
        for item in cast(list[dict[str, Any]], gates["targets"])
    }
    points = {
        cast(str, aggregate["target_id"]): (
            cast(int, aggregate["worst_trial_quality_errors"]),
            cast(float, aggregate["mean_cost_per_article_usd"]),
            cast(float, aggregate["median_trial_p95_article_latency_ms"]),
        )
        for aggregate in aggregates
    }
    all_frontier = _non_dominated(points)
    eligible_points = {
        target_id: point for target_id, point in points.items() if eligibility[target_id]
    }
    eligible_frontier = _non_dominated(eligible_points)
    return {
        "dimensions": [
            "worst_trial_quality_errors",
            "mean_cost_per_article_usd",
            "median_trial_p95_article_latency_ms",
        ],
        "all_targets_frontier": all_frontier,
        "eligible_targets_frontier": eligible_frontier,
        "targets": [
            {
                "target_id": target_id,
                "worst_trial_quality_errors": point[0],
                "mean_cost_per_article_usd": point[1],
                "median_trial_p95_article_latency_ms": point[2],
                "deployment_eligible": eligibility[target_id],
                "non_dominated_among_all_targets": target_id in all_frontier,
                "non_dominated_among_eligible_targets": target_id in eligible_frontier,
            }
            for target_id, point in points.items()
        ],
    }


def _non_dominated(points: dict[str, tuple[int, float, float]]) -> list[str]:
    return [
        target_id
        for target_id, point in points.items()
        if not any(
            other_id != target_id
            and all(other <= current for other, current in zip(other_point, point, strict=True))
            and any(other < current for other, current in zip(other_point, point, strict=True))
            for other_id, other_point in points.items()
        )
    ]


def _inverse_binomial_survival(probability: float, successes: int, total: int) -> float:
    low = 0.0
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2
        survival = sum(
            math.comb(total, count) * midpoint**count * (1 - midpoint) ** (total - count)
            for count in range(successes, total + 1)
        )
        if survival < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise ValueError("Metric denominator cannot be zero")
    return numerator / denominator


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            _ = file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _metric_ci(row: dict[str, Any], metric: str) -> str:
    interval = _object(row[f"{metric}_95_exact_ci"], f"{metric} interval")
    return f"{row[metric]:.4f} ({interval['lower']:.4f}-{interval['upper']:.4f})"


def _mark(value: object) -> str:
    return "yes" if value is True else "no"


def _ids(value: object) -> str:
    values = cast(list[str], _list(value, "IDs"))
    return ", ".join(f"`{item}`" for item in values) if values else "none"


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
