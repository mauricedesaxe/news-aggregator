from __future__ import annotations

import json
import math
from collections.abc import Mapping
from itertools import combinations, pairwise
from pathlib import Path
from statistics import median
from typing import Any, cast

from romanian_news.analysis.percentiles import float_linear_percentile
from romanian_news.binary_benchmark import (
    BinaryBenchmarkDefinition,
    BinaryBenchmarkEvaluationResult,
    BinaryJudgmentResult,
    atomic_write,
)

CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL


def exact_binomial_interval(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Exact binomial counts must satisfy 0 <= successes <= total")
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


def analyze_binary_benchmark(
    definition: BinaryBenchmarkDefinition,
    result: BinaryBenchmarkEvaluationResult,
) -> dict[str, Any]:
    identity = result.identity
    if (
        identity.benchmark != definition.benchmark
        or identity.question_digests != definition.question_digests
        or identity.composition_digest != definition.composition_digest
        or len(identity.cases) != definition.case_count
    ):
        raise ValueError("Binary analysis input does not match its registered definition")

    by_run: dict[tuple[str, str], tuple[BinaryJudgmentResult, ...]] = {}
    for target in identity.targets:
        for trial_ref in identity.trial_refs:
            by_run[(target.target_id, trial_ref)] = tuple(
                row
                for row in result.results
                if row.target_id == target.target_id and row.trial_ref == trial_ref
            )

    trials = tuple(
        _trial_analysis(target_id, trial_ref, rows)
        for (target_id, trial_ref), rows in by_run.items()
    )
    aggregates = tuple(
        _target_aggregate(
            target.target_id,
            tuple(row for row in trials if row["target_id"] == target.target_id),
        )
        for target in identity.targets
    )
    paired = _paired_comparisons(result, by_run)
    stability = _stability(result, by_run)
    pareto = _pareto(aggregates)
    return {
        "schema_version": "binary-benchmark-analysis/v1",
        "identity_digest": result.identity_digest,
        "benchmark": definition.benchmark,
        "source_artifact_id": identity.source_artifact_id,
        "declared_manifest_version": identity.declared_manifest_version,
        "question_digests": list(identity.question_digests),
        "composition_digest": identity.composition_digest,
        "target_ids": [target.target_id for target in identity.targets],
        "trial_refs": list(identity.trial_refs),
        "case_count": len(identity.cases),
        "judgment_count_per_run": definition.calls_per_model_trial,
        "trials": list(trials),
        "target_aggregates": list(aggregates),
        "paired_comparisons": paired,
        "stability": stability,
        "pareto": pareto,
        "limitation": definition.limitation,
    }


def render_binary_analysis_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['benchmark']} binary benchmark analysis",
        "",
        f"- Identity: `{report['identity_digest']}`",
        f"- Source artifact: `{report['source_artifact_id']}`",
        f"- Shape: {len(_list(report['target_ids']))} targets x "
        f"{len(_list(report['trial_refs']))} trials x {report['case_count']} cases",
        f"- Limitation: {report['limitation']}",
        "",
        "## Trial results",
        "",
        "| Target | Trial | Completed | Failed | Correct | Accuracy (95% exact CI) | Attempts | Cost USD | p50/p95 ms |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for item in _list(report["trials"]):
        row = _mapping(item)
        interval_value = row["accuracy_95_exact_interval"]
        interval = _mapping(interval_value) if interval_value is not None else None
        interval_text = (
            "unavailable"
            if interval is None
            else f"{float(interval['lower']):.4f}-{float(interval['upper']):.4f}"
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['trial_ref']}` | {row['completed']} | "
            f"{row['failed']} | {row['correct']} | {float(row['accuracy']):.4f} "
            f"({interval_text}) | "
            f"{row['attempt_count']} | {float(row['cost_usd']):.9f} | "
            f"{float(row['p50_latency_ms']):.1f}/{float(row['p95_latency_ms']):.1f} |"
        )

    lines.extend(["", "## Paired comparisons", ""])
    for item in _list(report["paired_comparisons"]):
        row = _mapping(item)
        lines.append(
            f"- `{row['trial_ref']}` `{row['first_target_id']}` vs "
            f"`{row['second_target_id']}`: {row['paired_count']} paired, "
            f"{row['unavailable_count']} unavailable, exact McNemar "
            f"p={float(row['mcnemar_two_sided_exact_p_value']):.6g}."
        )

    lines.extend(["", "## Stability", ""])
    for item in _list(report["stability"]):
        row = _mapping(item)
        lines.append(
            f"- `{row['target_id']}`: {float(row['unanimous_rate']):.4f} unanimous, "
            f"{row['flip_count']} adjacent-trial flips, {row['unavailable_count']} unavailable."
        )

    pareto = _mapping(report["pareto"])
    frontier = ", ".join(f"`{value}`" for value in _list(pareto["frontier"])) or "none"
    lines.extend(["", "## Pareto", "", f"- Non-dominated targets: {frontier}."])
    return "\n".join(lines) + "\n"


def write_binary_analysis(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("Binary analysis JSON and Markdown paths must differ")
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write(json_path, content)
    atomic_write(markdown_path, render_binary_analysis_markdown(report).encode())


def _trial_analysis(
    target_id: str,
    trial_ref: str,
    rows: tuple[BinaryJudgmentResult, ...],
) -> dict[str, Any]:
    completed = tuple(row for row in rows if row.status == "completed")
    correct = sum(row.passed for row in completed)
    attempts = tuple(attempt for row in rows for attempt in row.attempts)
    latencies = tuple(attempt.latency_ms for attempt in attempts)
    cost = sum((row.cost_usd for row in rows), start=0)
    interval = exact_binomial_interval(correct, len(completed)) if completed else None
    return {
        "target_id": target_id,
        "trial_ref": trial_ref,
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "correct": correct,
        "accuracy": correct / len(completed) if completed else 0.0,
        "accuracy_95_exact_interval": interval,
        "attempt_count": len(attempts),
        "input_tokens": sum(attempt.input_tokens or 0 for attempt in attempts),
        "output_tokens": sum(attempt.output_tokens or 0 for attempt in attempts),
        "cost_usd": float(cost),
        "p50_latency_ms": float_linear_percentile(latencies, 0.50),
        "p95_latency_ms": float_linear_percentile(latencies, 0.95),
    }


def _target_aggregate(target_id: str, trials: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    error_counts = tuple(
        int(row["failed"]) + int(row["completed"]) - int(row["correct"]) for row in trials
    )
    costs = tuple(float(row["cost_usd"]) for row in trials)
    p50 = tuple(float(row["p50_latency_ms"]) for row in trials)
    p95 = tuple(float(row["p95_latency_ms"]) for row in trials)
    return {
        "target_id": target_id,
        "error_counts_by_trial": list(error_counts),
        "worst_trial_errors": max(error_counts),
        "total_attempt_count": sum(int(row["attempt_count"]) for row in trials),
        "total_input_tokens": sum(int(row["input_tokens"]) for row in trials),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in trials),
        "total_cost_usd": sum(costs),
        "mean_cost_usd": sum(costs) / len(costs),
        "median_trial_p50_latency_ms": median(p50),
        "median_trial_p95_latency_ms": median(p95),
    }


def _paired_comparisons(
    result: BinaryBenchmarkEvaluationResult,
    by_run: Mapping[tuple[str, str], tuple[BinaryJudgmentResult, ...]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    target_ids = tuple(target.target_id for target in result.identity.targets)
    for trial_ref in result.identity.trial_refs:
        for first_target, second_target in combinations(target_ids, 2):
            first = {
                (row.case_id, row.judgment_id): row for row in by_run[(first_target, trial_ref)]
            }
            second = {
                (row.case_id, row.judgment_id): row for row in by_run[(second_target, trial_ref)]
            }
            keys = tuple(sorted(first))
            pairs = tuple((first[key], second[key]) for key in keys)
            available = tuple(
                pair for pair in pairs if all(row.status == "completed" for row in pair)
            )
            first_only = sum(a.passed and not b.passed for a, b in available)
            second_only = sum(not a.passed and b.passed for a, b in available)
            comparisons.append(
                {
                    "trial_ref": trial_ref,
                    "first_target_id": first_target,
                    "second_target_id": second_target,
                    "paired_count": len(available),
                    "unavailable_count": len(pairs) - len(available),
                    "both_correct": sum(a.passed and b.passed for a, b in available),
                    "first_only_correct": first_only,
                    "second_only_correct": second_only,
                    "both_wrong": sum(not a.passed and not b.passed for a, b in available),
                    "mcnemar_two_sided_exact_p_value": exact_mcnemar_p_value(
                        first_only, second_only
                    ),
                }
            )
    return comparisons


def _stability(
    result: BinaryBenchmarkEvaluationResult,
    by_run: Mapping[tuple[str, str], tuple[BinaryJudgmentResult, ...]],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for target in result.identity.targets:
        trial_maps = tuple(
            {(row.case_id, row.judgment_id): row for row in by_run[(target.target_id, trial_ref)]}
            for trial_ref in result.identity.trial_refs
        )
        keys = tuple(sorted(trial_maps[0]))
        unanimous = 0
        flips = 0
        unavailable = 0
        unstable_ids: list[str] = []
        for key in keys:
            rows = tuple(trial[key] for trial in trial_maps)
            if any(row.status != "completed" for row in rows):
                unavailable += 1
                continue
            verdicts = tuple(bool(row.verdict) for row in rows)
            is_unanimous = len(set(verdicts)) == 1
            unanimous += is_unanimous
            flips += sum(first != second for first, second in pairwise(verdicts))
            if not is_unanimous:
                unstable_ids.append(f"{key[0]}:{key[1]}")
        available = len(keys) - unavailable
        analyses.append(
            {
                "target_id": target.target_id,
                "judgment_count": len(keys),
                "unanimous_count": unanimous,
                "unanimous_rate": unanimous / available if available else 0.0,
                "flip_count": flips,
                "unavailable_count": unavailable,
                "unstable_ids": unstable_ids,
            }
        )
    return analyses


def _pareto(aggregates: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    points = {
        str(row["target_id"]): (
            int(row["worst_trial_errors"]),
            float(row["mean_cost_usd"]),
            float(row["median_trial_p95_latency_ms"]),
        )
        for row in aggregates
    }
    frontier = tuple(
        target_id
        for target_id, point in points.items()
        if not any(
            other_id != target_id
            and all(other <= current for other, current in zip(other_point, point, strict=True))
            and any(other < current for other, current in zip(other_point, point, strict=True))
            for other_id, other_point in points.items()
        )
    )
    return {
        "dimensions": [
            "worst_trial_errors",
            "mean_cost_usd",
            "median_trial_p95_latency_ms",
        ],
        "frontier": list(frontier),
        "targets": [
            {
                **row,
                "non_dominated": str(row["target_id"]) in frontier,
            }
            for row in aggregates
        ],
    }


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


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Binary analysis value must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("Binary analysis value must be an array")
    return cast(list[Any], value)
