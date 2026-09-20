#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from romanian_news.binary_benchmark import (
    REGISTERED_BINARY_TARGETS,
    BenchmarkId,
    BinaryBenchmarkEvaluationResult,
    BinaryEvaluator,
    BinaryExecutionIdentity,
    BinaryJudgmentResult,
    BinarySpendLedger,
    TargetId,
    atomic_write,
    build_binary_evaluators,
    build_execution_identity,
    load_binary_checkpoint,
    reusable_case_results,
    run_registered_binary_benchmark,
    trial_references,
    write_binary_checkpoint,
)
from romanian_news.derived_binary_benchmark import (
    DERIVED_BINARY_DISPATCH,
    DerivedBinaryWorkload,
    load_v11_derived_binary_workload,
)
from romanian_news.evaluation import PIN_PATH


@dataclass(frozen=True)
class RunnerArguments:
    concern: BenchmarkId
    execution_ref: str
    output: Path
    trials: int
    resume: bool
    retry_errors: bool
    dry_run: bool
    prior_spend_usd: Decimal


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed registered derived binary benchmark."
    )
    _ = parser.add_argument("--concern", required=True, choices=tuple(DERIVED_BINARY_DISPATCH))
    _ = parser.add_argument("--execution-ref", required=True, type=_non_empty)
    _ = parser.add_argument("--output", required=True, type=Path)
    _ = parser.add_argument("--trials", type=_positive_int, default=3)
    _ = parser.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--retry-errors", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--prior-spend-usd", type=_non_negative_decimal, default=Decimal(0))
    parsed = parser.parse_args(argv)
    retry_errors = cast(bool, parsed.retry_errors)
    resume = cast(bool, parsed.resume)
    if retry_errors and not resume:
        parser.error("--retry-errors requires --resume")
    return RunnerArguments(
        concern=cast(BenchmarkId, parsed.concern),
        execution_ref=cast(str, parsed.execution_ref),
        output=cast(Path, parsed.output),
        trials=cast(int, parsed.trials),
        resume=resume,
        retry_errors=retry_errors,
        dry_run=cast(bool, parsed.dry_run),
        prior_spend_usd=cast(Decimal, parsed.prior_spend_usd),
    )


def execute(
    arguments: RunnerArguments,
    *,
    workload: DerivedBinaryWorkload | None = None,
    evaluators: Mapping[TargetId, BinaryEvaluator] | None = None,
) -> BinaryBenchmarkEvaluationResult:
    if not arguments.dry_run and arguments.trials != 3:
        raise ValueError("Live derived binary benchmarks require exactly three trials")
    selected = workload or load_v11_derived_binary_workload(
        arguments.concern, PIN_PATH.read_bytes()
    )
    if selected.definition != DERIVED_BINARY_DISPATCH[arguments.concern].definition:
        raise ValueError("Derived binary workload does not match the selected concern")
    if arguments.output.exists() and not arguments.resume:
        raise ValueError(f"Output already exists; pass --resume to reuse it: {arguments.output}")

    targets = REGISTERED_BINARY_TARGETS
    trial_refs = trial_references(
        arguments.execution_ref, arguments.trials, dry_run=arguments.dry_run
    )
    identity = build_execution_identity(
        selected.definition,
        source_artifact_id=selected.source_artifact_id,
        declared_manifest_version=selected.declared_manifest_version,
        cases=selected.cases,
        targets=targets,
        trial_refs=trial_refs,
        execution_mode="dry_run" if arguments.dry_run else "live",
        execution_ref=arguments.execution_ref,
    )
    reusable, recorded_spend = _load_reusable_results(
        arguments.output,
        identity,
        resume=arguments.resume,
        retry_errors=arguments.retry_errors,
    )
    actual_spend = recorded_spend if arguments.resume else arguments.prior_spend_usd
    spend = BinarySpendLedger(Decimal("10.00"), actual_spend_usd=actual_spend)
    selected_evaluators = dict(
        evaluators
        if evaluators is not None
        else build_binary_evaluators(targets, dry_run=arguments.dry_run)
    )
    accumulated = dict(reusable)

    def checkpoint(result: BinaryJudgmentResult) -> None:
        accumulated[result.request_id] = result
        write_binary_checkpoint(arguments.output, identity, spend, tuple(accumulated.values()))

    result = run_registered_binary_benchmark(
        selected.definition,
        identity,
        selected.cases,
        selected_evaluators,
        reusable_results=reusable,
        on_result=checkpoint,
        spend=spend,
    )
    atomic_write(arguments.output, result.model_dump_json(indent=2).encode())
    _print_summary(result)
    return result


def _load_reusable_results(
    output: Path,
    identity: BinaryExecutionIdentity,
    *,
    resume: bool,
    retry_errors: bool,
) -> tuple[dict[str, BinaryJudgmentResult], Decimal]:
    if not resume:
        return {}, Decimal(0)
    if not output.is_file():
        raise ValueError(f"Cannot resume because checkpoint does not exist: {output}")
    content = output.read_bytes()
    try:
        final = BinaryBenchmarkEvaluationResult.model_validate_json(content, strict=True)
    except ValidationError:
        checkpoint, reusable = load_binary_checkpoint(output, identity, retry_failed=retry_errors)
        return reusable, checkpoint.actual_spend_usd
    if final.identity != identity or final.identity_digest != identity.digest:
        raise ValueError("Binary checkpoint identity does not match this execution")
    reusable = reusable_case_results(
        final.results,
        retry_failed=retry_errors,
        request_id=lambda result: result.request_id,
        completed=lambda result: result.status == "completed",
    )
    return reusable, sum((result.cost_usd for result in final.results), Decimal(0))


def _print_summary(result: BinaryBenchmarkEvaluationResult) -> None:
    completed = sum(item.status == "completed" for item in result.results)
    failed = len(result.results) - completed
    print(
        (
            f"binary {result.identity.benchmark}: {completed}/{len(result.results)} completed, "
            f"{failed} failed, {len(result.identity.targets)} targets x "
            f"{len(result.identity.trial_refs)} trials"
        ),
        file=sys.stderr,
    )


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_decimal(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative decimal")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    _ = execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
