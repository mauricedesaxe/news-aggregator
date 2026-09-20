#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_grouping import build_grouping_binary_request
from romanian_news.binary_benchmark import (
    REGISTERED_BINARY_TARGETS,
    BinaryEvaluator,
    BinaryExecutionItem,
    BinarySpendLedger,
    BinaryTarget,
    TargetId,
    atomic_write,
    build_binary_evaluators,
    execution_identity_hash,
    reusable_case_results,
    rotated_execution_plan,
    trial_references,
)
from romanian_news.binary_grouping_evaluation import (
    BinaryGroupingCaseResult,
    BinaryGroupingEvaluationResult,
    BinaryGroupingSource,
    load_v11_binary_grouping_source,
    run_binary_grouping_evaluation,
)
from romanian_news.binary_relevance_evaluation import V11_MANIFEST_VERSION, V11_SOURCE_ARTIFACT_ID
from romanian_news.derived_binary_protocol import (
    DERIVED_BINARY_BENCHMARKS,
    GROUPING_BINARY_QUESTION,
)
from romanian_news.evaluation import PIN_PATH

TARGETS = REGISTERED_BINARY_TARGETS
ExecutionItem = BinaryExecutionItem


@dataclass(frozen=True)
class RunnerArguments:
    execution_ref: str
    output: Path
    trials: int
    resume: bool
    retry_errors: bool
    dry_run: bool


class BinaryGroupingCheckpoint(NewsModel):
    schema_version: Literal["binary-grouping-checkpoint-v1"] = "binary-grouping-checkpoint-v1"
    identity_digest: Sha256
    execution_ref: str
    execution_mode: Literal["live", "dry_run"]
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    question_id: str
    question_digest: Sha256
    targets: tuple[BinaryTarget, ...]
    trial_refs: tuple[str, ...]
    case_ids: tuple[str, ...]
    cases: tuple[BinaryGroupingCaseResult, ...]


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    parser = argparse.ArgumentParser(
        description="Run the checkpointed paired News binary grouping benchmark."
    )
    _ = parser.add_argument("--execution-ref", required=True, type=_non_empty)
    _ = parser.add_argument("--output", required=True, type=Path)
    _ = parser.add_argument("--trials", type=_positive_int, default=3)
    _ = parser.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--retry-errors", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)
    retry_errors = cast(bool, parsed.retry_errors)
    resume = cast(bool, parsed.resume)
    if retry_errors and not resume:
        parser.error("--retry-errors requires --resume")
    return RunnerArguments(
        execution_ref=cast(str, parsed.execution_ref),
        output=cast(Path, parsed.output),
        trials=cast(int, parsed.trials),
        resume=resume,
        retry_errors=retry_errors,
        dry_run=cast(bool, parsed.dry_run),
    )


def execute(
    arguments: RunnerArguments,
    *,
    source: BinaryGroupingSource | None = None,
    evaluators: Mapping[TargetId, BinaryEvaluator] | None = None,
) -> BinaryGroupingEvaluationResult:
    selected_source = source or load_v11_binary_grouping_source(PIN_PATH.read_bytes())
    if arguments.output.exists() and not arguments.resume:
        raise ValueError(f"Output already exists; pass --resume to reuse it: {arguments.output}")
    trial_refs = trial_references(
        arguments.execution_ref, arguments.trials, dry_run=arguments.dry_run
    )
    case_ids = tuple(case.case_id for case in selected_source.cases)
    case_identities = tuple(
        {
            "case_id": case.case_id,
            "left_article_version_id": case.left_article.version_id,
            "right_article_version_id": case.right_article.version_id,
            "question_digest": request.question.semantic_digest,
            "state_digest": request.state_digest,
        }
        for case in selected_source.cases
        for request in (build_grouping_binary_request(case),)
    )
    identity = _identity_digest(
        arguments.execution_ref,
        TARGETS,
        trial_refs,
        case_ids,
        case_identities=case_identities,
        dry_run=arguments.dry_run,
    )
    reusable = _load_reusable_cases(
        arguments.output,
        resume=arguments.resume,
        retry_errors=arguments.retry_errors,
        identity_digest=identity,
        execution_ref=arguments.execution_ref,
        execution_mode="dry_run" if arguments.dry_run else "live",
        targets=TARGETS,
        trial_refs=trial_refs,
        case_ids=case_ids,
    )
    selected_evaluators = dict(
        evaluators
        if evaluators is not None
        else _build_evaluators(TARGETS, dry_run=arguments.dry_run)
    )
    accumulated = dict(reusable)
    actual_spend = sum((case.cost_usd for case in reusable.values()), Decimal(0))
    spend = BinarySpendLedger(
        DERIVED_BINARY_BENCHMARKS["grouping"].spend_ceiling_usd,
        actual_spend_usd=actual_spend,
    )

    def checkpoint(result: BinaryGroupingCaseResult) -> None:
        accumulated[result.request_id] = result
        _atomic_write(
            arguments.output,
            BinaryGroupingCheckpoint(
                identity_digest=identity,
                execution_ref=arguments.execution_ref,
                execution_mode="dry_run" if arguments.dry_run else "live",
                source_artifact_id=V11_SOURCE_ARTIFACT_ID,
                declared_manifest_version=V11_MANIFEST_VERSION,
                question_id=GROUPING_BINARY_QUESTION.question_id,
                question_digest=GROUPING_BINARY_QUESTION.semantic_digest,
                targets=TARGETS,
                trial_refs=trial_refs,
                case_ids=case_ids,
                cases=tuple(accumulated.values()),
            )
            .model_dump_json(indent=2)
            .encode(),
        )

    plan = rotated_execution_plan(TARGETS, trial_refs, case_ids)
    result = run_binary_grouping_evaluation(
        selected_source,
        targets=TARGETS,
        trial_refs=trial_refs,
        evaluators=selected_evaluators,
        execution_plan=tuple((item.target_id, item.trial_ref, item.case_id) for item in plan),
        reusable_cases=reusable,
        on_case_result=checkpoint,
        spend=spend,
        dry_run=arguments.dry_run,
    )
    _atomic_write(arguments.output, result.model_dump_json(indent=2).encode())
    _print_summary(result)
    return result


def _build_evaluators(
    targets: tuple[BinaryTarget, ...], *, dry_run: bool
) -> dict[TargetId, BinaryEvaluator]:
    return build_binary_evaluators(targets, dry_run=dry_run)


def _load_reusable_cases(
    output: Path,
    *,
    resume: bool,
    retry_errors: bool,
    identity_digest: Sha256,
    execution_ref: str,
    execution_mode: Literal["live", "dry_run"],
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    case_ids: tuple[str, ...],
) -> dict[Sha256, BinaryGroupingCaseResult]:
    if not resume:
        return {}
    if not output.is_file():
        raise ValueError(f"Cannot resume because checkpoint does not exist: {output}")
    content = output.read_bytes()
    try:
        final = BinaryGroupingEvaluationResult.model_validate_json(content, strict=True)
    except ValidationError:
        checkpoint = BinaryGroupingCheckpoint.model_validate_json(content, strict=True)
        if (
            checkpoint.identity_digest != identity_digest
            or checkpoint.execution_ref != execution_ref
            or checkpoint.execution_mode != execution_mode
            or checkpoint.question_digest != GROUPING_BINARY_QUESTION.semantic_digest
            or checkpoint.targets != targets
            or checkpoint.trial_refs != trial_refs
            or checkpoint.case_ids != case_ids
        ):
            raise ValueError(
                "Binary grouping checkpoint identity does not match this execution"
            ) from None
        cases = checkpoint.cases
    else:
        if (
            final.question_digest != GROUPING_BINARY_QUESTION.semantic_digest
            or final.targets != targets
            or final.trial_refs != trial_refs
            or final.case_ids != case_ids
        ):
            raise ValueError("Binary grouping checkpoint identity does not match this execution")
        cases = tuple(case for run in final.runs for case in run.cases)
    return reusable_case_results(
        cases,
        retry_failed=retry_errors,
        request_id=lambda case: case.request_id,
        completed=lambda case: case.status == "completed",
    )


def _identity_digest(
    execution_ref: str,
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    case_ids: tuple[str, ...],
    *,
    case_identities: tuple[object, ...] | None = None,
    dry_run: bool,
) -> Sha256:
    return execution_identity_hash(
        benchmark="grouping",
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        question_digests=(GROUPING_BINARY_QUESTION.semantic_digest,),
        composition_digest=None,
        case_identities=case_identities or tuple({"case_id": case_id} for case_id in case_ids),
        targets=targets,
        trial_refs=trial_refs,
        execution_mode="dry_run" if dry_run else "live",
        execution_ref=execution_ref,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    atomic_write(path, content)


def _print_summary(result: BinaryGroupingEvaluationResult) -> None:
    completed = sum(run.metrics.completed_cases for run in result.runs)
    failed = sum(run.metrics.failed_cases for run in result.runs)
    summary = (
        f"binary grouping: {completed}/{completed + failed} completed, {failed} failed, "
        + f"{len(result.targets)} targets x {len(result.trial_refs)} trials"
    )
    print(summary, file=sys.stderr)


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    _ = execute(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
