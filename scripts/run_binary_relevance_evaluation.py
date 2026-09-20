#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.binary_benchmark import (
    REGISTERED_BINARY_TARGETS,
    BinaryEvaluator,
    BinaryExecutionItem,
    BinaryTarget,
    TargetId,
    atomic_write,
    build_binary_evaluators,
    rotated_execution_plan,
    trial_references,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRelevanceCaseResult,
    BinaryRelevanceEvaluationResult,
    BinaryRelevanceSource,
    V11ManifestVersion,
    load_v11_binary_relevance_source,
    run_binary_relevance_evaluation,
)
from romanian_news.evaluation import (
    PIN_PATH,
    EvaluationSpecProvenance,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    RelevanceEvaluationSpec,
)

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


class BinaryRelevanceCheckpoint(NewsModel):
    schema_version: Literal["binary-relevance-checkpoint-v1"] = "binary-relevance-checkpoint-v1"
    identity_digest: Sha256
    execution_ref: str
    execution_mode: Literal["live", "dry_run"]
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: V11ManifestVersion
    question_id: str
    question_digest: Sha256
    targets: tuple[BinaryTarget, ...]
    trial_refs: tuple[str, ...]
    case_ids: tuple[str, ...]
    cases: tuple[BinaryRelevanceCaseResult, ...]


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    parser = argparse.ArgumentParser(
        description="Run the checkpointed paired News binary relevance benchmark."
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
    source: BinaryRelevanceSource | None = None,
    evaluators: Mapping[TargetId, BinaryEvaluator] | None = None,
) -> BinaryRelevanceEvaluationResult:
    request_loader: Callable[[ArtifactReference], BinaryRequest] | None = None
    if source is not None:
        selected_source = source
    elif arguments.dry_run:
        selected_source, dry_run_request = _dry_run_source()
        request_loader = _constant_request_loader(dry_run_request)
    else:
        selected_source = load_v11_binary_relevance_source(PIN_PATH.read_bytes())
    targets = TARGETS
    if arguments.output.exists() and not arguments.resume:
        raise ValueError(f"Output already exists; pass --resume to reuse it: {arguments.output}")
    trial_refs = trial_references(
        arguments.execution_ref, arguments.trials, dry_run=arguments.dry_run
    )
    case_ids = tuple(
        case.case_id for case in selected_source.manifest.cases if case.concern == "relevance"
    )
    identity_digest = _identity_digest(
        arguments.execution_ref,
        targets,
        trial_refs,
        case_ids,
        dry_run=arguments.dry_run,
    )
    reusable = _load_reusable_cases(
        arguments.output,
        resume=arguments.resume,
        retry_errors=arguments.retry_errors,
        identity_digest=identity_digest,
        execution_ref=arguments.execution_ref,
        execution_mode="dry_run" if arguments.dry_run else "live",
        targets=targets,
        trial_refs=trial_refs,
        case_ids=case_ids,
    )
    selected_evaluators = dict(
        evaluators
        if evaluators is not None
        else _build_evaluators(targets, dry_run=arguments.dry_run)
    )
    accumulated = dict(reusable)

    def checkpoint(result: BinaryRelevanceCaseResult) -> None:
        accumulated[result.request_id] = result
        _write_checkpoint(
            arguments.output,
            BinaryRelevanceCheckpoint(
                identity_digest=identity_digest,
                execution_ref=arguments.execution_ref,
                execution_mode="dry_run" if arguments.dry_run else "live",
                source_artifact_id=V11_SOURCE_ARTIFACT_ID,
                declared_manifest_version=V11_MANIFEST_VERSION,
                question_id=RELEVANCE_BINARY_QUESTION.question_id,
                question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
                targets=targets,
                trial_refs=trial_refs,
                case_ids=case_ids,
                cases=tuple(accumulated.values()),
            ),
        )

    plan = rotated_execution_plan(targets, trial_refs, case_ids)
    result = run_binary_relevance_evaluation(
        selected_source,
        targets=targets,
        trial_refs=trial_refs,
        evaluators=selected_evaluators,
        execution_plan=tuple((item.target_id, item.trial_ref, item.case_id) for item in plan),
        reusable_cases=reusable,
        on_case_result=checkpoint,
        request_loader=request_loader,
    )
    _atomic_write(arguments.output, result.model_dump_json(indent=2).encode())
    _print_summary(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _ = execute(parse_args(argv))
    return 0


def _build_evaluators(
    targets: tuple[BinaryTarget, ...],
    *,
    dry_run: bool,
) -> dict[TargetId, BinaryEvaluator]:
    return build_binary_evaluators(targets, dry_run=dry_run)


def _constant_request_loader(
    request: BinaryRequest,
) -> Callable[[ArtifactReference], BinaryRequest]:
    def load(_reference: ArtifactReference) -> BinaryRequest:
        return request

    return load


def _dry_run_source() -> tuple[BinaryRelevanceSource, BinaryRequest]:
    article = ArtifactReference(
        artifact_id="news:article:dry-run",
        version_id="1" * 64,
        content_digest="2" * 64,
        r2_key="dry-run/article.json",
    )
    model_output = ArtifactReference(
        artifact_id="news:relevance:dry-run",
        version_id="3" * 64,
        content_digest="4" * 64,
        r2_key="dry-run/relevance.json",
    )
    manifest_reference = ArtifactReference(
        artifact_id="news:evaluation-manifest:dry-run",
        version_id=V11_SOURCE_ARTIFACT_ID,
        content_digest="5" * 64,
        r2_key="dry-run/manifest.json",
    )
    manifest = NewsEvaluationManifest(
        version=V11_MANIFEST_VERSION,
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
        issue_url="https://example.test/dry-run",
        source_feedback_ids=(),
        reports=(),
        cases=(
            RelevanceEvaluationSpec(
                case_id="dry-run-smoke",
                provenance=EvaluationSpecProvenance(feedback_ids=()),
                expected_accepted=True,
                article=article,
                model_output=model_output,
            ),
        ),
    )
    state = "Dry-run Romanian news benchmark smoke case."
    return (
        BinaryRelevanceSource(
            pin=NewsEvaluationPin(
                manifest_version_id=V11_SOURCE_ARTIFACT_ID,
                baseline_version_id="6" * 64,
            ),
            manifest_reference=manifest_reference,
            manifest=manifest,
        ),
        BinaryRequest(
            question=RELEVANCE_BINARY_QUESTION,
            state=state,
            state_digest=binary_state_digest(state),
        ),
    )


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
) -> dict[Sha256, BinaryRelevanceCaseResult]:
    if not resume:
        return {}
    if not output.is_file():
        raise ValueError(f"Cannot resume because checkpoint does not exist: {output}")
    content = output.read_bytes()
    try:
        final = BinaryRelevanceEvaluationResult.model_validate_json(content, strict=True)
    except ValidationError:
        checkpoint = BinaryRelevanceCheckpoint.model_validate_json(content, strict=True)
        if (
            checkpoint.identity_digest != identity_digest
            or checkpoint.execution_ref != execution_ref
            or checkpoint.execution_mode != execution_mode
            or checkpoint.source_artifact_id != V11_SOURCE_ARTIFACT_ID
            or checkpoint.declared_manifest_version != V11_MANIFEST_VERSION
            or checkpoint.question_id != RELEVANCE_BINARY_QUESTION.question_id
            or checkpoint.question_digest != RELEVANCE_BINARY_QUESTION.semantic_digest
            or checkpoint.targets != targets
            or checkpoint.trial_refs != trial_refs
            or checkpoint.case_ids != case_ids
        ):
            raise ValueError(
                "Binary relevance checkpoint identity does not match this execution"
            ) from None
        cases = checkpoint.cases
    else:
        if (
            final.source_artifact_id != V11_SOURCE_ARTIFACT_ID
            or final.declared_manifest_version != V11_MANIFEST_VERSION
            or final.question_id != RELEVANCE_BINARY_QUESTION.question_id
            or final.question_digest != RELEVANCE_BINARY_QUESTION.semantic_digest
            or final.targets != targets
            or final.trial_refs != trial_refs
            or final.case_ids != case_ids
            or _identity_digest(
                execution_ref,
                targets,
                trial_refs,
                case_ids,
                dry_run=execution_mode == "dry_run",
            )
            != identity_digest
        ):
            raise ValueError("Binary relevance checkpoint identity does not match this execution")
        cases = tuple(case for run in final.runs for case in run.cases)
    if len({case.request_id for case in cases}) != len(cases):
        raise ValueError("Binary relevance checkpoint contains duplicate case executions")
    return {
        case.request_id: case for case in cases if not retry_errors or case.status == "completed"
    }


def _identity_digest(
    execution_ref: str,
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    case_ids: tuple[str, ...],
    *,
    dry_run: bool,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "case_ids": case_ids,
                "declared_manifest_version": V11_MANIFEST_VERSION,
                "execution_ref": execution_ref,
                "execution_mode": "dry_run" if dry_run else "live",
                "question_digest": RELEVANCE_BINARY_QUESTION.semantic_digest,
                "source_artifact_id": V11_SOURCE_ARTIFACT_ID,
                "targets": [target.model_dump(mode="json") for target in targets],
                "trial_refs": trial_refs,
            }
        )
    )


def _write_checkpoint(path: Path, checkpoint: BinaryRelevanceCheckpoint) -> None:
    _atomic_write(path, checkpoint.model_dump_json(indent=2).encode())


def _atomic_write(path: Path, content: bytes) -> None:
    atomic_write(path, content)


def _print_summary(result: BinaryRelevanceEvaluationResult) -> None:
    completed = sum(run.metrics.completed_cases for run in result.runs)
    failed = sum(run.metrics.failed_cases for run in result.runs)
    total = completed + failed
    summary = (
        f"binary relevance: {completed}/{total} completed, {failed} failed, "
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
