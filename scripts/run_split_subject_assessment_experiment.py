#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

import requests
from pydantic import ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.jev_relevance import JEV_EXECUTION_POLICY, jev_execution_policy_digest
from romanian_news.binary_benchmark import atomic_write
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS, TIER_COMPOSITION_DIGEST
from romanian_news.evaluation import (
    NewsEvaluationManifest,
    RankingEvaluationSpec,
    SubjectAssessmentEvaluationResult,
    TierEvaluationSpec,
    score_subject_assessments,
)
from romanian_news.split_subject_assessment_experiment import (
    V11_MANIFEST_VERSION_ID,
    V11_REPORT_VERSION_IDS,
    ArmOutcome,
    ArtifactReader,
    CompletedArm,
    ExperimentAttempt,
    load_frozen_v11_source,
    run_candidate_arm,
    run_incumbent_arm,
)
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import (
    PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
    DailySubjectAssessmentInput,
    SubjectAssessmentThemeSet,
    subject_assessment_policy_digest,
)

TRIAL_REFS = (
    "split-assessment-v1:trial-001",
    "split-assessment-v1:trial-002",
    "split-assessment-v1:trial-003",
)
ArmName = Literal["incumbent", "candidate"]
ARM_ORDER: tuple[tuple[ArmName, ...], ...] = (
    ("incumbent", "candidate"),
    ("candidate", "incumbent"),
    ("incumbent", "candidate"),
)
SPEND_CEILING_USD = Decimal("5.00")
JEV_REQUEST_RESERVE_USD = Decimal("0.008")
GEMINI_REQUEST_RESERVE_USD = Decimal("0.020")
SOURCE_DESCRIPTOR_PATH = (
    Path(__file__).parents[1] / "src" / "romanian_news" / "split_assessment_v11_source.json"
)
R2_PROXY_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RunnerArguments:
    execution_ref: str
    output: Path
    dry_run: bool
    resume: bool
    retry_failed: bool


class ArmRecord(NewsModel):
    trial_ref: str
    arm: ArmName
    outcome: ArmOutcome
    score: SubjectAssessmentEvaluationResult | None = None


class InFlightArm(NewsModel):
    trial_ref: str
    arm: ArmName
    report_version_id: Sha256
    attempts: tuple[ExperimentAttempt, ...] = ()


class SplitExperimentCheckpoint(NewsModel):
    schema_version: Literal["split-subject-assessment-checkpoint-v1"] = (
        "split-subject-assessment-checkpoint-v1"
    )
    identity_digest: Sha256
    execution_ref: str
    mode: Literal["dry_run", "live"]
    records: tuple[ArmRecord, ...] = ()
    in_flight: InFlightArm | None = None
    unknown_in_flight: tuple[InFlightArm, ...] = ()


class SplitExperimentResult(NewsModel):
    schema_version: Literal["split-subject-assessment-result-v1"] = (
        "split-subject-assessment-result-v1"
    )
    identity_digest: Sha256
    execution_ref: str
    mode: Literal["dry_run", "live"]
    manifest_version_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    report_version_ids: tuple[Sha256, ...]
    trial_refs: tuple[str, ...]
    arm_order: tuple[tuple[ArmName, ...], ...]
    spend_ceiling_usd: Decimal
    known_spend_usd: Decimal
    accounting_complete: bool
    records: tuple[ArmRecord, ...]
    unknown_in_flight: tuple[InFlightArm, ...]
    unknown_accounting: bool
    crash_window: str


ArmRunner = Callable[
    [Sha256, DailySubjectAssessmentInput, str, Callable[[ExperimentAttempt], None]], ArmOutcome
]


def parse_args(argv: Sequence[str] | None = None) -> RunnerArguments:
    parser = argparse.ArgumentParser(
        description="Run the frozen split subject-assessment experiment."
    )
    _ = parser.add_argument("--execution-ref", required=True, type=_non_empty)
    _ = parser.add_argument("--output", required=True, type=Path)
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--retry-failed", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.retry_failed and not parsed.resume:
        parser.error("--retry-failed requires --resume")
    return RunnerArguments(
        execution_ref=cast(str, parsed.execution_ref),
        output=cast(Path, parsed.output),
        dry_run=cast(bool, parsed.dry_run),
        resume=cast(bool, parsed.resume),
        retry_failed=cast(bool, parsed.retry_failed),
    )


def execute(
    arguments: RunnerArguments,
    *,
    manifest: NewsEvaluationManifest | None = None,
    inputs: tuple[tuple[Sha256, DailySubjectAssessmentInput], ...] | None = None,
    source_descriptor: bytes | Path | None = None,
    artifact_reader: ArtifactReader | None = None,
    incumbent_runner: ArmRunner | None = None,
    candidate_runner: ArmRunner | None = None,
) -> SplitExperimentResult:
    descriptor_source = source_descriptor or SOURCE_DESCRIPTOR_PATH
    descriptor_content = (
        descriptor_source.read_bytes() if isinstance(descriptor_source, Path) else descriptor_source
    )
    if manifest is None or inputs is None:
        _, frozen_manifest, frozen_inputs = load_frozen_v11_source(
            descriptor_content, artifact_reader or _experiment_artifact_reader()
        )
        selected_manifest = manifest or frozen_manifest
        selected_inputs = inputs or frozen_inputs
    else:
        selected_manifest = manifest
        selected_inputs = inputs
    if selected_manifest.version != "news-evaluation-2026-09-11-v11":
        raise ValueError("Experiment requires the exact frozen v11 manifest")
    if tuple(item[0] for item in selected_inputs) != V11_REPORT_VERSION_IDS:
        raise ValueError("Experiment inputs do not match the frozen report workload")
    identity = _identity(
        arguments.execution_ref,
        arguments.dry_run,
        source_descriptor_digest=hashlib.sha256(descriptor_content).hexdigest(),
    )
    records, unknown_in_flight = _resume_records(arguments, identity)
    if arguments.output.exists() and not arguments.resume:
        raise ValueError(f"Output already exists; pass --resume to reuse it: {arguments.output}")
    if arguments.dry_run:
        result = _result(arguments, identity, tuple(records), unknown_in_flight)
        atomic_write(arguments.output, result.model_dump_json(indent=2).encode())
        return result

    by_report = dict(selected_inputs)
    runners = {
        "incumbent": incumbent_runner or _incumbent,
        "candidate": candidate_runner or _candidate,
    }
    for trial_ref, order in zip(TRIAL_REFS, ARM_ORDER, strict=True):
        for arm in order:
            key = (trial_ref, arm)
            existing = next((item for item in records if (item.trial_ref, item.arm) == key), None)
            if existing is not None:
                if not (arguments.retry_failed and existing.outcome.status == "failed"):
                    continue
                records.remove(existing)
            report_id = V11_REPORT_VERSION_IDS[0]
            reserve = _arm_reserve(arm, len(by_report[report_id].theme_set.themes))
            known_spend = _known_spend(records, _unknown_attempts(unknown_in_flight))
            _require_arm_budget(known_spend, reserve)
            in_flight = InFlightArm(trial_ref=trial_ref, arm=arm, report_version_id=report_id)
            _checkpoint(arguments, identity, records, in_flight, unknown_in_flight)

            def attempt_checkpoint(attempt: ExperimentAttempt) -> None:
                nonlocal in_flight
                in_flight = in_flight.model_copy(
                    update={"attempts": (*in_flight.attempts, attempt)}
                )
                _checkpoint(arguments, identity, records, in_flight, unknown_in_flight)

            outcome = runners[arm](
                report_id, by_report[report_id], arguments.execution_ref, attempt_checkpoint
            )
            score = _score(selected_manifest, by_report[report_id], outcome)
            records.append(ArmRecord(trial_ref=trial_ref, arm=arm, outcome=outcome, score=score))
            _checkpoint(arguments, identity, records, None, unknown_in_flight)
    result = _result(arguments, identity, tuple(records), unknown_in_flight)
    atomic_write(arguments.output, result.model_dump_json(indent=2).encode())
    return result


def _score(
    manifest: NewsEvaluationManifest, value: DailySubjectAssessmentInput, outcome: ArmOutcome
) -> SubjectAssessmentEvaluationResult | None:
    if not isinstance(outcome, CompletedArm):
        return None
    cases = tuple(
        item
        for item in manifest.cases
        if isinstance(item, TierEvaluationSpec | RankingEvaluationSpec)
        and item.provenance.report_version_id == outcome.report_version_id
    )
    if not isinstance(value.theme_set, SubjectAssessmentThemeSet):
        raise ValueError("Frozen scoring requires a validated subject theme set")
    return score_subject_assessments(cases, value.theme_set, outcome.assessments)


def _incumbent(
    report_id: Sha256,
    value: DailySubjectAssessmentInput,
    execution_ref: str,
    on_attempt: Callable[[ExperimentAttempt], None],
) -> ArmOutcome:
    return run_incumbent_arm(report_id, value, execution_ref=execution_ref, on_attempt=on_attempt)


def _candidate(
    report_id: Sha256,
    value: DailySubjectAssessmentInput,
    execution_ref: str,
    on_attempt: Callable[[ExperimentAttempt], None],
) -> ArmOutcome:
    return run_candidate_arm(report_id, value, execution_ref=execution_ref, on_attempt=on_attempt)


def _checkpoint(
    arguments: RunnerArguments,
    identity: Sha256,
    records: list[ArmRecord],
    in_flight: InFlightArm | None,
    unknown_in_flight: tuple[InFlightArm, ...],
) -> None:
    checkpoint = SplitExperimentCheckpoint(
        identity_digest=identity,
        execution_ref=arguments.execution_ref,
        mode="dry_run" if arguments.dry_run else "live",
        records=tuple(records),
        in_flight=in_flight,
        unknown_in_flight=unknown_in_flight,
    )
    atomic_write(arguments.output, checkpoint.model_dump_json(indent=2).encode())


def _resume_records(
    arguments: RunnerArguments, identity: Sha256
) -> tuple[list[ArmRecord], tuple[InFlightArm, ...]]:
    if not arguments.resume:
        return [], ()
    if not arguments.output.is_file():
        raise ValueError(f"Cannot resume because output does not exist: {arguments.output}")
    content = arguments.output.read_bytes()
    try:
        result = SplitExperimentResult.model_validate_json(content, strict=True)
        recorded_identity, records = result.identity_digest, result.records
        unknown_in_flight = result.unknown_in_flight
    except ValidationError:
        checkpoint = SplitExperimentCheckpoint.model_validate_json(content, strict=True)
        recorded_identity, records = checkpoint.identity_digest, checkpoint.records
        unknown_in_flight = checkpoint.unknown_in_flight
        if checkpoint.in_flight is not None:
            if not arguments.retry_failed:
                raise ValueError(
                    "Checkpoint contains an in-flight arm with unknown spend; "
                    "pass --retry-failed to acknowledge and retry"
                ) from None
            unknown_in_flight = (*unknown_in_flight, checkpoint.in_flight)
    if recorded_identity != identity:
        raise ValueError("Split experiment checkpoint identity does not match this execution")
    return list(records), unknown_in_flight


def _result(
    arguments: RunnerArguments,
    identity: Sha256,
    records: tuple[ArmRecord, ...],
    unknown_in_flight: tuple[InFlightArm, ...],
) -> SplitExperimentResult:
    attempts = (
        *(attempt for record in records for attempt in record.outcome.attempts),
        *_unknown_attempts(unknown_in_flight),
    )
    return SplitExperimentResult(
        identity_digest=identity,
        execution_ref=arguments.execution_ref,
        mode="dry_run" if arguments.dry_run else "live",
        manifest_version_id=V11_MANIFEST_VERSION_ID,
        report_version_ids=V11_REPORT_VERSION_IDS,
        trial_refs=TRIAL_REFS,
        arm_order=ARM_ORDER,
        spend_ceiling_usd=SPEND_CEILING_USD,
        known_spend_usd=sum((item.cost_usd or Decimal(0) for item in attempts), Decimal(0)),
        accounting_complete=(
            not unknown_in_flight and all(item.accounting_complete for item in attempts)
        ),
        records=records,
        unknown_in_flight=unknown_in_flight,
        unknown_accounting=bool(unknown_in_flight),
        crash_window="Incumbent production calls expose attempts only after return; a process crash during that call leaves the pre-dispatch in-flight marker without per-attempt evidence.",
    )


def _known_spend(records: list[ArmRecord], extra: tuple[ExperimentAttempt, ...] = ()) -> Decimal:
    attempts = (*extra, *(attempt for record in records for attempt in record.outcome.attempts))
    return sum((item.cost_usd or Decimal(0) for item in attempts), Decimal(0))


def _unknown_attempts(values: tuple[InFlightArm, ...]) -> tuple[ExperimentAttempt, ...]:
    return tuple(attempt for value in values for attempt in value.attempts)


def _arm_reserve(arm: ArmName, subject_count: int) -> Decimal:
    if arm == "incumbent":
        return Decimal(2) * GEMINI_REQUEST_RESERVE_USD
    return (
        Decimal(subject_count * JEV_EXECUTION_POLICY.max_attempts) * JEV_REQUEST_RESERVE_USD
        + Decimal(2) * GEMINI_REQUEST_RESERVE_USD
    )


def _require_arm_budget(known_spend: Decimal, reserve: Decimal) -> None:
    if known_spend + reserve > SPEND_CEILING_USD:
        raise RuntimeError(
            "Experiment spend ceiling cannot admit arm reserve: "
            f"known={known_spend} reserve={reserve} ceiling={SPEND_CEILING_USD}"
        )


def _identity(
    execution_ref: str,
    dry_run: bool,
    *,
    source_descriptor_digest: Sha256 | None = None,
) -> Sha256:
    protocol_path = (
        Path(__file__).parents[1]
        / "src"
        / "romanian_news"
        / "split_assessment_experiment_protocol.json"
    )
    protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    tier_definition = DERIVED_BINARY_BENCHMARKS["tier"]
    payload = {
        "protocol": "split-subject-assessment-experiment/v1",
        "protocol_content_digest": protocol_digest,
        "source_descriptor_digest": source_descriptor_digest
        or hashlib.sha256(SOURCE_DESCRIPTOR_PATH.read_bytes()).hexdigest(),
        "manifest": V11_MANIFEST_VERSION_ID,
        "reports": V11_REPORT_VERSION_IDS,
        "tier_question_digests": tuple(
            question.semantic_digest for question in tier_definition.questions
        ),
        "tier_composition_digest": TIER_COMPOSITION_DIGEST,
        "jev_execution_policy_digest": jev_execution_policy_digest(),
        "incumbent_assessment_policy_digest": subject_assessment_policy_digest(
            PRODUCTION_SUBJECT_ASSESSMENT_POLICY
        ),
        "candidate_ranking_policy_digest": _candidate_ranking_policy_digest(),
        "trials": TRIAL_REFS,
        "arm_order": ARM_ORDER,
        "execution_ref": execution_ref,
        "mode": "dry_run" if dry_run else "live",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_ranking_policy_digest() -> Sha256:
    from romanian_news.subject_assessments import (
        ASSESSMENT_MAX_TOKENS,
        ASSESSMENT_MODEL,
        ASSESSMENT_PROMPT,
    )

    payload = {
        "model": ASSESSMENT_MODEL,
        "prompt": ASSESSMENT_PROMPT
        + " Tier membership is fixed. Only order within each supplied tier.",
        "temperature": 0,
        "max_tokens": ASSESSMENT_MAX_TOKENS,
        "reasoning_effort": "low",
        "correction_attempts": 1,
        "schema_policy": "fixed-tier-enum-and-exact-cardinality-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _experiment_artifact_reader() -> ArtifactReader:
    proxy_url = os.getenv("NEWS_R2_READ_PROXY_URL")
    if proxy_url is None:
        return lambda reference: read_verified_r2_object(reference.r2_key, reference.content_digest)

    def read(reference):
        response = requests.get(
            proxy_url,
            params={"key": reference.r2_key},
            timeout=R2_PROXY_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"R2 read proxy returned {response.status_code} for {reference.r2_key}"
            )
        if hashlib.sha256(response.content).hexdigest() != reference.content_digest:
            raise ValueError(f"R2 read proxy digest mismatch: {reference.r2_key}")
        return response.content

    return read


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must not be empty")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    result = execute(parse_args(argv))
    print(f"split assessment: {len(result.records)}/6 terminal arms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
