from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryAttemptEvidence,
    BinaryRequest,
    binary_attempt_totals,
    binary_decision,
    build_relevance_binary_request,
    validate_binary_attempts,
)
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.articles.models import ExtractedArticle
from romanian_news.binary_benchmark import (
    OPENROUTER_GEMINI_25_TARGET as OPENROUTER_GEMINI_25_TARGET,
)
from romanian_news.binary_benchmark import (
    OPENROUTER_GEMINI_38_TARGET as OPENROUTER_GEMINI_38_TARGET,
)
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET as TYPESAFE_JEV_TARGET,
)
from romanian_news.binary_benchmark import (
    BinaryEvaluator as BinaryEvaluator,
)
from romanian_news.binary_benchmark import (
    BinaryTarget,
    invoke_binary_evaluator,
)
from romanian_news.binary_benchmark import (
    CostBasis as CostBasis,
)
from romanian_news.binary_benchmark import (
    OpenRouterGeminiTarget as OpenRouterGeminiTarget,
)
from romanian_news.binary_benchmark import (
    TargetId as TargetId,
)
from romanian_news.binary_benchmark import (
    TypesafeJevTarget as TypesafeJevTarget,
)
from romanian_news.catalog.evaluations import read_news_evaluation_artifact_references
from romanian_news.evaluation import (
    NewsEvaluationManifest,
    NewsEvaluationPin,
    RelevanceEvaluationSpec,
)
from romanian_news.storage import read_verified_r2_object

V11_SOURCE_ARTIFACT_ID = "083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"
V11_MANIFEST_VERSION = "news-evaluation-2026-09-11-v11"

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
V11ManifestVersion = Literal["news-evaluation-2026-09-11-v11"]
BinaryExecutionKey = tuple[TargetId, str, str]

BinaryRelevanceTarget = BinaryTarget


class BinaryRelevanceSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    manifest: NewsEvaluationManifest

    @model_validator(mode="after")
    def require_pinned_v11_source(self) -> BinaryRelevanceSource:
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary relevance source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if self.manifest.version != V11_MANIFEST_VERSION:
            raise ValueError(
                f"Binary relevance declared manifest version must be {V11_MANIFEST_VERSION}"
            )
        return self


class BinaryRelevanceError(NewsModel):
    error_type: NonEmptyText
    message: NonEmptyText


class BinaryRelevanceCaseResult(NewsModel):
    status: Literal["completed", "failed"]
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: V11ManifestVersion
    case_id: NonEmptyText
    article_version_id: Sha256
    target_id: TargetId
    provider: Literal["openrouter", "typesafe"]
    trial_ref: NonEmptyText
    requested_model: NonEmptyText
    actual_model: NonEmptyText | None
    question_id: NonEmptyText
    question_digest: Sha256
    state_digest: Sha256
    execution_policy_id: NonEmptyText
    execution_policy_digest: Sha256
    request_id: Sha256
    adapter_request_id: Sha256 | None
    provider_request_id: NonEmptyText | None
    probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None
    verdict: bool | None
    expected: bool
    passed: bool
    control: bool
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    wall_latency_ms: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    cost_basis: CostBasis
    errors: tuple[BinaryRelevanceError, ...]
    attempts: Annotated[tuple[BinaryAttemptEvidence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> BinaryRelevanceCaseResult:
        validate_binary_attempts(self.attempts)
        input_tokens, output_tokens, cost_usd, latency_ms = binary_attempt_totals(self.attempts)
        if (
            self.input_tokens != input_tokens
            or self.output_tokens != output_tokens
            or self.cost_usd != cost_usd
            or self.wall_latency_ms != latency_ms
        ):
            raise ValueError("Binary relevance case accounting does not match its attempts")
        if self.status == "completed":
            if self.actual_model is None or self.probability is None or self.verdict is None:
                raise ValueError("Completed binary relevance cases require an observation")
            if self.adapter_request_id is None:
                raise ValueError("Completed binary relevance cases require an adapter request ID")
            if self.errors:
                raise ValueError("Completed binary relevance cases cannot contain errors")
            final = self.attempts[-1]
            if (
                final.status != "completed"
                or final.provider_request_id != self.provider_request_id
                or final.actual_model != self.actual_model
                or final.probability != self.probability
            ):
                raise ValueError("Completed binary relevance case does not match its final attempt")
            if self.passed != (self.verdict == self.expected):
                raise ValueError("Binary relevance pass result conflicts with its verdict")
        else:
            if any(
                value is not None
                for value in (
                    self.actual_model,
                    self.adapter_request_id,
                    self.provider_request_id,
                    self.probability,
                    self.verdict,
                )
            ):
                raise ValueError("Failed binary relevance cases cannot contain an observation")
            if self.passed or not self.errors:
                raise ValueError("Failed binary relevance cases require errors and cannot pass")
        return self


BinaryCaseResultCallback = Callable[[BinaryRelevanceCaseResult], None]


class BinaryRelevanceMetrics(NewsModel):
    passed_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    accuracy: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    precision: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    recall: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    positive_control_preservation: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    false_negative_ids: tuple[str, ...]
    request_count: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    total_cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    total_attempt_latency_ms: Annotated[int, Field(ge=0)]
    p50_wall_latency_ms: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    p95_wall_latency_ms: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> BinaryRelevanceMetrics:
        if self.completed_cases + self.failed_cases != self.total_cases:
            raise ValueError("Binary relevance completion counts must cover all cases")
        if self.passed_cases > self.completed_cases:
            raise ValueError("Binary relevance passed cases cannot exceed completed cases")
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("Binary relevance token totals are inconsistent")
        return self


class BinaryRelevanceRunResult(NewsModel):
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: V11ManifestVersion
    target: BinaryRelevanceTarget
    trial_ref: NonEmptyText
    question_id: NonEmptyText
    question_digest: Sha256
    cases: tuple[BinaryRelevanceCaseResult, ...]
    metrics: BinaryRelevanceMetrics

    @model_validator(mode="after")
    def require_consistent_run(self) -> BinaryRelevanceRunResult:
        for case in self.cases:
            if (
                case.source_artifact_id != self.source_artifact_id
                or case.declared_manifest_version != self.declared_manifest_version
                or case.target_id != self.target.target_id
                or case.provider != self.target.provider
                or case.trial_ref != self.trial_ref
                or case.requested_model != self.target.requested_model
                or case.execution_policy_id != self.target.execution_policy_id
                or case.execution_policy_digest != self.target.execution_policy_digest
                or case.cost_basis != self.target.cost_basis
                or case.question_id != self.question_id
                or case.question_digest != self.question_digest
            ):
                raise ValueError("Binary relevance case identity does not match its run")
        if self.metrics != binary_relevance_metrics(self.cases):
            raise ValueError("Binary relevance run metrics do not match its cases")
        return self


class BinaryRelevanceEvaluationResult(NewsModel):
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: V11ManifestVersion
    question_id: NonEmptyText
    question_digest: Sha256
    targets: tuple[BinaryRelevanceTarget, ...]
    trial_refs: tuple[NonEmptyText, ...]
    case_ids: tuple[NonEmptyText, ...]
    runs: tuple[BinaryRelevanceRunResult, ...]

    @model_validator(mode="after")
    def require_exact_ordered_coverage(self) -> BinaryRelevanceEvaluationResult:
        target_ids = tuple(target.target_id for target in self.targets)
        if target_ids != tuple(sorted(target_ids)) or len(set(target_ids)) != len(target_ids):
            raise ValueError("Binary relevance targets must be unique and ordered")
        if self.trial_refs != tuple(sorted(self.trial_refs)) or len(set(self.trial_refs)) != len(
            self.trial_refs
        ):
            raise ValueError("Binary relevance trial references must be unique and ordered")
        expected_runs = tuple(
            (target_id, trial_ref) for target_id in target_ids for trial_ref in self.trial_refs
        )
        if tuple((run.target.target_id, run.trial_ref) for run in self.runs) != expected_runs:
            raise ValueError("Binary relevance runs must exactly cover targets and trials")
        for run in self.runs:
            if (
                run.source_artifact_id != self.source_artifact_id
                or run.declared_manifest_version != self.declared_manifest_version
                or run.question_id != self.question_id
                or run.question_digest != self.question_digest
                or tuple(case.case_id for case in run.cases) != self.case_ids
            ):
                raise ValueError("Binary relevance run does not match its evaluation identity")
        return self


def load_v11_binary_relevance_source(pin_content: bytes | str) -> BinaryRelevanceSource:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    if pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID:
        raise ValueError(f"Binary relevance source artifact must be {V11_SOURCE_ARTIFACT_ID}")
    references = read_news_evaluation_artifact_references((V11_SOURCE_ARTIFACT_ID,))
    manifest_reference = references[V11_SOURCE_ARTIFACT_ID]
    manifest = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(
            manifest_reference.r2_key,
            manifest_reference.content_digest,
        ),
        strict=True,
    )
    return BinaryRelevanceSource(
        pin=pin,
        manifest_reference=manifest_reference,
        manifest=manifest,
    )


def run_binary_relevance_evaluation(
    source: BinaryRelevanceSource,
    *,
    targets: tuple[BinaryRelevanceTarget, ...],
    trial_refs: tuple[str, ...],
    evaluators: Mapping[TargetId, BinaryEvaluator],
    execution_plan: tuple[BinaryExecutionKey, ...] | None = None,
    reusable_cases: Mapping[Sha256, BinaryRelevanceCaseResult] | None = None,
    on_case_result: BinaryCaseResultCallback | None = None,
    request_loader: Callable[[ArtifactReference], BinaryRequest] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BinaryRelevanceEvaluationResult:
    ordered_targets = tuple(sorted(targets, key=lambda target: target.target_id))
    ordered_trial_refs = tuple(sorted(trial_refs))
    target_ids = tuple(target.target_id for target in ordered_targets)
    if not ordered_targets or len(set(target_ids)) != len(target_ids):
        raise ValueError("Binary relevance targets must be non-empty and unique")
    if not ordered_trial_refs or any(not ref.strip() for ref in ordered_trial_refs):
        raise ValueError("Binary relevance trial references must be non-empty")
    if len(set(ordered_trial_refs)) != len(ordered_trial_refs):
        raise ValueError("Binary relevance trial references must be unique")
    if set(evaluators) != set(target_ids):
        raise ValueError("Binary relevance evaluators must exactly match selected targets")

    cases = tuple(case for case in source.manifest.cases if case.concern == "relevance")
    load_request = request_loader or _load_request
    requests = tuple((case, load_request(case.article)) for case in cases)
    target_by_id: dict[TargetId, BinaryRelevanceTarget] = {
        target.target_id: target for target in ordered_targets
    }
    request_by_case_id = {case.case_id: (case, request) for case, request in requests}
    canonical_plan: tuple[BinaryExecutionKey, ...] = tuple(
        (target.target_id, trial_ref, case.case_id)
        for target in ordered_targets
        for trial_ref in ordered_trial_refs
        for case, _request in requests
    )
    selected_plan: tuple[BinaryExecutionKey, ...] = execution_plan or canonical_plan
    if len(selected_plan) != len(set(selected_plan)) or set(selected_plan) != set(canonical_plan):
        raise ValueError(
            "Binary relevance execution plan must exactly cover targets, trials, and cases"
        )

    reusable = dict(reusable_cases or {})
    expected_cases = {
        _request_id(target_by_id[target_id], trial_ref, case_id, request_by_case_id[case_id][1]): (
            target_by_id[target_id],
            trial_ref,
            request_by_case_id[case_id][0],
            request_by_case_id[case_id][1],
        )
        for target_id, trial_ref, case_id in canonical_plan
    }
    if not set(reusable).issubset(expected_cases):
        raise ValueError("Reusable binary relevance cases do not match this evaluation identity")
    for request_id, result in reusable.items():
        _validate_reusable_case(result, *expected_cases[request_id])

    results: dict[BinaryExecutionKey, BinaryRelevanceCaseResult] = {}
    for target_id, trial_ref, case_id in selected_plan:
        target = target_by_id[target_id]
        case, request = request_by_case_id[case_id]
        request_id = _request_id(target, trial_ref, case_id, request)
        result = reusable.get(request_id)
        if result is None:
            result = _evaluate_case(
                target,
                trial_ref,
                case,
                request,
                evaluators[target_id],
                clock,
            )
            if on_case_result is not None:
                on_case_result(result)
        results[(target_id, trial_ref, case_id)] = result

    runs = tuple(
        _build_run(
            target,
            trial_ref,
            tuple(results[(target.target_id, trial_ref, case.case_id)] for case, _ in requests),
        )
        for target in ordered_targets
        for trial_ref in ordered_trial_refs
    )
    return BinaryRelevanceEvaluationResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        question_id=RELEVANCE_BINARY_QUESTION.question_id,
        question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        targets=ordered_targets,
        trial_refs=ordered_trial_refs,
        case_ids=tuple(case.case_id for case in cases),
        runs=runs,
    )


def binary_relevance_metrics(
    cases: tuple[BinaryRelevanceCaseResult, ...],
) -> BinaryRelevanceMetrics:
    completed = tuple(case for case in cases if case.status == "completed")
    true_positives = sum(case.expected and case.verdict is True for case in completed)
    false_positives = sum(not case.expected and case.verdict is True for case in completed)
    expected_positives = tuple(case for case in completed if case.expected)
    false_negatives = tuple(case.case_id for case in expected_positives if case.verdict is False)
    positive_controls = tuple(case for case in completed if case.control and case.expected)
    preserved_controls = sum(case.verdict is True for case in positive_controls)
    attempts = tuple(attempt for case in cases for attempt in case.attempts)
    input_tokens = sum(attempt.input_tokens or 0 for attempt in attempts)
    output_tokens = sum(attempt.output_tokens or 0 for attempt in attempts)
    latencies = tuple(attempt.latency_ms for attempt in attempts)
    return BinaryRelevanceMetrics(
        passed_cases=sum(case.passed for case in cases),
        completed_cases=len(completed),
        failed_cases=len(cases) - len(completed),
        total_cases=len(cases),
        accuracy=_ratio(sum(case.passed for case in completed), len(completed)),
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, len(expected_positives)),
        positive_control_preservation=_ratio(preserved_controls, len(positive_controls)),
        false_negative_ids=false_negatives,
        request_count=len(attempts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_cost_usd=sum(
            (attempt.cost_usd or Decimal(0) for attempt in attempts), start=Decimal(0)
        ),
        total_attempt_latency_ms=sum(attempt.latency_ms for attempt in attempts),
        p50_wall_latency_ms=_percentile(latencies, Decimal("0.50")),
        p95_wall_latency_ms=_percentile(latencies, Decimal("0.95")),
    )


def _load_request(article_reference: ArtifactReference) -> BinaryRequest:
    article = ExtractedArticle.model_validate_json(
        read_verified_r2_object(article_reference.r2_key, article_reference.content_digest),
        strict=True,
    )
    return build_relevance_binary_request(
        ArticleAnalysisInput(reference=article_reference, article=article)
    )


def _build_run(
    target: BinaryRelevanceTarget,
    trial_ref: str,
    results: tuple[BinaryRelevanceCaseResult, ...],
) -> BinaryRelevanceRunResult:
    return BinaryRelevanceRunResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        target=target,
        trial_ref=trial_ref,
        question_id=RELEVANCE_BINARY_QUESTION.question_id,
        question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        cases=results,
        metrics=binary_relevance_metrics(results),
    )


def _validate_reusable_case(
    result: BinaryRelevanceCaseResult,
    target: BinaryRelevanceTarget,
    trial_ref: str,
    case: RelevanceEvaluationSpec,
    request: BinaryRequest,
) -> None:
    if (
        result.source_artifact_id != V11_SOURCE_ARTIFACT_ID
        or result.declared_manifest_version != V11_MANIFEST_VERSION
        or result.case_id != case.case_id
        or result.article_version_id != case.article.version_id
        or result.target_id != target.target_id
        or result.provider != target.provider
        or result.trial_ref != trial_ref
        or result.requested_model != target.requested_model
        or result.question_id != request.question.question_id
        or result.question_digest != request.question.semantic_digest
        or result.state_digest != request.state_digest
        or result.execution_policy_id != target.execution_policy_id
        or result.execution_policy_digest != target.execution_policy_digest
        or result.request_id != _request_id(target, trial_ref, case.case_id, request)
        or result.expected != case.expected_accepted
        or result.control != case.control
        or result.cost_basis != target.cost_basis
    ):
        raise ValueError("Reusable binary relevance case identity does not match its execution")


def _evaluate_case(
    target: BinaryRelevanceTarget,
    trial_ref: str,
    case: RelevanceEvaluationSpec,
    request: BinaryRequest,
    evaluator: BinaryEvaluator,
    clock: Callable[[], float],
) -> BinaryRelevanceCaseResult:
    case_id = case.case_id
    article = case.article
    expected = case.expected_accepted
    control = case.control
    request_id = _request_id(target, trial_ref, case_id, request)
    invocation = invoke_binary_evaluator(evaluator, request, trial_ref, clock=clock)
    attempts = invocation.attempts
    observation = invocation.observation
    error = invocation.error
    if error is not None:
        input_tokens, output_tokens, cost_usd, wall_latency_ms = binary_attempt_totals(attempts)
        return BinaryRelevanceCaseResult(
            status="failed",
            source_artifact_id=V11_SOURCE_ARTIFACT_ID,
            declared_manifest_version=V11_MANIFEST_VERSION,
            case_id=case_id,
            article_version_id=article.version_id,
            target_id=target.target_id,
            provider=target.provider,
            trial_ref=trial_ref,
            requested_model=target.requested_model,
            actual_model=None,
            question_id=request.question.question_id,
            question_digest=request.question.semantic_digest,
            state_digest=request.state_digest,
            execution_policy_id=target.execution_policy_id,
            execution_policy_digest=target.execution_policy_digest,
            request_id=request_id,
            adapter_request_id=None,
            provider_request_id=None,
            probability=None,
            verdict=None,
            expected=expected,
            passed=False,
            control=control,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_latency_ms=wall_latency_ms,
            cost_usd=cost_usd,
            cost_basis=target.cost_basis,
            errors=(
                BinaryRelevanceError(
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                ),
            ),
            attempts=attempts,
        )

    if observation is None:
        raise AssertionError("Successful binary invocation omitted its observation")
    verdict = binary_decision(observation.probability, request.question.threshold)
    input_tokens, output_tokens, cost_usd, wall_latency_ms = binary_attempt_totals(attempts)
    return BinaryRelevanceCaseResult(
        status="completed",
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        case_id=case_id,
        article_version_id=article.version_id,
        target_id=target.target_id,
        provider=target.provider,
        trial_ref=trial_ref,
        requested_model=target.requested_model,
        actual_model=observation.model,
        question_id=request.question.question_id,
        question_digest=request.question.semantic_digest,
        state_digest=request.state_digest,
        execution_policy_id=target.execution_policy_id,
        execution_policy_digest=target.execution_policy_digest,
        request_id=request_id,
        adapter_request_id=observation.request_id,
        provider_request_id=observation.provider_request_id,
        probability=observation.probability,
        verdict=verdict,
        expected=expected,
        passed=verdict == expected,
        control=control,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_latency_ms=wall_latency_ms,
        cost_usd=cost_usd,
        cost_basis=target.cost_basis,
        errors=(),
        attempts=attempts,
    )


def _request_id(
    target: BinaryRelevanceTarget,
    trial_ref: str,
    case_id: str,
    request: BinaryRequest,
) -> Sha256:
    return hashlib.sha256(
        _canonical_json(
            {
                "case_id": case_id,
                "execution_policy_digest": target.execution_policy_digest,
                "question_digest": request.question.semantic_digest,
                "source_artifact_id": V11_SOURCE_ARTIFACT_ID,
                "state_digest": request.state_digest,
                "target_id": target.target_id,
                "trial_ref": trial_ref,
            }
        )
    ).hexdigest()


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)


def _percentile(values: tuple[int, ...], quantile: Decimal) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return Decimal(ordered[lower])
    return Decimal(ordered[lower]) + Decimal(ordered[upper] - ordered[lower]) * (
        position - Decimal(lower)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
