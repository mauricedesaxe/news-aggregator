from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
    BinaryRequest,
    binary_attempt_totals,
    binary_decision,
    validate_binary_attempts,
)
from romanian_news.analysis.binary_grouping import BinaryGroupingCase, build_grouping_binary_request
from romanian_news.analysis.percentiles import decimal_linear_percentile
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_benchmark import (
    BINARY_TARGET_REGISTRY,
    BinaryBenchmarkCase,
    BinaryEvaluator,
    BinaryJudgmentCase,
    BinarySpendLedger,
    BinarySpendLimitExceeded,
    BinaryTarget,
    CostBasis,
    TargetId,
    invoke_binary_evaluator,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRelevanceError,
)
from romanian_news.catalog.evaluations import read_news_evaluation_artifact_references
from romanian_news.derived_binary_protocol import GROUPING_BINARY_QUESTION
from romanian_news.evaluation import (
    GroupingEvaluationSpec,
    NewsEvaluationManifest,
    NewsEvaluationPin,
)
from romanian_news.storage import read_verified_r2_object

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
BinaryGroupingExecutionKey = tuple[TargetId, str, str]
BinaryGroupingPreparedCase = tuple[BinaryGroupingCase, BinaryRequest]
BinaryGroupingPreparedExecution = tuple[
    BinaryTarget,
    str,
    BinaryGroupingCase,
    BinaryRequest,
]


class BinaryGroupingSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    cases: Annotated[tuple[BinaryGroupingCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_pinned_v11_source(self) -> BinaryGroupingSource:
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary grouping source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if self.declared_manifest_version != V11_MANIFEST_VERSION:
            raise ValueError(f"Binary grouping manifest version must be {V11_MANIFEST_VERSION}")
        return self


class BinaryGroupingCaseResult(NewsModel):
    status: Literal["completed", "failed"]
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    case_id: NonEmptyText
    left_article_version_id: Sha256
    right_article_version_id: Sha256
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
    predicted_same_group: bool | None
    expected_same_group: bool
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
    def require_consistent_outcome(self) -> BinaryGroupingCaseResult:
        validate_binary_attempts(self.attempts)
        accounting = binary_attempt_totals(self.attempts)
        if accounting != (
            self.input_tokens,
            self.output_tokens,
            self.cost_usd,
            self.wall_latency_ms,
        ):
            raise ValueError("Binary grouping case accounting does not match its attempts")
        if self.status == "completed":
            if None in (
                self.actual_model,
                self.adapter_request_id,
                self.probability,
                self.predicted_same_group,
            ):
                raise ValueError("Completed binary grouping cases require an observation")
            if (self.errors, self.passed) != (
                (),
                self.predicted_same_group == self.expected_same_group,
            ):
                raise ValueError("Completed binary grouping case has an inconsistent verdict")
            final = self.attempts[-1]
            if (
                final.status,
                final.provider_request_id,
                final.actual_model,
                final.probability,
            ) != (
                "completed",
                self.provider_request_id,
                self.actual_model,
                self.probability,
            ):
                raise ValueError("Completed binary grouping case does not match its final attempt")
        elif (
            (
                self.actual_model,
                self.adapter_request_id,
                self.provider_request_id,
                self.probability,
                self.predicted_same_group,
            ),
            self.passed,
            bool(self.errors),
        ) != ((None, None, None, None, None), False, True):
            raise ValueError("Failed binary grouping cases require only error evidence")
        return self


class BinaryGroupingMetrics(NewsModel):
    passed_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    accuracy: Annotated[Decimal, Field(ge=0, le=1)]
    same_group_recall: Annotated[Decimal, Field(ge=0, le=1)]
    different_group_preservation: Annotated[Decimal, Field(ge=0, le=1)]
    control_preservation: Annotated[Decimal, Field(ge=0, le=1)]
    false_merge_ids: tuple[str, ...]
    false_split_ids: tuple[str, ...]
    request_count: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    total_cost_usd: Annotated[Decimal, Field(ge=0)]
    total_attempt_latency_ms: Annotated[int, Field(ge=0)]
    p50_wall_latency_ms: Annotated[Decimal, Field(ge=0)]
    p95_wall_latency_ms: Annotated[Decimal, Field(ge=0)]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> BinaryGroupingMetrics:
        if self.completed_cases + self.failed_cases != self.total_cases:
            raise ValueError("Binary grouping completion counts must cover all cases")
        if self.passed_cases > self.completed_cases:
            raise ValueError("Binary grouping passed cases cannot exceed completed cases")
        if self.input_tokens + self.output_tokens != self.total_tokens:
            raise ValueError("Binary grouping token totals are inconsistent")
        return self


class BinaryGroupingRunResult(NewsModel):
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    target: BinaryTarget
    trial_ref: NonEmptyText
    question_id: NonEmptyText
    question_digest: Sha256
    cases: tuple[BinaryGroupingCaseResult, ...]
    metrics: BinaryGroupingMetrics

    @model_validator(mode="after")
    def require_consistent_run(self) -> BinaryGroupingRunResult:
        if any(
            (
                case.source_artifact_id,
                case.declared_manifest_version,
                case.target_id,
                case.provider,
                case.trial_ref,
                case.requested_model,
                case.question_id,
                case.question_digest,
                case.execution_policy_id,
                case.execution_policy_digest,
                case.cost_basis,
            )
            != (
                self.source_artifact_id,
                self.declared_manifest_version,
                self.target.target_id,
                self.target.provider,
                self.trial_ref,
                self.target.requested_model,
                self.question_id,
                self.question_digest,
                self.target.execution_policy_id,
                self.target.execution_policy_digest,
                self.target.cost_basis,
            )
            for case in self.cases
        ):
            raise ValueError("Binary grouping case identity does not match its run")
        if self.metrics != binary_grouping_metrics(self.cases):
            raise ValueError("Binary grouping run metrics do not match its cases")
        return self


class BinaryGroupingEvaluationResult(NewsModel):
    source_artifact_id: Literal["083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"]
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    question_id: NonEmptyText
    question_digest: Sha256
    targets: tuple[BinaryTarget, ...]
    trial_refs: tuple[NonEmptyText, ...]
    case_ids: tuple[NonEmptyText, ...]
    runs: tuple[BinaryGroupingRunResult, ...]

    @model_validator(mode="after")
    def require_exact_ordered_coverage(self) -> BinaryGroupingEvaluationResult:
        target_ids = tuple(target.target_id for target in self.targets)
        if (target_ids, len(set(target_ids))) != (tuple(sorted(target_ids)), len(target_ids)):
            raise ValueError("Binary grouping targets must be unique and ordered")
        if (self.trial_refs, len(set(self.trial_refs))) != (
            tuple(sorted(self.trial_refs)),
            len(self.trial_refs),
        ):
            raise ValueError("Binary grouping trial references must be unique and ordered")
        expected_runs = tuple(
            (target_id, trial_ref) for target_id in target_ids for trial_ref in self.trial_refs
        )
        if tuple((run.target.target_id, run.trial_ref) for run in self.runs) != expected_runs:
            raise ValueError("Binary grouping runs must exactly cover targets and trials")
        if any(
            (
                run.source_artifact_id,
                run.declared_manifest_version,
                run.question_id,
                run.question_digest,
                tuple(case.case_id for case in run.cases),
            )
            != (
                self.source_artifact_id,
                self.declared_manifest_version,
                self.question_id,
                self.question_digest,
                self.case_ids,
            )
            for run in self.runs
        ):
            raise ValueError("Binary grouping runs must exactly cover the frozen cases")
        return self


def load_v11_binary_grouping_source(pin_content: bytes | str) -> BinaryGroupingSource:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    if pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID:
        raise ValueError(f"Binary grouping source artifact must be {V11_SOURCE_ARTIFACT_ID}")
    manifest_reference = read_news_evaluation_artifact_references((V11_SOURCE_ARTIFACT_ID,))[
        V11_SOURCE_ARTIFACT_ID
    ]
    manifest = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(
            manifest_reference.r2_key,
            manifest_reference.content_digest,
        ),
        strict=True,
    )
    specs = tuple(case for case in manifest.cases if isinstance(case, GroupingEvaluationSpec))
    if len(specs) != 23:
        raise ValueError(f"Binary grouping benchmark requires 23 cases, received {len(specs)}")
    cases = tuple(_load_grouping_case(spec) for spec in specs)
    return BinaryGroupingSource(
        pin=pin,
        manifest_reference=manifest_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
    )


def adapt_binary_grouping_case(case: BinaryGroupingCase) -> BinaryBenchmarkCase:
    request = build_grouping_binary_request(case)
    articles = tuple(
        sorted((case.left_article, case.right_article), key=lambda item: item.version_id)
    )
    return BinaryBenchmarkCase(
        case_id=case.case_id,
        identity=(
            f"day:{case.day.isoformat()}",
            *(f"article:{article.version_id}:{article.content_digest}" for article in articles),
            f"expected-same-group:{str(case.expected_same_group).lower()}",
        ),
        control=case.control,
        judgments=(
            BinaryJudgmentCase(
                judgment_id=request.question.question_id,
                request=request,
                expected=case.expected_same_group,
            ),
        ),
    )


def adapt_binary_grouping_cases(
    cases: tuple[BinaryGroupingCase, ...],
) -> tuple[BinaryBenchmarkCase, ...]:
    return tuple(adapt_binary_grouping_case(case) for case in cases)


def _load_grouping_case(spec: GroupingEvaluationSpec) -> BinaryGroupingCase:
    articles = {item.article.version_id: item.article for item in spec.articles}
    left = articles[spec.left_article_version_id]
    right = articles[spec.right_article_version_id]
    return BinaryGroupingCase(
        case_id=spec.case_id,
        control=spec.control,
        day=spec.day,
        left_article=left,
        left_value=_load_article(left),
        right_article=right,
        right_value=_load_article(right),
        expected_same_group=spec.expected_same_group,
    )


def _load_article(reference: ArtifactReference) -> ExtractedArticle:
    return ExtractedArticle.model_validate_json(
        read_verified_r2_object(reference.r2_key, reference.content_digest), strict=True
    )


def run_binary_grouping_evaluation(
    source: BinaryGroupingSource,
    *,
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    evaluators: Mapping[TargetId, BinaryEvaluator],
    execution_plan: tuple[BinaryGroupingExecutionKey, ...] | None = None,
    reusable_cases: Mapping[Sha256, BinaryGroupingCaseResult] | None = None,
    on_case_result: Callable[[BinaryGroupingCaseResult], None] | None = None,
    spend: BinarySpendLedger | None = None,
    dry_run: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> BinaryGroupingEvaluationResult:
    ordered_targets, ordered_trials = _ordered_evaluation_inputs(targets, trial_refs, evaluators)

    prepared = tuple((case, build_grouping_binary_request(case)) for case in source.cases)
    target_by_id: dict[TargetId, BinaryTarget] = {
        target.target_id: target for target in ordered_targets
    }
    request_by_case = {case.case_id: (case, request) for case, request in prepared}
    canonical_plan: tuple[BinaryGroupingExecutionKey, ...] = tuple(
        (target.target_id, trial, case.case_id)
        for target in ordered_targets
        for trial in ordered_trials
        for case, _request in prepared
    )
    selected_plan: tuple[BinaryGroupingExecutionKey, ...] = execution_plan or canonical_plan
    if len(selected_plan) != len(set(selected_plan)) or set(selected_plan) != set(canonical_plan):
        raise ValueError(
            "Binary grouping execution plan must exactly cover targets, trials and cases"
        )

    reusable = dict(reusable_cases or {})
    expected = {
        _request_id(target_by_id[target_id], trial, case_id, request_by_case[case_id][1]): (
            target_by_id[target_id],
            trial,
            request_by_case[case_id][0],
            request_by_case[case_id][1],
        )
        for target_id, trial, case_id in canonical_plan
    }
    _validate_reusable_cases(reusable, expected)
    results = _execute_plan(
        selected_plan,
        target_by_id=target_by_id,
        request_by_case=request_by_case,
        reusable=reusable,
        evaluators=evaluators,
        on_case_result=on_case_result,
        spend=spend,
        dry_run=dry_run,
        clock=clock,
    )

    runs = tuple(
        _build_run(
            target,
            trial,
            tuple(results[(target.target_id, trial, case.case_id)] for case, _ in prepared),
        )
        for target in ordered_targets
        for trial in ordered_trials
    )
    return BinaryGroupingEvaluationResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        question_id=GROUPING_BINARY_QUESTION.question_id,
        question_digest=GROUPING_BINARY_QUESTION.semantic_digest,
        targets=ordered_targets,
        trial_refs=ordered_trials,
        case_ids=tuple(case.case_id for case in source.cases),
        runs=runs,
    )


def binary_grouping_metrics(
    cases: tuple[BinaryGroupingCaseResult, ...],
) -> BinaryGroupingMetrics:
    completed = _matching_cases(cases, lambda case: case.status == "completed")
    positives = _matching_cases(completed, lambda case: case.expected_same_group)
    negatives = _matching_cases(completed, lambda case: not case.expected_same_group)
    controls = _matching_cases(completed, lambda case: case.control)
    attempts = _all_attempts(cases)
    latencies = tuple(attempt.latency_ms for attempt in attempts)
    input_tokens, output_tokens = _attempt_token_totals(attempts)
    return BinaryGroupingMetrics(
        passed_cases=_count_cases(cases, lambda case: case.passed),
        completed_cases=len(completed),
        failed_cases=len(cases) - len(completed),
        total_cases=len(cases),
        accuracy=_ratio(_count_cases(completed, lambda case: case.passed), len(completed)),
        same_group_recall=_ratio(
            _count_cases(positives, lambda case: case.predicted_same_group is True),
            len(positives),
        ),
        different_group_preservation=_ratio(
            _count_cases(negatives, lambda case: case.predicted_same_group is False),
            len(negatives),
        ),
        control_preservation=_ratio(
            _count_cases(controls, lambda case: case.passed), len(controls)
        ),
        false_merge_ids=_matching_case_ids(
            negatives, lambda case: case.predicted_same_group is True
        ),
        false_split_ids=_matching_case_ids(
            positives, lambda case: case.predicted_same_group is False
        ),
        request_count=len(attempts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_cost_usd=sum(
            (attempt.cost_usd or Decimal(0) for attempt in attempts), start=Decimal(0)
        ),
        total_attempt_latency_ms=sum(latencies),
        p50_wall_latency_ms=decimal_linear_percentile(latencies, Decimal("0.50")),
        p95_wall_latency_ms=decimal_linear_percentile(latencies, Decimal("0.95")),
    )


def _ordered_evaluation_inputs(
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    evaluators: Mapping[TargetId, BinaryEvaluator],
) -> tuple[tuple[BinaryTarget, ...], tuple[str, ...]]:
    ordered_targets = tuple(sorted(targets, key=lambda target: target.target_id))
    ordered_trials = tuple(sorted(trial_refs))
    target_ids = tuple(target.target_id for target in ordered_targets)
    if not ordered_targets or len(set(target_ids)) != len(target_ids):
        raise ValueError("Binary grouping targets must be non-empty and unique")
    if not ordered_trials or any(not value.strip() for value in ordered_trials):
        raise ValueError("Binary grouping trial references must be non-empty")
    if len(set(ordered_trials)) != len(ordered_trials):
        raise ValueError("Binary grouping trial references must be unique")
    if set(evaluators) != set(target_ids):
        raise ValueError("Binary grouping evaluators must exactly match selected targets")
    return ordered_targets, ordered_trials


def _validate_reusable_cases(
    reusable: Mapping[Sha256, BinaryGroupingCaseResult],
    expected: Mapping[Sha256, BinaryGroupingPreparedExecution],
) -> None:
    if not set(reusable).issubset(expected):
        raise ValueError("Reusable binary grouping cases do not match this evaluation")
    for request_id, result in reusable.items():
        _validate_reusable(result, *expected[request_id])


def _execute_plan(
    selected_plan: tuple[BinaryGroupingExecutionKey, ...],
    *,
    target_by_id: Mapping[TargetId, BinaryTarget],
    request_by_case: Mapping[str, BinaryGroupingPreparedCase],
    reusable: Mapping[Sha256, BinaryGroupingCaseResult],
    evaluators: Mapping[TargetId, BinaryEvaluator],
    on_case_result: Callable[[BinaryGroupingCaseResult], None] | None,
    spend: BinarySpendLedger | None,
    dry_run: bool,
    clock: Callable[[], float],
) -> dict[BinaryGroupingExecutionKey, BinaryGroupingCaseResult]:
    results: dict[BinaryGroupingExecutionKey, BinaryGroupingCaseResult] = {}
    for target_id, trial, case_id in selected_plan:
        target = target_by_id[target_id]
        case, request = request_by_case[case_id]
        request_id = _request_id(target, trial, case_id, request)
        result = reusable.get(request_id)
        if result is None:
            if spend is not None:
                spend.reserve(
                    Decimal(0)
                    if dry_run
                    else BINARY_TARGET_REGISTRY[target_id].maximum_request_cost_usd
                )
            result = _evaluate_case(target, trial, case, request, evaluators[target_id], clock)
            if spend is not None:
                try:
                    spend.settle(result.cost_usd)
                except BinarySpendLimitExceeded:
                    if on_case_result is not None:
                        on_case_result(result)
                    raise
            if on_case_result is not None:
                on_case_result(result)
        results[(target_id, trial, case_id)] = result
    return results


def _matching_cases(
    cases: tuple[BinaryGroupingCaseResult, ...],
    predicate: Callable[[BinaryGroupingCaseResult], bool],
) -> tuple[BinaryGroupingCaseResult, ...]:
    return tuple(case for case in cases if predicate(case))


def _matching_case_ids(
    cases: tuple[BinaryGroupingCaseResult, ...],
    predicate: Callable[[BinaryGroupingCaseResult], bool],
) -> tuple[str, ...]:
    return tuple(case.case_id for case in cases if predicate(case))


def _count_cases(
    cases: tuple[BinaryGroupingCaseResult, ...],
    predicate: Callable[[BinaryGroupingCaseResult], bool],
) -> int:
    return sum(predicate(case) for case in cases)


def _all_attempts(
    cases: tuple[BinaryGroupingCaseResult, ...],
) -> tuple[BinaryAttemptEvidence, ...]:
    return tuple(attempt for case in cases for attempt in case.attempts)


def _attempt_token_totals(attempts: tuple[BinaryAttemptEvidence, ...]) -> tuple[int, int]:
    return (
        sum(attempt.input_tokens or 0 for attempt in attempts),
        sum(attempt.output_tokens or 0 for attempt in attempts),
    )


def _evaluate_case(
    target: BinaryTarget,
    trial_ref: str,
    case: BinaryGroupingCase,
    request: BinaryRequest,
    evaluator: BinaryEvaluator,
    clock: Callable[[], float],
) -> BinaryGroupingCaseResult:
    request_id = _request_id(target, trial_ref, case.case_id, request)
    invocation = invoke_binary_evaluator(evaluator, request, trial_ref, clock=clock)
    observation = invocation.observation
    error = invocation.error
    if error is not None:
        return _case_result(
            target,
            trial_ref,
            case,
            request,
            request_id,
            invocation.attempts,
            error=error,
        )
    if observation is None:
        raise AssertionError("Successful binary invocation omitted its observation")
    predicted = binary_decision(observation.probability, request.question.threshold)
    return _case_result(
        target,
        trial_ref,
        case,
        request,
        request_id,
        invocation.attempts,
        observation=observation,
        predicted=predicted,
    )


def _case_result(
    target: BinaryTarget,
    trial_ref: str,
    case: BinaryGroupingCase,
    request: BinaryRequest,
    request_id: Sha256,
    attempts: tuple[BinaryAttemptEvidence, ...],
    *,
    observation: BinaryProbabilityObservation | None = None,
    predicted: bool | None = None,
    error: Exception | None = None,
) -> BinaryGroupingCaseResult:
    input_tokens, output_tokens, cost, latency = binary_attempt_totals(attempts)
    return BinaryGroupingCaseResult(
        status="completed" if error is None else "failed",
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        case_id=case.case_id,
        left_article_version_id=case.left_article.version_id,
        right_article_version_id=case.right_article.version_id,
        target_id=target.target_id,
        provider=target.provider,
        trial_ref=trial_ref,
        requested_model=target.requested_model,
        actual_model=observation.model if observation else None,
        question_id=request.question.question_id,
        question_digest=request.question.semantic_digest,
        state_digest=request.state_digest,
        execution_policy_id=target.execution_policy_id,
        execution_policy_digest=target.execution_policy_digest,
        request_id=request_id,
        adapter_request_id=observation.request_id if observation else None,
        provider_request_id=observation.provider_request_id if observation else None,
        probability=observation.probability if observation else None,
        predicted_same_group=predicted,
        expected_same_group=case.expected_same_group,
        passed=predicted == case.expected_same_group if predicted is not None else False,
        control=case.control,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_latency_ms=latency,
        cost_usd=cost,
        cost_basis=target.cost_basis,
        errors=(
            ()
            if error is None
            else (
                BinaryRelevanceError(
                    error_type=type(error).__name__, message=str(error) or type(error).__name__
                ),
            )
        ),
        attempts=attempts,
    )


def _validate_reusable(
    result: BinaryGroupingCaseResult,
    target: BinaryTarget,
    trial_ref: str,
    case: BinaryGroupingCase,
    request: BinaryRequest,
) -> None:
    actual_identity = (
        result.source_artifact_id,
        result.declared_manifest_version,
        result.case_id,
        result.left_article_version_id,
        result.right_article_version_id,
        result.target_id,
        result.provider,
        result.trial_ref,
        result.requested_model,
        result.question_id,
        result.question_digest,
        result.state_digest,
        result.execution_policy_id,
        result.execution_policy_digest,
        result.request_id,
        result.expected_same_group,
        result.control,
        result.cost_basis,
    )
    expected_identity = (
        V11_SOURCE_ARTIFACT_ID,
        V11_MANIFEST_VERSION,
        case.case_id,
        case.left_article.version_id,
        case.right_article.version_id,
        target.target_id,
        target.provider,
        trial_ref,
        target.requested_model,
        request.question.question_id,
        request.question.semantic_digest,
        request.state_digest,
        target.execution_policy_id,
        target.execution_policy_digest,
        _request_id(target, trial_ref, case.case_id, request),
        case.expected_same_group,
        case.control,
        target.cost_basis,
    )
    if actual_identity != expected_identity:
        raise ValueError("Reusable binary grouping case identity does not match its execution")


def _build_run(
    target: BinaryTarget,
    trial_ref: str,
    cases: tuple[BinaryGroupingCaseResult, ...],
) -> BinaryGroupingRunResult:
    return BinaryGroupingRunResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        target=target,
        trial_ref=trial_ref,
        question_id=GROUPING_BINARY_QUESTION.question_id,
        question_digest=GROUPING_BINARY_QUESTION.semantic_digest,
        cases=cases,
        metrics=binary_grouping_metrics(cases),
    )


def _request_id(
    target: BinaryTarget,
    trial_ref: str,
    case_id: str,
    request: BinaryRequest,
) -> Sha256:
    return hashlib.sha256(
        "\0".join(
            (
                V11_SOURCE_ARTIFACT_ID,
                "grouping",
                case_id,
                target.target_id,
                target.execution_policy_digest,
                trial_ref,
                request.question.semantic_digest,
                request.state_digest,
            )
        ).encode()
    ).hexdigest()


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)
