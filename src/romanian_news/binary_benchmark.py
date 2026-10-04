from __future__ import annotations

import inspect
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptCallback,
    BinaryAttemptError,
    BinaryAttemptEvidence,
    BinaryProbabilityObservation,
    BinaryQuestion,
    BinaryRequest,
    binary_attempt_totals,
    binary_decision,
    validate_binary_attempts,
)
from romanian_news.analysis.gemini_binary import (
    GEMINI_25_FLASH_BINARY_EXECUTION_POLICY,
    GEMINI_38_FLASH_BINARY_EXECUTION_POLICY,
    evaluate_gemini_binary,
    gemini_binary_execution_policy_digest,
)
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    evaluate_jev_relevance,
    jev_execution_policy_digest,
)
from romanian_news.config import OPENROUTER_API_KEY, TYPESAFE_API_KEY
from romanian_news.identity import canonical_json, sha256

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
TargetId = Literal[
    "openrouter-gemini-2.5-flash",
    "openrouter-gemini-3.8-flash",
    "typesafe-jev",
]
CostBasis = Literal["estimated-input-rate", "provider-reported"]
ExecutionMode = Literal["live", "dry_run"]
BenchmarkId = Literal["grouping", "ranking", "tier", "confidence", "daily_theme"]
BinaryEvaluator = Callable[..., BinaryProbabilityObservation]


class TypesafeJevTarget(NewsModel):
    target_id: Literal["typesafe-jev"] = "typesafe-jev"
    provider: Literal["typesafe"] = "typesafe"
    requested_model: Literal["jev-1.13.0"] = "jev-1.13.0"
    execution_policy_id: Literal["typesafe-jev-1.13.0-v1"] = "typesafe-jev-1.13.0-v1"
    execution_policy_digest: Sha256
    cost_basis: Literal["estimated-input-rate"] = "estimated-input-rate"


class OpenRouterGeminiTarget(NewsModel):
    target_id: Literal["openrouter-gemini-2.5-flash", "openrouter-gemini-3.8-flash"]
    provider: Literal["openrouter"] = "openrouter"
    requested_model: Literal["google/gemini-2.5-flash", "google/gemini-3.8-flash"]
    execution_policy_id: Literal["openrouter-gemini-binary-v1"] = "openrouter-gemini-binary-v1"
    execution_policy_digest: Sha256
    cost_basis: Literal["provider-reported"] = "provider-reported"

    @model_validator(mode="after")
    def require_matching_model_identity(self) -> OpenRouterGeminiTarget:
        expected_model = {
            "openrouter-gemini-2.5-flash": "google/gemini-2.5-flash",
            "openrouter-gemini-3.8-flash": "google/gemini-3.8-flash",
        }[self.target_id]
        if self.requested_model != expected_model:
            raise ValueError("OpenRouter Gemini target ID and model must match")
        return self


BinaryTarget = Annotated[
    TypesafeJevTarget | OpenRouterGeminiTarget,
    Field(discriminator="target_id"),
]

TYPESAFE_JEV_TARGET = TypesafeJevTarget(execution_policy_digest=jev_execution_policy_digest())
OPENROUTER_GEMINI_25_TARGET = OpenRouterGeminiTarget(
    target_id="openrouter-gemini-2.5-flash",
    requested_model="google/gemini-2.5-flash",
    execution_policy_digest=gemini_binary_execution_policy_digest(),
)
OPENROUTER_GEMINI_38_TARGET = OpenRouterGeminiTarget(
    target_id="openrouter-gemini-3.8-flash",
    requested_model="google/gemini-3.8-flash",
    execution_policy_digest=gemini_binary_execution_policy_digest(
        GEMINI_38_FLASH_BINARY_EXECUTION_POLICY
    ),
)


class BinaryTargetRegistration(NewsModel):
    target: BinaryTarget
    maximum_request_cost_usd: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]


_TARGET_REGISTRATIONS = (
    BinaryTargetRegistration(
        target=OPENROUTER_GEMINI_25_TARGET,
        maximum_request_cost_usd=Decimal("0.020"),
    ),
    BinaryTargetRegistration(
        target=OPENROUTER_GEMINI_38_TARGET,
        maximum_request_cost_usd=Decimal("0.020"),
    ),
    BinaryTargetRegistration(
        target=TYPESAFE_JEV_TARGET,
        maximum_request_cost_usd=Decimal("0.008"),
    ),
)
BINARY_TARGET_REGISTRY: Mapping[TargetId, BinaryTargetRegistration] = MappingProxyType(
    {registration.target.target_id: registration for registration in _TARGET_REGISTRATIONS}
)
REGISTERED_BINARY_TARGETS = tuple(registration.target for registration in _TARGET_REGISTRATIONS)


class BinaryBenchmarkDefinition(NewsModel):
    benchmark: BenchmarkId
    case_count: Annotated[int, Field(gt=0)]
    scored_unit_count: Annotated[int, Field(gt=0)]
    questions: Annotated[tuple[BinaryQuestion, ...], Field(min_length=1)]
    calls_per_model_trial: Annotated[int, Field(gt=0)]
    state_format: NonEmptyText
    metrics: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    limitation: NonEmptyText
    composition_digest: Sha256 | None = None
    spend_ceiling_usd: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)] = Decimal("10.00")

    @model_validator(mode="after")
    def require_consistent_registration(self) -> BinaryBenchmarkDefinition:
        question_ids = tuple(question.question_id for question in self.questions)
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Binary benchmark question IDs must be unique")
        if self.benchmark == "tier":
            if len(self.questions) != 2 or self.composition_digest is None:
                raise ValueError("Tier requires two judgments and a composition digest")
        elif len(self.questions) != 1 or self.composition_digest is not None:
            raise ValueError("Non-tier binary benchmarks require one judgment and no composition")
        return self

    @property
    def question_digests(self) -> tuple[Sha256, ...]:
        return tuple(question.semantic_digest for question in self.questions)

    @property
    def concern(self) -> BenchmarkId:
        return self.benchmark


class BinaryJudgmentCase(NewsModel):
    judgment_id: NonEmptyText
    request: BinaryRequest
    expected: bool


class BinaryBenchmarkCase(NewsModel):
    case_id: NonEmptyText
    identity: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    control: bool = False
    judgments: Annotated[tuple[BinaryJudgmentCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_judgments(self) -> BinaryBenchmarkCase:
        judgment_ids = tuple(judgment.judgment_id for judgment in self.judgments)
        if len(set(judgment_ids)) != len(judgment_ids):
            raise ValueError("Binary benchmark judgment IDs must be unique within a case")
        return self


class BinaryCaseIdentity(NewsModel):
    case_id: NonEmptyText
    identity: tuple[NonEmptyText, ...]
    judgments: tuple[tuple[NonEmptyText, Sha256, Sha256], ...]


class BinaryExecutionIdentity(NewsModel):
    benchmark: BenchmarkId
    source_artifact_id: NonEmptyText
    declared_manifest_version: NonEmptyText
    question_digests: tuple[Sha256, ...]
    composition_digest: Sha256 | None
    cases: tuple[BinaryCaseIdentity, ...]
    targets: tuple[BinaryTarget, ...]
    trial_refs: tuple[NonEmptyText, ...]
    execution_mode: ExecutionMode
    execution_ref: NonEmptyText

    @property
    def digest(self) -> Sha256:
        return canonical_identity_hash(self.model_dump(mode="json"))


class BinaryJudgmentResult(NewsModel):
    status: Literal["completed", "failed"]
    request_id: Sha256
    case_id: NonEmptyText
    judgment_id: NonEmptyText
    target_id: TargetId
    trial_ref: NonEmptyText
    question_id: NonEmptyText
    question_digest: Sha256
    state_digest: Sha256
    expected: bool
    verdict: bool | None
    probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None
    passed: bool
    actual_model: NonEmptyText | None
    provider_request_id: NonEmptyText | None
    attempts: Annotated[tuple[BinaryAttemptEvidence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_consistent_result(self) -> BinaryJudgmentResult:
        validate_binary_attempts(self.attempts)
        final = self.attempts[-1]
        if self.status == "completed":
            if self.verdict is None or self.probability is None or self.actual_model is None:
                raise ValueError("Completed binary judgments require an observation")
            if (
                final.status != "completed"
                or final.actual_model != self.actual_model
                or final.provider_request_id != self.provider_request_id
                or final.probability != self.probability
            ):
                raise ValueError("Completed binary judgment does not match its final attempt")
            if self.passed != (self.verdict == self.expected):
                raise ValueError("Binary judgment pass result conflicts with its verdict")
        elif (
            self.verdict is not None
            or self.probability is not None
            or self.actual_model is not None
            or self.provider_request_id is not None
            or self.passed
            or final.status != "terminal_error"
        ):
            raise ValueError("Failed binary judgments cannot contain a verdict or pass")
        return self

    @property
    def cost_usd(self) -> Decimal:
        return binary_attempt_totals(self.attempts)[2]

    @property
    def latency_ms(self) -> int:
        return binary_attempt_totals(self.attempts)[3]


class BinaryBenchmarkEvaluationResult(NewsModel):
    identity: BinaryExecutionIdentity
    identity_digest: Sha256
    results: tuple[BinaryJudgmentResult, ...]

    @model_validator(mode="after")
    def require_exact_identity_and_coverage(self) -> BinaryBenchmarkEvaluationResult:
        if self.identity_digest != self.identity.digest:
            raise ValueError("Binary benchmark result identity digest does not match its identity")
        expected = {
            (target.target_id, trial_ref, case.case_id, judgment[0])
            for target in self.identity.targets
            for trial_ref in self.identity.trial_refs
            for case in self.identity.cases
            for judgment in case.judgments
        }
        actual = {
            (result.target_id, result.trial_ref, result.case_id, result.judgment_id)
            for result in self.results
        }
        if len(actual) != len(self.results) or actual != expected:
            raise ValueError("Binary benchmark results must exactly cover registered executions")
        return self


@dataclass(frozen=True)
class BinaryExecutionItem:
    target_id: TargetId
    trial_ref: str
    case_id: str
    judgment_id: str = "judgment"


@dataclass(frozen=True)
class BinaryInvocation:
    observation: BinaryProbabilityObservation | None
    attempts: tuple[BinaryAttemptEvidence, ...]
    error: Exception | None


class BinarySpendLimitExceeded(RuntimeError):
    pass


class BinaryModelIdentityMismatch(RuntimeError):
    pass


class BinarySpendLedger:
    ceiling_usd: Decimal
    actual_spend_usd: Decimal

    def __init__(
        self,
        ceiling_usd: Decimal,
        *,
        actual_spend_usd: Decimal | None = None,
    ) -> None:
        actual = Decimal(0) if actual_spend_usd is None else actual_spend_usd
        if ceiling_usd <= 0 or actual < 0:
            raise ValueError("Binary spend values must be non-negative with a positive ceiling")
        if actual > ceiling_usd:
            raise ValueError("Binary actual spend cannot exceed the ceiling")
        self.ceiling_usd = ceiling_usd
        self.actual_spend_usd = actual
        self._reservation: Decimal | None = None

    def reserve(self, maximum_cost_usd: Decimal) -> None:
        if maximum_cost_usd < 0 or self._reservation is not None:
            raise ValueError("Binary spend reservation is invalid")
        if self.actual_spend_usd + maximum_cost_usd > self.ceiling_usd:
            raise BinarySpendLimitExceeded(
                "Binary benchmark stopped before the next request could exceed the USD ceiling"
            )
        self._reservation = maximum_cost_usd

    def settle(self, actual_cost_usd: Decimal) -> None:
        reserved = self._reservation
        if reserved is None or actual_cost_usd < 0:
            raise ValueError("Binary spend settlement requires an active reservation")
        self._reservation = None
        self.actual_spend_usd += actual_cost_usd
        if actual_cost_usd > reserved or self.actual_spend_usd > self.ceiling_usd:
            raise BinarySpendLimitExceeded(
                "Provider accounting exceeded the registered maximum request cost"
            )

    def cancel(self) -> None:
        if self._reservation is None:
            raise ValueError("Binary spend cancellation requires an active reservation")
        self._reservation = None


CaseResultT = TypeVar("CaseResultT", bound=NewsModel)


class BinaryCheckpoint(NewsModel, Generic[CaseResultT]):
    schema_version: Literal["binary-benchmark-checkpoint-v1"] = "binary-benchmark-checkpoint-v1"
    identity: BinaryExecutionIdentity
    identity_digest: Sha256
    actual_spend_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    cases: tuple[CaseResultT, ...]

    @model_validator(mode="after")
    def require_matching_identity(self) -> BinaryCheckpoint[CaseResultT]:
        if self.identity_digest != self.identity.digest:
            raise ValueError("Binary checkpoint identity digest does not match its identity")
        return self


def build_execution_identity(
    definition: BinaryBenchmarkDefinition,
    *,
    source_artifact_id: str,
    declared_manifest_version: str,
    cases: tuple[BinaryBenchmarkCase, ...],
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    execution_mode: ExecutionMode,
    execution_ref: str,
) -> BinaryExecutionIdentity:
    _validate_benchmark_inputs(definition, cases, targets, trial_refs)
    if not execution_ref.strip():
        raise ValueError("Binary benchmark execution reference must not be empty")
    return BinaryExecutionIdentity(
        benchmark=definition.benchmark,
        source_artifact_id=source_artifact_id,
        declared_manifest_version=declared_manifest_version,
        question_digests=definition.question_digests,
        composition_digest=definition.composition_digest,
        cases=tuple(
            BinaryCaseIdentity(
                case_id=case.case_id,
                identity=case.identity,
                judgments=tuple(
                    (
                        judgment.judgment_id,
                        judgment.request.question.semantic_digest,
                        judgment.request.state_digest,
                    )
                    for judgment in case.judgments
                ),
            )
            for case in cases
        ),
        targets=tuple(sorted(targets, key=lambda target: target.target_id)),
        trial_refs=tuple(sorted(trial_refs)),
        execution_mode=execution_mode,
        execution_ref=execution_ref,
    )


def run_registered_binary_benchmark(
    definition: BinaryBenchmarkDefinition,
    identity: BinaryExecutionIdentity,
    cases: tuple[BinaryBenchmarkCase, ...],
    evaluators: Mapping[TargetId, BinaryEvaluator],
    *,
    execution_plan: tuple[BinaryExecutionItem, ...] | None = None,
    reusable_results: Mapping[Sha256, BinaryJudgmentResult] | None = None,
    on_result: Callable[[BinaryJudgmentResult], None] | None = None,
    spend: BinarySpendLedger | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BinaryBenchmarkEvaluationResult:
    targets = identity.targets
    trials = identity.trial_refs
    _validate_benchmark_inputs(definition, cases, targets, trials)
    expected_identity = build_execution_identity(
        definition,
        source_artifact_id=identity.source_artifact_id,
        declared_manifest_version=identity.declared_manifest_version,
        cases=cases,
        targets=targets,
        trial_refs=trials,
        execution_mode=identity.execution_mode,
        execution_ref=identity.execution_ref,
    )
    if identity != expected_identity:
        raise ValueError("Binary benchmark identity does not match its registered inputs")
    if set(evaluators) != {target.target_id for target in targets}:
        raise ValueError("Binary benchmark evaluators must exactly match selected targets")

    case_by_id = {case.case_id: case for case in cases}
    judgment_by_key = {
        (case.case_id, judgment.judgment_id): judgment
        for case in cases
        for judgment in case.judgments
    }
    canonical_plan = rotated_execution_plan(
        targets,
        trials,
        tuple((case.case_id, tuple(j.judgment_id for j in case.judgments)) for case in cases),
    )
    selected_plan = execution_plan or canonical_plan
    if len(selected_plan) != len(set(selected_plan)) or set(selected_plan) != set(canonical_plan):
        raise ValueError("Binary benchmark execution plan must exactly cover registered judgments")

    expected_results = {
        _judgment_request_id(
            identity,
            item,
            case_by_id[item.case_id],
            judgment_by_key[(item.case_id, item.judgment_id)],
        ): item
        for item in canonical_plan
    }
    reusable = _validated_reusable_results(reusable_results, expected_results, judgment_by_key)

    ledger = spend or BinarySpendLedger(definition.spend_ceiling_usd)
    target_by_id = {target.target_id: target for target in targets}
    results = dict(reusable)
    for item in selected_plan:
        case = case_by_id[item.case_id]
        judgment = judgment_by_key[(item.case_id, item.judgment_id)]
        request_id = _judgment_request_id(identity, item, case, judgment)
        if request_id in results:
            continue
        result = _evaluate_binary_item(
            request_id,
            item,
            judgment,
            evaluators[item.target_id],
            target_by_id[item.target_id],
            identity.execution_mode,
            ledger,
            on_result,
            clock,
        )
        results[request_id] = result
        if on_result is not None:
            on_result(result)

    return BinaryBenchmarkEvaluationResult(
        identity=identity,
        identity_digest=identity.digest,
        results=tuple(
            results[
                _judgment_request_id(
                    identity,
                    item,
                    case_by_id[item.case_id],
                    judgment_by_key[(item.case_id, item.judgment_id)],
                )
            ]
            for item in canonical_plan
        ),
    )


def _evaluate_binary_item(
    request_id: Sha256,
    item: BinaryExecutionItem,
    judgment: BinaryJudgmentCase,
    evaluator: BinaryEvaluator,
    target: BinaryTarget,
    execution_mode: ExecutionMode,
    ledger: BinarySpendLedger,
    on_result: Callable[[BinaryJudgmentResult], None] | None,
    clock: Callable[[], float],
) -> BinaryJudgmentResult:
    maximum_cost = (
        Decimal(0)
        if execution_mode == "dry_run"
        else BINARY_TARGET_REGISTRY[item.target_id].maximum_request_cost_usd
    )
    ledger.reserve(maximum_cost)
    invocation = invoke_binary_evaluator(evaluator, judgment.request, item.trial_ref, clock=clock)
    if invocation.observation is not None and not actual_model_matches_target(
        target, invocation.observation.model
    ):
        ledger.settle(binary_attempt_totals(invocation.attempts)[2])
        raise BinaryModelIdentityMismatch(
            f"Actual model {invocation.observation.model!r} does not match "
            + f"registered target {item.target_id!r}"
        )
    result = _judgment_result(request_id, item, judgment, invocation)
    try:
        ledger.settle(result.cost_usd)
    except BinarySpendLimitExceeded:
        if on_result is not None:
            on_result(result)
        raise
    return result


def _validated_reusable_results(
    reusable_results: Mapping[Sha256, BinaryJudgmentResult] | None,
    expected_results: Mapping[Sha256, BinaryExecutionItem],
    judgment_by_key: Mapping[tuple[str, str], BinaryJudgmentCase],
) -> dict[Sha256, BinaryJudgmentResult]:
    reusable = dict(reusable_results or {})
    if not set(reusable).issubset(expected_results):
        raise ValueError("Reusable binary judgments do not match this evaluation identity")
    for request_id, result in reusable.items():
        item = expected_results[request_id]
        judgment = judgment_by_key[(item.case_id, item.judgment_id)]
        _validate_reusable_result(result, request_id, item, judgment)
    return reusable


def trial_references(execution_ref: str, trials: int, *, dry_run: bool) -> tuple[str, ...]:
    if not execution_ref.strip() or trials <= 0:
        raise ValueError("Binary trial references require a non-empty execution and positive count")
    mode = "dry-run:" if dry_run else ""
    return tuple(f"{execution_ref}:{mode}trial-{index:03d}" for index in range(1, trials + 1))


def rotated_execution_plan(
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    cases: tuple[str, ...] | tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[BinaryExecutionItem, ...]:
    ordered_targets = tuple(sorted(targets, key=lambda target: target.target_id))
    if not ordered_targets:
        raise ValueError("Binary execution plan requires at least one target")
    normalized_cases = tuple(
        (case, ("judgment",)) if isinstance(case, str) else case for case in cases
    )
    return tuple(
        BinaryExecutionItem(target.target_id, trial_ref, case_id, judgment_id)
        for trial_index, trial_ref in enumerate(trial_refs)
        for target in ordered_targets[trial_index % len(ordered_targets) :]
        + ordered_targets[: trial_index % len(ordered_targets)]
        for case_id, judgment_ids in normalized_cases
        for judgment_id in judgment_ids
    )


def build_binary_evaluators(
    targets: tuple[BinaryTarget, ...], *, dry_run: bool
) -> dict[TargetId, BinaryEvaluator]:
    target_ids: set[TargetId] = {target.target_id for target in targets}
    if target_ids != set(BINARY_TARGET_REGISTRY).intersection(target_ids):
        raise ValueError("Binary evaluator construction received an unregistered target")
    if dry_run:
        return {target.target_id: _fake_evaluator(target) for target in targets}
    missing: list[str] = []
    if "typesafe-jev" in target_ids and not TYPESAFE_API_KEY:
        missing.append("TYPESAFE_API_KEY")
    if (
        any(target_id.startswith("openrouter-") for target_id in target_ids)
        and not OPENROUTER_API_KEY
    ):
        missing.append("OPENROUTER_API_KEY")
    if missing:
        raise ValueError(f"Missing credentials for selected targets: {', '.join(missing)}")
    registered: dict[TargetId, BinaryEvaluator] = {
        "typesafe-jev": _evaluate_jev,
        "openrouter-gemini-2.5-flash": _evaluate_gemini_25,
        "openrouter-gemini-3.8-flash": _evaluate_gemini_38,
    }
    return {target_id: registered[target_id] for target_id in target_ids}


def actual_model_matches_target(target: BinaryTarget, actual_model: str) -> bool:
    if target.target_id == "typesafe-jev":
        return actual_model == target.requested_model
    return actual_model == target.requested_model or actual_model.startswith(
        f"{target.requested_model}-"
    )


def invoke_binary_evaluator(
    evaluator: BinaryEvaluator,
    request: BinaryRequest,
    trial_ref: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> BinaryInvocation:
    captured: list[BinaryAttemptEvidence] = []
    supports_callback = _supports_attempt_callback(evaluator)
    started = _clock_value(clock())
    try:
        observation = _invoke_evaluator(
            evaluator,
            request,
            trial_ref,
            captured.append if supports_callback else None,
        )
        if captured and tuple(captured) != observation.attempts:
            raise ValueError("Binary evaluator callback evidence conflicts with its observation")
        attempts = tuple(captured) or observation.attempts
        verdict = binary_decision(observation.probability, request.question.threshold)
        if observation.predicted_accepted != verdict:
            raise ValueError("Binary evaluator verdict conflicts with its probability")
        return BinaryInvocation(observation=observation, attempts=attempts, error=None)
    except Exception as error:
        if not captured:
            captured.append(
                BinaryAttemptEvidence(
                    attempt_number=1,
                    status="terminal_error",
                    provider_request_id=None,
                    actual_model=None,
                    http_status=None,
                    error=BinaryAttemptError(
                        error_type=type(error).__name__,
                        message=str(error) or type(error).__name__,
                        retryable=False,
                    ),
                    input_tokens=None,
                    output_tokens=None,
                    cost_usd=None,
                    latency_ms=_elapsed_ms(started, _clock_value(clock())),
                    probability=None,
                )
            )
        return BinaryInvocation(observation=None, attempts=tuple(captured), error=error)


def reusable_case_results(
    cases: Sequence[CaseResultT],
    *,
    retry_failed: bool,
    request_id: Callable[[CaseResultT], Sha256],
    completed: Callable[[CaseResultT], bool],
) -> dict[Sha256, CaseResultT]:
    keys = tuple(request_id(case) for case in cases)
    if len(set(keys)) != len(keys):
        raise ValueError("Binary checkpoint contains duplicate case executions")
    return {
        key: case
        for key, case in zip(keys, cases, strict=True)
        if not retry_failed or completed(case)
    }


def load_binary_checkpoint(
    path: Path,
    identity: BinaryExecutionIdentity,
    *,
    retry_failed: bool,
) -> tuple[BinaryCheckpoint[BinaryJudgmentResult], dict[Sha256, BinaryJudgmentResult]]:
    if not path.is_file():
        raise ValueError(f"Cannot resume because checkpoint does not exist: {path}")
    checkpoint = BinaryCheckpoint[BinaryJudgmentResult].model_validate_json(
        path.read_bytes(), strict=True
    )
    if checkpoint.identity != identity or checkpoint.identity_digest != identity.digest:
        raise ValueError("Binary checkpoint identity does not match this execution")
    reusable = reusable_case_results(
        checkpoint.cases,
        retry_failed=retry_failed,
        request_id=lambda result: result.request_id,
        completed=lambda result: result.status == "completed",
    )
    return checkpoint, reusable


def write_binary_checkpoint(
    path: Path,
    identity: BinaryExecutionIdentity,
    spend: BinarySpendLedger,
    cases: Sequence[BinaryJudgmentResult],
) -> None:
    checkpoint = BinaryCheckpoint[BinaryJudgmentResult](
        identity=identity,
        identity_digest=identity.digest,
        actual_spend_usd=spend.actual_spend_usd,
        cases=tuple(cases),
    )
    atomic_write(path, checkpoint.model_dump_json(indent=2).encode())


def atomic_write(path: Path, content: bytes) -> None:
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


def canonical_identity_hash(value: object) -> Sha256:
    return sha256(canonical_json(value))


def execution_identity_hash(
    *,
    benchmark: str,
    source_artifact_id: str,
    declared_manifest_version: str,
    question_digests: tuple[Sha256, ...],
    composition_digest: Sha256 | None,
    case_identities: tuple[object, ...],
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    execution_mode: ExecutionMode,
    execution_ref: str,
) -> Sha256:
    return canonical_identity_hash(
        {
            "benchmark": benchmark,
            "source_artifact_id": source_artifact_id,
            "declared_manifest_version": declared_manifest_version,
            "question_digests": question_digests,
            "composition_digest": composition_digest,
            "case_identities": case_identities,
            "targets": [target.model_dump(mode="json") for target in targets],
            "trial_refs": trial_refs,
            "execution_mode": execution_mode,
            "execution_ref": execution_ref,
        }
    )


def _validate_benchmark_inputs(
    definition: BinaryBenchmarkDefinition,
    cases: tuple[BinaryBenchmarkCase, ...],
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
) -> None:
    if len(cases) != definition.case_count:
        raise ValueError("Binary benchmark case count does not match its registration")
    if sum(len(case.judgments) for case in cases) != definition.calls_per_model_trial:
        raise ValueError("Binary benchmark judgment count does not match its registration")
    expected_questions = set(definition.question_digests)
    if any(
        {judgment.request.question.semantic_digest for judgment in case.judgments}
        != expected_questions
        for case in cases
    ):
        raise ValueError("Binary benchmark cases must use every registered question exactly once")
    target_ids = tuple(target.target_id for target in targets)
    if not targets or len(set(target_ids)) != len(target_ids):
        raise ValueError("Binary benchmark targets must be non-empty and unique")
    if any(BINARY_TARGET_REGISTRY[target.target_id].target != target for target in targets):
        raise ValueError("Binary benchmark target identity does not match the registry")
    if not trial_refs or any(not trial.strip() for trial in trial_refs):
        raise ValueError("Binary benchmark trial references must be non-empty")
    if len(set(trial_refs)) != len(trial_refs):
        raise ValueError("Binary benchmark trial references must be unique")


def _judgment_request_id(
    identity: BinaryExecutionIdentity,
    item: BinaryExecutionItem,
    case: BinaryBenchmarkCase,
    judgment: BinaryJudgmentCase,
) -> Sha256:
    return canonical_identity_hash(
        {
            "benchmark": identity.benchmark,
            "source_artifact_id": identity.source_artifact_id,
            "declared_manifest_version": identity.declared_manifest_version,
            "composition_digest": identity.composition_digest,
            "case_id": case.case_id,
            "case_identity": case.identity,
            "judgment_id": judgment.judgment_id,
            "question_digest": judgment.request.question.semantic_digest,
            "state_digest": judgment.request.state_digest,
            "target_id": item.target_id,
            "execution_policy_digest": BINARY_TARGET_REGISTRY[
                item.target_id
            ].target.execution_policy_digest,
            "trial_ref": item.trial_ref,
            "execution_mode": identity.execution_mode,
            "execution_ref": identity.execution_ref,
        }
    )


def _validate_reusable_result(
    result: BinaryJudgmentResult,
    request_id: Sha256,
    item: BinaryExecutionItem,
    judgment: BinaryJudgmentCase,
) -> None:
    if (
        result.request_id != request_id
        or result.case_id != item.case_id
        or result.judgment_id != item.judgment_id
        or result.target_id != item.target_id
        or result.trial_ref != item.trial_ref
        or result.question_id != judgment.request.question.question_id
        or result.question_digest != judgment.request.question.semantic_digest
        or result.state_digest != judgment.request.state_digest
        or result.expected != judgment.expected
    ):
        raise ValueError("Reusable binary judgment identity does not match its execution")


def _judgment_result(
    request_id: Sha256,
    item: BinaryExecutionItem,
    judgment: BinaryJudgmentCase,
    invocation: BinaryInvocation,
) -> BinaryJudgmentResult:
    observation = invocation.observation
    verdict = (
        binary_decision(observation.probability, judgment.request.question.threshold)
        if observation is not None
        else None
    )
    return BinaryJudgmentResult(
        status="completed" if observation is not None else "failed",
        request_id=request_id,
        case_id=item.case_id,
        judgment_id=item.judgment_id,
        target_id=item.target_id,
        trial_ref=item.trial_ref,
        question_id=judgment.request.question.question_id,
        question_digest=judgment.request.question.semantic_digest,
        state_digest=judgment.request.state_digest,
        expected=judgment.expected,
        verdict=verdict,
        probability=observation.probability if observation is not None else None,
        passed=verdict == judgment.expected if verdict is not None else False,
        actual_model=observation.model if observation is not None else None,
        provider_request_id=observation.provider_request_id if observation is not None else None,
        attempts=invocation.attempts,
    )


def _fake_evaluator(target: BinaryTarget) -> BinaryEvaluator:
    def evaluate(
        request: BinaryRequest,
        trial_ref: str,
        *,
        on_attempt: BinaryAttemptCallback | None = None,
    ) -> BinaryProbabilityObservation:
        _ = on_attempt
        seed = canonical_identity_hash(
            {
                "state_digest": request.state_digest,
                "target_id": target.target_id,
                "trial_ref": trial_ref,
            }
        )
        probability = Decimal(int(seed[:8], 16) % 1001) / Decimal(1000)
        return BinaryProbabilityObservation(
            request_id=seed,
            provider_request_id=f"dry-run:{seed[:24]}",
            model=target.requested_model,
            probability=probability,
            predicted_accepted=probability >= request.question.threshold,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            estimated_cost_usd=Decimal(0),
        )

    return evaluate


def _evaluate_jev(
    request: BinaryRequest,
    trial_ref: str,
    *,
    on_attempt: BinaryAttemptCallback | None = None,
) -> BinaryProbabilityObservation:
    return evaluate_jev_relevance(
        request,
        execution_ref=trial_ref,
        policy=JEV_EXECUTION_POLICY,
        on_attempt=on_attempt,
    )


def _evaluate_gemini_25(
    request: BinaryRequest,
    trial_ref: str,
    *,
    on_attempt: BinaryAttemptCallback | None = None,
) -> BinaryProbabilityObservation:
    return evaluate_gemini_binary(
        request,
        execution_ref=trial_ref,
        policy=GEMINI_25_FLASH_BINARY_EXECUTION_POLICY,
        on_attempt=on_attempt,
    )


def _evaluate_gemini_38(
    request: BinaryRequest,
    trial_ref: str,
    *,
    on_attempt: BinaryAttemptCallback | None = None,
) -> BinaryProbabilityObservation:
    return evaluate_gemini_binary(
        request,
        execution_ref=trial_ref,
        policy=GEMINI_38_FLASH_BINARY_EXECUTION_POLICY,
        on_attempt=on_attempt,
    )


def _supports_attempt_callback(evaluator: BinaryEvaluator) -> bool:
    try:
        parameters = inspect.signature(evaluator).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "on_attempt" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _invoke_evaluator(
    evaluator: BinaryEvaluator,
    request: BinaryRequest,
    trial_ref: str,
    on_attempt: BinaryAttemptCallback | None,
) -> BinaryProbabilityObservation:
    if on_attempt is None:
        return evaluator(request, trial_ref)
    return evaluator(request, trial_ref, on_attempt=on_attempt)


def _clock_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("Binary benchmark latency clock is invalid")
    return float(value)


def _elapsed_ms(started: float, finished: float) -> int:
    if finished < started:
        raise ValueError("Binary benchmark latency cannot be negative")
    return round((finished - started) * 1000)
