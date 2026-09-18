from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cache, partial
from time import sleep as _sleep
from typing import Annotated, Literal, Protocol, TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from langfuse import Langfuse
from langfuse.api.commons.types.dataset_item import DatasetItem
from langfuse.api.core.api_error import ApiError
from langfuse.experiment import Evaluation, EvaluatorFunction
from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import (
    RELEVANCE_POLICY_V1,
    RELEVANCE_POLICY_V2,
    ArticleAnalysisInput,
    RelevanceDecision,
    RelevancePolicy,
    analyze_relevance,
    relevance_is_accepted,
    relevance_policy_digest,
    relevance_request_id,
)
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ImpactDecision,
    RelevanceV3Policy,
    analyze_relevance_v3,
    context_is_accepted,
    relevance_v3_is_accepted,
    relevance_v3_policy_digest,
    relevance_v3_request_id,
)
from romanian_news.articles.models import ExtractedArticle
from romanian_news.catalog.evaluations import (
    LoadedNewsEvaluationRelease,
    NewsEvaluationDatasetReceipt,
    NewsEvaluationExperimentReceipt,
    NewsRelevanceExperimentReceipt,
    NewsThemeExperimentReceipt,
    load_news_evaluation_release,
    read_evaluation_relevance_observation,
    read_news_evaluation_projection_receipt,
    read_news_relevance_experiment_receipt,
    read_news_theme_experiment_receipt,
    record_news_evaluation_projection_receipt,
    record_news_relevance_experiment_receipt,
    record_news_theme_experiment_receipt,
)
from romanian_news.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PROJECT_ID,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from romanian_news.evaluation import (
    PIN_PATH,
    EvaluationCaseResult,
    NewsEvaluationManifest,
    NewsEvaluationSpec,
    RankingEvaluationCase,
    RelevanceEvaluationSpec,
    RelevanceV3EvaluationDecision,
    ReportEvaluationSnapshot,
    SubjectAssessmentEvaluationResult,
    ThemeEvaluationDayCase,
    ThemePolicyDayMetrics,
    TierEvaluationCase,
    compare_news_evaluation_to_baseline,
    evaluate_news_dataset,
    evaluate_subject_assessment,
    evaluate_theme_output,
    subject_tier_label_conflicts,
)
from romanian_news.groups import EmbeddedArticleReference
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import (
    DailySubjectAssessmentInput,
    DailySubjectAssessmentSet,
    SubjectEvidenceInput,
    SubjectSummaryInput,
    construct_daily_subject_assessments,
)
from romanian_news.themes import (
    LEGACY_SPARSE_THEME_DEFINITION,
    PRODUCTION_THEME_DEFINITION,
    DailyThemeOutput,
    ReaderSubjectDailyThemeSet,
    SparseDailyThemeSet,
    SparseThemeConstruction,
    SparseThemePolicyDefinition,
    construct_daily_themes,
    sparse_theme_policy_digest,
)

_PageT = TypeVar("_PageT")

EvaluationSplit = Literal["executable"]
_PROVIDER = "langfuse"
_RUN_VISIBILITY_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)
_ACTIVE_RUN_RETRY_DELAYS = (30.0,) * 20
_EXPERIMENT_QUERY_START = datetime(2020, 1, 1, tzinfo=UTC)


class FreshEvaluationPlan(NewsModel):
    implementation_ref: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    trial_count: Annotated[int, Field(ge=3)] = 3

    @property
    def implementation_refs(self) -> tuple[str, ...]:
        return tuple(
            f"{self.implementation_ref}:fresh-trial-{position}-of-{self.trial_count}"
            for position in range(1, self.trial_count + 1)
        )


class NewsEvaluationDatasetProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    dataset_id: str
    dataset_name: str
    created_examples: Annotated[int, Field(ge=0)]
    reused_examples: Annotated[int, Field(ge=0)]


class NewsEvaluationExperimentProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None = None


class FreshRelevanceCaseResult(NewsModel):
    case_id: str
    article_version_id: Sha256
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256
    expected_accepted: bool
    control: bool
    accepted: bool
    passed: bool
    decision: RelevanceDecision


class FreshRelevanceMetrics(NewsModel):
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    positive_control_preservation: Annotated[float, Field(ge=0, le=1)]


class RelevancePolicyExperimentProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None = None
    implementation_ref: str
    policy_id: str
    policy_digest: Sha256
    case_results: tuple[FreshRelevanceCaseResult, ...]
    metrics: FreshRelevanceMetrics


class RelevanceComparisonTrial(NewsModel):
    ordinal: Annotated[int, Field(ge=1)]
    implementation_ref: str
    baseline: RelevancePolicyExperimentProjection
    candidate: RelevancePolicyExperimentProjection
    promotion_failures: tuple[str, ...]
    passed: bool

    @model_validator(mode="after")
    def require_consistent_verdict(self) -> RelevanceComparisonTrial:
        if (self.baseline.implementation_ref, self.candidate.implementation_ref) != (
            self.implementation_ref,
            self.implementation_ref,
        ):
            raise ValueError("Trial experiments must use the trial implementation reference")
        expected_failures = relevance_promotion_failures(self.baseline, self.candidate)
        if self.promotion_failures != expected_failures:
            raise ValueError("Trial promotion failures do not match its experiments")
        if self.passed != (not expected_failures):
            raise ValueError("Trial verdict does not match its promotion failures")
        return self


class RelevanceComparisonResult(NewsModel):
    manifest_artifact_version_id: Sha256
    plan: FreshEvaluationPlan
    trials: tuple[RelevanceComparisonTrial, ...]
    promotion_failures: tuple[str, ...]
    promoted: bool

    @model_validator(mode="after")
    def require_complete_trials(self) -> RelevanceComparisonResult:
        expected_refs = self.plan.implementation_refs
        if tuple(trial.ordinal for trial in self.trials) != tuple(
            range(1, self.plan.trial_count + 1)
        ):
            raise ValueError("Relevance comparison trials must have exact ordinal coverage")
        if tuple(trial.implementation_ref for trial in self.trials) != expected_refs:
            raise ValueError("Relevance comparison trials must have exact reference coverage")
        if any(
            trial.baseline.manifest_artifact_version_id != self.manifest_artifact_version_id
            or trial.candidate.manifest_artifact_version_id != self.manifest_artifact_version_id
            for trial in self.trials
        ):
            raise ValueError("Relevance comparison trial manifests must match the group")
        expected_failures = tuple(
            f"trial {trial.ordinal} of {self.plan.trial_count}: {failure}"
            for trial in self.trials
            for failure in trial.promotion_failures
        )
        if self.promotion_failures != expected_failures:
            raise ValueError("Grouped promotion failures do not match its trials")
        if self.promoted != (not expected_failures):
            raise ValueError("Grouped verdict does not match its promotion failures")
        return self


class FreshThemeDayResult(NewsModel):
    case_id: str
    day: date
    source_report_version_id: Sha256
    cluster_set_version_id: Sha256
    ordered_summary_version_ids: tuple[Sha256, ...]
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256
    theme_content_digest: Sha256
    assignment_response_id: str
    assignment_attempt_id: Sha256
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet
    metrics: ThemePolicyDayMetrics
    assessment_set: DailySubjectAssessmentSet | None = None
    assessment_metrics: SubjectAssessmentEvaluationResult | None = None

    @model_validator(mode="after")
    def require_assessment_for_reader_subjects(self) -> FreshThemeDayResult:
        has_assessment = self.assessment_set is not None and self.assessment_metrics is not None
        if (self.assessment_set is None) != (self.assessment_metrics is None):
            raise ValueError("Fresh subject assessment observation must be complete")
        if isinstance(self.theme_set, ReaderSubjectDailyThemeSet) != has_assessment:
            raise ValueError("Fresh reader-subject output must include its assessment")
        return self


class FreshThemeMetrics(NewsModel):
    must_link_recall: Annotated[float, Field(ge=0, le=1)]
    must_separate_preservation: Annotated[float, Field(ge=0, le=1)]
    singleton_rate: Annotated[float, Field(ge=0, le=1)]
    complete_report_usefulness: Annotated[float, Field(ge=0, le=1)]


class ThemePolicyExperimentProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None = None
    implementation_ref: str
    policy_id: str
    policy_digest: Sha256
    day_results: tuple[FreshThemeDayResult, ...]
    metrics: FreshThemeMetrics


class ThemeComparisonTrial(NewsModel):
    ordinal: Annotated[int, Field(ge=1)]
    implementation_ref: str
    baseline: ThemePolicyExperimentProjection
    candidate: ThemePolicyExperimentProjection
    promotion_failures: tuple[str, ...]
    passed: bool


class ThemeComparisonResult(NewsModel):
    manifest_artifact_version_id: Sha256
    plan: FreshEvaluationPlan
    trials: tuple[ThemeComparisonTrial, ...]
    promotion_failures: tuple[str, ...]
    promoted: bool


class RelevanceV3SliceMetrics(NewsModel):
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]


class RelevanceV3CaseResult(NewsModel):
    case_id: str
    article_version_id: Sha256
    request_id: Sha256
    execution_ref: str
    expected_accepted: bool
    control: bool
    slice: Literal["prior", "new"]
    accepted: bool
    passed: bool
    context: ContextDecision
    impact: ImpactDecision
    context_early_exit: bool
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    observability_complete: bool


class RelevanceV3ExperimentProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None = None
    implementation_ref: str
    policy_id: str
    policy_digest: Sha256
    case_results: tuple[RelevanceV3CaseResult, ...]
    full_metrics: RelevanceV3SliceMetrics
    prior_metrics: RelevanceV3SliceMetrics
    new_metrics: RelevanceV3SliceMetrics
    false_negative_ids: tuple[str, ...]
    context_early_exit_rate: Annotated[float, Field(ge=0, le=1)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    observability_complete: bool


class RelevanceV3EvaluationProjection(NewsModel):
    manifest_artifact_version_id: Sha256
    plan: FreshEvaluationPlan
    runs: tuple[RelevanceV3ExperimentProjection, ...]
    false_negative_ids: tuple[str, ...]
    threshold_failures: tuple[str, ...]
    passed: bool

    @model_validator(mode="after")
    def require_complete_trials(self) -> RelevanceV3EvaluationProjection:
        if tuple(run.implementation_ref for run in self.runs) != self.plan.implementation_refs:
            raise ValueError("V3 runs must have exact reference coverage")
        if any(
            run.manifest_artifact_version_id != self.manifest_artifact_version_id
            for run in self.runs
        ):
            raise ValueError("V3 run manifests must match the group")
        expected_false_negatives = tuple(
            sorted({case_id for run in self.runs for case_id in run.false_negative_ids})
        )
        if self.false_negative_ids != expected_false_negatives:
            raise ValueError("V3 false negatives do not match its runs")
        expected_failures = tuple(
            failure
            for position, run in enumerate(self.runs, start=1)
            for failure in _relevance_v3_threshold_failures(run, position)
        )
        if self.threshold_failures != expected_failures:
            raise ValueError("V3 threshold failures do not match its runs")
        if self.passed != (not expected_failures):
            raise ValueError("V3 verdict does not match its threshold failures")
        return self


class _ProjectedExample(NewsModel):
    example_id: str
    inputs: dict[str, object]
    outputs: dict[str, object]
    metadata: dict[str, object]
    split: EvaluationSplit


class _ExperimentInput(NewsModel):
    case_id: str
    artifact_references: tuple[ArtifactReference, ...]


class _ExperimentOutput(NewsModel):
    case_id: str
    passed: bool


class _FreshRelevanceRunOutput(NewsModel):
    case_id: str
    article_version_id: Sha256
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256
    accepted: bool
    decision: RelevanceDecision


class _ExpectedAcceptedOutput(NewsModel):
    expected_accepted: bool


class _FreshRelevanceExpected(NewsModel):
    example_id: str
    case_id: str
    article: ArtifactReference
    expected_accepted: bool
    control: bool
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256


class _RelevanceV3RunOutput(NewsModel):
    case_id: str
    article_version_id: Sha256
    execution_ref: str
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256
    accepted: bool
    context: ContextDecision
    impact: ImpactDecision
    context_early_exit: bool
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    observability_complete: bool


class _RelevanceV3ExpectedOutput(NewsModel):
    expected_accepted: bool


class _RelevanceV3Expected(NewsModel):
    example_id: str
    case_id: str
    article: ArtifactReference
    execution_ref: str
    expected_accepted: bool
    control: bool
    slice: Literal["prior", "new"]
    policy_id: str
    policy_digest: Sha256
    request_id: Sha256


def project_news_evaluation_release(
    release: LoadedNewsEvaluationRelease,
) -> NewsEvaluationDatasetProjection:
    """Reconcile one exact evaluation release into its immutable provider dataset."""
    _require_langfuse_credentials()
    client = _langfuse_client()
    dataset_name = _dataset_name(release.manifest_reference.version_id)
    dataset = _read_or_create_dataset(client, release, dataset_name)
    expected = _projected_examples(release)
    existing = _dataset_items_by_id(dataset.items)
    _require_examples_match(existing, expected)

    missing = [item for item in expected if item.example_id not in existing]
    for item in missing:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item.example_id,
            input=item.inputs,
            expected_output=item.outputs,
            metadata={**item.metadata, "split": item.split},
        )

    reconciled = client.get_dataset(dataset_name)
    _require_examples_match(_dataset_items_by_id(reconciled.items), expected)
    if receipt := read_news_evaluation_projection_receipt(
        _PROVIDER, release.manifest_reference.version_id, "dataset", ""
    ):
        if (receipt.dataset_id, receipt.dataset_name) != (dataset.id, dataset_name):
            raise ValueError("Dataset receipt conflicts with the manifest projection")
    else:
        record_news_evaluation_projection_receipt(
            NewsEvaluationDatasetReceipt(
                provider=_PROVIDER,
                manifest_artifact_version_id=release.manifest_reference.version_id,
                dataset_id=dataset.id,
                dataset_name=dataset_name,
            )
        )
    return NewsEvaluationDatasetProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        dataset_id=dataset.id,
        dataset_name=dataset_name,
        created_examples=len(missing),
        reused_examples=len(expected) - len(missing),
    )


def run_news_evaluation_experiment(
    release: LoadedNewsEvaluationRelease,
    implementation_ref: str,
) -> NewsEvaluationExperimentProjection:
    """Publish deterministic case results as one provider experiment without model calls."""
    _require_langfuse_credentials()
    if not implementation_ref.strip():
        raise ValueError("Implementation reference is required")
    dataset_projection = project_news_evaluation_release(release)
    experiment_name = _experiment_name(release.manifest_reference.version_id, implementation_ref)
    client = _langfuse_client()
    dataset = client.get_dataset(dataset_projection.dataset_name)
    examples = tuple(
        item
        for item in dataset.items
        if _item_split(item) == "executable"
        and isinstance(item.metadata, dict)
        and item.metadata.get("concern") != "daily_theme"
    )
    expected_outputs = _expected_run_outputs(examples, _results_by_case(release))

    def load_result(*, item: DatasetItem, **_kwargs: object) -> dict[str, object]:
        _ExperimentInput.model_validate(item.input, strict=False)
        return expected_outputs[item.id]

    def deterministic_pass(*, output: object, **_kwargs: object) -> Evaluation:
        projected = _ExperimentOutput.model_validate(output, strict=False)
        return Evaluation(name="deterministic_pass", value=int(projected.passed))

    execution = _run_or_reuse_experiment(
        client,
        dataset_name=dataset_projection.dataset_name,
        experiment_name=experiment_name,
        examples=examples,
        expected_outputs=expected_outputs,
        task=load_result,
        evaluator=deterministic_pass,
        metadata=_experiment_metadata(release, implementation_ref),
        description="Deterministic Romanian news evaluation over one published manifest.",
    )
    receipt = read_news_evaluation_projection_receipt(
        _PROVIDER, release.manifest_reference.version_id, "experiment", implementation_ref
    )
    if receipt is not None:
        _require_projection_receipt_matches(
            receipt, dataset_projection, experiment_name, implementation_ref
        )
        if _required(receipt.experiment_id, "experiment ID") != execution.experiment_id:
            raise ValueError("Experiment receipt conflicts with the remote run")
        return NewsEvaluationExperimentProjection(
            manifest_artifact_version_id=release.manifest_reference.version_id,
            dataset_id=receipt.dataset_id,
            dataset_name=receipt.dataset_name,
            experiment_id=execution.experiment_id,
            experiment_name=experiment_name,
            experiment_url=receipt.experiment_url or execution.experiment_url,
        )
    record_news_evaluation_projection_receipt(
        NewsEvaluationExperimentReceipt(
            provider=_PROVIDER,
            manifest_artifact_version_id=release.manifest_reference.version_id,
            dataset_id=dataset_projection.dataset_id,
            dataset_name=dataset_projection.dataset_name,
            experiment_id=execution.experiment_id,
            experiment_name=experiment_name,
            experiment_url=execution.experiment_url,
            implementation_ref=implementation_ref,
        )
    )
    return NewsEvaluationExperimentProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        dataset_id=dataset_projection.dataset_id,
        dataset_name=dataset_projection.dataset_name,
        experiment_id=execution.experiment_id,
        experiment_name=experiment_name,
        experiment_url=execution.experiment_url,
    )


def compare_relevance_policies(
    release: LoadedNewsEvaluationRelease,
    plan: FreshEvaluationPlan,
    baseline_policy: RelevancePolicy = RELEVANCE_POLICY_V1,
    candidate_policy: RelevancePolicy = RELEVANCE_POLICY_V2,
) -> RelevanceComparisonResult:
    """Run paired fresh baseline and candidate relevance trials."""
    trials = tuple(
        _run_relevance_comparison_trial(
            release,
            ordinal,
            implementation_ref,
            baseline_policy,
            candidate_policy,
        )
        for ordinal, implementation_ref in enumerate(plan.implementation_refs, start=1)
    )
    failures = tuple(
        f"trial {trial.ordinal} of {plan.trial_count}: {failure}"
        for trial in trials
        for failure in trial.promotion_failures
    )
    return RelevanceComparisonResult(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        plan=plan,
        trials=trials,
        promotion_failures=failures,
        promoted=not failures,
    )


def _run_relevance_comparison_trial(
    release: LoadedNewsEvaluationRelease,
    ordinal: int,
    implementation_ref: str,
    baseline_policy: RelevancePolicy,
    candidate_policy: RelevancePolicy,
) -> RelevanceComparisonTrial:
    baseline = _run_relevance_policy_experiment(release, baseline_policy, implementation_ref)
    candidate = _run_relevance_policy_experiment(release, candidate_policy, implementation_ref)
    failures = relevance_promotion_failures(baseline, candidate)
    return RelevanceComparisonTrial(
        ordinal=ordinal,
        implementation_ref=implementation_ref,
        baseline=baseline,
        candidate=candidate,
        promotion_failures=failures,
        passed=not failures,
    )


def compare_theme_policies(
    release: LoadedNewsEvaluationRelease,
    plan: FreshEvaluationPlan,
    baseline: SparseThemePolicyDefinition = LEGACY_SPARSE_THEME_DEFINITION,
    candidate: SparseThemePolicyDefinition = PRODUCTION_THEME_DEFINITION,
) -> ThemeComparisonResult:
    """Run paired fresh reader-subject trials over complete frozen days."""
    if sparse_theme_policy_digest(baseline.policy) == sparse_theme_policy_digest(candidate.policy):
        raise ValueError("Theme comparison policies must have different identities")
    trials = tuple(
        _run_theme_comparison_trial(
            release,
            ordinal,
            implementation_ref,
            baseline,
            candidate,
        )
        for ordinal, implementation_ref in enumerate(plan.implementation_refs, start=1)
    )
    failures = [
        f"trial {trial.ordinal} of {plan.trial_count}: {failure}"
        for trial in trials
        for failure in trial.promotion_failures
    ]
    response_ids = [
        result.assignment_response_id
        for trial in trials
        for experiment in (trial.baseline, trial.candidate)
        for result in experiment.day_results
    ]
    attempt_ids = [
        result.assignment_attempt_id
        for trial in trials
        for experiment in (trial.baseline, trial.candidate)
        for result in experiment.day_results
    ]
    if len(set(response_ids)) != len(response_ids) or len(set(attempt_ids)) != len(attempt_ids):
        failures.append("Fresh theme trials reused assignment provider evidence")
    return ThemeComparisonResult(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        plan=plan,
        trials=trials,
        promotion_failures=tuple(failures),
        promoted=not failures,
    )


def _run_theme_comparison_trial(
    release: LoadedNewsEvaluationRelease,
    ordinal: int,
    implementation_ref: str,
    baseline: SparseThemePolicyDefinition,
    candidate: SparseThemePolicyDefinition,
) -> ThemeComparisonTrial:
    baseline_result = _run_theme_policy_experiment(release, baseline, implementation_ref)
    candidate_result = _run_theme_policy_experiment(release, candidate, implementation_ref)
    failures = tuple(
        f"{result.day.isoformat()}: {failure}"
        for result in candidate_result.day_results
        for failure in (
            *result.metrics.usefulness_failures,
            *(
                f"subject assessment failed: {case.case_id}: {case.detail}"
                for case in (
                    result.assessment_metrics.case_results
                    if result.assessment_metrics is not None
                    else ()
                )
                if not case.passed
            ),
        )
    )
    return ThemeComparisonTrial(
        ordinal=ordinal,
        implementation_ref=implementation_ref,
        baseline=baseline_result,
        candidate=candidate_result,
        promotion_failures=failures,
        passed=not failures,
    )


def _run_theme_policy_experiment(
    release: LoadedNewsEvaluationRelease,
    definition: SparseThemePolicyDefinition,
    implementation_ref: str,
) -> ThemePolicyExperimentProjection:
    _require_langfuse_credentials()
    if not implementation_ref.strip():
        raise ValueError("Implementation reference is required")
    cases = {
        case.case_id: case
        for case in release.dataset.cases
        if isinstance(case, ThemeEvaluationDayCase) and case.report_expectation is not None
    }
    reports = {snapshot.report.version_id: snapshot for snapshot in release.dataset.reports}
    assessment_cases = tuple(
        case
        for case in release.dataset.cases
        if isinstance(case, RankingEvaluationCase | TierEvaluationCase)
    )
    if not cases:
        raise ValueError("Theme comparison requires frozen report-level cases")
    dataset_projection = project_news_evaluation_release(release)
    client = _langfuse_client()
    dataset = client.get_dataset(dataset_projection.dataset_name)
    examples = tuple(
        item
        for item in dataset.items
        if _item_split(item) == "executable"
        and isinstance(item.metadata, dict)
        and item.metadata.get("concern") == "daily_theme"
        and _ExperimentInput.model_validate(item.input, strict=False).case_id in cases
    )
    projected_case_ids = {
        _ExperimentInput.model_validate(item.input, strict=False).case_id for item in examples
    }
    if projected_case_ids != set(cases):
        raise ValueError("Projected theme cases do not match the frozen comparison workload")
    policy = definition.policy
    policy_digest = sparse_theme_policy_digest(policy)
    experiment_name = _theme_experiment_name(
        release.manifest_reference.version_id,
        policy_digest,
        implementation_ref,
    )

    def generate(*, item: DatasetItem, **_kwargs: object) -> dict[str, object]:
        projected = _ExperimentInput.model_validate(item.input, strict=False)
        case = cases[projected.case_id]
        output = construct_daily_themes(case.input, definition)
        theme_set = output.theme_set
        if not isinstance(theme_set, SparseDailyThemeSet | ReaderSubjectDailyThemeSet):
            raise ValueError("Fresh theme comparison requires sparse theme output")
        daily_assessment_set = None
        assessment_metrics = None
        if isinstance(theme_set, ReaderSubjectDailyThemeSet):
            day_cases = tuple(
                assessment_case
                for assessment_case in assessment_cases
                if assessment_case.provenance.report == case.source_report
            )
            conflicts = subject_tier_label_conflicts(day_cases, theme_set)
            unresolvable = tuple(
                conflict for conflict in conflicts if conflict.governing_case_id is None
            )
            if unresolvable:
                raise ValueError(
                    "Conflicting frozen tier labels without a reviewed merge rule: "
                    + "; ".join(
                        f"subject={conflict.subject_id} labels={conflict.labels} "
                        f"groups={sorted(conflict.group_ids)}"
                        for conflict in unresolvable
                    )
                )
            assessment_output = construct_daily_subject_assessments(
                _fresh_subject_assessment_input(
                    case,
                    output,
                    reports[case.source_report.version_id],
                    implementation_ref,
                )
            )
            daily_assessment_set = assessment_output.assessment_set
            assessment_metrics = evaluate_subject_assessment(
                day_cases, theme_set, daily_assessment_set
            )
        construction = theme_set.construction
        if not isinstance(construction, SparseThemeConstruction):
            raise ValueError("Fresh theme comparison requires assignment evidence")
        accepted = construction.assignment.attempts[-1]
        result = FreshThemeDayResult(
            case_id=case.case_id,
            day=case.input.day,
            source_report_version_id=case.source_report.version_id,
            cluster_set_version_id=case.input.cluster_set.version_id,
            ordered_summary_version_ids=tuple(
                item.summary.version_id for item in case.input.groups
            ),
            policy_id=policy.policy_id,
            policy_digest=policy_digest,
            request_id=theme_set.request_id,
            theme_content_digest=output.content_digest,
            assignment_response_id=accepted.response_id,
            assignment_attempt_id=accepted.attempt_id,
            theme_set=theme_set,
            metrics=evaluate_theme_output(case, theme_set),
            assessment_set=daily_assessment_set,
            assessment_metrics=assessment_metrics,
        )
        return result.model_dump(mode="json")

    def complete_report_useful(*, output: object, **_kwargs: object) -> Evaluation:
        result = FreshThemeDayResult.model_validate(output, strict=False)
        return Evaluation(name="complete_report_useful", value=int(result.metrics.useful))

    def validate(example: DatasetItem, output: object) -> dict[str, object]:
        projected = _ExperimentInput.model_validate(example.input, strict=False)
        result = FreshThemeDayResult.model_validate(output, strict=False)
        case = cases[projected.case_id]
        if result.case_id != case.case_id or result.day != case.input.day:
            raise ValueError("Fresh theme result identifies another day case")
        if (
            result.policy_id != policy.policy_id
            or result.policy_digest != policy_digest
            or result.theme_set.policy != policy
        ):
            raise ValueError("Fresh theme result identifies another policy")
        construction = result.theme_set.construction
        if not isinstance(construction, SparseThemeConstruction):
            raise ValueError("Fresh theme result has no assignment evidence")
        accepted = construction.assignment.attempts[-1]
        expected_identity = (
            case.source_report.version_id,
            case.input.cluster_set.version_id,
            tuple(item.summary.version_id for item in case.input.groups),
            result.theme_set.request_id,
            _theme_set_content_digest(result.theme_set),
            accepted.response_id,
            accepted.attempt_id,
        )
        actual_identity = (
            result.source_report_version_id,
            result.cluster_set_version_id,
            result.ordered_summary_version_ids,
            result.request_id,
            result.theme_content_digest,
            result.assignment_response_id,
            result.assignment_attempt_id,
        )
        if actual_identity != expected_identity:
            raise ValueError("Fresh theme result conflicts with its embedded evidence")
        expected_metrics = evaluate_theme_output(case, result.theme_set)
        if result.metrics != expected_metrics:
            raise ValueError("Fresh theme metrics do not match the complete output")
        if isinstance(result.theme_set, ReaderSubjectDailyThemeSet):
            if result.assessment_set is None or result.assessment_metrics is None:
                raise ValueError("Fresh reader-subject result omits its assessment")
            if result.assessment_set.themes.version_id != result.theme_content_digest:
                raise ValueError("Fresh assessment references another theme output")
            day_cases = tuple(
                assessment_case
                for assessment_case in assessment_cases
                if assessment_case.provenance.report == case.source_report
            )
            expected_assessment_metrics = evaluate_subject_assessment(
                day_cases, result.theme_set, result.assessment_set
            )
            if result.assessment_metrics != expected_assessment_metrics:
                raise ValueError("Fresh assessment metrics do not match the complete output")
        return result.model_dump(mode="json")

    execution = _run_or_reuse_experiment(
        client,
        dataset_name=dataset_projection.dataset_name,
        experiment_name=experiment_name,
        examples=examples,
        expected_outputs=None,
        task=generate,
        evaluator=complete_report_useful,
        metadata=_theme_experiment_metadata(release, definition, implementation_ref),
        description="Fresh Romanian news reader-subject evaluation over complete frozen days.",
        validate_output=validate,
    )
    day_results = tuple(
        FreshThemeDayResult.model_validate(execution.outputs[item.id], strict=False)
        for item in examples
    )
    experiment_url = _theme_experiment_url(
        release,
        dataset_projection,
        definition,
        policy_digest,
        implementation_ref,
        experiment_name,
        execution,
    )
    return ThemePolicyExperimentProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        dataset_id=dataset_projection.dataset_id,
        dataset_name=dataset_projection.dataset_name,
        experiment_id=execution.experiment_id,
        experiment_name=experiment_name,
        experiment_url=experiment_url,
        implementation_ref=implementation_ref,
        policy_id=policy.policy_id,
        policy_digest=policy_digest,
        day_results=day_results,
        metrics=_fresh_theme_metrics(day_results),
    )


def _fresh_subject_assessment_input(
    case: ThemeEvaluationDayCase,
    output: DailyThemeOutput,
    snapshot: ReportEvaluationSnapshot,
    implementation_ref: str,
) -> DailySubjectAssessmentInput:
    theme_set = output.theme_set
    if not isinstance(theme_set, ReaderSubjectDailyThemeSet):
        raise ValueError("Subject assessment requires final reader subjects")
    if snapshot.cluster_set != case.input.cluster_set:
        raise ValueError("Fresh assessment report and theme inputs use different clusters")
    themes = ArtifactReference(
        artifact_id=f"evaluation:themes:{case.case_id}:{implementation_ref}",
        version_id=output.content_digest,
        content_digest=output.content_digest,
        r2_key=f"evaluation://themes/{case.case_id}/{implementation_ref}",
    )
    summaries = tuple(
        SubjectSummaryInput(
            group_id=item.group.id,
            reference=item.summary,
            summary=item.value,
        )
        for item in case.input.groups
    )
    summary_by_group = {item.group_id: item.reference for item in summaries}
    articles = {item.article.version_id: item for item in snapshot.cluster_articles}
    theme_by_article = {
        article_id: theme.id
        for theme in theme_set.themes
        for article_id in theme.article_version_ids
    }
    group_by_article = {
        article_id: group.id
        for group in theme_set.groups
        for article_id in group.article_version_ids
    }
    relevance = tuple(item.relevance for item in snapshot.cluster_articles)
    evidence = tuple(
        SubjectEvidenceInput(
            theme_id=theme_by_article[item.article.version_id],
            group_id=group_by_article[item.article.version_id],
            article=item.article,
            relevance=item.relevance,
            summary=summary_by_group[group_by_article[item.article.version_id]],
            evidence_quote=_assessment_evidence_quote(item),
        )
        for item in snapshot.cluster_articles
    )
    if set(articles) != set(theme_by_article):
        raise ValueError("Fresh assessment themes do not exactly cover report articles")
    return DailySubjectAssessmentInput(
        day=case.input.day,
        themes=themes,
        theme_set=theme_set,
        summaries=summaries,
        relevance=relevance,
        evidence=evidence,
    )


def _theme_set_content_digest(
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet,
) -> Sha256:
    return hashlib.sha256(
        json.dumps(
            theme_set.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _assessment_evidence_quote(article: EmbeddedArticleReference) -> str:
    decision = read_evaluation_relevance_observation(article.relevance, article.article.version_id)
    if isinstance(decision, RelevanceV3EvaluationDecision):
        return (decision.impact or decision.context).evidence_quote
    return decision.evidence_quote


def _theme_experiment_url(
    release: LoadedNewsEvaluationRelease,
    dataset_projection: NewsEvaluationDatasetProjection,
    definition: SparseThemePolicyDefinition,
    policy_digest: Sha256,
    implementation_ref: str,
    experiment_name: str,
    execution: _ExperimentExecution,
) -> str | None:
    receipt = read_news_theme_experiment_receipt(
        _PROVIDER, release.manifest_reference.version_id, policy_digest, implementation_ref
    )
    if receipt is not None:
        if (
            receipt.policy_id != definition.policy.policy_id
            or receipt.dataset_id != dataset_projection.dataset_id
            or receipt.dataset_name != dataset_projection.dataset_name
            or receipt.experiment_id != execution.experiment_id
            or receipt.experiment_name != experiment_name
        ):
            raise ValueError("Experiment receipt conflicts with this theme run")
        return receipt.experiment_url or execution.experiment_url
    record_news_theme_experiment_receipt(
        NewsThemeExperimentReceipt(
            provider=_PROVIDER,
            manifest_artifact_version_id=release.manifest_reference.version_id,
            policy_digest=policy_digest,
            implementation_ref=implementation_ref,
            policy_id=definition.policy.policy_id,
            dataset_id=dataset_projection.dataset_id,
            dataset_name=dataset_projection.dataset_name,
            experiment_id=execution.experiment_id,
            experiment_name=experiment_name,
            experiment_url=execution.experiment_url,
        )
    )
    return execution.experiment_url


def _fresh_theme_metrics(results: tuple[FreshThemeDayResult, ...]) -> FreshThemeMetrics:
    must_link_passed = sum(item.metrics.must_link_passed for item in results)
    must_link_total = sum(item.metrics.must_link_total for item in results)
    must_separate_passed = sum(item.metrics.must_separate_passed for item in results)
    must_separate_total = sum(item.metrics.must_separate_total for item in results)
    singleton_groups = sum(item.metrics.singleton_groups for item in results)
    total_groups = sum(item.metrics.total_groups for item in results)
    return FreshThemeMetrics(
        must_link_recall=must_link_passed / must_link_total,
        must_separate_preservation=must_separate_passed / must_separate_total,
        singleton_rate=singleton_groups / total_groups,
        complete_report_usefulness=sum(item.metrics.useful for item in results) / len(results),
    )


def _run_relevance_policy_experiment(
    release: LoadedNewsEvaluationRelease,
    policy: RelevancePolicy,
    implementation_ref: str,
) -> RelevancePolicyExperimentProjection:
    """Run or resume one fresh relevance experiment without publishing analysis artifacts."""
    _require_langfuse_credentials()
    if not implementation_ref.strip():
        raise ValueError("Implementation reference is required")
    dataset_projection = project_news_evaluation_release(release)
    client = _langfuse_client()
    dataset = client.get_dataset(dataset_projection.dataset_name)
    examples = tuple(
        item
        for item in dataset.items
        if _item_split(item) == "executable"
        and isinstance(item.metadata, dict)
        and item.metadata.get("concern") == "relevance"
    )
    expected = _fresh_relevance_expected(release, examples, policy)
    expected_by_case = {item.case_id: item for item in expected.values()}
    policy_digest = relevance_policy_digest(policy)
    experiment_name = _relevance_experiment_name(
        release.manifest_reference.version_id,
        policy_digest,
        implementation_ref,
    )

    def generate(*, item: DatasetItem, **_kwargs: object) -> dict[str, object]:
        projected = _ExperimentInput.model_validate(item.input, strict=False)
        expected_case = expected_by_case[projected.case_id]
        content = read_verified_r2_object(
            expected_case.article.r2_key,
            expected_case.article.content_digest,
        )
        article = ExtractedArticle.model_validate_json(content, strict=True)
        output = analyze_relevance(
            ArticleAnalysisInput(reference=expected_case.article, article=article),
            policy,
        )
        return _FreshRelevanceRunOutput(
            case_id=expected_case.case_id,
            article_version_id=expected_case.article.version_id,
            policy_id=output.policy.policy_id,
            policy_digest=output.policy_digest,
            request_id=output.request_id,
            accepted=output.accepted,
            decision=output.decision,
        ).model_dump(mode="json")

    def fresh_relevance_pass(
        *, output: object, expected_output: object, **_kwargs: object
    ) -> Evaluation:
        projected = _FreshRelevanceRunOutput.model_validate(output, strict=False)
        expected = _ExpectedAcceptedOutput.model_validate(expected_output, strict=False)
        return Evaluation(
            name="fresh_relevance_pass",
            value=int(projected.accepted == expected.expected_accepted),
        )

    def validate(example: DatasetItem, output: object) -> dict[str, object]:
        validated = _validated_relevance_run_output(output, expected[example.id], policy)
        return validated.model_dump(mode="json")

    execution = _run_or_reuse_experiment(
        client,
        dataset_name=dataset_projection.dataset_name,
        experiment_name=experiment_name,
        examples=examples,
        expected_outputs=None,
        task=generate,
        evaluator=fresh_relevance_pass,
        metadata=_relevance_experiment_metadata(release, policy, implementation_ref),
        description="Fresh Romanian news relevance generation over exact manifest articles.",
        validate_output=validate,
    )
    outputs = execution.outputs
    results = tuple(
        _fresh_relevance_case_result(expected[item.id], outputs[item.id], policy)
        for item in examples
    )
    receipt = read_news_relevance_experiment_receipt(
        _PROVIDER,
        release.manifest_reference.version_id,
        policy_digest,
        implementation_ref,
    )
    if receipt is not None:
        if (
            receipt.policy_id != policy.policy_id
            or receipt.dataset_id != dataset_projection.dataset_id
            or receipt.dataset_name != dataset_projection.dataset_name
            or receipt.experiment_id != execution.experiment_id
            or receipt.experiment_name != experiment_name
        ):
            raise ValueError("Experiment receipt conflicts with this relevance run")
        experiment_url = receipt.experiment_url or execution.experiment_url
    else:
        experiment_url = execution.experiment_url
        record_news_relevance_experiment_receipt(
            NewsRelevanceExperimentReceipt(
                provider=_PROVIDER,
                manifest_artifact_version_id=release.manifest_reference.version_id,
                policy_digest=policy_digest,
                implementation_ref=implementation_ref,
                policy_id=policy.policy_id,
                dataset_id=dataset_projection.dataset_id,
                dataset_name=dataset_projection.dataset_name,
                experiment_id=execution.experiment_id,
                experiment_name=experiment_name,
                experiment_url=experiment_url,
            )
        )
    return RelevancePolicyExperimentProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        dataset_id=dataset_projection.dataset_id,
        dataset_name=dataset_projection.dataset_name,
        experiment_id=execution.experiment_id,
        experiment_name=experiment_name,
        experiment_url=experiment_url,
        implementation_ref=implementation_ref,
        policy_id=policy.policy_id,
        policy_digest=policy_digest,
        case_results=results,
        metrics=_fresh_relevance_metrics(results),
    )


def _run_relevance_v3_experiment(
    release: LoadedNewsEvaluationRelease,
    implementation_ref: str,
    policy: RelevanceV3Policy = RELEVANCE_V3_POLICY,
) -> RelevanceV3ExperimentProjection:
    _require_langfuse_credentials()
    if not implementation_ref.strip():
        raise ValueError("Implementation reference is required")
    dataset_projection = project_news_evaluation_release(release)
    client = _langfuse_client()
    dataset = client.get_dataset(dataset_projection.dataset_name)
    examples = tuple(
        item
        for item in dataset.items
        if _item_split(item) == "executable"
        and isinstance(item.metadata, dict)
        and item.metadata.get("concern") == "relevance"
    )
    expected = _relevance_v3_expected(release, examples, policy, implementation_ref)
    expected_by_case = {item.case_id: item for item in expected.values()}
    policy_digest = relevance_v3_policy_digest(policy)
    experiment_name = _relevance_experiment_name(
        release.manifest_reference.version_id,
        policy_digest,
        implementation_ref,
    )

    execution = _run_or_reuse_experiment(
        client,
        dataset_name=dataset_projection.dataset_name,
        experiment_name=experiment_name,
        examples=examples,
        expected_outputs=None,
        task=partial(
            _generate_relevance_v3_output,
            expected_by_case=expected_by_case,
            policy=policy,
            implementation_ref=implementation_ref,
        ),
        evaluator=_score_relevance_v3_output,
        metadata=_relevance_v3_experiment_metadata(release, policy, implementation_ref),
        description="Fresh two-gate Romanian news relevance V3 evaluation.",
        validate_output=partial(
            _validate_relevance_v3_example,
            expected=expected,
            policy=policy,
        ),
    )
    return _project_relevance_v3_experiment(
        release,
        dataset_projection,
        implementation_ref,
        policy,
        policy_digest,
        experiment_name,
        examples,
        expected,
        execution,
    )


def _generate_relevance_v3_output(
    *,
    item: DatasetItem,
    expected_by_case: dict[str, _RelevanceV3Expected],
    policy: RelevanceV3Policy,
    implementation_ref: str,
    **_kwargs: object,
) -> dict[str, object]:
    projected = _ExperimentInput.model_validate(item.input, strict=False)
    expected_case = expected_by_case[projected.case_id]
    content = read_verified_r2_object(
        expected_case.article.r2_key,
        expected_case.article.content_digest,
    )
    article = ExtractedArticle.model_validate_json(content, strict=True)
    output = analyze_relevance_v3(
        ArticleAnalysisInput(reference=expected_case.article, article=article),
        mode="full_evaluation",
        policy=policy,
        execution_ref=implementation_ref,
    )
    if output.impact is None:
        raise ValueError("Full V3 evaluation omitted the impact gate")
    if output.execution_ref is None:
        raise ValueError("Full V3 evaluation omitted the execution reference")
    return _RelevanceV3RunOutput(
        case_id=expected_case.case_id,
        article_version_id=expected_case.article.version_id,
        execution_ref=output.execution_ref,
        policy_id=output.policy.policy_id,
        policy_digest=output.policy_digest,
        request_id=output.request_id,
        accepted=output.accepted,
        context=output.context.decision,
        impact=output.impact.decision,
        context_early_exit=output.context_early_exit,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        cost_usd=output.cost_usd,
        observability_complete=output.observability_complete,
    ).model_dump(mode="json")


def _score_relevance_v3_output(
    *, output: object, expected_output: object, **_kwargs: object
) -> Evaluation:
    projected = _RelevanceV3RunOutput.model_validate(output, strict=True)
    expected = _RelevanceV3ExpectedOutput.model_validate(expected_output, strict=True)
    return Evaluation(
        name="relevance_v3_pass",
        value=int(projected.accepted == expected.expected_accepted),
    )


def _validate_relevance_v3_example(
    example: DatasetItem,
    output: object,
    *,
    expected: dict[str, _RelevanceV3Expected],
    policy: RelevanceV3Policy,
) -> dict[str, object]:
    validated = _validated_relevance_v3_output(output, expected[example.id], policy)
    return validated.model_dump(mode="json")


def _project_relevance_v3_experiment(
    release: LoadedNewsEvaluationRelease,
    dataset: NewsEvaluationDatasetProjection,
    implementation_ref: str,
    policy: RelevanceV3Policy,
    policy_digest: Sha256,
    experiment_name: str,
    examples: tuple[DatasetItem, ...],
    expected: dict[str, _RelevanceV3Expected],
    execution: _ExperimentExecution,
) -> RelevanceV3ExperimentProjection:
    results = tuple(
        _relevance_v3_case_result(expected[item.id], execution.outputs[item.id], policy)
        for item in examples
    )
    experiment_url = _relevance_v3_experiment_url(
        release,
        dataset,
        implementation_ref,
        policy,
        policy_digest,
        experiment_name,
        execution,
    )
    return RelevanceV3ExperimentProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        experiment_id=execution.experiment_id,
        experiment_name=experiment_name,
        experiment_url=experiment_url,
        implementation_ref=implementation_ref,
        policy_id=policy.policy_id,
        policy_digest=policy_digest,
        case_results=results,
        full_metrics=_relevance_v3_metrics(results),
        prior_metrics=_relevance_v3_metrics(
            tuple(item for item in results if item.slice == "prior")
        ),
        new_metrics=_relevance_v3_metrics(tuple(item for item in results if item.slice == "new")),
        false_negative_ids=tuple(
            item.case_id for item in results if item.expected_accepted and not item.accepted
        ),
        context_early_exit_rate=_ratio(
            sum(item.context_early_exit for item in results), len(results)
        ),
        input_tokens=sum(item.input_tokens for item in results),
        output_tokens=sum(item.output_tokens for item in results),
        cost_usd=round(sum(item.cost_usd for item in results), 8),
        observability_complete=all(item.observability_complete for item in results),
    )


def _relevance_v3_experiment_url(
    release: LoadedNewsEvaluationRelease,
    dataset: NewsEvaluationDatasetProjection,
    implementation_ref: str,
    policy: RelevanceV3Policy,
    policy_digest: Sha256,
    experiment_name: str,
    execution: _ExperimentExecution,
) -> str | None:
    receipt = read_news_relevance_experiment_receipt(
        _PROVIDER,
        release.manifest_reference.version_id,
        policy_digest,
        implementation_ref,
    )
    if receipt is not None:
        if (
            receipt.policy_id != policy.policy_id
            or receipt.dataset_id != dataset.dataset_id
            or receipt.dataset_name != dataset.dataset_name
            or receipt.experiment_id != execution.experiment_id
            or receipt.experiment_name != experiment_name
        ):
            raise ValueError("Experiment receipt conflicts with this V3 relevance run")
        experiment_url = receipt.experiment_url or execution.experiment_url
    else:
        experiment_url = execution.experiment_url
        record_news_relevance_experiment_receipt(
            NewsRelevanceExperimentReceipt(
                provider=_PROVIDER,
                manifest_artifact_version_id=release.manifest_reference.version_id,
                policy_digest=policy_digest,
                implementation_ref=implementation_ref,
                policy_id=policy.policy_id,
                dataset_id=dataset.dataset_id,
                dataset_name=dataset.dataset_name,
                experiment_id=execution.experiment_id,
                experiment_name=experiment_name,
                experiment_url=experiment_url,
            )
        )
    return experiment_url


def run_relevance_v3_evaluation(
    release: LoadedNewsEvaluationRelease,
    plan: FreshEvaluationPlan,
    policy: RelevanceV3Policy = RELEVANCE_V3_POLICY,
) -> RelevanceV3EvaluationProjection:
    """Run or resume every fresh V3 relevance trial in one validated group."""
    runs = tuple(
        _run_relevance_v3_experiment(release, implementation_ref, policy)
        for implementation_ref in plan.implementation_refs
    )
    failures = tuple(
        failure
        for position, run in enumerate(runs, start=1)
        for failure in _relevance_v3_threshold_failures(run, position)
    )
    return RelevanceV3EvaluationProjection(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        plan=plan,
        runs=runs,
        false_negative_ids=tuple(
            sorted({case_id for run in runs for case_id in run.false_negative_ids})
        ),
        threshold_failures=failures,
        passed=not failures,
    )


def relevance_promotion_failures(
    baseline: RelevancePolicyExperimentProjection,
    candidate: RelevancePolicyExperimentProjection,
) -> tuple[str, ...]:
    """Return each failed promotion gate for a fresh relevance comparison."""
    identity_failures = _relevance_comparison_identity_failures(baseline, candidate)
    if identity_failures:
        return identity_failures
    failures = []
    for result in candidate.case_results:
        if not result.expected_accepted and result.accepted:
            failures.append(f"expected-negative case accepted: {result.case_id}")
        if result.control and result.expected_accepted and not result.accepted:
            failures.append(f"positive control rejected: {result.case_id}")
    if candidate.metrics.precision <= baseline.metrics.precision:
        failures.append(
            "candidate precision did not strictly improve: "
            f"{baseline.metrics.precision:g} to {candidate.metrics.precision:g}"
        )
    if candidate.metrics.recall < baseline.metrics.recall:
        failures.append(
            f"candidate recall regressed: {baseline.metrics.recall:g} to "
            f"{candidate.metrics.recall:g}"
        )
    if (
        candidate.metrics.positive_control_preservation
        < baseline.metrics.positive_control_preservation
    ):
        failures.append(
            "candidate positive-control preservation regressed: "
            f"{baseline.metrics.positive_control_preservation:g} to "
            f"{candidate.metrics.positive_control_preservation:g}"
        )
    return tuple(failures)


def _relevance_comparison_identity_failures(
    baseline: RelevancePolicyExperimentProjection,
    candidate: RelevancePolicyExperimentProjection,
) -> tuple[str, ...]:
    failures = []
    if baseline.manifest_artifact_version_id != candidate.manifest_artifact_version_id:
        failures.append("baseline and candidate manifest identities differ")
    if baseline.implementation_ref != candidate.implementation_ref:
        failures.append("baseline and candidate implementation references differ")
    if (baseline.dataset_id, baseline.dataset_name) != (
        candidate.dataset_id,
        candidate.dataset_name,
    ):
        failures.append("baseline and candidate dataset identities differ")
    baseline_cases = tuple(
        (item.case_id, item.article_version_id, item.control, item.expected_accepted)
        for item in baseline.case_results
    )
    candidate_cases = tuple(
        (item.case_id, item.article_version_id, item.control, item.expected_accepted)
        for item in candidate.case_results
    )
    if baseline_cases != candidate_cases:
        failures.append("baseline and candidate case identities differ")
    return tuple(failures)


def _relevance_v3_expected(
    release: LoadedNewsEvaluationRelease,
    examples: tuple[DatasetItem, ...],
    policy: RelevanceV3Policy,
    execution_ref: str,
) -> dict[str, _RelevanceV3Expected]:
    specs = {case.case_id: case for case in release.manifest.cases if case.concern == "relevance"}
    prior_ids = _prior_relevance_case_ids(release, specs)
    expected = {}
    digest = relevance_v3_policy_digest(policy)
    for example in examples:
        projected = _ExperimentInput.model_validate(example.input, strict=False)
        case = specs.get(projected.case_id)
        if case is None:
            raise ValueError(f"V3 relevance example has no manifest case: {projected.case_id}")
        if case.article not in projected.artifact_references:
            raise ValueError(f"V3 relevance example omits its exact article: {case.case_id}")
        expected[example.id] = _RelevanceV3Expected(
            example_id=example.id,
            case_id=case.case_id,
            article=case.article,
            execution_ref=execution_ref,
            expected_accepted=case.expected_accepted,
            control=case.control,
            slice="prior" if case.case_id in prior_ids else "new",
            policy_id=policy.policy_id,
            policy_digest=digest,
            request_id=relevance_v3_request_id(
                case.article,
                mode="full_evaluation",
                policy=policy,
                execution_ref=execution_ref,
            ),
        )
    if len(expected) != len(specs):
        raise ValueError("V3 relevance examples do not exactly match manifest relevance cases")
    return expected


def _prior_relevance_case_ids(
    release: LoadedNewsEvaluationRelease,
    current_specs: dict[str, RelevanceEvaluationSpec],
) -> frozenset[str]:
    reference = release.manifest.prior_manifest
    if reference is None:
        raise ValueError("V3 relevance evaluation requires a prior manifest reference")
    prior = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(reference.r2_key, reference.content_digest),
        strict=True,
    )
    prior_specs = {case.case_id: case for case in prior.cases if case.concern == "relevance"}
    missing = set(prior_specs) - set(current_specs)
    if missing:
        raise ValueError(f"Current manifest omits prior relevance cases: {sorted(missing)}")
    for case_id, prior_case in prior_specs.items():
        current_case = current_specs[case_id]
        if (
            current_case.article != prior_case.article
            or current_case.expected_accepted != prior_case.expected_accepted
            or current_case.control != prior_case.control
        ):
            raise ValueError(f"Prior relevance case identity changed: {case_id}")
    return frozenset(prior_specs)


def _validated_relevance_v3_output(
    output: object,
    expected: _RelevanceV3Expected,
    policy: RelevanceV3Policy,
) -> _RelevanceV3RunOutput:
    try:
        validated = _RelevanceV3RunOutput.model_validate(output, strict=True)
    except ValueError as error:
        raise ValueError(
            f"V3 relevance run has an invalid decision for {expected.case_id}"
        ) from error
    identity = (
        validated.case_id,
        validated.article_version_id,
        validated.execution_ref,
        validated.policy_id,
        validated.policy_digest,
        validated.request_id,
    )
    expected_identity = (
        expected.case_id,
        expected.article.version_id,
        expected.execution_ref,
        expected.policy_id,
        expected.policy_digest,
        expected.request_id,
    )
    if identity != expected_identity:
        raise ValueError(f"V3 relevance run identity conflicts for {expected.case_id}")
    accepted = relevance_v3_is_accepted(
        validated.context,
        validated.impact,
        policy.acceptance,
    )
    if validated.accepted != accepted:
        raise ValueError(f"V3 relevance run acceptance conflicts for {expected.case_id}")
    expected_early_exit = not context_is_accepted(validated.context, policy.acceptance.context)
    if validated.context_early_exit != expected_early_exit:
        raise ValueError(f"V3 relevance early-exit result conflicts for {expected.case_id}")
    return validated


def _relevance_v3_case_result(
    expected: _RelevanceV3Expected,
    output: object,
    policy: RelevanceV3Policy,
) -> RelevanceV3CaseResult:
    validated = _validated_relevance_v3_output(output, expected, policy)
    return RelevanceV3CaseResult(
        case_id=expected.case_id,
        article_version_id=expected.article.version_id,
        request_id=expected.request_id,
        execution_ref=expected.execution_ref,
        expected_accepted=expected.expected_accepted,
        control=expected.control,
        slice=expected.slice,
        accepted=validated.accepted,
        passed=validated.accepted == expected.expected_accepted,
        context=validated.context,
        impact=validated.impact,
        context_early_exit=validated.context_early_exit,
        input_tokens=validated.input_tokens,
        output_tokens=validated.output_tokens,
        cost_usd=validated.cost_usd,
        observability_complete=validated.observability_complete,
    )


def _relevance_v3_metrics(
    results: tuple[RelevanceV3CaseResult, ...],
) -> RelevanceV3SliceMetrics:
    true_positives = sum(item.expected_accepted and item.accepted for item in results)
    false_positives = sum(not item.expected_accepted and item.accepted for item in results)
    false_negatives = sum(item.expected_accepted and not item.accepted for item in results)
    return RelevanceV3SliceMetrics(
        passed_cases=sum(item.passed for item in results),
        total_cases=len(results),
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, true_positives + false_negatives),
    )


def _relevance_v3_threshold_failures(
    run: RelevanceV3ExperimentProjection,
    repetition: int,
) -> tuple[str, ...]:
    failures = []
    if not run.observability_complete:
        failures.append(f"run {repetition} has incomplete observability")
    if run.full_metrics.precision < 0.90:
        failures.append(f"run {repetition} precision below 0.90: {run.full_metrics.precision:g}")
    if run.full_metrics.recall < 0.90:
        failures.append(f"run {repetition} recall below 0.90: {run.full_metrics.recall:g}")
    return tuple(failures)


def _fresh_relevance_expected(
    release: LoadedNewsEvaluationRelease,
    examples: tuple[DatasetItem, ...],
    policy: RelevancePolicy,
) -> dict[str, _FreshRelevanceExpected]:
    specs = {case.case_id: case for case in release.manifest.cases if case.concern == "relevance"}
    expected = {}
    digest = relevance_policy_digest(policy)
    for example in examples:
        projected = _ExperimentInput.model_validate(example.input, strict=False)
        case = specs.get(projected.case_id)
        if case is None:
            raise ValueError(f"Fresh relevance example has no manifest case: {projected.case_id}")
        if case.article not in projected.artifact_references:
            raise ValueError(f"Fresh relevance example omits its exact article: {case.case_id}")
        expected[example.id] = _FreshRelevanceExpected(
            example_id=example.id,
            case_id=case.case_id,
            article=case.article,
            expected_accepted=case.expected_accepted,
            control=case.control,
            policy_id=policy.policy_id,
            policy_digest=digest,
            request_id=relevance_request_id(case.article, policy),
        )
    if len(expected) != len(specs):
        raise ValueError("Fresh relevance examples do not exactly match manifest relevance cases")
    return expected


def _fresh_relevance_case_result(
    expected: _FreshRelevanceExpected,
    output: object,
    policy: RelevancePolicy,
) -> FreshRelevanceCaseResult:
    validated = _validated_relevance_run_output(output, expected, policy)
    return FreshRelevanceCaseResult(
        case_id=expected.case_id,
        article_version_id=expected.article.version_id,
        policy_id=expected.policy_id,
        policy_digest=expected.policy_digest,
        request_id=expected.request_id,
        expected_accepted=expected.expected_accepted,
        control=expected.control,
        accepted=validated.accepted,
        passed=validated.accepted == expected.expected_accepted,
        decision=validated.decision,
    )


def _fresh_relevance_metrics(
    results: tuple[FreshRelevanceCaseResult, ...],
) -> FreshRelevanceMetrics:
    true_positives = sum(item.expected_accepted and item.accepted for item in results)
    false_positives = sum(not item.expected_accepted and item.accepted for item in results)
    false_negatives = sum(item.expected_accepted and not item.accepted for item in results)
    controls = tuple(item for item in results if item.control and item.expected_accepted)
    return FreshRelevanceMetrics(
        passed_cases=sum(item.passed for item in results),
        total_cases=len(results),
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, true_positives + false_negatives),
        positive_control_preservation=_ratio(
            sum(item.accepted for item in controls),
            len(controls),
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_or_create_dataset(
    client: Langfuse,
    release: LoadedNewsEvaluationRelease,
    dataset_name: str,
):
    identity = _dataset_metadata(release)
    try:
        dataset = client.get_dataset(dataset_name)
    except ApiError as error:
        if error.status_code != 404:
            raise
        client.create_dataset(
            name=dataset_name,
            description="Reference-only projection of one immutable Romanian news evaluation release.",
            metadata=identity,
        )
        dataset = client.get_dataset(dataset_name)
    if dataset.metadata != identity:
        raise ValueError("Dataset name already belongs to another projection")
    return dataset


def _validated_relevance_run_output(
    output: object,
    expected: _FreshRelevanceExpected,
    policy: RelevancePolicy,
) -> _FreshRelevanceRunOutput:
    try:
        validated = _FreshRelevanceRunOutput.model_validate(output, strict=False)
    except ValueError as error:
        raise ValueError(
            f"Relevance run has an invalid typed decision for {expected.case_id}"
        ) from error
    identity = (
        validated.case_id,
        validated.article_version_id,
        validated.policy_id,
        validated.policy_digest,
        validated.request_id,
    )
    expected_identity = (
        expected.case_id,
        expected.article.version_id,
        expected.policy_id,
        expected.policy_digest,
        expected.request_id,
    )
    if identity != expected_identity:
        raise ValueError(f"Relevance run identity conflicts for {expected.case_id}")
    accepted = relevance_is_accepted(validated.decision, policy)
    if validated.accepted != accepted:
        raise ValueError(f"Relevance run acceptance conflicts for {expected.case_id}")
    return validated


def _projected_examples(
    release: LoadedNewsEvaluationRelease,
) -> tuple[_ProjectedExample, ...]:
    reports = {item.report.version_id: item.report for item in release.manifest.reports}
    return tuple(
        example
        for case in release.manifest.cases
        for example in _project_cases(release, case, reports)
    )


def _project_cases(
    release: LoadedNewsEvaluationRelease,
    case: NewsEvaluationSpec,
    reports: dict[Sha256, ArtifactReference],
) -> tuple[_ProjectedExample, ...]:
    if (
        case.concern != "daily_theme"
        or case.model_output is None
        or case.report_expectation is not None
    ):
        return (_project_case(release, case, reports),)
    references = list(_case_references(case))
    report_version_id = case.provenance.report_version_id
    if report_version_id is None:
        raise ValueError("Executable theme cases require a frozen report")
    references.insert(0, reports[report_version_id])
    inputs = [item.model_dump(mode="json") for item in _unique_references(references)]
    return tuple(
        _ProjectedExample(
            example_id=_example_id(
                release.manifest_reference.version_id,
                "case",
                expectation.case_id,
            ),
            inputs={"case_id": expectation.case_id, "artifact_references": inputs},
            outputs={"expected_same_group": expectation.expected_same_theme},
            metadata=_example_metadata(
                release,
                concern="grouping",
                control=expectation.control,
                feedback_ids=expectation.feedback_ids,
                report_version_id=report_version_id,
                group_ids=(expectation.left_group_id, expectation.right_group_id),
            ),
            split="executable",
        )
        for expectation in case.expectations
    )


def _project_case(
    release: LoadedNewsEvaluationRelease,
    case: NewsEvaluationSpec,
    reports: dict[Sha256, ArtifactReference],
) -> _ProjectedExample:
    references = list(_case_references(case))
    report_version_id = case.provenance.report_version_id
    if report_version_id is not None:
        references.insert(0, reports[report_version_id])
    inputs: dict[str, object] = {
        "case_id": case.case_id,
        "artifact_references": [
            item.model_dump(mode="json") for item in _unique_references(references)
        ],
    }
    if case.concern == "relevance":
        outputs: dict[str, object] = {"expected_accepted": case.expected_accepted}
        group_ids = _optional_ids(case.provenance.group_id)
    elif case.concern == "summary_format":
        outputs = {"expected_valid_format": True}
        group_ids = _optional_ids(case.provenance.group_id)
    elif case.concern == "grouping":
        outputs = {"expected_same_group": case.expected_same_group}
        group_ids = _optional_ids(case.provenance.group_id)
    elif case.concern == "ranking":
        outputs = {
            "expected_higher_group_id": case.higher_group_id,
            "expected_lower_group_id": case.lower_group_id,
        }
        group_ids = _unique_strings(
            (case.provenance.group_id, case.higher_group_id, case.lower_group_id)
        )
    elif case.concern == "tier":
        outputs = {"expected_tier": case.expected_tier}
        group_ids = _optional_ids(case.provenance.group_id)
    elif case.concern == "confidence":
        outputs = {"expected_sufficient": case.expected_sufficient}
        group_ids = _optional_ids(case.provenance.group_id)
    elif case.concern == "daily_theme":
        outputs = {
            "expectations": [
                expectation.model_dump(mode="json") for expectation in case.expectations
            ],
            "report_expectation": (
                case.report_expectation.model_dump(mode="json")
                if case.report_expectation is not None
                else None
            ),
        }
        group_ids = _unique_strings(
            tuple(
                group_id
                for expectation in case.expectations
                for group_id in (expectation.left_group_id, expectation.right_group_id)
            )
        )
    else:
        outputs = {"expected_reader_presentation": case.expectation.model_dump(mode="json")}
        group_ids = _optional_ids(case.provenance.group_id)
    return _ProjectedExample(
        example_id=_example_id(release.manifest_reference.version_id, "case", case.case_id),
        inputs=inputs,
        outputs=outputs,
        metadata=_example_metadata(
            release,
            concern=case.concern,
            control=case.control,
            feedback_ids=case.provenance.feedback_ids,
            report_version_id=report_version_id,
            group_ids=group_ids,
        ),
        split="executable",
    )


def _case_references(case: NewsEvaluationSpec) -> tuple[ArtifactReference, ...]:
    if case.concern == "relevance":
        return case.article, case.model_output
    if case.concern == "summary_format":
        return *case.articles, case.model_output
    if case.concern == "grouping":
        return (
            case.source_cluster_set,
            *(
                reference
                for article in case.articles
                for reference in (article.article, article.relevance, article.embedding)
            ),
        )
    if case.concern == "daily_theme":
        references = (case.source_report, case.cluster_set, *case.summaries)
        return (*references, case.model_output) if case.model_output is not None else references
    if case.concern == "ranking":
        return tuple(
            reference for item in case.relevance for reference in (item.article, item.model_output)
        )
    return ()


def _example_metadata(
    release: LoadedNewsEvaluationRelease,
    *,
    concern: str,
    control: bool,
    feedback_ids: tuple[UUID, ...],
    report_version_id: Sha256 | None,
    group_ids: tuple[Sha256, ...],
) -> dict[str, object]:
    return {
        "concern": concern,
        "control": control,
        "feedback_ids": [str(item) for item in feedback_ids],
        "report_version_id": report_version_id,
        "group_ids": list(group_ids),
        "manifest": release.manifest_reference.model_dump(mode="json"),
    }


def _dataset_metadata(release: LoadedNewsEvaluationRelease) -> dict[str, object]:
    return {
        "manifest": release.manifest_reference.model_dump(mode="json"),
        "dataset_version": release.manifest.version,
    }


def _experiment_metadata(
    release: LoadedNewsEvaluationRelease,
    implementation_ref: str,
) -> dict[str, str]:
    return _flatten_metadata(
        {
            "manifest": release.manifest_reference.model_dump(mode="json"),
            "implementation_ref": implementation_ref,
        }
    )


def _relevance_v3_experiment_metadata(
    release: LoadedNewsEvaluationRelease,
    policy: RelevanceV3Policy,
    implementation_ref: str,
) -> dict[str, str]:
    return _flatten_metadata(
        {
            "manifest": release.manifest_reference.model_dump(mode="json"),
            "manifest_version": release.manifest.version,
            "implementation_ref": implementation_ref,
            "policy_id": policy.policy_id,
            "policy_digest": relevance_v3_policy_digest(policy),
            "execution_mode": "full_evaluation",
        }
    )


def _relevance_experiment_metadata(
    release: LoadedNewsEvaluationRelease,
    policy: RelevancePolicy,
    implementation_ref: str,
) -> dict[str, str]:
    return _flatten_metadata(
        {
            "manifest": release.manifest_reference.model_dump(mode="json"),
            "manifest_version": release.manifest.version,
            "implementation_ref": implementation_ref,
            "policy_id": policy.policy_id,
            "policy_digest": relevance_policy_digest(policy),
        }
    )


def _theme_experiment_metadata(
    release: LoadedNewsEvaluationRelease,
    definition: SparseThemePolicyDefinition,
    implementation_ref: str,
) -> dict[str, str]:
    return _flatten_metadata(
        {
            "manifest": release.manifest_reference.model_dump(mode="json"),
            "manifest_version": release.manifest.version,
            "implementation_ref": implementation_ref,
            "policy_id": definition.policy.policy_id,
            "policy_digest": sparse_theme_policy_digest(definition.policy),
            "evaluation_kind": "daily_theme",
        }
    )


def _dataset_items_by_id(items: list[DatasetItem]) -> dict[str, DatasetItem]:
    by_id = {item.id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("Dataset returned duplicate item IDs")
    return by_id


def _item_split(item: DatasetItem) -> str | None:
    return item.metadata.get("split") if isinstance(item.metadata, dict) else None


def _require_examples_match(
    existing: dict[str, DatasetItem],
    expected: tuple[_ProjectedExample, ...],
) -> None:
    expected_by_id = {item.example_id: item for item in expected}
    unrelated = set(existing) - set(expected_by_id)
    if unrelated:
        raise ValueError(f"Dataset contains unrelated item IDs: {sorted(unrelated)}")
    for item_id in set(existing) & set(expected_by_id):
        actual = existing[item_id]
        projected = expected_by_id[item_id]
        expected_metadata = {**projected.metadata, "split": projected.split}
        if (
            actual.input != projected.inputs
            or actual.expected_output != projected.outputs
            or actual.metadata != expected_metadata
        ):
            raise ValueError(f"Dataset item identity conflict: {item_id}")


def _results_by_case(
    release: LoadedNewsEvaluationRelease,
) -> dict[str, EvaluationCaseResult]:
    report = evaluate_news_dataset(release.dataset)
    if report.dataset_version != release.manifest.version:
        raise ValueError("Evaluation report belongs to another manifest version")
    results = {item.case_id: item for item in report.case_results}
    if len(results) != len(report.case_results):
        raise ValueError("Evaluation report contains duplicate case IDs")
    expected = {
        case_id
        for item in release.manifest.cases
        for case_id in (
            tuple(expectation.case_id for expectation in item.expectations)
            if item.concern == "daily_theme" and item.model_output is not None
            else ()
            if item.concern == "daily_theme"
            else (item.case_id,)
        )
    }
    if set(results) != expected:
        raise ValueError("Evaluation report case IDs do not match the manifest")
    for case in release.manifest.cases:
        if case.concern == "daily_theme":
            if case.model_output is not None:
                for expectation in case.expectations:
                    result = results[expectation.case_id]
                    if (result.concern, result.control) != (
                        "grouping",
                        expectation.control,
                    ):
                        raise ValueError(
                            f"Evaluation report metadata differs for case {expectation.case_id}"
                        )
            continue
        result = results[case.case_id]
        if (result.concern, result.control) != (case.concern, case.control):
            raise ValueError(f"Evaluation report metadata differs for case {case.case_id}")
    return results


def _expected_run_outputs(
    examples: tuple[DatasetItem, ...],
    results_by_case: dict[str, EvaluationCaseResult],
) -> dict[str, dict[str, object]]:
    expected = {}
    represented = set()
    for example in examples:
        projected = _ExperimentInput.model_validate(example.input, strict=False)
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        if metadata.get("concern") == "daily_theme":
            raise ValueError("Deterministic experiments cannot include daily-theme cases")
        result = results_by_case[projected.case_id]
        represented.add(result.case_id)
        expected[example.id] = _ExperimentOutput(
            case_id=result.case_id,
            passed=result.passed,
        ).model_dump(mode="json")
    if len(expected) != len(examples) or represented != set(results_by_case):
        raise ValueError("Executable dataset items do not exactly match evaluation cases")
    return expected


@dataclass(frozen=True)
class _ExperimentExecution:
    experiment_id: str
    experiment_url: str | None
    outputs: dict[str, dict[str, object]]
    ended: bool
    item_count: int


def _run_missing_experiment_items(
    client: Langfuse,
    *,
    dataset_name: str,
    experiment_name: str,
    examples_by_id: dict[str, DatasetItem],
    outputs: dict[str, dict[str, object]],
    experiment_id: str | None,
    experiment_url: str | None,
    missing: tuple[DatasetItem, ...],
    task: Callable[..., dict[str, object]],
    evaluator: EvaluatorFunction,
    metadata: dict[str, str],
    description: str,
) -> tuple[dict[str, dict[str, object]], str | None, str | None]:
    """Run the dataset items a reused experiment has not produced yet."""
    if not missing:
        return outputs, experiment_id, experiment_url
    result = client.run_experiment(
        name=experiment_name,
        run_name=experiment_name,
        description=description,
        data=list(missing),
        task=task,
        evaluators=[evaluator],
        max_concurrency=1,
        metadata=metadata,
    )
    run_id = result.dataset_run_id or result.experiment_id
    if experiment_id is not None and experiment_id != run_id:
        raise ValueError("Experiment retry created a different remote run")
    experiment_id = run_id
    experiment_url = result.dataset_run_url or experiment_url
    for item_result in result.item_results:
        item = item_result.item
        if not isinstance(item, DatasetItem):
            raise ValueError("Experiment returned a local item instead of a dataset item")
        item_id = item.id
        if item_id in outputs:
            raise ValueError(f"Experiment returned a duplicate dataset item: {item_id}")
        outputs[item_id] = item_result.output
    return outputs, experiment_id, experiment_url


def _run_or_reuse_experiment(
    client: Langfuse,
    *,
    dataset_name: str,
    experiment_name: str,
    examples: tuple[DatasetItem, ...],
    expected_outputs: dict[str, dict[str, object]] | None,
    task: Callable[..., dict[str, object]],
    evaluator: EvaluatorFunction,
    metadata: dict[str, str],
    description: str,
    validate_output: Callable[[DatasetItem, object], dict[str, object]] | None = None,
) -> _ExperimentExecution:
    examples_by_id = {item.id: item for item in examples}
    existing = _read_remote_experiment_before_run(client, dataset_name, experiment_name, metadata)
    if existing is not None:
        existing = _wait_for_active_experiment_end(
            client,
            dataset_name,
            experiment_name,
            metadata,
            existing,
        )
        existing = _wait_for_existing_experiment_outputs(
            client,
            dataset_name,
            experiment_name,
            examples_by_id,
            expected_outputs,
            validate_output,
            metadata,
            existing,
        )
    outputs = dict(existing.outputs) if existing is not None else {}
    unrelated = set(outputs) - set(examples_by_id)
    if unrelated:
        raise ValueError(f"Experiment contains unrelated dataset items: {sorted(unrelated)}")
    _validate_experiment_outputs(examples_by_id, outputs, expected_outputs, validate_output)
    missing = tuple(item for item in examples if item.id not in outputs)
    experiment_id = existing.experiment_id if existing is not None else None
    experiment_url = existing.experiment_url if existing is not None else None
    outputs, experiment_id, experiment_url = _run_missing_experiment_items(
        client,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        examples_by_id=examples_by_id,
        outputs=outputs,
        experiment_id=experiment_id,
        experiment_url=experiment_url,
        missing=missing,
        task=task,
        evaluator=evaluator,
        metadata=metadata,
        description=description,
    )
    if experiment_id is None:
        raise ValueError("Experiment did not return a remote identity")
    remote = _read_complete_remote_experiment(
        client,
        dataset_name,
        experiment_name,
        examples_by_id,
        expected_outputs,
        validate_output,
        metadata,
    )
    if remote.experiment_id != experiment_id:
        raise ValueError("Experiment result conflicts with the remote run")
    outputs = remote.outputs
    missing_count = len(set(examples_by_id) - set(outputs))
    if missing_count:
        raise ValueError(
            f"Experiment is incomplete: missing {missing_count} of {len(examples_by_id)} items"
        )
    return _ExperimentExecution(
        experiment_id=remote.experiment_id,
        experiment_url=experiment_url or remote.experiment_url,
        outputs=outputs,
        ended=remote.ended,
        item_count=remote.item_count,
    )


def _wait_for_active_experiment_end(
    client: Langfuse,
    dataset_name: str,
    experiment_name: str,
    metadata: dict[str, str],
    existing: _ExperimentExecution,
) -> _ExperimentExecution:
    current = existing
    if current.ended:
        return current
    for delay in _ACTIVE_RUN_RETRY_DELAYS:
        _sleep(delay)
        visible = _read_remote_experiment(client, dataset_name, experiment_name, metadata)
        if visible is not None:
            current = visible
            if current.ended:
                return current
    raise ValueError(
        "Matching experiment remained active after wait: "
        f"items={current.item_count}, outputs={len(current.outputs)}"
    )


def _wait_for_existing_experiment_outputs(
    client: Langfuse,
    dataset_name: str,
    experiment_name: str,
    examples: dict[str, DatasetItem],
    expected_outputs: dict[str, dict[str, object]] | None,
    validate_output: Callable[[DatasetItem, object], dict[str, object]] | None,
    metadata: dict[str, str],
    existing: _ExperimentExecution,
) -> _ExperimentExecution:
    current = existing
    _validate_experiment_outputs(examples, current.outputs, expected_outputs, validate_output)
    if _remote_outputs_complete(current, examples):
        return current
    for delay in _RUN_VISIBILITY_RETRY_DELAYS:
        _sleep(delay)
        visible = _read_remote_experiment(client, dataset_name, experiment_name, metadata)
        if visible is not None:
            current = visible
            _validate_experiment_outputs(
                examples, current.outputs, expected_outputs, validate_output
            )
            if _remote_outputs_complete(current, examples):
                return current
    return current


def _read_complete_remote_experiment(
    client: Langfuse,
    dataset_name: str,
    experiment_name: str,
    examples: dict[str, DatasetItem],
    expected_outputs: dict[str, dict[str, object]] | None,
    validate_output: Callable[[DatasetItem, object], dict[str, object]] | None,
    metadata: dict[str, str],
) -> _ExperimentExecution:
    remote = None
    for attempt in range(len(_RUN_VISIBILITY_RETRY_DELAYS) + 1):
        remote = _read_remote_experiment(client, dataset_name, experiment_name, metadata)
        if remote is not None:
            _validate_experiment_outputs(
                examples, remote.outputs, expected_outputs, validate_output
            )
            if _remote_outputs_complete(remote, examples):
                return remote
        if attempt < len(_RUN_VISIBILITY_RETRY_DELAYS):
            _sleep(_RUN_VISIBILITY_RETRY_DELAYS[attempt])
    visible = 0 if remote is None else len(remote.outputs)
    raise ValueError(
        f"Experiment remained incomplete after read retries: "
        f"missing {len(examples) - visible} of {len(examples)} items"
    )


def _remote_outputs_complete(
    remote: _ExperimentExecution,
    examples: dict[str, DatasetItem],
) -> bool:
    return (
        remote.ended and remote.item_count >= len(examples) and set(remote.outputs) == set(examples)
    )


def _validate_experiment_outputs(
    examples: dict[str, DatasetItem],
    outputs: dict[str, dict[str, object]],
    expected_outputs: dict[str, dict[str, object]] | None,
    validate_output: Callable[[DatasetItem, object], dict[str, object]] | None,
) -> None:
    unrelated = set(outputs) - set(examples)
    if unrelated:
        raise ValueError(f"Experiment contains unrelated dataset items: {sorted(unrelated)}")
    for item_id, output in outputs.items():
        if expected_outputs is not None and output != expected_outputs[item_id]:
            raise ValueError(f"Experiment output conflicts for {item_id}")
        if validate_output is not None:
            validate_output(examples[item_id], output)


def _read_remote_experiment_before_run(
    client: Langfuse,
    dataset_name: str,
    experiment_name: str,
    metadata: dict[str, str],
) -> _ExperimentExecution | None:
    for delay in (*_RUN_VISIBILITY_RETRY_DELAYS, None):
        remote = _read_remote_experiment(client, dataset_name, experiment_name, metadata)
        if remote is not None:
            return remote
        if delay is not None:
            _sleep(delay)
    return None


def _read_remote_experiment(
    client: Langfuse,
    dataset_name: str,
    experiment_name: str,
    metadata: dict[str, str],
) -> _ExperimentExecution | None:
    dataset_id = client.get_dataset(dataset_name).id
    experiments = _experiment_pages(
        lambda cursor: client.api.experiments.list(
            from_start_time=_EXPERIMENT_QUERY_START,
            fields="core,metadata",
            limit=100,
            cursor=cursor,
            dataset_id=dataset_id,
        )
    )
    expected_metadata = _flatten_metadata(metadata)
    matches = tuple(
        experiment
        for experiment in experiments
        if experiment.dataset_id == dataset_id and experiment.metadata == expected_metadata
    )
    if not matches:
        return None
    experiment, *extra_matches = matches
    if extra_matches:
        raise ValueError("Experiment identity is not unique")
    run_items = _experiment_pages(
        lambda cursor: client.api.experiments.list_items(
            from_start_time=_EXPERIMENT_QUERY_START,
            fields="dataset,io",
            limit=100,
            cursor=cursor,
            experiment_id=experiment.id,
            dataset_id=dataset_id,
        )
    )
    outputs: dict[str, dict[str, object]] = {}
    for item in run_items:
        if item.experiment_id != experiment.id:
            raise ValueError("Experiment item belongs to another experiment")
        if (
            item.end_time is None
            or item.level == "ERROR"
            or item.output is None
            or (isinstance(item.output, str) and not item.output.strip())
        ):
            continue
        if item.experiment_item_id in outputs:
            raise ValueError(f"Experiment contains duplicate item {item.experiment_item_id}")
        outputs[item.experiment_item_id] = _experiment_output(item.output)
    return _ExperimentExecution(
        experiment_id=experiment.id,
        experiment_url=_experiment_url(dataset_id, experiment.id),
        outputs=outputs,
        ended=experiment.end_time is not None,
        item_count=int(experiment.item_count or 0),
    )


class _ExperimentPageMeta(Protocol):
    cursor: str | None


class _ExperimentPage(Protocol[_PageT]):
    @property
    def data(self) -> Sequence[_PageT]: ...

    @property
    def meta(self) -> _ExperimentPageMeta: ...


def _experiment_pages(fetch: Callable[[str | None], _ExperimentPage[_PageT]]) -> tuple[_PageT, ...]:
    values = []
    cursor = None
    while True:
        response = fetch(cursor)
        values.extend(response.data)
        cursor = response.meta.cursor
        if cursor is None:
            return tuple(values)


def _flatten_metadata(value: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Flatten and stringify metadata exactly like the Langfuse SDK stores it."""
    flattened: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_metadata(item, name))
        elif item is None:
            continue
        else:
            flattened[name] = item if isinstance(item, str) else json.dumps(item)
    return flattened


def _experiment_output(value: object) -> dict[str, object]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Experiment item output is not an object")
    return parsed


def _experiment_url(dataset_id: str, experiment_id: str) -> str | None:
    if not LANGFUSE_PROJECT_ID:
        return None
    base_url = (LANGFUSE_BASE_URL or "https://cloud.langfuse.com").rstrip("/")
    return f"{base_url}/project/{LANGFUSE_PROJECT_ID}/datasets/{dataset_id}/runs/{experiment_id}"


def _require_projection_receipt_matches(
    receipt: NewsEvaluationExperimentReceipt,
    dataset: NewsEvaluationDatasetProjection,
    experiment_name: str,
    implementation_ref: str,
) -> None:
    if (
        receipt.dataset_id != dataset.dataset_id
        or receipt.dataset_name != dataset.dataset_name
        or receipt.experiment_name != experiment_name
        or receipt.implementation_ref != implementation_ref
    ):
        raise ValueError("Experiment receipt conflicts with this evaluation run")


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"Receipt omits {name}")
    return value


def _unique_references(
    references: list[ArtifactReference],
) -> tuple[ArtifactReference, ...]:
    by_version: dict[Sha256, ArtifactReference] = {}
    for reference in references:
        existing = by_version.get(reference.version_id)
        if existing is not None and existing != reference:
            raise ValueError("Projection has conflicting artifact reference identities")
        by_version[reference.version_id] = reference
    return tuple(by_version.values())


def _optional_ids(value: Sha256 | None) -> tuple[Sha256, ...]:
    return (value,) if value is not None else ()


def _unique_strings(values: tuple[Sha256 | None, ...]) -> tuple[Sha256, ...]:
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _dataset_name(manifest_version_id: Sha256) -> str:
    return f"chartly-news-evaluation-{manifest_version_id}"


def _example_id(manifest_version_id: Sha256, kind: str, item_id: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"chartly:news-evaluation:{manifest_version_id}:{kind}:{item_id}",
        )
    )


def _experiment_name(manifest_version_id: Sha256, implementation_ref: str) -> str:
    implementation_digest = hashlib.sha256(implementation_ref.encode()).hexdigest()
    return f"chartly-news-evaluation-{manifest_version_id}-{implementation_digest}"


def _relevance_experiment_name(
    manifest_version_id: Sha256,
    policy_digest: Sha256,
    implementation_ref: str,
) -> str:
    implementation_digest = hashlib.sha256(implementation_ref.encode()).hexdigest()
    return f"chartly-news-relevance-{manifest_version_id}-{policy_digest}-{implementation_digest}"


def _theme_experiment_name(
    manifest_version_id: Sha256,
    policy_digest: Sha256,
    implementation_ref: str,
) -> str:
    implementation_digest = hashlib.sha256(implementation_ref.encode()).hexdigest()
    return f"chartly-news-theme-{manifest_version_id[:16]}-{policy_digest}-{implementation_digest}"


def _require_langfuse_credentials() -> None:
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for evaluation projection"
        )


@cache
def _langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        base_url=LANGFUSE_BASE_URL,
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-ref", required=True, type=_non_empty)
    parser.add_argument("--compare-relevance", action="store_true")
    parser.add_argument("--evaluate-relevance-v3", action="store_true")
    parser.add_argument("--trial-count", type=int, default=None)
    arguments = parser.parse_args(argv)
    release = load_news_evaluation_release(PIN_PATH.read_bytes())
    if arguments.compare_relevance:
        plan = FreshEvaluationPlan(
            implementation_ref=arguments.implementation_ref,
            trial_count=3 if arguments.trial_count is None else arguments.trial_count,
        )
        comparison = compare_relevance_policies(release, plan)
        print(json.dumps(comparison.model_dump(mode="json"), indent=2))
        return 0 if comparison.promoted else 1
    if arguments.evaluate_relevance_v3:
        plan = FreshEvaluationPlan(
            implementation_ref=arguments.implementation_ref,
            trial_count=3 if arguments.trial_count is None else arguments.trial_count,
        )
        evaluation = run_relevance_v3_evaluation(release, plan)
        print(json.dumps(evaluation.model_dump(mode="json"), indent=2))
        return 1 if evaluation.threshold_failures else 0
    if arguments.trial_count is not None:
        parser.error("--trial-count is only valid for fresh evaluation modes")
    report = evaluate_news_dataset(release.dataset)
    regressions = compare_news_evaluation_to_baseline(
        report,
        release.baseline,
        release.manifest_reference,
    )
    if regressions:
        raise RuntimeError("Evaluation regressed: " + "; ".join(regressions))
    dataset = project_news_evaluation_release(release)
    experiment = run_news_evaluation_experiment(release, arguments.implementation_ref)
    print(
        json.dumps(
            {
                "dataset": dataset.model_dump(mode="json"),
                "experiment": experiment.model_dump(mode="json"),
            },
            indent=2,
        )
    )
    return 0


def _non_empty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("implementation reference must not be empty")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
