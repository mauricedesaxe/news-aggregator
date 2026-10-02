from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    build_relevance_binary_request,
)
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    JevExecutionPolicy,
    evaluate_jev_relevance,
)
from romanian_news.analysis.percentiles import float_linear_percentile
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.evaluations import read_news_evaluation_artifact_references
from romanian_news.evaluation import NewsEvaluationManifest, NewsEvaluationPin
from romanian_news.evaluation_projection import FreshEvaluationPlan
from romanian_news.storage import read_verified_r2_object


class LoadedJevRelevanceRelease(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    manifest: NewsEvaluationManifest


class JevRelevanceCaseResult(NewsModel):
    case_id: str
    article_version_id: Sha256
    execution_ref: str
    expected_accepted: bool
    predicted_accepted: bool
    pass_probability: Annotated[float, Field(ge=0, le=1)]
    control: bool
    passed: bool
    model: str
    request_id: Sha256
    provider_request_id: str | None
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    latency_ms: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def require_consistent_verdict(self) -> JevRelevanceCaseResult:
        if self.passed != (self.predicted_accepted == self.expected_accepted):
            raise ValueError("Jev case pass result conflicts with its prediction")
        return self


class JevRelevanceRunMetrics(NewsModel):
    passed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    positive_control_preservation: Annotated[float, Field(ge=0, le=1)]
    false_negative_ids: tuple[str, ...]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Annotated[float, Field(ge=0)]
    p50_latency_ms: Annotated[float, Field(ge=0)]
    p95_latency_ms: Annotated[float, Field(ge=0)]


class JevRelevanceRunResult(NewsModel):
    manifest_artifact_version_id: Sha256
    execution_ref: str
    policy_id: str
    policy_digest: Sha256
    case_results: tuple[JevRelevanceCaseResult, ...]
    metrics: JevRelevanceRunMetrics

    @model_validator(mode="after")
    def require_consistent_execution(self) -> JevRelevanceRunResult:
        if any(result.execution_ref != self.execution_ref for result in self.case_results):
            raise ValueError("Jev case results must use the run execution reference")
        if self.metrics != jev_relevance_metrics(self.case_results):
            raise ValueError("Jev run metrics do not match its case results")
        return self


class JevRelevanceEvaluationResult(NewsModel):
    manifest_artifact_version_id: Sha256
    plan: FreshEvaluationPlan
    runs: tuple[JevRelevanceRunResult, ...]

    @model_validator(mode="after")
    def require_complete_trials(self) -> JevRelevanceEvaluationResult:
        if tuple(run.execution_ref for run in self.runs) != self.plan.implementation_refs:
            raise ValueError("Jev runs must have exact FreshEvaluationPlan reference coverage")
        if any(
            run.manifest_artifact_version_id != self.manifest_artifact_version_id
            for run in self.runs
        ):
            raise ValueError("Jev run manifests must match the grouped result")
        if len({(run.policy_id, run.policy_digest) for run in self.runs}) > 1:
            raise ValueError("Jev runs must use one frozen policy")
        return self


def run_jev_relevance_evaluation(
    release: LoadedJevRelevanceRelease,
    plan: FreshEvaluationPlan,
    policy: JevExecutionPolicy = JEV_EXECUTION_POLICY,
) -> JevRelevanceEvaluationResult:
    runs = tuple(
        _run_jev_relevance_trial(release, execution_ref, policy)
        for execution_ref in plan.implementation_refs
    )
    return JevRelevanceEvaluationResult(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        plan=plan,
        runs=runs,
    )


def load_jev_relevance_release(pin_content: bytes | str) -> LoadedJevRelevanceRelease:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    references = read_news_evaluation_artifact_references((pin.manifest_version_id,))
    manifest_reference = references[pin.manifest_version_id]
    manifest = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(
            manifest_reference.r2_key,
            manifest_reference.content_digest,
        ),
        strict=True,
    )
    return LoadedJevRelevanceRelease(
        pin=pin,
        manifest_reference=manifest_reference,
        manifest=manifest,
    )


def jev_relevance_metrics(
    results: tuple[JevRelevanceCaseResult, ...],
) -> JevRelevanceRunMetrics:
    true_positives = sum(item.expected_accepted and item.predicted_accepted for item in results)
    false_positives = sum(
        not item.expected_accepted and item.predicted_accepted for item in results
    )
    false_negatives = tuple(
        item.case_id for item in results if item.expected_accepted and not item.predicted_accepted
    )
    controls = tuple(item for item in results if item.control and item.expected_accepted)
    latencies = tuple(item.latency_ms for item in results)
    return JevRelevanceRunMetrics(
        passed_cases=sum(item.passed for item in results),
        total_cases=len(results),
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, true_positives + len(false_negatives)),
        positive_control_preservation=_ratio(
            sum(item.predicted_accepted for item in controls), len(controls)
        ),
        false_negative_ids=false_negatives,
        input_tokens=sum(item.input_tokens for item in results),
        output_tokens=sum(item.output_tokens for item in results),
        estimated_cost_usd=sum(item.estimated_cost_usd for item in results),
        p50_latency_ms=float_linear_percentile(latencies, 0.50),
        p95_latency_ms=float_linear_percentile(latencies, 0.95),
    )


def _run_jev_relevance_trial(
    release: LoadedJevRelevanceRelease,
    execution_ref: str,
    policy: JevExecutionPolicy,
) -> JevRelevanceRunResult:
    specs = tuple(case for case in release.manifest.cases if case.concern == "relevance")
    results: list[JevRelevanceCaseResult] = []
    for case in specs:
        content = read_verified_r2_object(case.article.r2_key, case.article.content_digest)
        article = ExtractedArticle.model_validate_json(content, strict=True)
        request = build_relevance_binary_request(
            ArticleAnalysisInput(reference=case.article, article=article)
        )
        observation = evaluate_jev_relevance(
            request,
            execution_ref=execution_ref,
            policy=policy,
        )
        results.append(
            JevRelevanceCaseResult(
                case_id=case.case_id,
                article_version_id=case.article.version_id,
                execution_ref=execution_ref,
                expected_accepted=case.expected_accepted,
                predicted_accepted=observation.predicted_accepted,
                pass_probability=(
                    float(observation.probability)
                    if case.expected_accepted
                    else 1 - float(observation.probability)
                ),
                control=case.control,
                passed=observation.predicted_accepted == case.expected_accepted,
                model=observation.model,
                request_id=observation.request_id,
                provider_request_id=observation.provider_request_id,
                input_tokens=observation.input_tokens,
                output_tokens=observation.output_tokens,
                latency_ms=observation.latency_ms,
                estimated_cost_usd=float(observation.estimated_cost_usd),
            )
        )
    case_results = tuple(results)
    return JevRelevanceRunResult(
        manifest_artifact_version_id=release.manifest_reference.version_id,
        execution_ref=execution_ref,
        policy_id=RELEVANCE_BINARY_QUESTION.question_id,
        policy_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        case_results=case_results,
        metrics=jev_relevance_metrics(case_results),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
