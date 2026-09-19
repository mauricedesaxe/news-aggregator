import dagster as dg
from dagster import OpExecutionContext

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.tracing import flush_langfuse_traces
from romanian_news.catalog.evaluations import load_news_evaluation_release
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.evaluation import PIN_PATH
from romanian_news.evaluation_projection import (
    FreshEvaluationPlan,
    RelevanceComparisonTrial,
    RelevancePolicyExperimentProjection,
    compare_relevance_policies,
)


class RelevanceExperimentSummary(NewsModel):
    policy_id: str
    policy_digest: Sha256
    precision: float
    recall: float
    positive_control_preservation: float
    experiment_url: str | None


class RelevanceComparisonTrialSummary(NewsModel):
    ordinal: int
    implementation_ref: str
    baseline: RelevanceExperimentSummary
    candidate: RelevanceExperimentSummary
    promotion_failures: tuple[str, ...]
    passed: bool


class FreshRelevanceComparisonSummary(NewsModel):
    manifest_artifact_version_id: Sha256
    implementation_ref: str
    trial_count: int
    trials: tuple[RelevanceComparisonTrialSummary, ...]
    promotion_failures: tuple[str, ...]
    promoted: bool


@dg.op(pool="news_model")
def fresh_relevance_comparison_op(
    context: OpExecutionContext,
) -> FreshRelevanceComparisonSummary:
    ensure_news_catalog_schema()
    try:
        release = load_news_evaluation_release(PIN_PATH.read_bytes())
        plan = FreshEvaluationPlan(implementation_ref=IMPLEMENTATION_REF)
        comparison = compare_relevance_policies(release, plan)
        summary = FreshRelevanceComparisonSummary(
            manifest_artifact_version_id=comparison.manifest_artifact_version_id,
            implementation_ref=plan.implementation_ref,
            trial_count=plan.trial_count,
            trials=tuple(_trial_summary(trial) for trial in comparison.trials),
            promotion_failures=comparison.promotion_failures,
            promoted=comparison.promoted,
        )
        context.log.info(summary.model_dump_json())
        metadata = {
            "trials": dg.MetadataValue.json(
                [trial.model_dump(mode="json") for trial in summary.trials]
            ),
            "promotion_failures": dg.MetadataValue.json(summary.promotion_failures),
        }
        if comparison.promotion_failures:
            raise dg.Failure(
                description=(
                    "Fresh relevance comparison failed promotion gates: "
                    + "; ".join(comparison.promotion_failures)
                ),
                metadata=metadata,
            )
        context.add_output_metadata(metadata)
        return summary
    finally:
        flush_langfuse_traces()


@dg.job
def fresh_relevance_comparison() -> None:
    fresh_relevance_comparison_op()


def _trial_summary(trial: RelevanceComparisonTrial) -> RelevanceComparisonTrialSummary:
    return RelevanceComparisonTrialSummary(
        ordinal=trial.ordinal,
        implementation_ref=trial.implementation_ref,
        baseline=_experiment_summary(trial.baseline),
        candidate=_experiment_summary(trial.candidate),
        promotion_failures=trial.promotion_failures,
        passed=trial.passed,
    )


def _experiment_summary(
    experiment: RelevancePolicyExperimentProjection,
) -> RelevanceExperimentSummary:
    return RelevanceExperimentSummary(
        policy_id=experiment.policy_id,
        policy_digest=experiment.policy_digest,
        precision=experiment.metrics.precision,
        recall=experiment.metrics.recall,
        positive_control_preservation=experiment.metrics.positive_control_preservation,
        experiment_url=experiment.experiment_url,
    )
