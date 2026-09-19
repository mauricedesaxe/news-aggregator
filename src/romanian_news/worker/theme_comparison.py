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
    FreshThemeMetrics,
    ThemeComparisonTrial,
    compare_theme_policies,
)


class ThemeExperimentSummary(NewsModel):
    policy_id: str
    policy_digest: Sha256
    metrics: FreshThemeMetrics
    experiment_url: str | None


class ThemeComparisonTrialSummary(NewsModel):
    ordinal: int
    implementation_ref: str
    baseline: ThemeExperimentSummary
    candidate: ThemeExperimentSummary
    promotion_failures: tuple[str, ...]
    passed: bool


class FreshThemeComparisonSummary(NewsModel):
    manifest_artifact_version_id: Sha256
    implementation_ref: str
    trial_count: int
    trials: tuple[ThemeComparisonTrialSummary, ...]
    promotion_failures: tuple[str, ...]
    promoted: bool


@dg.op(pool="news_model")
def fresh_theme_comparison_op(context: OpExecutionContext) -> FreshThemeComparisonSummary:
    ensure_news_catalog_schema()
    try:
        release = load_news_evaluation_release(PIN_PATH.read_bytes())
        implementation_ref = IMPLEMENTATION_REF
        plan = FreshEvaluationPlan(implementation_ref=implementation_ref)
        comparison = compare_theme_policies(release, plan)
        summary = FreshThemeComparisonSummary(
            manifest_artifact_version_id=comparison.manifest_artifact_version_id,
            implementation_ref=implementation_ref,
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
                    "Fresh theme comparison failed promotion gates: "
                    + "; ".join(comparison.promotion_failures)
                ),
                metadata=metadata,
            )
        context.add_output_metadata(metadata)
        return summary
    finally:
        flush_langfuse_traces()


@dg.job
def fresh_theme_comparison() -> None:
    fresh_theme_comparison_op()


def _trial_summary(trial: ThemeComparisonTrial) -> ThemeComparisonTrialSummary:
    return ThemeComparisonTrialSummary(
        ordinal=trial.ordinal,
        implementation_ref=trial.implementation_ref,
        baseline=ThemeExperimentSummary(
            policy_id=trial.baseline.policy_id,
            policy_digest=trial.baseline.policy_digest,
            metrics=trial.baseline.metrics,
            experiment_url=trial.baseline.experiment_url,
        ),
        candidate=ThemeExperimentSummary(
            policy_id=trial.candidate.policy_id,
            policy_digest=trial.candidate.policy_digest,
            metrics=trial.candidate.metrics,
            experiment_url=trial.candidate.experiment_url,
        ),
        promotion_failures=trial.promotion_failures,
        passed=trial.passed,
    )
