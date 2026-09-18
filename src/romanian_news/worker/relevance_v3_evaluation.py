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
    RelevanceV3ExperimentProjection,
    RelevanceV3SliceMetrics,
    run_relevance_v3_evaluation,
)


class RelevanceV3RunSummary(NewsModel):
    implementation_ref: str
    policy_id: str
    policy_digest: Sha256
    experiment_url: str | None
    full: RelevanceV3SliceMetrics
    prior: RelevanceV3SliceMetrics
    new: RelevanceV3SliceMetrics
    false_negative_ids: tuple[str, ...]
    context_early_exit_rate: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    observability_complete: bool


class RelevanceV3EvaluationSummary(NewsModel):
    manifest_artifact_version_id: Sha256
    implementation_ref: str
    runs: tuple[RelevanceV3RunSummary, ...]
    false_negative_ids: tuple[str, ...]
    threshold_failures: tuple[str, ...]


@dg.op(pool="news_model")
def relevance_v3_evaluation_op(context: OpExecutionContext) -> RelevanceV3EvaluationSummary:
    ensure_news_catalog_schema()
    try:
        release = load_news_evaluation_release(PIN_PATH.read_bytes())
        plan = FreshEvaluationPlan(implementation_ref=IMPLEMENTATION_REF)
        evaluation = run_relevance_v3_evaluation(release, plan)
        summary = RelevanceV3EvaluationSummary(
            manifest_artifact_version_id=evaluation.manifest_artifact_version_id,
            implementation_ref=plan.implementation_ref,
            runs=tuple(_run_summary(run) for run in evaluation.runs),
            false_negative_ids=evaluation.false_negative_ids,
            threshold_failures=evaluation.threshold_failures,
        )
        context.log.info(summary.model_dump_json())
        metadata = {
            "runs": dg.MetadataValue.json([run.model_dump(mode="json") for run in summary.runs]),
            "false_negative_ids": dg.MetadataValue.json(summary.false_negative_ids),
            "threshold_failures": dg.MetadataValue.json(summary.threshold_failures),
        }
        if summary.threshold_failures:
            raise dg.Failure(
                description="Relevance V3 failed thresholds: "
                + "; ".join(summary.threshold_failures),
                metadata=metadata,
            )
        context.add_output_metadata(metadata)
        return summary
    finally:
        flush_langfuse_traces()


@dg.job
def relevance_v3_evaluation() -> None:
    relevance_v3_evaluation_op()


def _run_summary(run: RelevanceV3ExperimentProjection) -> RelevanceV3RunSummary:
    return RelevanceV3RunSummary(
        implementation_ref=run.implementation_ref,
        policy_id=run.policy_id,
        policy_digest=run.policy_digest,
        experiment_url=run.experiment_url,
        full=run.full_metrics,
        prior=run.prior_metrics,
        new=run.new_metrics,
        false_negative_ids=run.false_negative_ids,
        context_early_exit_rate=run.context_early_exit_rate,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.cost_usd,
        observability_complete=run.observability_complete,
    )
