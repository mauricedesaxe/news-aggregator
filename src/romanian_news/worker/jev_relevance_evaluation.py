import dagster as dg
from dagster import OpExecutionContext

from romanian_news.catalog.evaluations import load_news_evaluation_release
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.evaluation import PIN_PATH
from romanian_news.evaluation_projection import FreshEvaluationPlan
from romanian_news.jev_relevance_evaluation import (
    JevRelevanceEvaluationResult,
    run_jev_relevance_evaluation,
)


@dg.op(pool="news_model")
def jev_relevance_evaluation_op(context: OpExecutionContext) -> JevRelevanceEvaluationResult:
    ensure_news_catalog_schema()
    release = load_news_evaluation_release(PIN_PATH.read_bytes())
    result = run_jev_relevance_evaluation(
        release,
        FreshEvaluationPlan(implementation_ref=IMPLEMENTATION_REF),
    )
    context.log.info(result.model_dump_json())
    context.add_output_metadata(
        {"runs": dg.MetadataValue.json([run.model_dump(mode="json") for run in result.runs])}
    )
    return result


@dg.job
def jev_relevance_evaluation() -> None:
    jev_relevance_evaluation_op()
