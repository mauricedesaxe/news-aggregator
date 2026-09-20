from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.config import IMPLEMENTATION_REF
from romanian_news.evaluation import PIN_PATH
from romanian_news.evaluation_projection import FreshEvaluationPlan
from romanian_news.jev_relevance_evaluation import (
    load_jev_relevance_release,
    run_jev_relevance_evaluation,
)


def main() -> None:
    ensure_news_catalog_schema()
    release = load_jev_relevance_release(PIN_PATH.read_bytes())
    result = run_jev_relevance_evaluation(
        release,
        FreshEvaluationPlan(implementation_ref=IMPLEMENTATION_REF),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
