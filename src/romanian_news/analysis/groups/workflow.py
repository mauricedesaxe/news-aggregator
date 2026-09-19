from concurrent.futures import ThreadPoolExecutor

from openai import OpenAIError

from romanian_news.analysis.groups.models import GroupAnalysisInput, GroupAnalysisOutput
from romanian_news.analysis.groups.sentiment import score_group_sentiment
from romanian_news.analysis.groups.summary import summarize_group


def analyze_group(value: GroupAnalysisInput) -> GroupAnalysisOutput:
    """Run a group's missing summary and sentiment branches concurrently."""
    article_context = _article_context(value)
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = (
            executor.submit(summarize_group, value, article_context)
            if value.summary_needed
            else None
        )
        sentiment_future = (
            executor.submit(score_group_sentiment, value) if value.sentiment_needed else None
        )
        errors = []
        try:
            summary = summary_future.result() if summary_future else None
        except (OpenAIError, RuntimeError, ValueError) as error:
            summary = None
            errors.append(f"summary: {error}")
        try:
            sentiment = sentiment_future.result() if sentiment_future else None
        except (OpenAIError, RuntimeError, ValueError) as error:
            sentiment = None
            errors.append(f"sentiment: {error}")
        return GroupAnalysisOutput(summary=summary, sentiment=sentiment, errors=tuple(errors))


def _article_context(value: GroupAnalysisInput) -> str:
    return "\n\n".join(
        f"ARTICLE_ID: a{index}\n"
        f"OUTLET_ID: {article.outlet_id}\n"
        f"PUBLISHED_AT: {article.published_at.isoformat()}\n"
        f"TITLE: {article.title}\nTEXT:\n{article.body[:12000]}"
        for index, (_reference, article) in enumerate(value.articles, start=1)
    )
