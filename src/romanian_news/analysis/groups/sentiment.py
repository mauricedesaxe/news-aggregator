from __future__ import annotations

import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import ValidationError

from romanian_news import GROUP_ANALYSIS_MODEL, Sha256
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.groups.models import (
    ArticleSentiment,
    ArticleSentimentResponse,
    GroupAnalysisInput,
    GroupSentiment,
    GroupSentimentOutput,
    SentimentAssessment,
)
from romanian_news.analysis.tracing import (
    ProviderCallResult,
    ProviderChatRequest,
    trace_provider_call,
)
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256

ARTICLE_SENTIMENT_PROMPT = (
    "Assess the sentiment and practical direction of this Romanian news article. Sentiment "
    "describes the likely political or economic effect, not the emotional writing style. Every "
    "sentence carries an ID like [S4]. Set evidence_quote to the sentence ID of the one sentence "
    "that best supports your assessment. Never invent evidence. Write generated text values in "
    "English. Fields ending in _ro are legacy field names and do not request Romanian output."
)
GROUP_OVERALL_PROMPT = (
    "Assess the overall sentiment and practical direction of this group of related Romanian news "
    "articles. Sentiment describes the likely political or economic effect, not the emotional "
    "writing style. Set evidence_quote to the literal string group. Write generated text values "
    "in English. Fields ending in _ro are legacy field names and do not request Romanian output."
)
SENTIMENT_MAX_TOKENS = 4000
SENTIMENT_CORRECTION_POLICY = "per-article-calls-v11"
SENTIMENT_TEXT_POLICY = "per-article-sentence-index-title-body-12000-v3"
_ROMANIAN_COMMA_VARIANTS = str.maketrans({"ş": "ș", "ţ": "ț"})
_EVIDENCE_MARKER = re.compile(r"^[Ss](\d+)$")


def score_group_sentiment(
    value: GroupAnalysisInput,
) -> GroupSentimentOutput:
    request_id = sentiment_request_id(value.group)
    started = time.monotonic()
    attempts = _AttemptLog(request_id)
    with ThreadPoolExecutor(max_workers=min(len(value.articles), 8)) as executor:
        articles = list(
            executor.map(
                lambda item: _score_article_sentiment(item[0], item[1], attempts),
                value.articles,
            )
        )
    overall = _score_group_overall(value, attempts)
    sentiment = GroupSentiment(overall=overall, articles=tuple(articles))
    call = ModelCall(
        response_id=attempts.responses[-1].id,
        model=attempts.responses[-1].model,
        input_tokens=attempts.input_tokens,
        output_tokens=attempts.output_tokens,
        latency_ms=round((time.monotonic() - started) * 1000),
    )
    payload = _canonical_json(
        {
            "article_version_ids": sorted(reference.version_id for reference, _ in value.articles),
            "call": call.model_dump(mode="json"),
            "cluster_set_version_id": value.cluster_set.version_id,
            "group_id": value.group.id,
            "prompt_digest": _sha256(ARTICLE_SENTIMENT_PROMPT.encode()),
            "provider_responses": [item.model_dump(mode="json") for item in attempts.responses],
            "request_id": request_id,
            "sentiment": sentiment.model_dump(mode="json"),
        }
    )
    return GroupSentimentOutput(
        request_id=request_id,
        cluster_set=value.cluster_set,
        group_id=value.group.id,
        articles=tuple(reference for reference, _ in value.articles),
        sentiment=sentiment,
        call=call,
        content=payload,
    )


def _score_article_sentiment(
    reference: ArtifactReference,
    article: ExtractedArticle,
    attempts: _AttemptLog,
) -> ArticleSentiment:
    sentences = _article_sentences(article)
    response_schema = ArticleSentimentResponse.model_json_schema()
    response_schema["properties"]["evidence_quote"]["enum"] = list(sentences)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": ARTICLE_SENTIMENT_PROMPT},
        {"role": "user", "content": _numbered_article_text(article)},
    ]
    for attempt_index in range(2):
        response, provider_call, latency_ms = _complete(messages, response_schema, attempts)
        content = response.choices[0].message.content
        try:
            if not content:
                raise ValueError("Article sentiment model returned no content")
            assessment = _parse_article_assessment(content, sentences)
            _validate_article_evidence(assessment.evidence_quote, article)
        except (ValidationError, ValueError) as error:
            attempts.record(response, provider_call, latency_ms, "rejected", str(error))
            if attempt_index == 1:
                raise ValueError(
                    f"Article sentiment remained invalid after correction: {error}"
                ) from error
            messages = [
                *messages,
                {"role": "assistant", "content": content or ""},
                {
                    "role": "user",
                    "content": (
                        "The response was invalid. Return complete JSON and set evidence_quote "
                        f"to one of these sentence IDs: {', '.join(sentences)}. "
                        f"Validation error: {error}"
                    ),
                },
            ]
            continue
        attempts.record(response, provider_call, latency_ms, "accepted", None)
        return ArticleSentiment(
            article_version_id=reference.version_id,
            **assessment.model_dump(),
        )
    raise RuntimeError("Article sentiment correction loop did not return")


def _score_group_overall(value: GroupAnalysisInput, attempts: _AttemptLog) -> SentimentAssessment:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": GROUP_OVERALL_PROMPT},
        {"role": "user", "content": _group_overview(value)},
    ]
    for attempt_index in range(2):
        response, provider_call, latency_ms = _complete(
            messages, ArticleSentimentResponse.model_json_schema(), attempts
        )
        content = response.choices[0].message.content
        try:
            if not content:
                raise ValueError("Group overall sentiment model returned no content")
            assessment = ArticleSentimentResponse.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            attempts.record(response, provider_call, latency_ms, "rejected", str(error))
            if attempt_index == 1:
                raise ValueError(
                    f"Group overall sentiment remained invalid after correction: {error}"
                ) from error
            messages = [
                *messages,
                {"role": "assistant", "content": content or ""},
                {
                    "role": "user",
                    "content": (
                        "The response was invalid. Return complete JSON and set evidence_quote "
                        f"to the literal string group. Validation error: {error}"
                    ),
                },
            ]
            continue
        attempts.record(response, provider_call, latency_ms, "accepted", None)
        return SentimentAssessment(**assessment.model_dump(exclude={"evidence_quote"}))
    raise RuntimeError("Group overall sentiment correction loop did not return")


def _complete(
    messages: list[ChatCompletionMessageParam],
    schema: dict[str, object],
    attempts: _AttemptLog,
) -> tuple[ChatCompletion, ProviderCallResult[ChatCompletion], int]:
    provider_inputs: ProviderChatRequest = {
        "model": GROUP_ANALYSIS_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": SENTIMENT_MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "romanian_news_sentiment_assessment",
                "strict": True,
                "schema": schema,
            },
        },
        "extra_body": {"provider": {"require_parameters": True}},
    }
    started = time.monotonic()
    provider_call = trace_provider_call(
        "news.score_group_sentiment",
        attempts.request_id,
        provider_inputs,
        lambda provider_inputs=provider_inputs: (
            openrouter_client().chat.completions.create(**provider_inputs)
        ),
    )
    return provider_call.response, provider_call, round((time.monotonic() - started) * 1000)


class _AttemptLog:
    def __init__(self, request_id: Sha256) -> None:
        self.request_id = request_id
        self.responses: list[ChatCompletion] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._index = 0

    def record(
        self,
        response: ChatCompletion,
        provider_call: ProviderCallResult[ChatCompletion],
        latency_ms: int,
        status: Literal["accepted", "rejected"],
        error: str | None,
    ) -> None:
        record_model_attempt(
            response,
            request_id=self.request_id,
            operation_key="news.score_group_sentiment",
            attempt_index=self._index,
            latency_ms=latency_ms,
            status=status,
            error=error,
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        self._index += 1
        self.responses.append(response)
        usage = response.usage
        self.input_tokens += usage.prompt_tokens if usage else 0
        self.output_tokens += usage.completion_tokens if usage else 0


def _parse_article_assessment(content: str, sentences: dict[str, str]) -> ArticleSentimentResponse:
    payload = json.loads(content)
    if isinstance(payload, dict):
        marker = payload.get("evidence_quote")
        if isinstance(marker, str) and (match := _EVIDENCE_MARKER.match(marker.strip())):
            sentence = sentences.get(f"S{match.group(1)}")
            if sentence is not None:
                payload["evidence_quote"] = sentence
    return ArticleSentimentResponse.model_validate(payload)


def _validate_article_evidence(evidence: str, article: ExtractedArticle) -> None:
    haystack = f"{article.title}\n{article.body}"
    if not evidence or _normalize_quote(evidence) not in _normalize_quote(haystack):
        raise ValueError(f"Article sentiment evidence is not present in the article: {evidence}")


def _article_sentences(article: ExtractedArticle) -> dict[str, str]:
    text = f"{article.title}\n{article.body[:12000]}"
    result: dict[str, str] = {}
    position = 0
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        position += 1
        result[f"S{position}"] = sentence
    return result


def _numbered_article_text(article: ExtractedArticle) -> str:
    sentences = _article_sentences(article)
    return (
        f"OUTLET_ID: {article.outlet_id}\n"
        f"PUBLISHED_AT: {article.published_at.isoformat()}\n"
        f"TITLE: {article.title}\n"
        "NUMBERED TEXT:\n" + "\n".join(f"[{marker}] {text}" for marker, text in sentences.items())
    )


def _group_overview(value: GroupAnalysisInput) -> str:
    return "\n".join(
        f"OUTLET_ID: {article.outlet_id}\nTITLE: {article.title}"
        for _reference, article in value.articles
    )


def sentiment_request_id(group: NewsGroup) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "group_id": group.id,
                "max_tokens": SENTIMENT_MAX_TOKENS,
                "model": GROUP_ANALYSIS_MODEL,
                "prompt_digest": _sha256(ARTICLE_SENTIMENT_PROMPT.encode()),
                "purpose": "news.sentiment_group",
                "semantic_correction_attempts": 1,
                "schema": ArticleSentimentResponse.model_json_schema(),
                "temperature": 0,
                "text_policy": SENTIMENT_TEXT_POLICY,
            }
        )
    )


def validate_sentiment_evidence(
    sentiment: GroupSentiment,
    value: GroupAnalysisInput,
) -> None:
    articles = {reference.version_id: article for reference, article in value.articles}
    if len(sentiment.articles) != len(articles) or {
        item.article_version_id for item in sentiment.articles
    } != set(articles):
        raise ValueError("Group sentiment must include every exact article input once")
    for item in sentiment.articles:
        article = articles[item.article_version_id]
        evidence = _normalize_quote(item.evidence_quote)
        if not evidence or evidence not in _normalize_quote(f"{article.title}\n{article.body}"):
            raise ValueError(
                f"Article sentiment evidence is not present in {item.article_version_id}: "
                f"{item.evidence_quote}"
            )


def _normalize_quote(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().translate(_ROMANIAN_COMMA_VARIANTS)
    return " ".join(re.findall(r"\w+", normalized))
