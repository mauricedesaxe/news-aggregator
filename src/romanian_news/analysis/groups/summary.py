from __future__ import annotations

import hashlib
import json
import time

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import ValidationError

from romanian_news import GROUP_ANALYSIS_MODEL, Sha256
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.groups.models import (
    GroupAnalysisInput,
    GroupSummary,
    GroupSummaryOutput,
)
from romanian_news.analysis.tracing import ProviderChatRequest, trace_provider_call
from romanian_news.groups import NewsGroup

SUMMARY_TITLE_POLICY = "strip-leading-news-labels-v1"
SUMMARY_COMPARISON_POLICY = "distinct-outlets-only-v1"
SUMMARY_CORRECTION_POLICY = "complete-json-once-v1"
SUMMARY_SCHEMA_POLICY = "all-properties-required-v1"
SUMMARY_MAX_TOKENS = 4000

SUMMARY_PROMPT = (
    "Summarize this group of Romanian news articles. Write all generated text values in English. "
    "Fields ending in _ro are legacy field names and do not request Romanian output. State what "
    "happened, the important facts, differences between sources, and material uncertainty. "
    "Differences compare only distinct supplied outlets. Set disagreements_ro to an empty list "
    "when the supplied articles contain fewer than two distinct outlets. Use a direct 4-12 "
    "word title without BREAKING, LIVE, VIDEO, or UPDATE labels. When the supplied articles "
    "report an unconfirmed claim, use a hedged title that keeps the claim conditional, such as "
    "'PSD floats candidate for PM', rather than restating the outlet headline as fact. Use a "
    "1-2 sentence, 60-word maximum summary and no more than 3 concise key points. Do not add "
    "facts "
    "that the supplied articles do not support. Do not infer today's date or call a period "
    "incomplete unless a supplied article says so. Report uncertainty only when the supplied "
    "articles support it. Cite only the supplied ARTICLE_ID handles in the structured result."
)

SUMMARY_TEXT_POLICY = "article-handle-outlet-published-title-body-12000-v2"


def _accept_summary(
    content: str | None,
    handle_ids: tuple[str, ...],
    article_handles: dict[str, Sha256],
    value: GroupAnalysisInput,
) -> GroupSummary:
    """Validate one model response and translate cited handles to exact versions."""
    if not content:
        raise ValueError("Group summary model returned no content")
    summary_document = json.loads(content)
    cited_handles = summary_document.get("cited_article_version_ids")
    if (
        not isinstance(cited_handles, list)
        or not cited_handles
        or not set(cited_handles) <= set(handle_ids)
    ):
        raise ValueError("Group summary cites an article outside its exact inputs")
    summary_document["cited_article_version_ids"] = [
        article_handles[handle] for handle in cited_handles
    ]
    summary = GroupSummary.model_validate_json(json.dumps(summary_document))
    if len({article.outlet_id for _reference, article in value.articles}) < 2:
        return summary.model_copy(update={"disagreements_ro": ()})
    return summary


def summarize_group(value: GroupAnalysisInput, article_context: str) -> GroupSummaryOutput:
    request_id = summary_request_id(value.group)
    started = time.monotonic()
    valid_ids = {reference.version_id for reference, _ in value.articles}
    article_handles = {
        f"a{index}": reference.version_id
        for index, (reference, _article) in enumerate(value.articles, start=1)
    }
    handle_ids = tuple(article_handles)
    response_schema = GroupSummary.model_json_schema()
    response_schema["required"] = list(response_schema["properties"])
    response_schema["properties"]["cited_article_version_ids"]["items"] = {
        "type": "string",
        "enum": list(handle_ids),
    }
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": article_context},
    ]
    responses: list[ChatCompletion] = []
    for attempt in range(2):
        provider_inputs: ProviderChatRequest = {
            "model": GROUP_ANALYSIS_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": SUMMARY_MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "romanian_news_group_summary",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "extra_body": {"provider": {"require_parameters": True}},
        }
        attempt_started = time.monotonic()
        provider_call = trace_provider_call(
            "news.summarize_group",
            request_id,
            provider_inputs,
            lambda provider_inputs=provider_inputs: (
                openrouter_client().chat.completions.create(**provider_inputs)
            ),
        )
        response = provider_call.response
        latency_ms = round((time.monotonic() - attempt_started) * 1000)
        responses.append(response)
        content = response.choices[0].message.content
        try:
            summary = _accept_summary(content, handle_ids, article_handles, value)
        except (ValidationError, ValueError) as error:
            record_model_attempt(
                response,
                request_id=request_id,
                operation_key="news.summarize_group",
                attempt_index=attempt,
                latency_ms=latency_ms,
                status="rejected",
                error=str(error),
                fallback_response_id=str(provider_call.call_id),
                trace=provider_call.trace,
            )
            if attempt == 1:
                raise ValueError(
                    f"Group summary remained invalid after correction: {error}"
                ) from error
            messages.extend(
                (
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": (
                            "The response was invalid. Return complete JSON and cite only these exact "
                            f"ARTICLE_ID handles: {sorted(handle_ids)}. Validation error: {error}"
                        ),
                    },
                )
            )
            continue
        record_model_attempt(
            response,
            request_id=request_id,
            operation_key="news.summarize_group",
            attempt_index=attempt,
            latency_ms=latency_ms,
            status="accepted",
            error=None,
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        break
    else:
        raise RuntimeError("Summary correction loop did not return")
    call = ModelCall(
        response_id=response.id,
        model=response.model,
        input_tokens=sum(item.usage.prompt_tokens if item.usage else 0 for item in responses),
        output_tokens=sum(item.usage.completion_tokens if item.usage else 0 for item in responses),
        latency_ms=round((time.monotonic() - started) * 1000),
    )
    payload = _canonical_json(
        {
            "article_version_ids": sorted(valid_ids),
            "call": call.model_dump(mode="json"),
            "cluster_set_version_id": value.cluster_set.version_id,
            "comparison_policy": SUMMARY_COMPARISON_POLICY,
            "group_id": value.group.id,
            "prompt_digest": _sha256(SUMMARY_PROMPT.encode()),
            "title_policy": SUMMARY_TITLE_POLICY,
            "provider_responses": [item.model_dump(mode="json") for item in responses],
            "request_id": request_id,
            "summary": summary.model_dump(mode="json"),
        }
    )
    return GroupSummaryOutput(
        request_id=request_id,
        cluster_set=value.cluster_set,
        group_id=value.group.id,
        articles=tuple(reference for reference, _ in value.articles),
        summary=summary,
        call=call,
        content=payload,
    )


def summary_request_id(group: NewsGroup) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "group_id": group.id,
                "comparison_policy": SUMMARY_COMPARISON_POLICY,
                "correction_policy": SUMMARY_CORRECTION_POLICY,
                "max_tokens": SUMMARY_MAX_TOKENS,
                "model": GROUP_ANALYSIS_MODEL,
                "prompt_digest": _sha256(SUMMARY_PROMPT.encode()),
                "purpose": "news.summary_group",
                "semantic_correction_attempts": 1,
                "schema": GroupSummary.model_json_schema(),
                "schema_policy": SUMMARY_SCHEMA_POLICY,
                "temperature": 0,
                "text_policy": SUMMARY_TEXT_POLICY,
                "title_policy": SUMMARY_TITLE_POLICY,
            }
        )
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
