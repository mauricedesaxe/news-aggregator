from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Annotated, Literal

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import Field

from romanian_news import GENERATION_MODEL, NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.tracing import ProviderChatRequest, trace_provider_call
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import existing_current_artifact_ids
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import read_verified_r2_object

RELEVANCE_PROMPT = (
    "Decide whether this article has a material nationwide consequence for Romania's political "
    "situation or economy. Set national_reach to none for foreign news without a Romanian "
    "consequence, sector_limited for local, individual, or sector-limited consequences, and "
    "nationwide only for consequences across Romania. Set consequence_magnitude to narrow for "
    "administrative fees, permits, certificates, filings, and procedural changes unless they "
    "materially affect households, businesses, public finances, or institutions. Set it to routine "
    "for a material policy or economic consequence and major for a broad fiscal, political, or "
    "institutional consequence. Acceptance requires nationwide reach, routine or major consequence, "
    "strong Romania relevance, and strong political or economic relevance. Reject sports, "
    "entertainment, lifestyle, and crime without national policy significance. Return one exact "
    "quote from the supplied article that best supports the decision. Write the short reason in "
    "Romanian."
)
RELEVANCE_EVIDENCE_POLICY = "verbatim-then-article-title-v1"
RELEVANCE_CORRECTION_INSTRUCTION = (
    "The response was invalid. Return complete JSON and copy evidence_quote verbatim from the "
    "article title or body. Validation error: {error}"
)
RELEVANCE_PROMPT_V2 = (
    "Decide whether this article states a concrete material effect on Romanian national policy, "
    "government power, public finances, major economic conditions, or national institutions. "
    "Romanian publication or general Romanian interest is insufficient. Choose the lowest supported "
    "national_reach and consequence_magnitude when the source is ambiguous. Nationwide describes "
    "the breadth of the consequence, not the geography of a report or the national status of an "
    "institution. A condition, report, recommendation, or plan confined to one public service, one "
    "institution family, one industry, one locality, or one demographic group is sector_limited even "
    "when it applies across Romania. Reports about police custody or prisons remain sector_limited "
    "when they concern detention conditions, capacity, staffing, abuse, or operational "
    "recommendations, even when they quantify a nationwide problem or ask central authorities to "
    "act. Set nationwide only when the article's main subject states a "
    "current event, enacted decision, or proposed national decision that materially changes central "
    "government power, broad public finances, major economic conditions, or several national "
    "systems. An expert forecast, recommendation, or general trend does not qualify by itself. It "
    "qualifies only when the article explains a concrete material Romanian consequence under these "
    "rules. Advice about what governments generally will need to do is not evidence of a Romanian "
    "government decision. A brief Romanian statistic does not promote an otherwise global article "
    "to nationwide relevance. If the headline and most of the article concern a global trend, keep "
    "it below nationwide even when one Romanian passage names service adaptation, pension pressure, "
    "or other possible future effects. Nationwide requires the article to focus principally on "
    "Romania and establish the Romanian consequence from Romanian evidence. For example, reject an "
    "article primarily about global population aging that includes Romanian statistics and general "
    "pension or service recommendations. Accept a Romania-focused demographic article whose central "
    "argument calls for coordinated planning across education, health, and pensions. Forming or "
    "dismissing Romania's government, and an active "
    "parliamentary-majority dispute that determines who can govern, are nationwide political "
    "consequences. They do not require an enacted policy or completed coalition agreement. Set "
    "national_reach to none for foreign or global news without that Romanian consequence. "
    "Do not infer a Romanian consequence from global treaties, references to world governments, or "
    "risks shared by all countries. Set consequence_magnitude to narrow for administrative or "
    "procedural changes without a material national effect, routine for a material policy or economic "
    "effect, and major for a broad fiscal, political, or institutional effect. Strong political OR "
    "strong economic relevance suffices. Preserve important single-source reporting when its stated "
    "effect meets these rules. Return one exact quote from the supplied article that best supports "
    "the decision. Write the short reason in Romanian."
)

NationalReach = Literal["none", "sector_limited", "nationwide"]
ConsequenceMagnitude = Literal["narrow", "routine", "major"]
RelevanceStrength = Literal["none", "weak", "strong"]


class RelevancePolicy(NewsModel):
    policy_id: Annotated[str, Field(min_length=1)]
    prompt: Annotated[str, Field(min_length=1)]
    model: str
    temperature: float
    max_tokens: Annotated[int, Field(gt=0)]
    response_schema_name: Annotated[str, Field(min_length=1)]
    semantic_correction_attempts: Annotated[int, Field(ge=0)]
    correction_instruction: Annotated[str, Field(min_length=1)]
    correction_fallback: Literal["article_title"]
    evidence_policy: Annotated[str, Field(min_length=1)]
    text_policy: Annotated[str, Field(min_length=1)]
    max_body_characters: Annotated[int, Field(gt=0)]
    provider_require_parameters: bool
    accepted_national_reach: tuple[NationalReach, ...]
    accepted_consequence_magnitudes: tuple[ConsequenceMagnitude, ...]
    accepted_romania_relevance: tuple[RelevanceStrength, ...]
    accepted_topic_relevance: tuple[RelevanceStrength, ...]


RELEVANCE_POLICY_V1 = RelevancePolicy(
    policy_id="relevance-v1",
    prompt=RELEVANCE_PROMPT,
    model=GENERATION_MODEL,
    temperature=0,
    max_tokens=800,
    response_schema_name="romanian_news_relevance",
    semantic_correction_attempts=1,
    correction_instruction=RELEVANCE_CORRECTION_INSTRUCTION,
    correction_fallback="article_title",
    evidence_policy=RELEVANCE_EVIDENCE_POLICY,
    text_policy="title-body-24000-v1",
    max_body_characters=24000,
    provider_require_parameters=True,
    accepted_national_reach=("nationwide",),
    accepted_consequence_magnitudes=("routine", "major"),
    accepted_romania_relevance=("strong",),
    accepted_topic_relevance=("strong",),
)
RELEVANCE_POLICY_V2 = RELEVANCE_POLICY_V1.model_copy(
    update={
        "policy_id": "relevance-v2",
        "prompt": RELEVANCE_PROMPT_V2,
        "model": "google/gemini-2.5-flash",
    }
)
PRODUCTION_RELEVANCE_POLICY = RELEVANCE_POLICY_V2


class ArticleAnalysisInput(NewsModel):
    reference: ArtifactReference
    article: ExtractedArticle


class ArticleAnalysisReference(NewsModel):
    reference: ArtifactReference
    bucharest_day: date


class RelevanceDecision(NewsModel):
    national_reach: NationalReach
    consequence_magnitude: ConsequenceMagnitude
    political_relevance: RelevanceStrength
    economic_relevance: RelevanceStrength
    romania_relevance: RelevanceStrength
    confidence: Annotated[float, Field(ge=0, le=1)]
    evidence_quote: Annotated[str, Field(min_length=1)]
    reason_ro: Annotated[str, Field(min_length=1)]


class RelevanceOutput(NewsModel):
    request_id: Sha256
    policy: RelevancePolicy
    article: ArtifactReference
    decision: RelevanceDecision
    call: ModelCall
    content: bytes

    @property
    def policy_digest(self) -> Sha256:
        return relevance_policy_digest(self.policy)

    @property
    def accepted(self) -> bool:
        return relevance_is_accepted(self.decision, self.policy)


def read_pending_relevance(
    limit: int | None = None,
    through_day: date | None = None,
    *,
    day: date | None = None,
    request_id_for_article: Callable[[ArtifactReference], Sha256] | None = None,
) -> tuple[ArticleAnalysisInput, ...]:
    """Read current article versions that lack this relevance configuration."""
    pending = read_pending_relevance_references(
        through_day,
        day=day,
        request_id_for_article=request_id_for_article,
    )
    if limit is not None:
        pending = pending[:limit]
    references = tuple(value.reference for value in pending)
    if not references:
        return ()
    with ThreadPoolExecutor(max_workers=min(16, len(references))) as executor:
        contents = tuple(
            executor.map(
                lambda reference: read_verified_r2_object(
                    reference.r2_key,
                    reference.content_digest,
                ),
                references,
            )
        )
    return tuple(
        ArticleAnalysisInput(
            reference=reference,
            article=ExtractedArticle.model_validate_json(content, strict=True),
        )
        for reference, content in zip(references, contents, strict=True)
    )


def read_pending_relevance_references(
    through_day: date | None = None,
    *,
    day: date | None = None,
    request_id_for_article: Callable[[ArtifactReference], Sha256] | None = None,
) -> tuple[ArticleAnalysisReference, ...]:
    """Read pending relevance identities without downloading article bodies."""
    values = _current_article_references(through_day, day=day)
    identify = request_id_for_article or relevance_request_id
    artifact_ids = {
        value.reference.version_id: f"news:relevance:{identify(value.reference)}"
        for value in values
    }
    existing = existing_current_artifact_ids(tuple(artifact_ids.values()))
    return tuple(
        value for value in values if artifact_ids[value.reference.version_id] not in existing
    )


def load_article_analysis_input(value: ArticleAnalysisReference) -> ArticleAnalysisInput:
    """Load one exact article input from its verified immutable reference."""
    content = read_verified_r2_object(value.reference.r2_key, value.reference.content_digest)
    return ArticleAnalysisInput(
        reference=value.reference,
        article=ExtractedArticle.model_validate_json(content, strict=True),
    )


def analyze_relevance(
    value: ArticleAnalysisInput,
    policy: RelevancePolicy = PRODUCTION_RELEVANCE_POLICY,
) -> RelevanceOutput:
    """Classify one immutable article version through a strict provider boundary."""
    request_id = relevance_request_id(value.reference, policy)
    policy_digest = relevance_policy_digest(policy)
    started = time.monotonic()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": policy.prompt},
        {
            "role": "user",
            "content": (
                f"Titlu: {value.article.title}\n\n"
                f"Articol:\n{value.article.body[: policy.max_body_characters]}"
            ),
        },
    ]
    responses: list[ChatCompletion] = []
    for attempt in range(policy.semantic_correction_attempts + 1):
        provider_inputs: ProviderChatRequest = {
            "model": policy.model,
            "messages": messages,
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": policy.response_schema_name,
                    "strict": True,
                    "schema": RelevanceDecision.model_json_schema(),
                },
            },
            "extra_body": {"provider": {"require_parameters": policy.provider_require_parameters}},
        }
        attempt_started = time.monotonic()
        provider_call = trace_provider_call(
            "news.relevance",
            request_id,
            provider_inputs,
            lambda provider_inputs=provider_inputs: (
                openrouter_client().chat.completions.create(**provider_inputs)
            ),
        )
        response = provider_call.response
        responses.append(response)
        latency_ms = round((time.monotonic() - attempt_started) * 1000)
        content = response.choices[0].message.content
        decision = None
        try:
            if not content:
                raise ValueError("Relevance model returned no content")
            decision = RelevanceDecision.model_validate_json(content)
            source_text = f"{value.article.title}\n{value.article.body}"
            evidence = normalize_evidence_quote(decision.evidence_quote)
            if not evidence or evidence not in normalize_evidence_quote(source_text):
                raise ValueError("Relevance evidence quote is not present in the article")
        except ValueError as error:
            if attempt == policy.semantic_correction_attempts and decision is not None:
                decision = decision.model_copy(update={"evidence_quote": value.article.title})
                record_model_attempt(
                    response,
                    request_id=request_id,
                    operation_key="news.relevance",
                    attempt_index=attempt,
                    latency_ms=latency_ms,
                    status="accepted",
                    error=None,
                    fallback_response_id=str(provider_call.call_id),
                    trace=provider_call.trace,
                )
                break
            record_model_attempt(
                response,
                request_id=request_id,
                operation_key="news.relevance",
                attempt_index=attempt,
                latency_ms=latency_ms,
                status="rejected",
                error=str(error),
                fallback_response_id=str(provider_call.call_id),
                trace=provider_call.trace,
            )
            if attempt == policy.semantic_correction_attempts:
                raise ValueError(f"Relevance remained invalid after correction: {error}") from error
            messages.extend(
                (
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": policy.correction_instruction.format(error=error),
                    },
                )
            )
            continue
        record_model_attempt(
            response,
            request_id=request_id,
            operation_key="news.relevance",
            attempt_index=attempt,
            latency_ms=latency_ms,
            status="accepted",
            error=None,
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        break
    else:
        raise RuntimeError("Relevance correction loop did not return")
    call = _model_call(responses, round((time.monotonic() - started) * 1000))
    accepted = relevance_is_accepted(decision, policy)
    payload = canonical_json(
        {
            "accepted": accepted,
            "article_version_id": value.reference.version_id,
            "call": call.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "policy": relevance_policy_payload(policy),
            "policy_digest": policy_digest,
            "provider_responses": [item.model_dump(mode="json") for item in responses],
            "request_id": request_id,
        }
    )
    return RelevanceOutput(
        request_id=request_id,
        policy=policy,
        article=value.reference,
        decision=decision,
        call=call,
        content=payload,
    )


def relevance_is_accepted(decision: RelevanceDecision, policy: RelevancePolicy) -> bool:
    """Apply one explicit acceptance policy to a typed relevance decision."""
    return (
        decision.national_reach in policy.accepted_national_reach
        and decision.consequence_magnitude in policy.accepted_consequence_magnitudes
        and decision.romania_relevance in policy.accepted_romania_relevance
        and (
            decision.political_relevance in policy.accepted_topic_relevance
            or decision.economic_relevance in policy.accepted_topic_relevance
        )
    )


def relevance_policy_payload(policy: RelevancePolicy) -> dict[str, object]:
    """Return every behavior-changing policy input in canonical identity form."""
    return {
        **policy.model_dump(mode="json"),
        "response_schema": RelevanceDecision.model_json_schema(),
    }


def relevance_policy_digest(policy: RelevancePolicy) -> Sha256:
    return sha256(canonical_json(relevance_policy_payload(policy)))


def relevance_request_id(
    article: ArtifactReference,
    policy: RelevancePolicy = PRODUCTION_RELEVANCE_POLICY,
) -> Sha256:
    return sha256(
        canonical_json(
            {
                "article_version_id": article.version_id,
                "policy": relevance_policy_payload(policy),
                "policy_digest": relevance_policy_digest(policy),
                "purpose": "news.relevance",
            }
        )
    )


def normalize_evidence_quote(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized))


def _current_article_references(
    through_day: date | None = None,
    *,
    day: date | None = None,
) -> tuple[ArticleAnalysisReference, ...]:
    if through_day is not None and day is not None:
        raise ValueError("Choose either an exact relevance day or a latest relevance day")
    from romanian_news.catalog.analysis_inputs import read_current_article_analysis_references

    return tuple(
        ArticleAnalysisReference(
            reference=value.reference,
            bucharest_day=value.bucharest_day,
        )
        for value in read_current_article_analysis_references(through_day, day=day)
    )


def _model_call(responses: list[ChatCompletion], latency_ms: int) -> ModelCall:
    response = responses[-1]
    return ModelCall(
        response_id=response.id,
        model=response.model,
        input_tokens=sum(item.usage.prompt_tokens if item.usage else 0 for item in responses),
        output_tokens=sum(item.usage.completion_tokens if item.usage else 0 for item in responses),
        latency_ms=latency_ms,
    )
