from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from romanian_news import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.relevance import (
    ArticleAnalysisInput,
    ArticleAnalysisReference,
)
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.analysis.tracing import ProviderEmbeddingRequest, trace_provider_call
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import existing_current_artifact_ids
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256
from romanian_news.storage import read_verified_r2_object

EMBEDDING_TEXT_POLICY = "title-body-1000-v1"


class EmbeddingInput(NewsModel):
    article: ArticleAnalysisInput
    relevance: ArtifactReference


class EmbeddingReference(NewsModel):
    article: ArticleAnalysisReference
    relevance: ArtifactReference


class EmbeddingOutput(NewsModel):
    request_id: Sha256
    article: ArtifactReference
    relevance: ArtifactReference
    vector: tuple[float, ...]
    call: ModelCall
    content: bytes


def read_pending_embeddings(
    limit: int | None = None,
    through_day: date | None = None,
    *,
    day: date | None = None,
) -> tuple[EmbeddingInput, ...]:
    """Read accepted relevance outputs that lack this embedding configuration."""
    pending = read_pending_embedding_references(through_day, day=day)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return ()
    with ThreadPoolExecutor(max_workers=min(16, len(pending))) as executor:
        contents = tuple(
            executor.map(
                lambda value: read_verified_r2_object(
                    value.article.reference.r2_key,
                    value.article.reference.content_digest,
                ),
                pending,
            )
        )
    return tuple(
        EmbeddingInput(
            article=ArticleAnalysisInput(
                reference=value.article.reference,
                article=ExtractedArticle.model_validate_json(content, strict=True),
            ),
            relevance=value.relevance,
        )
        for value, content in zip(pending, contents, strict=True)
    )


def read_pending_embedding_references(
    through_day: date | None = None,
    *,
    day: date | None = None,
) -> tuple[EmbeddingReference, ...]:
    """Read pending embedding identities without downloading article bodies."""
    if through_day is not None and day is not None:
        raise ValueError("Choose either an exact embedding day or a latest embedding day")
    from romanian_news.catalog.analysis_inputs import read_embedding_candidates

    candidates = tuple(
        EmbeddingReference(
            article=ArticleAnalysisReference(
                reference=value.article.reference,
                bucharest_day=value.article.bucharest_day,
            ),
            relevance=value.relevance,
        )
        for value in read_embedding_candidates(through_day, day=day)
    )
    candidates = tuple(
        value
        for value in candidates
        if value.relevance.artifact_id
        == f"news:relevance:{production_relevance_v3_request_id(value.article.reference)}"
    )
    embedding_ids = {
        value.article.reference.version_id: (
            f"news:embedding:{embedding_request_id(value.article.reference, value.relevance)}"
        )
        for value in candidates
    }
    existing = existing_current_artifact_ids(tuple(embedding_ids.values()))
    return tuple(
        value
        for value in candidates
        if embedding_ids[value.article.reference.version_id] not in existing
    )


def load_embedding_input(value: EmbeddingReference) -> EmbeddingInput:
    """Load one exact accepted article input from immutable references."""
    content = read_verified_r2_object(
        value.article.reference.r2_key,
        value.article.reference.content_digest,
    )
    return EmbeddingInput(
        article=ArticleAnalysisInput(
            reference=value.article.reference,
            article=ExtractedArticle.model_validate_json(content, strict=True),
        ),
        relevance=value.relevance,
    )


def embed_article(value: EmbeddingInput) -> EmbeddingOutput:
    """Embed one accepted article with a fixed model and vector size."""
    request_id = embedding_request_id(value.article.reference, value.relevance)
    text = f"{value.article.article.title}\n\n{value.article.article.body[:1000]}"
    provider_inputs: ProviderEmbeddingRequest = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "dimensions": EMBEDDING_DIMENSIONS,
        "encoding_format": "float",
    }
    started = time.monotonic()
    provider_call = trace_provider_call(
        "news.embed",
        request_id,
        provider_inputs,
        lambda: openrouter_client().embeddings.create(**provider_inputs),
    )
    response = provider_call.response
    latency_ms = round((time.monotonic() - started) * 1000)
    call = ModelCall(
        response_id=response.model_dump().get("id") or request_id,
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=0,
        latency_ms=latency_ms,
    )
    vector = tuple(response.data[0].embedding)
    if len(vector) != EMBEDDING_DIMENSIONS:
        error = ValueError(f"Embedding has {len(vector)} dimensions")
        record_model_attempt(
            response,
            request_id=request_id,
            operation_key="news.embed",
            attempt_index=0,
            latency_ms=latency_ms,
            status="rejected",
            error=str(error),
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        raise error
    record_model_attempt(
        response,
        request_id=request_id,
        operation_key="news.embed",
        attempt_index=0,
        latency_ms=latency_ms,
        status="accepted",
        error=None,
        fallback_response_id=str(provider_call.call_id),
        trace=provider_call.trace,
    )
    payload = _canonical_json(
        {
            "article_version_id": value.article.reference.version_id,
            "call": call.model_dump(mode="json"),
            "dimensions": EMBEDDING_DIMENSIONS,
            "relevance_version_id": value.relevance.version_id,
            "provider_response": response.model_dump(mode="json"),
            "request_id": request_id,
            "vector": vector,
        }
    )
    return EmbeddingOutput(
        request_id=request_id,
        article=value.article.reference,
        relevance=value.relevance,
        vector=vector,
        call=call,
        content=payload,
    )


def embedding_request_id(
    article: ArtifactReference,
    relevance: ArtifactReference,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "article_version_id": article.version_id,
                "dimensions": EMBEDDING_DIMENSIONS,
                "model": EMBEDDING_MODEL,
                "purpose": "news.embed",
                "relevance_version_id": relevance.version_id,
                "text_policy": EMBEDDING_TEXT_POLICY,
            }
        )
    )
