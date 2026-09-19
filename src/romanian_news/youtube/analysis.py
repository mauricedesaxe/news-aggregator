from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, TypedDict

import requests
from pydantic import HttpUrl, TypeAdapter, ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelAttempt, model_attempt_from_payload
from romanian_news.analysis.tracing import trace_provider_call
from romanian_news.catalog.artifacts import ArtifactFile, canonical_json, sha256
from romanian_news.catalog.youtube import (
    next_receipt_attempt_index,
    publish_accepted_clip,
    publish_accepted_merge,
    publish_receipt,
    read_accepted_clip_references,
    read_unhandled_receipt,
    read_youtube_merge,
    reject_receipt,
)
from romanian_news.config import GEMINI_API_KEY
from romanian_news.youtube.errors import (
    YouTubeBudgetError,
    YouTubeConfigurationError,
    YouTubeDeterministicError,
    YouTubeEvidenceError,
    YouTubeInfrastructureError,
    YouTubeMergeError,
    YouTubeProviderPayloadError,
    YouTubeRateCardError,
)
from romanian_news.youtube.models import (
    AUTO_VIDEO_BUDGET_USD,
    ESTIMATED_EXTRACTION_USD_PER_MINUTE,
    MERGE_RESERVE_USD,
    YOUTUBE_CLIP_SECONDS,
    YOUTUBE_MODEL,
    YOUTUBE_RATE_CARDS,
    AcceptedClipExtraction,
    ClipEvidence,
    ClipRange,
    GeminiHttpResponse,
    GeminiRateCard,
    GeminiResponse,
    YouTubeClipExtraction,
    YouTubeMergedArticle,
    YouTubeModelReceipt,
    YouTubeVideoMetadata,
)


class _RawEvidence(NewsModel):
    kind: Literal[
        "reported_fact",
        "attributed_claim",
        "visual_observation",
        "on_screen_data",
        "publisher_framing",
    ]
    text: str
    attribution: str
    start_seconds: float
    end_seconds: float


class _RawClipExtraction(NewsModel):
    title: str
    standfirst: str
    narrative: str
    evidence: tuple[_RawEvidence, ...]
    source_stated_uncertainties: tuple[str, ...]


class GeminiUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    thought_tokens: int
    cost: float


_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"
CLIP_PROMPT = """Transform this bounded Romanian news video clip into a standalone English mini-article.

Write all generated prose in English for a reader who cannot watch the video. Preserve the Romanian
source material as evidence, including direct quotations when material. Preserve proper names as
given by the source. Preserve the current event, material context, specific facts, numbers, named
people and institutions, attributed claims, and information carried by images, documents, captions,
charts, or ambient sound. Paraphrase rather than transcribe.
Do not decide whether the subject is relevant to Romania. Do not add background knowledge.
Attribute allegations, opinions, forecasts, and disputed statements. Use neutral English.
Every evidence window must satisfy 0 <= start_seconds < end_seconds <= {duration}.
The clip starts at {absolute_start} in the full video and lasts {duration} seconds.
"""
MERGE_PROMPT = """Combine these consecutive mini-articles from one YouTube video into one compact,
standalone English article. Write all generated prose in English. Preserve Romanian source material
as evidence, including direct quotations when material, and preserve proper names as given by the
source. Preserve chronology, dates, quantities, people, institutions, and attributed claims. Remove
repetition. Do not add facts. Return a title, standfirst, narrative, covered_clip_starts, and
omitted_clips. Account for every input clip. An omitted clip must add no distinct material
information.
"""
CLIP_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "standfirst": {"type": "string"},
        "narrative": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "reported_fact",
                            "attributed_claim",
                            "visual_observation",
                            "on_screen_data",
                            "publisher_framing",
                        ],
                    },
                    "text": {"type": "string"},
                    "attribution": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                },
                "required": ["kind", "text", "attribution", "start_seconds", "end_seconds"],
                "additionalProperties": False,
            },
        },
        "source_stated_uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "standfirst", "narrative", "evidence", "source_stated_uncertainties"],
    "additionalProperties": False,
}
MERGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "standfirst": {"type": "string"},
        "narrative": {"type": "string"},
        "covered_clip_starts": {"type": "array", "items": {"type": "integer"}},
        "omitted_clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"clip_start": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["clip_start", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "standfirst", "narrative", "covered_clip_starts", "omitted_clips"],
    "additionalProperties": False,
}


def enforce_video_budget(duration_seconds: int, model: str = YOUTUBE_MODEL) -> None:
    rate_card_for_model(model)
    estimate = duration_seconds / 60 * ESTIMATED_EXTRACTION_USD_PER_MINUTE + MERGE_RESERVE_USD
    if estimate > AUTO_VIDEO_BUDGET_USD:
        raise YouTubeBudgetError(
            f"YouTube video estimated cost ${estimate:.2f} exceeds ${AUTO_VIDEO_BUDGET_USD:.2f} budget"
        )


def rate_card_for_model(model: str) -> GeminiRateCard:
    try:
        return YOUTUBE_RATE_CARDS[model]
    except KeyError as error:
        raise YouTubeRateCardError(f"YouTube model has no reviewed rate card: {model}") from error


def plan_clip_ranges(duration_seconds: int) -> tuple[ClipRange, ...]:
    if duration_seconds <= 0:
        raise ValueError("YouTube video duration must be positive")
    return tuple(
        ClipRange(
            start_second=start,
            duration_seconds=min(YOUTUBE_CLIP_SECONDS, duration_seconds - start),
        )
        for start in range(0, duration_seconds, YOUTUBE_CLIP_SECONDS)
    )


def parse_clip_extraction(content: str, requested: ClipRange) -> YouTubeClipExtraction:
    try:
        raw = _RawClipExtraction.model_validate_json(content, strict=True)
    except (ValidationError, ValueError) as error:
        raise YouTubeEvidenceError("Gemini clip evidence is invalid") from error
    if not raw.evidence:
        raise YouTubeEvidenceError("Gemini clip evidence is required")
    evidence: list[ClipEvidence] = []
    for item in raw.evidence:
        if not 0 <= item.start_seconds < item.end_seconds <= requested.duration_seconds:
            raise YouTubeEvidenceError(
                f"Gemini evidence window escapes the requested clip: {item.start_seconds}-{item.end_seconds}"
            )
        evidence.append(
            ClipEvidence(
                kind=item.kind,
                text=item.text,
                attribution=item.attribution,
                start_second=item.start_seconds,
                duration_seconds=item.end_seconds - item.start_seconds,
            )
        )
    try:
        return YouTubeClipExtraction(
            title=raw.title,
            standfirst=raw.standfirst,
            narrative=raw.narrative,
            evidence=tuple(evidence),
            source_stated_uncertainties=raw.source_stated_uncertainties,
        )
    except ValidationError as error:
        raise YouTubeEvidenceError("Gemini clip evidence is invalid") from error


def complete_clip_coverage(
    duration_seconds: int, clips: tuple[AcceptedClipExtraction, ...]
) -> tuple[AcceptedClipExtraction, ...]:
    ordered = tuple(sorted(clips, key=lambda value: value.clip_range.start_second))
    if tuple(value.clip_range for value in ordered) != plan_clip_ranges(duration_seconds):
        raise YouTubeEvidenceError("Accepted clips do not provide complete gap-free video coverage")
    return ordered


def parse_merge(content: str, clips: tuple[AcceptedClipExtraction, ...]) -> YouTubeMergedArticle:
    try:
        article = YouTubeMergedArticle.model_validate_json(content, strict=True)
    except (ValidationError, ValueError) as error:
        raise YouTubeMergeError("Merged YouTube article is invalid") from error
    expected = {clip.clip_range.start_second for clip in clips}
    covered = set(article.covered_clip_starts)
    omitted = {item.clip_start for item in article.omitted_clips}
    if (
        len(article.covered_clip_starts) != len(covered)
        or len(article.omitted_clips) != len(omitted)
        or covered & omitted
        or covered | omitted != expected
    ):
        raise YouTubeMergeError("Merged YouTube article does not account for every input clip")
    if not article.covered_clip_starts:
        raise YouTubeMergeError("Merged YouTube article must cover at least one input clip")
    return article


def youtube_clip_request_id(metadata: YouTubeVideoMetadata, clip_range: ClipRange) -> Sha256:
    return sha256(
        canonical_json(
            {
                "source_id": metadata.source.source_id,
                "video_id": metadata.video_id,
                "video_url": str(metadata.url),
                "duration_seconds": metadata.duration_seconds,
                "range": clip_range.model_dump(mode="json"),
                "request_digest": clip_request_digest(),
                "model": YOUTUBE_MODEL,
            }
        )
    )


def clip_request_digest() -> Sha256:
    return sha256(
        canonical_json(
            {
                "prompt": CLIP_PROMPT,
                "schema": CLIP_SCHEMA,
                "model": YOUTUBE_MODEL,
                "max_output_tokens": 8000,
                "temperature": 0,
                "fps": 1.0,
            }
        )
    )


def analyze_and_publish_clip(
    metadata: YouTubeVideoMetadata,
    metadata_file: ArtifactFile,
    clip_range: ClipRange,
    implementation_ref: str,
    renew_lease: Callable[[], None],
) -> AcceptedClipExtraction:
    request_id = youtube_clip_request_id(metadata, clip_range)
    prompt = CLIP_PROMPT.format(
        duration=clip_range.duration_seconds, absolute_start=_format_time(clip_range.start_second)
    )
    receipt = _receipt_for_request(
        request_id,
        "news.youtube.extract_clip",
        metadata.video_id,
        lambda: _send_gemini_request(
            metadata.url,
            prompt,
            CLIP_SCHEMA,
            start_second=clip_range.start_second,
            end_second=clip_range.end_second,
            max_output_tokens=8000,
        ),
        renew_lease,
    )
    response = None
    try:
        response = _parse_gemini_response(receipt.response, includes_video=True)
        extraction = parse_clip_extraction(_gemini_text(response.payload), clip_range)
    except (
        YouTubeEvidenceError,
        YouTubeInfrastructureError,
        YouTubeProviderPayloadError,
    ) as error:
        _reject_receipt(receipt, response, error)
        raise
    accepted = AcceptedClipExtraction(
        source=metadata.source,
        video_id=metadata.video_id,
        metadata_version_id=metadata_file.version_id,
        video_url=metadata.url,
        clip_range=clip_range,
        request_digest=clip_request_digest(),
        model=YOUTUBE_MODEL,
        extraction=extraction,
        provider_response=response.payload,
    )
    attempt = _accepted_attempt(receipt, response)
    return publish_accepted_clip(
        metadata, accepted, metadata_file, receipt, attempt, implementation_ref
    )


def analyze_and_publish_merge(
    metadata: YouTubeVideoMetadata,
    clips: tuple[AcceptedClipExtraction, ...],
    implementation_ref: str,
    renew_lease: Callable[[], None],
) -> tuple[YouTubeMergedArticle, ArtifactFile]:
    metadata_version_ids = {clip.metadata_version_id for clip in clips}
    if any(clip.source != metadata.source for clip in clips):
        raise YouTubeMergeError("YouTube clips do not share the metadata source")
    if len(metadata_version_ids) != 1:
        raise YouTubeMergeError("YouTube clips do not share one metadata version")
    metadata_version_id = next(iter(metadata_version_ids))
    references = read_accepted_clip_references(
        metadata, metadata_version_id, YOUTUBE_MODEL, clip_request_digest()
    )
    versions = {
        (reference.start_second, reference.duration_seconds): reference.artifact_version_id
        for reference in references
    }
    input_payload = [
        {
            "clip_range": clip.clip_range.model_dump(mode="json"),
            "title": clip.extraction.title,
            "standfirst": clip.extraction.standfirst,
            "narrative": clip.extraction.narrative,
            "evidence": [value.model_dump(mode="json") for value in clip.extraction.evidence],
            "source_stated_uncertainties": clip.extraction.source_stated_uncertainties,
        }
        for clip in clips
    ]
    request_digest = sha256(
        canonical_json(
            {
                "prompt": MERGE_PROMPT,
                "schema": MERGE_SCHEMA,
                "model": YOUTUBE_MODEL,
                "max_output_tokens": 10000,
                "temperature": 0,
            }
        )
    )
    clip_version_ids = tuple(
        versions[(clip.clip_range.start_second, clip.clip_range.duration_seconds)] for clip in clips
    )
    request_id = sha256(
        canonical_json(
            {
                "source_id": metadata.source.source_id,
                "clip_version_ids": clip_version_ids,
                "request_digest": request_digest,
                "model": YOUTUBE_MODEL,
            }
        )
    )
    existing = read_youtube_merge(request_id, metadata.source, metadata.title)
    if existing is not None:
        return existing
    receipt = _receipt_for_request(
        request_id,
        "news.youtube.merge_video",
        metadata.video_id,
        lambda: _send_gemini_request(
            None,
            f"{MERGE_PROMPT}\n\nInput clips:\n{json.dumps(input_payload, ensure_ascii=False)}",
            MERGE_SCHEMA,
            start_second=None,
            end_second=None,
            max_output_tokens=10000,
        ),
        renew_lease,
    )
    response = None
    try:
        response = _parse_gemini_response(receipt.response, includes_video=False)
        article = parse_merge(_gemini_text(response.payload), clips)
    except (YouTubeInfrastructureError, YouTubeMergeError, YouTubeProviderPayloadError) as error:
        _reject_receipt(receipt, response, error)
        raise
    attempt = _accepted_attempt(receipt, response)
    return publish_accepted_merge(
        metadata,
        clips,
        clip_version_ids,
        article,
        request_id,
        request_digest,
        response,
        receipt,
        attempt,
        implementation_ref,
    )


def _receipt_for_request(
    request_id: Sha256,
    operation_key: str,
    video_id: str,
    call: Callable[[], GeminiHttpResponse],
    renew_lease: Callable[[], None],
) -> YouTubeModelReceipt:
    existing = read_unhandled_receipt(request_id, operation_key)
    if existing is not None:
        return existing
    attempt_index = next_receipt_attempt_index(request_id, operation_key)
    renew_lease()
    provider = trace_provider_call(
        operation_key, request_id, {"model": YOUTUBE_MODEL, "video_id": video_id}, call
    )
    receipt = YouTubeModelReceipt(
        request_id=request_id,
        operation_key=operation_key,
        attempt_index=attempt_index,
        response=provider.response,
        trace=provider.trace,
        received_at=datetime.now(UTC),
    )
    return publish_receipt(receipt)


def _reject_receipt(
    receipt: YouTubeModelReceipt, response: GeminiResponse | None, error: Exception
) -> None:
    observed = datetime.now(UTC)
    attempt = (
        model_attempt_from_payload(
            response.attempt_payload(),
            request_id=receipt.request_id,
            operation_key=receipt.operation_key,
            attempt_index=receipt.attempt_index,
            latency_ms=response.latency_ms,
            status="rejected",
            error=str(error),
            observed_at=observed,
        )
        if response is not None
        else None
    )
    reject_receipt(receipt, attempt, observed, str(error))


def _accepted_attempt(receipt: YouTubeModelReceipt, response: GeminiResponse) -> ModelAttempt:
    return model_attempt_from_payload(
        response.attempt_payload(),
        request_id=receipt.request_id,
        operation_key=receipt.operation_key,
        attempt_index=receipt.attempt_index,
        latency_ms=response.latency_ms,
        status="accepted",
        error=None,
        observed_at=datetime.now(UTC),
    )


def _send_gemini_request(
    video_url: HttpUrl | None,
    prompt: str,
    schema: dict[str, object],
    *,
    start_second: int | None,
    end_second: int | None,
    max_output_tokens: int,
    model: str = YOUTUBE_MODEL,
) -> GeminiHttpResponse:
    rate_card_for_model(model)
    if not GEMINI_API_KEY:
        raise YouTubeConfigurationError("GEMINI_API_KEY is required")
    parts: list[dict[str, object]] = [{"text": prompt}]
    if video_url is not None:
        if start_second is None or end_second is None:
            raise YouTubeDeterministicError("Gemini video requests require an exact range")
        parts.append(
            {
                "fileData": {"fileUri": str(video_url)},
                "videoMetadata": {
                    "startOffset": f"{start_second}s",
                    "endOffset": f"{end_second}s",
                    "fps": 1.0,
                },
            }
        )
    started = time.monotonic()
    try:
        response = requests.post(
            f"{_GEMINI_API}/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                    "maxOutputTokens": max_output_tokens,
                    "temperature": 0,
                },
            },
            timeout=300,
        )
        latency_ms = round((time.monotonic() - started) * 1000)
    except requests.RequestException as error:
        raise YouTubeInfrastructureError("Gemini request failed") from error
    return GeminiHttpResponse(
        requested_model=model,
        status_code=response.status_code,
        body_base64=base64.b64encode(response.content).decode(),
        latency_ms=latency_ms,
    )


def _parse_gemini_response(response: GeminiHttpResponse, *, includes_video: bool) -> GeminiResponse:
    if response.status_code >= 400:
        error = f"Gemini request failed with HTTP {response.status_code}"
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise YouTubeInfrastructureError(error)
        raise YouTubeProviderPayloadError(error)
    body = response.body_bytes()
    rate_card = rate_card_for_model(response.requested_model)
    try:
        payload_value: object = json.loads(body)
        payload = TypeAdapter(dict[str, object]).validate_python(payload_value, strict=True)
        usage = gemini_usage(payload.get("usageMetadata"), rate_card, includes_video=includes_video)
    except (ValidationError, ValueError, TypeError) as error:
        if isinstance(error, YouTubeProviderPayloadError):
            raise
        raise YouTubeProviderPayloadError("Gemini response is invalid") from error
    return GeminiResponse(
        response_id=str(payload.get("responseId") or sha256(body)),
        model=response.requested_model,
        payload=payload,
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        thought_tokens=usage["thought_tokens"],
        cost_usd=usage["cost"],
        rate_card_version=rate_card.version,
        latency_ms=response.latency_ms,
    )


def gemini_usage(value: object, rate_card: GeminiRateCard, *, includes_video: bool) -> GeminiUsage:
    if not isinstance(value, dict):
        raise YouTubeProviderPayloadError("Gemini usage metadata is missing")
    input_tokens = _nonnegative_int(value.get("promptTokenCount"), "input tokens")
    output_tokens = _nonnegative_int(value.get("candidatesTokenCount"), "candidate tokens")
    thought_tokens = _nonnegative_int(value.get("thoughtsTokenCount", 0), "thought tokens")
    cost = 0.0
    detailed = 0
    details = value.get("promptTokensDetails")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                raise YouTubeProviderPayloadError("Gemini prompt token details are invalid")
            tokens = _nonnegative_int(item.get("tokenCount"), "modality tokens")
            detailed += tokens
            rate = (
                rate_card.audio_input_usd_per_million
                if item.get("modality") == "AUDIO"
                else rate_card.text_input_usd_per_million
            )
            cost += tokens * rate / 1_000_000
    if detailed > input_tokens:
        raise YouTubeProviderPayloadError("Gemini prompt token details exceed prompt token count")
    fallback_rate = (
        rate_card.audio_input_usd_per_million
        if includes_video
        else rate_card.text_input_usd_per_million
    )
    cost += (input_tokens - detailed) * fallback_rate / 1_000_000
    billed_output = output_tokens + (
        thought_tokens if rate_card.thought_tokens_billed_as_output else 0
    )
    cost += billed_output * rate_card.output_usd_per_million / 1_000_000
    return GeminiUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        thought_tokens=thought_tokens,
        cost=cost,
    )


def _gemini_text(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise YouTubeProviderPayloadError("Gemini response must contain one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
        reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
        raise YouTubeProviderPayloadError(f"Gemini response did not finish normally: {reason}")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = (
        parts[0].get("text")
        if isinstance(parts, list) and parts and isinstance(parts[0], dict)
        else None
    )
    if not isinstance(text, str):
        raise YouTubeProviderPayloadError("Gemini response has no JSON text")
    return text


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise YouTubeProviderPayloadError(f"Gemini {label} is invalid")
    return value


def _format_time(seconds: int) -> str:
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{remaining:02d}" if hours else f"{minutes}:{remaining:02d}"
