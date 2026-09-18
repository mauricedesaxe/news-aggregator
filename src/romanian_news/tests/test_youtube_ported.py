import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import HttpUrl, ValidationError

from romanian_news.youtube import analysis as analysis_module
from romanian_news.youtube import discovery as discovery_module
from romanian_news.youtube.analysis import (
    complete_clip_coverage,
    gemini_usage,
    parse_clip_extraction,
    plan_clip_ranges,
    rate_card_for_model,
    youtube_clip_request_id,
)
from romanian_news.youtube.discovery import (
    acquire_youtube_poll,
    decide_youtube_poll,
    fetch_video_metadata,
    parse_youtube_feed,
)
from romanian_news.youtube.errors import YouTubeInfrastructureError, YouTubeProviderPayloadError
from romanian_news.youtube.models import (
    YOUTUBE_MODEL,
    YOUTUBE_SOURCES,
    AcceptedClipExtraction,
    ClipEvidence,
    ClipRange,
    GeminiHttpResponse,
    YouTubeClipExtraction,
    YouTubeFeed,
    YouTubeFeedEntry,
    YouTubePollStatus,
    YouTubeVideoId,
    YouTubeVideoMetadata,
)

RECORDER_SOURCE = YOUTUBE_SOURCES.source("recorder-youtube")
RECORDER_CHANNEL_ID = RECORDER_SOURCE.channel_id
VIDEO_ID = YouTubeVideoId("abcdefghijk")
PUBLISHED_AT = datetime(2026, 9, 9, 8, tzinfo=UTC)
RECORDER_FEED_FIXTURE = Path(__file__).parent / "fixtures/recorder_uploads_playlist.json"


def test_parses_fixed_recorder_feed_response() -> None:
    content = RECORDER_FEED_FIXTURE.read_bytes()
    feed = parse_youtube_feed(RECORDER_SOURCE, content)
    assert feed.entries[0].video_id == "rYzsi1A-aYc"
    assert feed.entries[0].published_at == datetime(2026, 7, 14, 9, 25, 8, tzinfo=UTC)


def test_acquires_feed_from_uploads_playlist(monkeypatch) -> None:
    content = RECORDER_FEED_FIXTURE.read_bytes()
    requests = []

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    response = Response(content)
    monkeypatch.setattr(discovery_module, "YOUTUBE_API_KEY", "test")
    monkeypatch.setattr(
        discovery_module.requests,
        "get",
        lambda url, **kwargs: requests.append((url, kwargs)) or response,
    )

    capture = acquire_youtube_poll(RECORDER_SOURCE, PUBLISHED_AT)

    assert capture.content == content
    assert capture.feed.entries[0].video_id == "rYzsi1A-aYc"
    assert requests == [
        (
            "https://www.googleapis.com/youtube/v3/playlistItems",
            {
                "params": {
                    "part": "snippet,contentDetails",
                    "playlistId": RECORDER_SOURCE.uploads_playlist_id,
                    "maxResults": 50,
                },
                "headers": {"X-Goog-Api-Key": "test"},
                "timeout": 30,
            },
        )
    ]


def test_feed_failure_does_not_put_api_key_in_request_url(monkeypatch) -> None:
    api_key = "test-secret"

    def forbidden(url, **kwargs):
        kwargs.pop("timeout")
        request = discovery_module.requests.Request("GET", url, **kwargs).prepare()
        response = discovery_module.requests.Response()
        response.status_code = 403
        response.request = request
        response.url = request.url or url
        return response

    monkeypatch.setattr(discovery_module, "YOUTUBE_API_KEY", api_key)
    monkeypatch.setattr(discovery_module.requests, "get", forbidden)

    with pytest.raises(YouTubeInfrastructureError) as captured:
        acquire_youtube_poll(RECORDER_SOURCE, PUBLISHED_AT)

    assert captured.value.__cause__ is not None
    assert api_key not in str(captured.value.__cause__)


def test_first_poll_baselines_then_discovers_only_new_ids() -> None:
    first = decide_youtube_poll(_feed(VIDEO_ID), frozenset(), baseline_exists=False)
    repeated = decide_youtube_poll(_feed(VIDEO_ID), frozenset({VIDEO_ID}), baseline_exists=True)
    discovered = decide_youtube_poll(
        _feed("lmnopqrstuv", VIDEO_ID), frozenset({VIDEO_ID}), baseline_exists=True
    )
    assert first.status == YouTubePollStatus.BASELINE
    assert first.pending_video_ids == ()
    assert repeated.pending_video_ids == ()
    assert discovered.pending_video_ids == ("lmnopqrstuv",)


def test_later_poll_without_known_overlap_refuses_backfill() -> None:
    decision = decide_youtube_poll(
        _feed("lmnopqrstuv"), frozenset({VIDEO_ID}), baseline_exists=True
    )
    assert decision.status == YouTubePollStatus.OVERLAP_LOST
    assert decision.pending_video_ids == ()


def test_fetch_video_metadata_parses_production_shaped_json_bytes(monkeypatch) -> None:
    content = _youtube_response_bytes()
    requests = []

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    response = Response(content)
    monkeypatch.setattr(discovery_module, "YOUTUBE_API_KEY", "test")
    monkeypatch.setattr(
        discovery_module.requests,
        "get",
        lambda url, **kwargs: requests.append((url, kwargs)) or response,
    )
    metadata = fetch_video_metadata(RECORDER_SOURCE, VIDEO_ID, "b" * 64)
    assert metadata.duration_seconds == 754
    assert metadata.published_at == PUBLISHED_AT
    assert metadata.provider_response["kind"] == "youtube#videoListResponse"
    assert metadata.raw_provider_response == content
    assert requests == [
        (
            "https://www.googleapis.com/youtube/v3/videos",
            {
                "headers": {"X-Goog-Api-Key": "test"},
                "params": {"part": "snippet,contentDetails", "id": VIDEO_ID},
                "timeout": 30,
            },
        )
    ]


def test_recorder_models_reject_naive_publication_times() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        YouTubeFeedEntry(
            source=RECORDER_SOURCE,
            video_id=VIDEO_ID,
            title="Recorder",
            published_at=datetime(2026, 9, 9, 8),
        )
    with pytest.raises(ValidationError, match="timezone"):
        YouTubeVideoMetadata(
            source=RECORDER_SOURCE,
            video_id=VIDEO_ID,
            url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
            title="Recorder",
            published_at=datetime(2026, 9, 9, 8),
            duration_seconds=300,
            feed_snapshot_version_id="b" * 64,
            retrieved_at=PUBLISHED_AT,
            provider_response={},
            raw_provider_response=b"{}",
        )


def test_plans_ranges_and_rejects_escaped_evidence() -> None:
    assert plan_clip_ranges(601) == (
        ClipRange(start_second=0, duration_seconds=300),
        ClipRange(start_second=300, duration_seconds=300),
        ClipRange(start_second=600, duration_seconds=1),
    )
    payload = _clip_payload()
    cast(list[dict[str, object]], payload["evidence"]).append(
        {
            "kind": "reported_fact",
            "text": "Leaked",
            "attribution": "Recorder",
            "start_seconds": 299,
            "end_seconds": 301,
        }
    )
    with pytest.raises(ValueError, match="escapes"):
        parse_clip_extraction(json.dumps(payload), ClipRange(start_second=0, duration_seconds=300))


def test_clip_requires_structured_evidence_with_text_and_attribution() -> None:
    clip_range = ClipRange(start_second=0, duration_seconds=300)
    without_evidence = _clip_payload()
    without_evidence["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        parse_clip_extraction(json.dumps(without_evidence), clip_range)

    for field in ("text", "attribution"):
        blank = _clip_payload()
        cast(list[dict[str, object]], blank["evidence"])[0][field] = "   "
        with pytest.raises(ValueError, match=field):
            parse_clip_extraction(json.dumps(blank), clip_range)


def test_complete_coverage_requires_every_contiguous_range() -> None:
    clips = (_accepted(0, 300), _accepted(300, 300), _accepted(600, 1))
    assert complete_clip_coverage(601, clips) == clips
    with pytest.raises(ValueError, match="gap-free"):
        complete_clip_coverage(601, clips[::2])


def test_merge_rejects_all_omitted_clips() -> None:
    clips = (_accepted(0, 300), _accepted(300, 300))
    payload = {
        "title": "Titlu",
        "standfirst": "Introducere",
        "narrative": "Narațiune",
        "covered_clip_starts": [],
        "omitted_clips": [
            {"clip_start": 0, "reason": "Nimic distinct"},
            {"clip_start": 300, "reason": "Nimic distinct"},
        ],
    }

    with pytest.raises(ValueError, match="cover at least one"):
        analysis_module.parse_merge(json.dumps(payload), clips)


def test_clip_request_identity_survives_metadata_refresh() -> None:
    first = _metadata(duration_seconds=300)
    refreshed = first.model_copy(update={"retrieved_at": PUBLISHED_AT + timedelta(hours=1)})
    clip_range = ClipRange(start_second=0, duration_seconds=300)
    assert youtube_clip_request_id(first, clip_range) == youtube_clip_request_id(
        refreshed, clip_range
    )


def test_clip_request_identity_changes_with_response_schema(monkeypatch) -> None:
    metadata = _metadata(duration_seconds=300)
    clip_range = ClipRange(start_second=0, duration_seconds=300)
    original = youtube_clip_request_id(metadata, clip_range)
    monkeypatch.setattr(
        analysis_module,
        "CLIP_SCHEMA",
        {**analysis_module.CLIP_SCHEMA, "description": "revised contract"},
    )

    assert youtube_clip_request_id(metadata, clip_range) != original


def test_gemini_usage_uses_candidates_and_bills_thoughts_separately() -> None:
    usage = gemini_usage(
        {
            "promptTokenCount": 1_000,
            "candidatesTokenCount": 200,
            "thoughtsTokenCount": 300,
            "totalTokenCount": 9_999,
            "promptTokensDetails": [
                {"modality": "AUDIO", "tokenCount": 700},
                {"modality": "TEXT", "tokenCount": 200},
            ],
        },
        rate_card_for_model(YOUTUBE_MODEL),
        includes_video=True,
    )
    assert usage["prompt_tokens"] == 1_000
    assert usage["completion_tokens"] == 200
    assert usage["thought_tokens"] == 300
    assert usage["cost"] == pytest.approx((800 + 0.3 * 200 + 2.5 * 500) / 1_000_000)


def test_gemini_usage_rejects_missing_usage_and_candidate_count() -> None:
    card = rate_card_for_model(YOUTUBE_MODEL)
    with pytest.raises(ValueError, match="missing"):
        gemini_usage(None, card, includes_video=False)
    with pytest.raises(ValueError, match="candidate tokens"):
        gemini_usage(
            {"promptTokenCount": 1, "totalTokenCount": 2},
            card,
            includes_video=False,
        )


def test_gemini_usage_fallback_prices_video_as_audio_and_merge_as_text() -> None:
    metadata = {
        "promptTokenCount": 1_000,
        "candidatesTokenCount": 200,
        "thoughtsTokenCount": 300,
    }
    card = rate_card_for_model(YOUTUBE_MODEL)

    video = gemini_usage(metadata, card, includes_video=True)
    merge = gemini_usage(metadata, card, includes_video=False)

    assert video["completion_tokens"] == merge["completion_tokens"] == 200
    assert video["thought_tokens"] == merge["thought_tokens"] == 300
    assert video["cost"] == pytest.approx((1_000 + 2.5 * 500) / 1_000_000)
    assert merge["cost"] == pytest.approx((0.3 * 1_000 + 2.5 * 500) / 1_000_000)


def test_unknown_model_pricing_fails_before_provider_call(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(analysis_module, "GEMINI_API_KEY", "test")
    monkeypatch.setattr(analysis_module.requests, "post", lambda *_args, **_kwargs: calls.append(1))
    with pytest.raises(ValueError, match="no reviewed rate card"):
        analysis_module._send_gemini_request(
            None,
            "prompt",
            {},
            start_second=None,
            end_second=None,
            max_output_tokens=1,
            model="unknown-model",
        )
    assert calls == []


def test_direct_gemini_request_uses_native_offsets_and_usage(monkeypatch) -> None:
    captured = {}

    class Response:
        status_code = 200
        content = json.dumps(_gemini_payload(json.dumps(_clip_payload()))).encode()

    def post(_url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(analysis_module, "GEMINI_API_KEY", "test")
    monkeypatch.setattr(analysis_module.requests, "post", post)
    raw_response = analysis_module._send_gemini_request(
        HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        "prompt",
        analysis_module.CLIP_SCHEMA,
        start_second=300,
        end_second=600,
        max_output_tokens=8000,
    )
    response = analysis_module._parse_gemini_response(raw_response, includes_video=True)
    video = captured["contents"][0]["parts"][1]
    assert video["videoMetadata"] == {
        "startOffset": "300s",
        "endOffset": "600s",
        "fps": 1.0,
    }
    assert response.output_tokens == 5
    assert response.rate_card_version == "gemini-3.6-flash-2026-09-09"


def test_invalid_gemini_json_is_receipted_before_parsing(monkeypatch) -> None:
    events = []
    raw_response = _http_response(b"not-json")
    monkeypatch.setattr(analysis_module, "read_unhandled_receipt", lambda *_args: None)
    monkeypatch.setattr(analysis_module, "next_receipt_attempt_index", lambda *_args: 0)
    monkeypatch.setattr(
        analysis_module,
        "trace_provider_call",
        lambda _operation, _request, _inputs, call: SimpleNamespace(
            response=call(), call_id="call", trace=None
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "publish_receipt",
        lambda receipt: events.append("receipt") or receipt,
    )

    receipt = analysis_module._receipt_for_request(
        "a" * 64,
        "news.youtube.extract_clip",
        VIDEO_ID,
        lambda: events.append("paid") or raw_response,
        lambda: events.append("renew"),
    )

    with pytest.raises(ValueError, match="Gemini response is invalid"):
        analysis_module._parse_gemini_response(receipt.response, includes_video=True)
    assert events == ["renew", "paid", "receipt"]


def test_missing_gemini_usage_is_receipted_before_parsing(monkeypatch) -> None:
    events = []
    raw_response = _http_response({"responseId": "response-1"})
    monkeypatch.setattr(analysis_module, "read_unhandled_receipt", lambda *_args: None)
    monkeypatch.setattr(analysis_module, "next_receipt_attempt_index", lambda *_args: 0)
    monkeypatch.setattr(
        analysis_module,
        "trace_provider_call",
        lambda _operation, _request, _inputs, call: SimpleNamespace(
            response=call(), call_id="call", trace=None
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "publish_receipt",
        lambda receipt: events.append("receipt") or receipt,
    )

    receipt = analysis_module._receipt_for_request(
        "a" * 64,
        "news.youtube.extract_clip",
        VIDEO_ID,
        lambda: events.append("paid") or raw_response,
        lambda: events.append("renew"),
    )

    with pytest.raises(ValueError, match="usage metadata is missing"):
        analysis_module._parse_gemini_response(receipt.response, includes_video=True)
    assert events == ["renew", "paid", "receipt"]


def test_gemini_http_status_classifies_retryable_failures() -> None:
    with pytest.raises(YouTubeProviderPayloadError, match="HTTP 400"):
        analysis_module._parse_gemini_response(
            _http_response({}).model_copy(update={"status_code": 400}), includes_video=False
        )
    with pytest.raises(YouTubeInfrastructureError, match="HTTP 429"):
        analysis_module._parse_gemini_response(
            _http_response({}).model_copy(update={"status_code": 429}), includes_video=False
        )


def _feed(*video_ids: str) -> YouTubeFeed:
    return YouTubeFeed(
        source=RECORDER_SOURCE,
        entries=tuple(
            YouTubeFeedEntry(
                source=RECORDER_SOURCE,
                video_id=YouTubeVideoId(video_id),
                title=video_id,
                published_at=PUBLISHED_AT,
            )
            for video_id in video_ids
        ),
    )


def _youtube_response_bytes() -> bytes:
    return json.dumps(
        {
            "kind": "youtube#videoListResponse",
            "etag": "etag",
            "items": [
                {
                    "kind": "youtube#video",
                    "etag": "video-etag",
                    "id": VIDEO_ID,
                    "snippet": {
                        "publishedAt": "2026-09-09T08:00:00Z",
                        "channelId": RECORDER_CHANNEL_ID,
                        "title": "Investigație Recorder",
                    },
                    "contentDetails": {"duration": "PT12M34S"},
                }
            ],
            "pageInfo": {"totalResults": 1, "resultsPerPage": 1},
        },
        ensure_ascii=False,
    ).encode()


def _clip_payload() -> dict[str, object]:
    return {
        "title": "Titlu",
        "standfirst": "Introducere",
        "narrative": "Narațiune",
        "evidence": [
            {
                "kind": "reported_fact",
                "text": "Fapt",
                "attribution": "Recorder",
                "start_seconds": 1,
                "end_seconds": 2,
            }
        ],
        "source_stated_uncertainties": [],
    }


def _accepted(
    start: int, duration: int, *, uncertainties: tuple[str, ...] = ()
) -> AcceptedClipExtraction:
    return AcceptedClipExtraction(
        source=RECORDER_SOURCE,
        video_id=VIDEO_ID,
        metadata_version_id="b" * 64,
        video_url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        clip_range=ClipRange(start_second=start, duration_seconds=duration),
        request_digest="a" * 64,
        model=YOUTUBE_MODEL,
        extraction=YouTubeClipExtraction(
            title="Titlu",
            standfirst="Introducere",
            narrative="Narațiune",
            evidence=(
                ClipEvidence(
                    kind="reported_fact",
                    text="Fapt",
                    attribution="Recorder",
                    start_second=0,
                    duration_seconds=min(1, duration),
                ),
            ),
            source_stated_uncertainties=uncertainties,
        ),
        provider_response={},
    )


def _metadata(*, duration_seconds: int) -> YouTubeVideoMetadata:
    return YouTubeVideoMetadata(
        source=RECORDER_SOURCE,
        video_id=VIDEO_ID,
        url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        title="Recorder",
        published_at=PUBLISHED_AT,
        duration_seconds=duration_seconds,
        feed_snapshot_version_id="b" * 64,
        retrieved_at=PUBLISHED_AT,
        provider_response={"items": []},
        raw_provider_response=b'{"items":[]}',
    )


def _gemini_payload(text: str) -> dict[str, object]:
    return {
        "responseId": "response-1",
        "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


def _http_response(payload: dict[str, object] | bytes) -> GeminiHttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return GeminiHttpResponse(
        requested_model=YOUTUBE_MODEL,
        status_code=200,
        body_base64=base64.b64encode(body).decode(),
        latency_ms=1,
    )
