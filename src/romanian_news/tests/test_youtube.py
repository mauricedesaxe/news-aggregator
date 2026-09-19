import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import HttpUrl, ValidationError

from romanian_news import Sha256
from romanian_news.analysis.attempts import ModelAttempt, model_attempt_from_payload
from romanian_news.analysis.tracing import ModelTraceReference
from romanian_news.catalog import youtube as catalog_module
from romanian_news.catalog.artifacts import artifact_file, canonical_json, sha256
from romanian_news.catalog.youtube import (
    publish_accepted_clip,
    publish_accepted_merge,
    publish_next_approved_candidate,
    publish_youtube_candidate,
    reject_receipt,
)
from romanian_news.youtube import discovery as discovery_module
from romanian_news.youtube import workflow as workflow_module
from romanian_news.youtube.analysis import (
    CLIP_PROMPT,
    MERGE_PROMPT,
    clip_request_digest,
    complete_clip_coverage,
    enforce_video_budget,
    parse_clip_extraction,
    plan_clip_ranges,
    youtube_clip_request_id,
)
from romanian_news.youtube.discovery import (
    decide_youtube_poll,
    fetch_video_metadata,
    parse_youtube_feed,
)
from romanian_news.youtube.errors import (
    YouTubeBudgetError,
    YouTubeLeaseLostError,
    YouTubeProviderPayloadError,
    YouTubeRateCardError,
)
from romanian_news.youtube.models import (
    LIVE_MEDIA_REPLAY_LIMIT,
    YOUTUBE_MODEL,
    YOUTUBE_SOURCES,
    AcceptedClipExtraction,
    CandidateEvidence,
    ClipEvidence,
    ClipRange,
    GeminiHttpResponse,
    GeminiResponse,
    MeasuredCost,
    SystemApproval,
    UnmeasuredCost,
    YouTubeCandidate,
    YouTubeClipExtraction,
    YouTubeFeed,
    YouTubeFeedEntry,
    YouTubeMergedArticle,
    YouTubeModelReceipt,
    YouTubePollCapture,
    YouTubePollStatus,
    YouTubeSource,
    YouTubeVideoId,
    YouTubeVideoLease,
    YouTubeVideoMetadata,
)

RECORDER = YOUTUBE_SOURCES.source("recorder-youtube")
STAREA = YOUTUBE_SOURCES.source("starea-impostorilor-youtube")
VIDEO_ID = YouTubeVideoId("abcdefghijk")
PUBLISHED_AT = datetime(2026, 9, 9, 8, tzinfo=UTC)


def test_article_prompts_require_english_prose_and_preserve_source_evidence() -> None:
    for prompt in (CLIP_PROMPT, MERGE_PROMPT):
        normalized = " ".join(prompt.split())
        assert "generated prose in English" in normalized
        assert "source material as evidence" in normalized
        assert "proper names" in normalized


def test_source_registry_encodes_unique_identity() -> None:
    assert [
        (v.source_id, v.channel_id, v.outlet_id, v.display_name) for v in YOUTUBE_SOURCES.sources
    ] == [
        (
            "recorder-youtube",
            "UChDQ6nYN6XyRU-8IEgbym1g",
            "recorder",
            "Recorder",
        ),
        (
            "starea-impostorilor-youtube",
            "UCtK5Oe8sHjp6WPcwWuHUVpQ",
            "starea-impostorilor",
            "Starea Impostorilor",
        ),
        (
            "snoop-youtube",
            "UCi7oQBZ4Amu_xPxc78F970A",
            "snoop",
            "Snoop",
        ),
        (
            "stirile-protv-youtube",
            "UCEJf5cGtkBdZS8Jh2uSW9xw",
            "stirile-protv",
            "Știrile ProTV",
        ),
    ]
    with pytest.raises(ValidationError, match="duplicate source_id"):
        type(YOUTUBE_SOURCES)(sources=(RECORDER, RECORDER))


def test_parses_both_channel_feeds_with_source_identity() -> None:
    for source in (RECORDER, STAREA):
        feed = parse_youtube_feed(source, _feed_response(source))
        assert feed.source == source
        assert feed.entries[0].source == source
        assert feed.entries[0].video_id == VIDEO_ID


def test_feed_rejects_another_channel() -> None:
    with pytest.raises(YouTubeProviderPayloadError, match="another channel"):
        parse_youtube_feed(RECORDER, _feed_response(STAREA))


def test_first_poll_and_overlap_loss_are_independent_per_source() -> None:
    for source in (RECORDER, STAREA):
        first = decide_youtube_poll(_feed(source, VIDEO_ID), frozenset(), baseline_exists=False)
        current = decide_youtube_poll(
            _feed(source, "lmnopqrstuv", VIDEO_ID), frozenset({VIDEO_ID}), baseline_exists=True
        )
        lost = decide_youtube_poll(
            _feed(source, "lmnopqrstuv"), frozenset({VIDEO_ID}), baseline_exists=True
        )
        assert first.status == YouTubePollStatus.BASELINE and first.pending_video_ids == ()
        assert current.pending_video_ids == ("lmnopqrstuv",)
        assert lost.status == YouTubePollStatus.OVERLAP_LOST and lost.pending_video_ids == ()


def test_uploads_playlist_id_matches_the_synthetic_recorder_playlist() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "synthetic_uploads_playlist.json"
    fixture = json.loads(fixture_path.read_text())
    playlist_id = fixture["items"][0]["snippet"]["playlistId"]

    assert RECORDER.uploads_playlist_id == playlist_id
    assert playlist_id == f"UU{RECORDER.channel_id.removeprefix('UC')}"


def test_metadata_rejects_snippet_channel_mismatch_before_analysis(monkeypatch) -> None:
    class Response:
        content = _youtube_response(STAREA.channel_id)

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(discovery_module, "YOUTUBE_API_KEY", "test")
    monkeypatch.setattr(discovery_module.requests, "get", lambda *_args, **_kwargs: Response())
    with pytest.raises(YouTubeProviderPayloadError, match="another channel"):
        fetch_video_metadata(RECORDER, VIDEO_ID, "b" * 64)


def test_metadata_requires_exactly_one_matching_video(monkeypatch) -> None:
    monkeypatch.setattr(discovery_module, "YOUTUBE_API_KEY", "test")

    def video(id: str) -> dict[str, object]:
        return {
            "id": id,
            "snippet": {
                "publishedAt": "2026-09-09T08:00:00Z",
                "channelId": RECORDER.channel_id,
                "title": "Video",
            },
            "contentDetails": {"duration": "PT5M"},
        }

    for body, match in (
        ({"kind": "youtube#videoListResponse", "items": []}, "did not return"),
        (
            {
                "kind": "youtube#videoListResponse",
                "items": [video(VIDEO_ID), video("lmnopqrstuv")],
            },
            "did not return",
        ),
        (
            {"kind": "youtube#videoListResponse", "items": [video("lmnopqrstuv")]},
            "did not return",
        ),
        (
            {
                "kind": "youtube#videoListResponse",
                "items": [video(VIDEO_ID) | {"contentDetails": {"duration": "PT1H30M?"}}],
            },
            "duration is invalid",
        ),
        (
            {
                "kind": "youtube#videoListResponse",
                "items": [video(VIDEO_ID) | {"contentDetails": {"duration": "PT0S"}}],
            },
            "must be positive",
        ),
    ):

        class Response:
            content = json.dumps(body).encode()

            def raise_for_status(self) -> None:
                return None

        monkeypatch.setattr(discovery_module.requests, "get", lambda *_args, **_kwargs: Response())
        with pytest.raises(YouTubeProviderPayloadError, match=match):
            fetch_video_metadata(RECORDER, VIDEO_ID, "b" * 64)


def test_video_budget_rejects_unaffordable_videos_and_unknown_models() -> None:
    enforce_video_budget(600)

    with pytest.raises(YouTubeBudgetError, match="exceeds"):
        enforce_video_budget(30_000)
    with pytest.raises(YouTubeRateCardError, match="rate card"):
        enforce_video_budget(600, model="gemini-unknown-model")


def test_claimed_lease_carries_persisted_metadata_version(monkeypatch) -> None:
    monkeypatch.setattr(
        catalog_module,
        "catalog_mutation",
        lambda *_args: [
            {
                "video_id": VIDEO_ID,
                "owner_token": "owner",
                "lease_expires_at": "2026-09-09T08:10:00+00:00",
                "first_poll_version_id": "b" * 64,
                "metadata_version_id": "a" * 64,
                "deterministic_failure_fingerprint": None,
                "unchanged_deterministic_failures": 0,
            }
        ],
    )

    lease = catalog_module.claim_youtube_video(RECORDER, "owner", PUBLISHED_AT)

    assert lease is not None
    assert lease.metadata_version_id == "a" * 64


def test_retry_reuses_persisted_metadata_and_accepted_clips_without_paid_calls(
    monkeypatch,
) -> None:
    metadata = _metadata(RECORDER)
    metadata_file = _file("metadata")
    lease = _lease(RECORDER).model_copy(update={"metadata_version_id": metadata_file.version_id})
    accepted = _accepted(RECORDER).model_copy(
        update={"metadata_version_id": metadata_file.version_id}
    )
    published = []

    monkeypatch.setattr(workflow_module, "acquire_youtube_poll", lambda source, _at: _poll(source))
    monkeypatch.setattr(
        workflow_module,
        "publish_youtube_poll",
        lambda _capture: SimpleNamespace(status=YouTubePollStatus.CURRENT),
    )
    monkeypatch.setattr(workflow_module, "claim_youtube_video", lambda *_args: lease)
    monkeypatch.setattr(workflow_module, "renew_youtube_lease", lambda value, _now: value)
    monkeypatch.setattr(
        workflow_module, "read_youtube_metadata", lambda _lease: (metadata, metadata_file)
    )
    monkeypatch.setattr(
        workflow_module,
        "fetch_video_metadata",
        lambda *_args: pytest.fail("YouTube metadata ran again"),
    )
    monkeypatch.setattr(
        workflow_module,
        "publish_youtube_metadata",
        lambda *_args: pytest.fail("YouTube metadata was republished"),
    )
    monkeypatch.setattr(workflow_module, "read_accepted_clips", lambda *_args: (accepted,))
    monkeypatch.setattr(
        workflow_module,
        "analyze_and_publish_clip",
        lambda *_args: pytest.fail("Gemini clip extraction ran again"),
    )
    monkeypatch.setattr(
        workflow_module,
        "analyze_and_publish_merge",
        lambda *_args: (_merged(), _file("merge")),
    )
    monkeypatch.setattr(
        workflow_module,
        "publish_youtube_candidate",
        lambda *_args: published.append("candidate") or "c" * 64,
    )
    monkeypatch.setattr(workflow_module, "flush_langfuse_traces", lambda: None)

    result = workflow_module.materialize_youtube_source(RECORDER, PUBLISHED_AT, "git:test")

    assert result.candidate_version_id == "c" * 64
    assert published == ["candidate"]


def test_stale_owner_cannot_start_paid_analysis(monkeypatch) -> None:
    stale_lease = _lease(RECORDER).model_copy(update={"owner_token": "stale-owner"})
    monkeypatch.setattr(workflow_module, "acquire_youtube_poll", lambda source, _at: _poll(source))
    monkeypatch.setattr(
        workflow_module,
        "publish_youtube_poll",
        lambda _capture: SimpleNamespace(status=YouTubePollStatus.CURRENT),
    )
    monkeypatch.setattr(workflow_module, "claim_youtube_video", lambda *_args: stale_lease)
    monkeypatch.setattr(catalog_module, "catalog_mutation", lambda *_args: [])
    monkeypatch.setattr(
        workflow_module,
        "fetch_video_metadata",
        lambda *_args: pytest.fail("stale worker fetched metadata"),
    )
    monkeypatch.setattr(
        workflow_module,
        "analyze_and_publish_clip",
        lambda *_args: pytest.fail("stale worker invoked Gemini"),
    )
    monkeypatch.setattr(workflow_module, "flush_langfuse_traces", lambda: None)

    with pytest.raises(YouTubeLeaseLostError, match="YouTube video lease was lost"):
        workflow_module.materialize_youtube_source(RECORDER, PUBLISHED_AT, "git:test")


def test_stored_metadata_rejects_source_or_video_identity_mismatch(monkeypatch) -> None:
    metadata = _metadata(STAREA)
    content = canonical_json(metadata.model_dump(mode="json"))
    value = artifact_file(
        artifact_id=f"news:youtube-video:{STAREA.source_id}:{VIDEO_ID}",
        artifact_kind="youtube_video_metadata",
        title=metadata.title,
        content=content,
        r2_key="news/youtube/metadata.json",
        media_type="application/json",
    )
    lease = _lease(RECORDER).model_copy(update={"metadata_version_id": value.version_id})
    monkeypatch.setattr(
        catalog_module,
        "catalog_query",
        lambda *_args: [
            {
                "artifact_id": value.artifact_id,
                "artifact_kind": value.artifact_kind,
                "title": value.title,
                "content_digest": value.content_digest,
                "r2_key": value.r2_key,
                "media_type": value.media_type,
            }
        ],
    )
    monkeypatch.setattr(catalog_module, "read_verified_r2_object", lambda *_args: content)

    with pytest.raises(ValueError, match="another source or video"):
        catalog_module.read_youtube_metadata(lease)


def test_candidate_lease_error_survives_catalog_error_wrapping() -> None:
    error = catalog_module.ResearchCatalogError("PostgreSQL catalog request failed")
    error.__cause__ = RuntimeError("YouTube candidate lease was lost")

    assert catalog_module._exception_contains(error, "youtube candidate lease was lost")


def test_candidate_is_complete_and_system_approved(monkeypatch) -> None:
    metadata_file, merge_file, clip_file = _file("metadata"), _file("merge"), _file("clip")
    batches: list[list[tuple[str, list[object]]]] = []
    published: list[tuple[str, bytes]] = []

    def query(sql: str, _params=None):
        if "FROM youtube_model_receipts receipt" in sql:
            assert "LEFT JOIN news_model_attempts" in sql
            return [
                {"receipt_version_id": "6" * 64, "cost_usd": 0.02},
                {"receipt_version_id": "7" * 64, "cost_usd": 0.12},
                {"receipt_version_id": "8" * 64, "cost_usd": 0.03},
            ]
        if "FROM youtube_clip_analyses" in sql:
            return [
                {
                    "artifact_version_id": clip_file.version_id,
                    "model_attempt_id": "a",
                    "receipt_version_id": "7" * 64,
                    "cost_usd": 0.12,
                }
            ]
        if "FROM youtube_merge_analyses" in sql:
            return [{"model_attempt_id": "b", "receipt_version_id": "8" * 64, "cost_usd": 0.03}]
        if "SELECT content_digest FROM artifact_versions" in sql:
            return [{"content_digest": "d" * 64}]
        if "FROM youtube_candidate_decisions" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(catalog_module, "catalog_query", query)
    monkeypatch.setattr(catalog_module, "catalog_batch", lambda rows: batches.append(list(rows)))
    monkeypatch.setattr(
        catalog_module, "publish_immutable_r2_objects", lambda values: published.extend(values)
    )
    current_clip = _accepted(RECORDER).model_copy(
        update={"metadata_version_id": metadata_file.version_id}
    )
    candidate_version_id = publish_youtube_candidate(
        _metadata(RECORDER),
        metadata_file,
        (current_clip,),
        merge_file,
        _merged(),
        "git:test",
        _lease(RECORDER),
    )
    candidate = _published_candidate(published, candidate_version_id)
    assert candidate.receipt_version_ids == ("6" * 64, "7" * 64, "8" * 64)
    assert candidate.cost == MeasuredCost(usd=0.17)
    assert candidate.replay_limitation == LIVE_MEDIA_REPLAY_LIMIT
    decisions = [
        parameters
        for batch in batches
        for sql, parameters in batch
        if "INSERT INTO youtube_candidate_decisions" in sql
    ]
    assert len(decisions) == 1
    assert decisions[0][3] == "automatic_approved"
    assert decisions[0][4] == "system"
    assert candidate_version_id == decisions[0][1]


def _published_candidate(
    published: list[tuple[str, bytes]], candidate_version_id: str
) -> YouTubeCandidate:
    for _key, content in published:
        digest = sha256(content)
        identity = sha256(
            f"news:youtube-candidate:{RECORDER.source_id}:{VIDEO_ID}\0{digest}".encode()
        )
        if identity == candidate_version_id:
            return YouTubeCandidate.model_validate_json(content, strict=True)
    raise AssertionError("Published objects do not contain the candidate artifact")


def test_candidate_accounting_uses_only_the_supplied_clip_generation(monkeypatch) -> None:
    metadata_file, merge_file = _file("metadata"), _file("merge")
    old_file, current_file = _file("old"), _file("current")
    old_digest, current_digest = "e" * 64, "f" * 64
    clip_generations = {
        old_digest: {
            "artifact_version_id": old_file.version_id,
            "model_attempt_id": "old-attempt",
            "receipt_version_id": "7" * 64,
            "cost_usd": 0.02,
        },
        current_digest: {
            "artifact_version_id": current_file.version_id,
            "model_attempt_id": "current-attempt",
            "receipt_version_id": "8" * 64,
            "cost_usd": 0.03,
        },
    }
    receipt_generations = {
        old_digest: {"receipt_version_id": "7" * 64, "cost_usd": 0.02},
        current_digest: {"receipt_version_id": "8" * 64, "cost_usd": 0.03},
    }
    current = _accepted(RECORDER).model_copy(
        update={
            "metadata_version_id": metadata_file.version_id,
            "request_digest": current_digest,
        }
    )

    def query(sql: str, params: list[object] | None = None):
        assert params is not None
        if "FROM youtube_model_receipts receipt" in sql:
            return [receipt_generations[str(params[4])]]
        if "FROM youtube_clip_analyses" in sql:
            return [clip_generations[str(params[-1])]]
        if "FROM youtube_merge_analyses" in sql:
            return [{"model_attempt_id": "merge", "receipt_version_id": "9" * 64, "cost_usd": 0.01}]
        if "SELECT content_digest FROM artifact_versions" in sql:
            return [{"content_digest": "d" * 64}]
        if "FROM youtube_candidate_decisions" in sql:
            return []
        raise AssertionError(sql)

    monkeypatch.setattr(catalog_module, "catalog_query", query)
    published: list[tuple[str, bytes]] = []
    monkeypatch.setattr(
        catalog_module, "publish_immutable_r2_objects", lambda values: published.extend(values)
    )
    monkeypatch.setattr(catalog_module, "catalog_batch", lambda *_args, **_kwargs: None)

    candidate_version_id = publish_youtube_candidate(
        _metadata(RECORDER),
        metadata_file,
        (current,),
        merge_file,
        _merged(),
        "git:test",
        _lease(RECORDER),
    )

    candidate = _published_candidate(published, candidate_version_id)
    assert candidate.clip_version_ids == (current_file.version_id,)
    assert candidate.receipt_version_ids == ("8" * 64,)


def test_candidate_rejects_mixed_clip_generations_before_accounting(monkeypatch) -> None:
    current = _accepted(RECORDER)
    old = current.model_copy(update={"request_digest": "f" * 64})
    monkeypatch.setattr(
        catalog_module,
        "catalog_query",
        lambda *_args: pytest.fail("mixed generations reached accounting"),
    )
    with pytest.raises(ValueError, match="one analysis generation"):
        publish_youtube_candidate(
            _metadata(RECORDER),
            _file("metadata"),
            (current, old),
            _file("merge"),
            _merged(),
            "git:test",
            _lease(RECORDER),
        )


def test_publication_without_a_decided_candidate_publishes_nothing(monkeypatch) -> None:
    monkeypatch.setattr(catalog_module, "catalog_query", lambda *_args: [])
    assert publish_next_approved_candidate("git:test") is None


def test_system_decision_reuse_keeps_one_decision_per_candidate(monkeypatch) -> None:
    decided_at = datetime.fromisoformat("2026-09-09T08:00:00+00:00")
    row = {
        "kind": "automatic_approved",
        "source_id": STAREA.source_id,
        "actor": "system",
        "decided_at": "2026-09-09T08:00:00+00:00",
    }
    monkeypatch.setattr(
        catalog_module,
        "catalog_query",
        lambda *_args, **_kwargs: [row],
    )
    monkeypatch.setattr(
        catalog_module,
        "publish_immutable_r2_objects",
        lambda _values: pytest.fail("existing decision was republished"),
    )
    monkeypatch.setattr(
        catalog_module,
        "catalog_batch",
        lambda _values: pytest.fail("existing decision was rewritten"),
    )
    decision = catalog_module._record_decision("c" * 64, STAREA)
    assert decision == SystemApproval(
        candidate_version_id="c" * 64,
        source_id=STAREA.source_id,
        decided_at=decided_at,
    )


def test_publication_survives_lease_expiry(monkeypatch) -> None:
    candidate = _candidate(STAREA)

    def query(sql: str, _params=None):
        if "LEFT JOIN youtube_candidate_decisions" in sql:
            return []
        if "FROM youtube_publications publication" in sql:
            return []
        if "FROM youtube_candidates candidate" in sql:
            return [{"candidate_version_id": "c" * 64, "decision_version_id": "e" * 64}]
        if "SELECT content_digest FROM artifact_versions" in sql:
            return [{"content_digest": "d" * 64}]
        raise AssertionError(sql)

    monkeypatch.setattr(catalog_module, "catalog_query", query)
    monkeypatch.setattr(catalog_module, "read_youtube_candidate", lambda _version: candidate)
    monkeypatch.setattr(catalog_module, "publish_immutable_r2_objects", lambda _objects: None)
    monkeypatch.setattr(catalog_module, "catalog_batch", lambda _values: None)
    result = publish_next_approved_candidate("git:restart")
    assert result is not None
    assert result.source_id == STAREA.source_id
    assert result.article_version_id


def test_unknown_cost_is_explicit_not_zero() -> None:
    candidate = _candidate(STAREA).model_copy(
        update={"cost": UnmeasuredCost(reason="Provider omitted billing data")}
    )
    assert candidate.cost.kind == "unmeasured"
    assert "0" not in candidate.cost.model_dump_json()


def test_analysis_requires_gap_free_clips_and_bounded_evidence() -> None:
    assert plan_clip_ranges(601) == (
        ClipRange(start_second=0, duration_seconds=300),
        ClipRange(start_second=300, duration_seconds=300),
        ClipRange(start_second=600, duration_seconds=1),
    )
    assert complete_clip_coverage(300, (_accepted(RECORDER),)) == (_accepted(RECORDER),)
    payload = _clip_payload()
    cast(list[dict[str, object]], payload["evidence"])[0]["end_seconds"] = 301
    with pytest.raises(ValueError, match="escapes"):
        parse_clip_extraction(json.dumps(payload), ClipRange(start_second=0, duration_seconds=300))


def test_accepted_clip_publication_keeps_receipt_attempt_run_and_checkpoint_atomic(
    monkeypatch,
) -> None:
    metadata_file = _file("metadata")
    accepted = _accepted(RECORDER).model_copy(
        update={
            "metadata_version_id": metadata_file.version_id,
            "request_digest": clip_request_digest(),
        }
    )
    metadata = _metadata(RECORDER)
    request_id = youtube_clip_request_id(metadata, accepted.clip_range)
    receipt, _response, attempt = _receipt_response_and_attempt(
        "news.youtube.extract_clip", request_id
    )
    batches = []
    monkeypatch.setattr(
        catalog_module,
        "catalog_query",
        lambda sql, _params=None: [{"content_digest": "d" * 64}]
        if "SELECT content_digest FROM artifact_versions" in sql
        else pytest.fail(sql),
    )
    monkeypatch.setattr(catalog_module, "publish_immutable_r2_objects", lambda _objects: None)
    monkeypatch.setattr(
        catalog_module,
        "catalog_batch",
        lambda statements, **kwargs: batches.append((statements, kwargs)),
    )
    monkeypatch.setattr(catalog_module, "read_accepted_clip_checkpoint", lambda *_args: accepted)

    assert (
        publish_accepted_clip(
            metadata,
            accepted,
            metadata_file,
            receipt,
            attempt,
            "git:test",
        )
        == accepted
    )

    assert len(batches) == 1


def test_accepted_clip_publication_rejects_cross_linked_model_records() -> None:
    metadata = _metadata(RECORDER)
    metadata_file = _file("metadata")
    accepted = _accepted(RECORDER).model_copy(
        update={
            "metadata_version_id": metadata_file.version_id,
            "request_digest": clip_request_digest(),
        }
    )
    request_id = youtube_clip_request_id(metadata, accepted.clip_range)
    receipt, _response, attempt = _receipt_response_and_attempt(
        "news.youtube.extract_clip", request_id
    )

    with pytest.raises(ValueError, match="metadata"):
        publish_accepted_clip(
            metadata.model_copy(update={"video_id": "lmnopqrstuv"}),
            accepted,
            metadata_file,
            receipt,
            attempt,
            "git:test",
        )
    with pytest.raises(ValueError, match="model receipt"):
        publish_accepted_clip(
            metadata,
            accepted,
            metadata_file,
            receipt,
            attempt.model_copy(update={"operation_key": "news.youtube.merge_video"}),
            "git:test",
        )


def test_accepted_merge_publication_keeps_receipt_attempt_run_and_checkpoint_atomic(
    monkeypatch,
) -> None:
    accepted = _accepted(RECORDER)
    request_digest = "d" * 64
    clip_version_ids = ("c" * 64,)
    request_id = sha256(
        canonical_json(
            {
                "source_id": RECORDER.source_id,
                "clip_version_ids": clip_version_ids,
                "request_digest": request_digest,
                "model": YOUTUBE_MODEL,
            }
        )
    )
    receipt, response, attempt = _receipt_response_and_attempt(
        "news.youtube.merge_video", request_id
    )
    batches = []
    monkeypatch.setattr(
        catalog_module,
        "catalog_query",
        lambda sql, _params=None: [{"content_digest": "d" * 64}]
        if "SELECT content_digest FROM artifact_versions" in sql
        else pytest.fail(sql),
    )
    monkeypatch.setattr(catalog_module, "publish_immutable_r2_objects", lambda _objects: None)
    monkeypatch.setattr(
        catalog_module,
        "catalog_batch",
        lambda statements, **kwargs: batches.append((statements, kwargs)),
    )

    article, _file_value = publish_accepted_merge(
        _metadata(RECORDER),
        (accepted,),
        clip_version_ids,
        _merged(),
        request_id,
        request_digest,
        response,
        receipt,
        attempt,
        "git:test",
    )

    assert article == _merged()
    assert len(batches) == 1


def test_accepted_merge_publication_rejects_cross_linked_response() -> None:
    accepted = _accepted(RECORDER)
    request_digest = "d" * 64
    clip_version_ids = ("c" * 64,)
    request_id = sha256(
        canonical_json(
            {
                "source_id": RECORDER.source_id,
                "clip_version_ids": clip_version_ids,
                "request_digest": request_digest,
                "model": YOUTUBE_MODEL,
            }
        )
    )
    receipt, response, attempt = _receipt_response_and_attempt(
        "news.youtube.merge_video", request_id
    )

    with pytest.raises(ValueError, match="does not match its receipt"):
        publish_accepted_merge(
            _metadata(RECORDER),
            (accepted,),
            clip_version_ids,
            _merged(),
            request_id,
            request_digest,
            response.model_copy(update={"payload": {"responseId": "other"}}),
            receipt,
            attempt,
            "git:test",
        )


def test_rejected_receipt_keeps_attempt_trace_and_transition_atomic(monkeypatch) -> None:
    receipt, _response, attempt = _receipt_response_and_attempt("news.youtube.extract_clip")
    batches = []
    monkeypatch.setattr(
        catalog_module,
        "catalog_batch",
        lambda statements, **kwargs: batches.append((statements, kwargs)),
    )

    reject_receipt(
        receipt,
        attempt.model_copy(update={"status": "rejected", "error": "bad"}),
        PUBLISHED_AT,
        "bad",
    )

    assert len(batches) == 1


def _receipt_response_and_attempt(
    operation_key: str,
    request_id: Sha256 = "a" * 64,
) -> tuple[YouTubeModelReceipt, GeminiResponse, ModelAttempt]:
    response = GeminiResponse(
        response_id=sha256(b"{}"),
        model=YOUTUBE_MODEL,
        payload={},
        input_tokens=10,
        output_tokens=5,
        thought_tokens=0,
        cost_usd=0.01,
        rate_card_version="test",
        latency_ms=20,
    )
    receipt = YouTubeModelReceipt(
        request_id=request_id,
        operation_key=operation_key,
        attempt_index=0,
        response=GeminiHttpResponse(
            requested_model=YOUTUBE_MODEL,
            status_code=200,
            body_base64="e30=",
            latency_ms=20,
        ),
        trace=ModelTraceReference(
            provider="langfuse",
            trace_id="trace",
            observation_id="observation",
            project_ref="project",
            recorded_at=PUBLISHED_AT,
        ),
        received_at=PUBLISHED_AT,
    )
    attempt = model_attempt_from_payload(
        response.attempt_payload(),
        request_id=receipt.request_id,
        operation_key=operation_key,
        attempt_index=0,
        latency_ms=response.latency_ms,
        status="accepted",
        error=None,
        observed_at=PUBLISHED_AT,
    )
    return receipt, response, attempt


def _feed_response(source: YouTubeSource) -> bytes:
    return json.dumps(
        {
            "kind": "youtube#playlistItemListResponse",
            "items": [
                {
                    "snippet": {
                        "channelId": source.channel_id,
                        "title": source.display_name,
                        "resourceId": {"videoId": VIDEO_ID},
                    },
                    "contentDetails": {"videoPublishedAt": "2026-09-09T08:00:00Z"},
                }
            ],
        }
    ).encode()


def _feed(source: YouTubeSource, *video_ids: str) -> YouTubeFeed:
    return YouTubeFeed(
        source=source,
        entries=tuple(
            YouTubeFeedEntry(
                source=source,
                video_id=YouTubeVideoId(value),
                title=value,
                published_at=PUBLISHED_AT,
            )
            for value in video_ids
        ),
    )


def _poll(source: YouTubeSource) -> YouTubePollCapture:
    return YouTubePollCapture(
        source=source,
        scheduled_at=PUBLISHED_AT,
        observed_at=PUBLISHED_AT,
        content=_feed_response(source),
        feed=_feed(source, VIDEO_ID),
    )


def _metadata(source: YouTubeSource) -> YouTubeVideoMetadata:
    return YouTubeVideoMetadata(
        source=source,
        video_id=VIDEO_ID,
        url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        title=source.display_name,
        published_at=PUBLISHED_AT,
        duration_seconds=300,
        feed_snapshot_version_id="b" * 64,
        retrieved_at=PUBLISHED_AT,
        provider_response={"items": []},
        raw_provider_response=b"{}",
    )


def _accepted(source: YouTubeSource) -> AcceptedClipExtraction:
    return AcceptedClipExtraction(
        source=source,
        video_id=VIDEO_ID,
        metadata_version_id="b" * 64,
        video_url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        clip_range=ClipRange(start_second=0, duration_seconds=300),
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
                    attribution=source.display_name,
                    start_second=1,
                    duration_seconds=2,
                ),
            ),
            source_stated_uncertainties=("Limită",),
        ),
        provider_response={},
    )


def _merged() -> YouTubeMergedArticle:
    return YouTubeMergedArticle(
        title="Titlu",
        standfirst="Introducere",
        narrative="Narațiune",
        covered_clip_starts=(0,),
        omitted_clips=(),
    )


def _candidate(source: YouTubeSource) -> YouTubeCandidate:
    return YouTubeCandidate(
        source=source,
        video_id=VIDEO_ID,
        video_url=HttpUrl(f"https://www.youtube.com/watch?v={VIDEO_ID}"),
        video_title=source.display_name,
        published_at=PUBLISHED_AT,
        feed_snapshot_version_id="1" * 64,
        metadata_version_id="2" * 64,
        clip_version_ids=("3" * 64,),
        merge_version_id="4" * 64,
        receipt_version_ids=("5" * 64,),
        article=_merged(),
        evidence=(
            CandidateEvidence(
                clip_version_id="3" * 64,
                kind="reported_fact",
                text="Fapt",
                attribution=source.display_name,
                start_second=1,
                duration_seconds=2,
            ),
        ),
        uncertainties=("Limită",),
        omissions=(),
        implementation_ref="git:test",
        model=YOUTUBE_MODEL,
        cost=MeasuredCost(usd=0.15),
    )


def _lease(source: YouTubeSource) -> YouTubeVideoLease:
    return YouTubeVideoLease(
        source=source,
        video_id=VIDEO_ID,
        owner_token="owner",
        lease_expires_at=PUBLISHED_AT + timedelta(minutes=10),
        first_poll_version_id="b" * 64,
        deterministic_failure_fingerprint=None,
        unchanged_deterministic_failures=0,
    )


def _file(label: str):
    return artifact_file(
        artifact_id=f"test:{label}",
        artifact_kind=f"test_{label}",
        title=label,
        content=label.encode(),
        r2_key=f"test/{label}.json",
        media_type="application/json",
    )


def _youtube_response(channel_id: str) -> bytes:
    return json.dumps(
        {
            "kind": "youtube#videoListResponse",
            "items": [
                {
                    "id": VIDEO_ID,
                    "snippet": {
                        "publishedAt": "2026-09-09T08:00:00Z",
                        "channelId": channel_id,
                        "title": "Video",
                    },
                    "contentDetails": {"duration": "PT5M"},
                }
            ],
        }
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
