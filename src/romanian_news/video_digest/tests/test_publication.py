import json
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from botocore.client import BaseClient
from pydantic import ValidationError

from romanian_news.storage import (
    PublicObjectConflict,
    PublicObjectUnavailable,
    PublicObjectVerification,
)
from romanian_news.video_digest import publication
from romanian_news.video_digest.models import (
    EditionId,
    PublicationAttemptReady,
    PublicationAttemptWaiting,
    PublicationCheckpointSuperseded,
    PublicationState,
    PublicationStatus,
    SlotId,
    SlotLease,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionWaiting,
    FailedPublicationSubtitles,
    MediaArtifactReference,
    PublicationHandoff,
    PublishAction,
)

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
EDITION_ID = EditionId("1" * 64)
VIDEO = b"video"
VIDEO_DIGEST = "2" * 64
VIDEO_VERSION = "3" * 64
LEASE = SlotLease(
    slot_id=SlotId("4" * 64),
    edition_id=EDITION_ID,
    owner_token="owner",
    expires_at=NOW + timedelta(minutes=10),
    claim_count=1,
)
HANDOFF = PublicationHandoff(
    slot_id=LEASE.slot_id,
    edition_id=EDITION_ID,
    slot_name="morning",
    scheduled_at=NOW,
    video=MediaArtifactReference(
        artifact_id="assembled",
        version_id=VIDEO_VERSION,
        content_digest=VIDEO_DIGEST,
        r2_key="private/assembled.mp4",
        byte_size=len(VIDEO),
        media_type="video/mp4",
    ),
    subtitles=FailedPublicationSubtitles(),
)
ACTION = PublishAction(lease=LEASE, handoff=HANDOFF)
CONFIG = publication.PublicMediaConfiguration(
    private_bucket="private",
    public_bucket="public",
    base_url="https://media.example.com",
)


@pytest.mark.parametrize(
    "values",
    (
        {"private_bucket": "", "public_bucket": "public", "base_url": "https://media.example.com"},
        {
            "private_bucket": "same",
            "public_bucket": "same",
            "base_url": "https://media.example.com",
        },
        {
            "private_bucket": "private",
            "public_bucket": "public",
            "base_url": "http://media.example.com",
        },
        {
            "private_bucket": "private",
            "public_bucket": "public",
            "base_url": "https://media.example.com/path",
        },
        {
            "private_bucket": "private",
            "public_bucket": "public",
            "base_url": "https://media.example.com/",
        },
    ),
)
def test_public_media_configuration_rejects_unsafe_boundaries(
    values: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        publication.PublicMediaConfiguration.model_validate(values)


def test_publication_intent_is_content_addressed_and_excludes_location() -> None:
    intent = publication.publication_intent_from_handoff(HANDOFF)

    assert intent.edition_id in intent.expected_video_key
    assert intent.video_digest in intent.expected_video_key
    assert intent.expected_video_key.endswith(".mp4")
    assert intent.cache_control == "public,max-age=31536000,immutable"
    assert intent.visibility == "public"
    assert intent.retention == "permanent"
    assert "bucket" not in intent.model_dump()
    assert "base_url" not in intent.model_dump()


def _configure_port(
    monkeypatch: pytest.MonkeyPatch,
    stage: PublicationState,
    *,
    attempt_index: int = 0,
) -> tuple[publication.R2PublicationPort, list[str], list[bytes]]:
    actions: list[str] = []
    evidence: list[bytes] = []

    def record(_lease: SlotLease, intent, *, recorded_at: datetime) -> PublicationStatus:
        actions.append("intent")
        assert recorded_at == NOW
        return PublicationStatus(
            publication_id=intent.publication_id,
            edition_id=intent.edition_id,
            stage=stage,
        )

    def begin(_lease: SlotLease, _publication_id, *, recorded_at: datetime):
        actions.append("attempt")
        assert recorded_at == NOW
        return PublicationAttemptReady(attempt_index=attempt_index)

    def checkpoint(
        _lease,
        publication_id,
        _attempt_index,
        progress,
        *,
        evidence_file=None,
        recorded_at,
    ):
        actions.append(progress.kind)
        return PublicationStatus(
            publication_id=publication_id,
            edition_id=EDITION_ID,
            stage=PublicationState(progress.kind),
        )

    def publish_evidence(objects, *, client, bucket) -> None:
        assert client is port_client
        assert bucket == "private"
        for _key, content in objects:
            evidence.append(content)

    def read_private(_key, _digest, *, client, bucket) -> bytes:
        assert client is port_client
        assert bucket == "private"
        return VIDEO

    def publish_public(client, bucket, _value) -> None:
        assert client is port_client
        assert bucket == "public"
        actions.append("r2")

    port_client = cast(BaseClient, object())

    monkeypatch.setattr(publication, "record_publication_intent", record)
    monkeypatch.setattr(publication, "begin_publication_attempt", begin)
    monkeypatch.setattr(publication, "checkpoint_publication_progress", checkpoint)
    monkeypatch.setattr(
        publication,
        "read_verified_r2_object",
        read_private,
    )
    monkeypatch.setattr(publication, "publish_public_r2_object", publish_public)
    monkeypatch.setattr(publication, "publish_immutable_r2_objects", publish_evidence)
    monkeypatch.setattr(
        publication,
        "verify_public_object_url",
        lambda _url, value, **_kwargs: PublicObjectVerification(
            key=value.key,
            content_digest=value.content_digest,
            byte_size=value.byte_size,
            content_type=value.content_type,
            cache_control=value.cache_control,
            visibility=value.visibility,
            retention=value.retention,
            source_lineage=value.source_lineage,
            validator='"validator"',
        ),
    )
    monkeypatch.setattr(
        publication,
        "complete_publication",
        lambda *_args, **_kwargs: actions.append("complete"),
    )
    return (
        publication.R2PublicationPort(
            CONFIG,
            client=port_client,
            now=lambda: NOW,
        ),
        actions,
        evidence,
    )


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        (
            PublicationState.PENDING,
            [
                "intent",
                "attempt",
                "uploading",
                "r2",
                "uploaded",
                "r2",
                "verified",
                "complete",
            ],
        ),
        (
            PublicationState.UPLOADED,
            ["intent", "attempt", "r2", "verified", "complete"],
        ),
        (
            PublicationState.VERIFIED,
            ["intent", "attempt", "complete"],
        ),
    ),
)
def test_publication_replays_from_each_durable_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    stage: PublicationState,
    expected: list[str],
) -> None:
    port, actions, evidence = _configure_port(monkeypatch, stage)

    assert port.execute(ACTION) == ActionAdvanced()

    assert actions == expected
    for content in evidence:
        body = json.loads(content)
        assert "uploaded" not in body
        assert "adopted" not in body


def test_successful_completion_replay_does_not_start_an_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, actions, _evidence = _configure_port(monkeypatch, PublicationState.PUBLISHED)

    assert port.execute(ACTION) == ActionAdvanced()
    assert actions == ["intent"]


def test_conflict_terminalizes_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, actions, _evidence = _configure_port(monkeypatch, PublicationState.UPLOADING)
    failures: list[tuple[int, PublicationState]] = []
    monkeypatch.setattr(
        publication,
        "publish_public_r2_object",
        lambda *_args: (_ for _ in ()).throw(PublicObjectConflict("conflict")),
    )
    monkeypatch.setattr(
        publication,
        "fail_publication",
        lambda _lease, _publication_id, attempt_index, *, state, **_kwargs: failures.append(
            (attempt_index, state)
        ),
    )

    assert port.execute(ACTION) == ActionAdvanced()

    assert failures == [(0, PublicationState.CONFLICT)]
    assert "complete" not in actions


@pytest.mark.parametrize("attempt_index", range(5))
def test_transient_failure_uses_four_retries_then_terminalizes_fifth(
    monkeypatch: pytest.MonkeyPatch,
    attempt_index: int,
) -> None:
    port, _actions, _evidence = _configure_port(
        monkeypatch,
        PublicationState.UPLOADING,
        attempt_index=attempt_index,
    )
    retries: list[int] = []
    failures: list[int] = []
    monkeypatch.setattr(
        publication,
        "publish_public_r2_object",
        lambda *_args: (_ for _ in ()).throw(PublicObjectUnavailable("offline")),
    )
    monkeypatch.setattr(
        publication,
        "checkpoint_publication_retry",
        lambda _lease, _publication_id, index, **_kwargs: (
            retries.append(index)
            or PublicationAttemptWaiting(
                attempt_index=index,
                retry_at=NOW + timedelta(seconds=60 * (2**index)),
            )
        ),
    )
    monkeypatch.setattr(
        publication,
        "fail_publication",
        lambda _lease, _publication_id, index, **_kwargs: failures.append(index),
    )

    outcome = port.execute(ACTION)

    if attempt_index < 4:
        assert isinstance(outcome, ActionWaiting)
        assert retries == [attempt_index]
        assert failures == []
    else:
        assert outcome == ActionAdvanced()
        assert retries == []
        assert failures == [4]


def test_waiting_for_480_second_retry_returns_current_cap_without_new_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = publication.publication_intent_from_handoff(HANDOFF)
    begin_calls = 0

    monkeypatch.setattr(
        publication,
        "record_publication_intent",
        lambda *_args, **_kwargs: PublicationStatus(
            publication_id=intent.publication_id,
            edition_id=intent.edition_id,
            stage=PublicationState.UPLOADING,
        ),
    )

    def begin(*_args, **_kwargs):
        nonlocal begin_calls
        begin_calls += 1
        return PublicationAttemptWaiting(
            attempt_index=3,
            retry_at=NOW + timedelta(seconds=480),
        )

    monkeypatch.setattr(publication, "begin_publication_attempt", begin)
    port = publication.R2PublicationPort(
        CONFIG,
        client=cast(BaseClient, object()),
        now=lambda: NOW,
    )

    outcome = port.execute(ACTION)

    assert outcome == ActionWaiting(
        reason="public media publication retry is not ready",
        retry_after_seconds=300,
    )
    assert begin_calls == 1


def test_superseded_attempt_preserves_the_catalog_retry_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, _actions, _evidence = _configure_port(monkeypatch, PublicationState.UPLOADING)
    intent = publication.publication_intent_from_handoff(HANDOFF)
    monkeypatch.setattr(
        publication,
        "checkpoint_publication_progress",
        lambda *_args, **_kwargs: PublicationCheckpointSuperseded(
            status=PublicationStatus(
                publication_id=intent.publication_id,
                edition_id=intent.edition_id,
                stage=PublicationState.UPLOADING,
            ),
            retry_at=NOW + timedelta(seconds=60),
        ),
    )

    assert port.execute(ACTION) == ActionWaiting(
        reason="public media publication retry is not ready",
        retry_after_seconds=60,
    )
