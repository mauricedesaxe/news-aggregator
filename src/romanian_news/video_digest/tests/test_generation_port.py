from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from botocore.exceptions import ClientError

from romanian_news import storage
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import sha256
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import (
    checkpoint_edition_verification,
    checkpoint_generation_request,
    checkpoint_generation_response,
    checkpoint_generation_submission,
    read_generation_attempts,
    read_slot_resume_state,
)
from romanian_news.storage import read_verified_r2_object
from romanian_news.video_digest import generation, generation_port, media, preflight, references
from romanian_news.video_digest.generation import (
    CandidateReady,
    CandidateReference,
    FalH3Provider,
    GenerationComplete,
    GenerationFailed,
    GenerationInProgress,
    GenerationRetryAvailable,
    H3ReferencePack,
)
from romanian_news.video_digest.models import (
    EditionId,
    EstimatedAttemptCost,
    GenerationStage,
    ScheduledSlot,
    SlotFailureReason,
    SlotId,
    SlotLease,
    SlotName,
    UnknownAttemptCost,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionWaiting,
    FailedResume,
    GenerateAction,
)
from romanian_news.video_digest.planning import VerifiedDigestPlan, authorize_generation
from romanian_news.video_digest.tests.test_generation import _prepared, _references
from romanian_news.video_digest.tests.test_media import _probe_payload
from tests.contracts.postgres_video_digest_generation import _GenerationPipeline
from tests.postgres_catalog import isolated_postgres_schema
from tests.worker.conftest import FakeR2Client

requires_postgres = pytest.mark.skipif(
    not os.getenv("NEWS_TEST_POSTGRES_DSN"),
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


def _action() -> GenerateAction:
    return GenerateAction(
        lease=SlotLease(
            slot_id=SlotId("1" * 64),
            edition_id=EditionId("2" * 64),
            owner_token="owner",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            claim_count=1,
        )
    )


def _pack() -> H3ReferencePack:
    video = ArtifactReference(
        artifact_id="video",
        version_id="3" * 64,
        content_digest="4" * 64,
        r2_key="references/video",
    )
    audio = ArtifactReference(
        artifact_id="audio",
        version_id="5" * 64,
        content_digest="6" * 64,
        r2_key="references/audio",
    )
    return H3ReferencePack(videos=(video,), audio=(audio,))


class _MediaAwareR2Client(FakeR2Client):
    def __init__(self) -> None:
        super().__init__()
        self.content_types: dict[str, str] = {}
        self.cache_controls: dict[str, str] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str],
        ContentType: str = "application/octet-stream",
        CacheControl: str = "private,no-store",
        **options: object,
    ) -> None:
        super().put_object(Bucket=Bucket, Key=Key, Body=Body, Metadata=Metadata)
        self.content_types[Key] = ContentType
        self.cache_controls[Key] = CacheControl

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        content = self.objects[Key]
        return {
            "ContentLength": len(content),
            "ContentType": self.content_types.get(Key, "application/octet-stream"),
            "CacheControl": self.cache_controls.get(Key, "private,no-store"),
            "Metadata": dict(self.metadata[Key]),
            "ETag": f'"{sha256(content)}"',
        }


@pytest.fixture
def generation_port_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with isolated_postgres_schema(monkeypatch, "generation_port_contract"):
        ensure_news_catalog_schema()
        yield


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> _MediaAwareR2Client:
    client = _MediaAwareR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    return client


@pytest.fixture
def media_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def process(
        arguments: tuple[str, ...], *, capture_output: bool
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        output = json.dumps(_probe_payload(duration="15.0")) if arguments[0] == "ffprobe" else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(media, "_run", process)
    monkeypatch.setattr(media, "clip_narration_matches", lambda _path, _text: True)


def _unused_provider() -> FalH3Provider:
    return cast(FalH3Provider, object())


def _configured(monkeypatch: pytest.MonkeyPatch, outcome: object) -> None:
    monkeypatch.setattr(generation_port, "read_h3_reference_pack", lambda _pack_id: _pack())
    monkeypatch.setattr(
        generation_port,
        "read_prepared_paid_generation",
        lambda _lease: SimpleNamespace(
            plan=SimpleNamespace(stories=("story",)),
            verified_plan=SimpleNamespace(
                plan=SimpleNamespace(stories=(SimpleNamespace(narration="English narration"),))
            ),
        ),
    )
    monkeypatch.setattr(generation_port, "read_generation_attempts", lambda _edition: ())
    monkeypatch.setattr(generation_port, "generate_next_candidate", lambda *_args, **_kw: outcome)


def _import_reference_pack(tmp_path: Path) -> tuple[str, H3ReferencePack]:
    video = tmp_path / "host.mp4"
    video.write_bytes(b"\0\0\0\x18ftypisomvideo")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF\x08\0\0\0WAVEaudio")
    pack_id = references.import_h3_reference_pack(
        (video,), (audio,), approval_ref="generation-port-approved-media"
    )
    return pack_id, references.read_h3_reference_pack(pack_id)


class _ProcessingEdition:
    def __init__(
        self,
        fake_r2: _MediaAwareR2Client,
        tmp_path: Path,
        *,
        seed: int,
    ) -> None:
        self.pipeline = _GenerationPipeline(seed=seed)
        self.pack_id, pack = _import_reference_pack(tmp_path)
        plan, plan_file = self.pipeline.build_plan()
        self.pipeline.checkpoint_plan()
        fake_r2.objects[plan_file.r2_key] = plan_file.content
        self.pipeline.verify_story(0)
        verified = VerifiedDigestPlan.model_validate_json(plan_file.content, strict=True)
        authorization = preflight._authorization_manifest_file(authorize_generation(verified))
        fake_r2.objects[authorization.r2_key] = authorization.content
        checkpoint_edition_verification(
            self.pipeline.lease, manifest_file=authorization, recorded_at=self._now()
        )
        prepared = preflight.read_prepared_paid_generation(self.pipeline.lease)
        _request, request_file, identity = generation._generation_request(
            prepared, pack, generation.PRODUCTION_GENERATION_POLICY, 0, 0
        )
        fake_r2.objects[request_file.r2_key] = request_file.content
        checkpoint_generation_request(
            self.pipeline.lease,
            identity,
            request_file=request_file,
            admission=self.pipeline.generation_admission(),
            recorded_at=self._now(),
        )
        receipt = self.pipeline._file(
            f"fal-{identity.request_id}",
            "video_digest_provider_receipt",
            title="Provider receipt 0/0",
        )
        fake_r2.objects[receipt.r2_key] = receipt.content
        checkpoint_generation_submission(
            self.pipeline.lease,
            identity.request_id,
            provider_receipt_id=f"fal-{identity.request_id}",
            receipt_file=receipt,
            cost=EstimatedAttemptCost(usd=Decimal("1.2500")),
            recorded_at=self._now(),
        )
        self.response = self.pipeline._file(
            f"{identity.request_id}:response",
            "video_digest_generation_response",
            title="Generation response 0/0",
        )
        fake_r2.objects[self.response.r2_key] = self.response.content
        checkpoint_generation_response(
            self.pipeline.lease,
            identity.request_id,
            response_file=self.response,
            recorded_at=self._now(),
        )
        self.identity = identity
        self.plan = plan

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def candidate(self, clip: bytes) -> CandidateReady:
        return CandidateReady(
            request_id=self.identity.request_id,
            story_id=self.plan.stories[0].story_id,
            story_position=0,
            attempt_index=0,
            response_artifact_version_id=self.response.version_id,
            candidate=CandidateReference(
                r2_key=f"news/video-digest/{self.pipeline.edition.edition_id}/candidates/fixture.mp4",
                content_digest=sha256(clip),
                byte_size=len(clip),
            ),
            cost=UnknownAttemptCost(reason="provider did not report cost"),
        )


def test_missing_configuration_fails_before_provider_or_catalog_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_port, "NEWS_H3_REFERENCE_PACK_ID", None)
    monkeypatch.setattr(
        generation_port,
        "read_h3_reference_pack",
        lambda _pack_id: pytest.fail("missing pack configuration reached the catalog"),
    )
    with pytest.raises(ValueError, match="NEWS_H3_REFERENCE_PACK_ID"):
        generation_port.ProductionH3GenerationPort().execute(_action())


def test_provider_credentials_are_checked_before_pack_or_request_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generation_port,
        "FalH3Client",
        lambda: (_ for _ in ()).throw(ValueError("FAL_KEY is required")),
    )
    monkeypatch.setattr(
        generation_port,
        "read_h3_reference_pack",
        lambda _pack_id: pytest.fail("missing Fal credentials reached the catalog"),
    )
    with pytest.raises(ValueError, match="FAL_KEY"):
        generation_port.ProductionH3GenerationPort(pack_id="7" * 64).execute(_action())


@requires_postgres
def test_generation_port_records_the_accepted_clip_in_the_durable_catalog(
    generation_port_schema: None,
    fake_r2: _MediaAwareR2Client,
    media_process: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del generation_port_schema, media_process
    edition = _ProcessingEdition(fake_r2, tmp_path, seed=51)
    clip = b"fixture-generated-clip"
    fake_r2.objects[edition.candidate(clip).candidate.r2_key] = clip
    monkeypatch.setattr(
        generation_port,
        "generate_next_candidate",
        lambda *_args, **_kwargs: edition.candidate(clip),
    )

    outcome = generation_port.ProductionH3GenerationPort(
        provider=_unused_provider(), pack_id=edition.pack_id
    ).execute(GenerateAction(lease=edition.pipeline.lease))

    assert outcome == ActionAdvanced(durable_progress=True)
    attempts = read_generation_attempts(edition.pipeline.edition.edition_id)
    assert [attempt.stage for attempt in attempts] == [GenerationStage.ACCEPTED]
    accepted_clip = attempts[0].accepted_clip
    assert accepted_clip is not None
    assert accepted_clip.content_digest == sha256(clip)
    assert accepted_clip.byte_size == len(clip)
    assert read_verified_r2_object(accepted_clip.r2_key, accepted_clip.content_digest) == clip
    validation = attempts[0].validation_evidence
    assert validation is not None
    evidence = json.loads(read_verified_r2_object(validation.r2_key, validation.content_digest))
    assert evidence["request_id"] == str(edition.identity.request_id)
    assert evidence["candidate_r2_key"] == edition.candidate(clip).candidate.r2_key
    assert evidence["probe"]["duration_ms"] == 15000
    assert read_slot_resume_state(edition.pipeline.slot.slot_id).kind == "generation"


@requires_postgres
def test_generation_port_waits_for_fal_and_advances_to_assembly_when_complete(
    generation_port_schema: None,
    fake_r2: _MediaAwareR2Client,
    media_process: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del generation_port_schema, media_process
    edition = _ProcessingEdition(fake_r2, tmp_path, seed=52)
    clip = b"fixture-completing-clip"
    fake_r2.objects[edition.candidate(clip).candidate.r2_key] = clip
    waiting = GenerationInProgress(
        request_id=edition.identity.request_id,
        provider_receipt_id=f"fal-{edition.identity.request_id}",
        provider_status="IN_QUEUE",
    )
    accepted = edition.candidate(clip)
    complete = GenerationComplete(edition_id=edition.pipeline.edition.edition_id)
    monkeypatch.setattr(generation_port, "generate_next_candidate", lambda *_a, **_k: waiting)
    port = generation_port.ProductionH3GenerationPort(
        provider=_unused_provider(), pack_id=edition.pack_id
    )

    assert port.execute(GenerateAction(lease=edition.pipeline.lease)) == ActionWaiting(
        reason="Fal H3 request IN_QUEUE", retry_after_seconds=60
    )
    assert read_generation_attempts(edition.pipeline.edition.edition_id)[0].stage is (
        GenerationStage.PROCESSING
    )

    monkeypatch.setattr(generation_port, "generate_next_candidate", lambda *_a, **_k: accepted)
    assert port.execute(GenerateAction(lease=edition.pipeline.lease)) == ActionAdvanced(
        durable_progress=True
    )
    assert read_generation_attempts(edition.pipeline.edition.edition_id)[0].stage is (
        GenerationStage.ACCEPTED
    )

    monkeypatch.setattr(generation_port, "generate_next_candidate", lambda *_a, **_k: complete)
    assert port.execute(GenerateAction(lease=edition.pipeline.lease)) == ActionAdvanced()
    assert read_slot_resume_state(edition.pipeline.slot.slot_id).kind == "assembly"


def test_terminal_generation_failure_requires_failed_catalog_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action()
    _configured(
        monkeypatch,
        GenerationFailed(edition_id=action.lease.edition_id, reason="both attempts failed"),
    )
    scheduled_at = datetime(2099, 9, 20, 5, tzinfo=UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, scheduled_at),
        name=SlotName.MORNING,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.date(),
    )
    monkeypatch.setattr(
        generation_port,
        "read_slot_resume_state",
        lambda _slot_id: FailedResume(slot=slot, reason=SlotFailureReason.TERMINAL_FAILURE),
    )
    port = generation_port.ProductionH3GenerationPort(provider=_unused_provider(), pack_id="7" * 64)
    assert isinstance(port.execute(action), ActionAdvanced)

    monkeypatch.setattr(generation_port, "read_slot_resume_state", lambda _slot_id: object())
    with pytest.raises(ValueError, match="did not fail its slot"):
        port.execute(action)


def test_retry_available_advances_from_durable_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action()
    _configured(
        monkeypatch,
        GenerationRetryAvailable(
            edition_id=action.lease.edition_id,
            story_position=0,
            reason="technical validation failed",
        ),
    )
    assert isinstance(
        generation_port.ProductionH3GenerationPort(
            provider=_unused_provider(), pack_id="7" * 64
        ).execute(action),
        ActionAdvanced,
    )


def test_generation_port_rejects_reference_pack_drift_from_prior_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    old_pack = _references()
    _request, file, _identity = generation._generation_request(
        prepared, old_pack, generation.PRODUCTION_GENERATION_POLICY, 0, 0
    )
    monkeypatch.setattr(
        generation_port,
        "read_generation_attempts",
        lambda _edition: (SimpleNamespace(request_evidence=generation._reference(file)),),
    )
    monkeypatch.setattr(
        generation_port,
        "read_verified_r2_object",
        lambda _key, _digest: file.content,
    )

    generation_port._require_same_references(prepared.plan.edition_id, old_pack)
    with pytest.raises(ValueError, match="differs from stored generation work"):
        generation_port._require_same_references(prepared.plan.edition_id, _pack())
