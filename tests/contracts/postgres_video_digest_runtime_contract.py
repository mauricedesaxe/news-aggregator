from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import psycopg
import pytest
from dagster import DefaultScheduleStatus

import romanian_news.catalog.schema as news_schema
from romanian_news.catalog import video_digest as catalog
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.generation_port import ProductionH3GenerationPort
from romanian_news.video_digest.models import (
    PublicationAttemptReady,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
    VerifiedPublicObject,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionWaiting,
    AssembleAction,
    DomainPorts,
    GenerateAction,
    NoAlert,
    PlanAction,
    PublishAction,
    PublishedResume,
    RunDeferred,
    RunPublished,
    SubtitleAction,
)
from romanian_news.video_digest.publication import publication_intent_from_handoff
from romanian_news.worker import video_digest, video_digest_runtime
from romanian_news.worker.video_digest_runtime import ProductionVideoDigestRuntime
from tests.contracts.postgres_video_digest_generation import _GenerationPipeline


class _CheckpointPorts:
    def __init__(self, pipeline: _GenerationPipeline) -> None:
        self.pipeline = pipeline
        self.calls: list[str] = []

    def execute(self, action: object) -> ActionAdvanced | ActionWaiting:
        self.calls.append(type(action).__name__)
        if isinstance(action, PlanAction):
            self.pipeline.lease = action.lease
            self.pipeline.checkpoint_plan()
        elif isinstance(action, GenerateAction):
            self.pipeline.lease = action.lease
            self.pipeline.generate(0)
            self.pipeline.accept(0)
            self.pipeline.assembly_ready()
        elif isinstance(action, AssembleAction):
            self.pipeline.lease = action.lease
            self.pipeline.assemble()
        elif isinstance(action, SubtitleAction):
            self.pipeline.lease = action.lease
            self.pipeline.record_failed_subtitles()
        elif isinstance(action, PublishAction):
            self._publish(action)
            return ActionAdvanced()
        else:
            raise AssertionError(f"Unexpected action: {action}")
        return ActionWaiting(reason="fixture process restart", retry_after_seconds=1)

    def _publish(self, action: PublishAction) -> None:
        lease = action.lease
        now = datetime.now(UTC)
        intent = publication_intent_from_handoff(action.handoff)
        catalog.record_publication_intent(lease, intent, recorded_at=now)
        attempt = catalog.begin_publication_attempt(lease, intent.publication_id, recorded_at=now)
        assert isinstance(attempt, PublicationAttemptReady)
        catalog.checkpoint_publication_progress(
            lease,
            intent.publication_id,
            attempt.attempt_index,
            UploadingPublication(),
            recorded_at=now,
        )
        upload = self.pipeline._file(
            f"{intent.publication_id}:upload",
            "video_digest_publication_upload",
            title="Fixture publication upload",
        )
        catalog.checkpoint_publication_progress(
            lease,
            intent.publication_id,
            attempt.attempt_index,
            UploadedPublication(evidence_artifact_version_id=upload.version_id),
            evidence_file=upload,
            recorded_at=now,
        )
        verification = self.pipeline._file(
            f"{intent.publication_id}:verification",
            "video_digest_publication_verification",
            title="Fixture publication verification",
        )
        catalog.checkpoint_publication_progress(
            lease,
            intent.publication_id,
            attempt.attempt_index,
            VerifiedPublication(
                evidence_artifact_version_id=verification.version_id,
                video=VerifiedPublicObject(
                    content_digest=intent.video_digest,
                    byte_size=intent.video_byte_size,
                    media_type=intent.video_media_type,
                    source_artifact_version_id=intent.source_video_version_id,
                ),
            ),
            evidence_file=verification,
            recorded_at=now,
        )
        catalog.complete_publication(
            lease, intent.publication_id, attempt.attempt_index, recorded_at=now
        )


def _ports(pipeline: _GenerationPipeline, calls: list[str]) -> DomainPorts:
    checkpoint = _CheckpointPorts(pipeline)
    checkpoint.calls = calls
    return DomainPorts(
        planning=checkpoint,
        generation=checkpoint,
        assembly=checkpoint,
        subtitles=checkpoint,
        publication=checkpoint,
    )


def _expire_lease(slot_id: str) -> None:
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            "UPDATE video_digest_slots SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 second', "
            "updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
            (slot_id,),
        )


def test_production_runtime_reacquires_every_persisted_stage_after_restart(
    postgres_news_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_news_catalog_schema()
    assert video_digest.scheduled_video_digest.default_status is DefaultScheduleStatus.STOPPED
    pipeline = _GenerationPipeline(seed=81)
    calls: list[str] = []
    monkeypatch.setattr(
        video_digest_runtime,
        "resolve_video_digest_run_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("A resumed slot must not resolve the current report")
        ),
    )

    for index, stage in enumerate(("generation", "assembly", "subtitles", "publication")):
        _expire_lease(pipeline.slot.slot_id)
        runtime = ProductionVideoDigestRuntime(
            generation=_CheckpointPorts(pipeline),
            ports=_ports(pipeline, calls),
        )
        outcome, alert = runtime.run(pipeline.slot.slot_id, owner_token=f"restarted-{index}")
        assert outcome == RunDeferred(reason="fixture process restart", retry_after_seconds=1)
        assert alert == NoAlert()
        assert catalog.read_slot_resume_state(pipeline.slot.slot_id).kind == stage

    assert news_schema.NEWS_POSTGRES_DSN is not None
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from romanian_news.catalog.video_digest import read_slot_resume_state; "
            "s=read_slot_resume_state(sys.argv[1]); "
            "print(json.dumps({'kind':s.kind,'claim_count':s.lease.claim_count}))",
            pipeline.slot.slot_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "NEWS_POSTGRES_DSN": news_schema.NEWS_POSTGRES_DSN},
    )
    assert json.loads(child.stdout) == {"kind": "publication", "claim_count": 5}

    _expire_lease(pipeline.slot.slot_id)
    runtime = ProductionVideoDigestRuntime(
        generation=_CheckpointPorts(pipeline),
        ports=_ports(pipeline, calls),
    )
    outcome, alert = runtime.run(pipeline.slot.slot_id, owner_token="restarted-4")
    assert outcome == RunPublished()
    assert alert == NoAlert()
    assert isinstance(catalog.read_slot_resume_state(pipeline.slot.slot_id), PublishedResume)
    assert calls == [
        "PlanAction",
        "GenerateAction",
        "AssembleAction",
        "SubtitleAction",
        "PublishAction",
    ]

    replayed, replay_alert = ProductionVideoDigestRuntime(
        generation=_CheckpointPorts(pipeline),
        ports=_ports(pipeline, calls),
    ).run(pipeline.slot.slot_id, owner_token="restarted-5")
    assert replayed == RunPublished()
    assert replay_alert == NoAlert()
    assert len(calls) == 5


def test_runtime_invalid_reference_pack_fails_before_paid_admission(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=82)
    pipeline.checkpoint_plan()
    _expire_lease(pipeline.slot.slot_id)
    checkpoint = _CheckpointPorts(pipeline)
    ports = DomainPorts(
        planning=checkpoint,
        generation=ProductionH3GenerationPort(pack_id="invalid"),
        assembly=checkpoint,
        subtitles=checkpoint,
        publication=checkpoint,
    )
    runtime = ProductionVideoDigestRuntime(
        generation=ProductionH3GenerationPort(pack_id="invalid"), ports=ports
    )

    with pytest.raises(ValueError):
        runtime.run(pipeline.slot.slot_id, owner_token="invalid-config")

    assert catalog.read_slot_resume_state(pipeline.slot.slot_id).kind == "generation"
    assert catalog.read_generation_attempts(pipeline.edition.edition_id) == ()
