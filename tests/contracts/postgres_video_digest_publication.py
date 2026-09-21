from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, LiteralString

import psycopg
import pytest
from psycopg import sql

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog.video_digest as video_digest_catalog
from romanian_news import BUCHAREST
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog_transport import ResearchCatalogError
from romanian_news.video_digest.errors import VideoDigestCheckpointConflictError
from romanian_news.video_digest.models import (
    AvailableSubtitles,
    ClaimedSlot,
    DigestPlan,
    EditionId,
    EditionIdentity,
    EstimatedAttemptCost,
    FailedSubtitles,
    GenerationRequestIdentity,
    MeasuredAttemptCost,
    PlannedStory,
    PublicationIntent,
    PublicationState,
    PublicationStatus,
    PublishedSubtitleAvailable,
    ScheduledSlot,
    SlotLease,
    SlotName,
    SubtitleObjectMetadata,
    TerminalSlot,
    TerminalSlotState,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
    VerifiedPublicObject,
    edition_id,
    generation_request_id,
    planned_story_id,
    publication_id,
    scheduled_slot_id,
)

PUBLIC_MEDIA_BASE_URL = "https://media.example.com"

_WalkStage = Literal[
    "planned",
    "request_pending",
    "publishing",
    "intent",
    "uploading",
    "uploaded",
    "verified",
    "published",
]


@dataclass(frozen=True)
class PublicationRun:
    recorded_at: datetime
    identity: EditionIdentity
    slot: ScheduledSlot
    lease: SlotLease
    stories: tuple[PlannedStory, ...]
    intent: PublicationIntent
    assembled_file: ArtifactFile
    subtitle_file: ArtifactFile | None
    upload_file: ArtifactFile
    verification_file: ArtifactFile
    publication_failure_file: ArtifactFile
    slot_failure_file: ArtifactFile


def _sha256_id(value: int) -> str:
    return f"{value:064x}"


def _record_artifact_versions(
    connection: psycopg.Connection[Any], start: int, count: int
) -> tuple[str, ...]:
    version_ids = tuple(_sha256_id(value) for value in range(start, start + count))
    for version_id in version_ids:
        artifact_id = f"artifact-{version_id}"
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Test artifact', 'test', 'active', 'private', CURRENT_TIMESTAMP)",
            (artifact_id,),
        )
        connection.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, schema_version, content_digest, created_at) "
            "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
            (version_id, artifact_id, version_id),
        )
    return version_ids


def _fetch_row(
    statement: LiteralString, parameters: tuple[object, ...]
) -> tuple[object, ...] | None:
    dsn = news_schema.NEWS_POSTGRES_DSN
    assert dsn is not None
    with psycopg.connect(dsn, autocommit=True) as connection:
        return connection.execute(statement, parameters).fetchone()


@contextmanager
def _rejected_slot_stage(stage: str) -> Iterator[None]:
    dsn = news_schema.NEWS_POSTGRES_DSN
    assert dsn is not None
    rejection = sql.SQL(
        "CREATE FUNCTION reject_publication_contract_stage() RETURNS trigger "
        "LANGUAGE plpgsql AS $$ BEGIN "
        "IF NEW.stage = {stage} THEN "
        "RAISE EXCEPTION 'publication contract rejection' USING ERRCODE = '23000'; "
        "END IF; RETURN NEW; END; $$"
    ).format(stage=sql.Literal(stage))
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(rejection)
        connection.execute(
            "CREATE TRIGGER reject_publication_contract_stage_trigger "
            "BEFORE UPDATE ON video_digest_slots "
            "FOR EACH ROW EXECUTE FUNCTION reject_publication_contract_stage()"
        )
    try:
        yield
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                "DROP TRIGGER reject_publication_contract_stage_trigger ON video_digest_slots"
            )
            connection.execute("DROP FUNCTION reject_publication_contract_stage")


def _corrupt_published_video_key(edition_id: str) -> None:
    dsn = news_schema.NEWS_POSTGRES_DSN
    assert dsn is not None
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            "ALTER TABLE video_digest_publication_intents "
            "DROP CONSTRAINT video_digest_publication_video_key_normalized"
        )
        connection.execute("ALTER TABLE video_digest_publication_intents DISABLE TRIGGER USER")
        connection.execute(
            "UPDATE video_digest_publication_intents SET expected_video_key = %s "
            "WHERE edition_id = %s",
            ("legacy/video.mp4?download=1", edition_id),
        )
        connection.execute("ALTER TABLE video_digest_publication_intents ENABLE TRIGGER USER")


def _evidence(seed: int, artifact_id: str, artifact_kind: str, purpose: str) -> ArtifactFile:
    return artifact_file(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        title=f"Evidence {purpose} {seed}",
        content=f"evidence {purpose} {seed}".encode(),
        r2_key=f"video-digest/{seed}/{purpose}.json",
        media_type="application/json",
    )


def _publication_intent(
    identity: EditionIdentity,
    assembled_file: ArtifactFile,
    subtitle_file: ArtifactFile | None,
    *,
    expected_video_key: str,
    expected_subtitle_key: str,
    source_video_version_id: str | None = None,
) -> PublicationIntent:
    video_source = source_video_version_id if source_video_version_id else assembled_file.version_id
    subtitle_source = subtitle_file.version_id if subtitle_file is not None else None
    subtitle = (
        SubtitleObjectMetadata(
            expected_key=expected_subtitle_key,
            content_digest=subtitle_file.content_digest,
            byte_size=len(subtitle_file.content),
            media_type=subtitle_file.media_type,
        )
        if subtitle_file is not None
        else None
    )
    return PublicationIntent(
        publication_id=publication_id(
            edition_id_value=identity.edition_id,
            expected_video_key=expected_video_key,
            video_digest=assembled_file.content_digest,
            video_byte_size=len(assembled_file.content),
            video_media_type=assembled_file.media_type,
            subtitle=subtitle,
            source_video_version_id=video_source,
            source_subtitle_version_id=subtitle_source,
        ),
        edition_id=identity.edition_id,
        expected_video_key=expected_video_key,
        video_digest=assembled_file.content_digest,
        video_byte_size=len(assembled_file.content),
        video_media_type=assembled_file.media_type,
        subtitle=subtitle,
        source_video_version_id=video_source,
        source_subtitle_version_id=subtitle_source,
    )


def _verified_publication(run: PublicationRun) -> VerifiedPublication:
    subtitle = run.intent.subtitle
    subtitle_source = run.intent.source_subtitle_version_id
    verified_subtitle = (
        VerifiedPublicObject(
            content_digest=subtitle.content_digest,
            byte_size=subtitle.byte_size,
            media_type=subtitle.media_type,
            source_artifact_version_id=subtitle_source,
        )
        if subtitle is not None and subtitle_source is not None
        else None
    )
    return VerifiedPublication(
        evidence_artifact_version_id=run.verification_file.version_id,
        video=VerifiedPublicObject(
            content_digest=run.intent.video_digest,
            byte_size=run.intent.video_byte_size,
            media_type=run.intent.video_media_type,
            source_artifact_version_id=run.intent.source_video_version_id,
        ),
        subtitle=verified_subtitle,
    )


def _walk_publication_pipeline(
    *,
    seed: int,
    stop_at: _WalkStage,
    subtitle_available: bool = False,
    story_count: int = 1,
    slot_name: SlotName = SlotName.MORNING,
    scheduled_at: datetime | None = None,
) -> PublicationRun:
    dsn = news_schema.NEWS_POSTGRES_DSN
    assert dsn is not None
    recorded_at = datetime.now(UTC)
    when = scheduled_at if scheduled_at is not None else datetime(2026, 9, 20, 6, tzinfo=UTC)
    with psycopg.connect(dsn, autocommit=True) as connection:
        report, policy = _record_artifact_versions(connection, seed, 2)
    identity = EditionIdentity(
        edition_id=edition_id(report, policy),
        daily_report_version_id=report,
        policy_bundle_version_id=policy,
    )
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(slot_name, when),
        name=slot_name,
        scheduled_at=when,
        bucharest_day=when.astimezone(BUCHAREST).date(),
    )
    video_digest_catalog.schedule_slot(slot, recorded_at=recorded_at)
    claimed = video_digest_catalog.claim_slot(
        slot.slot_id,
        identity,
        owner_token=f"owner-{seed}",
        now=recorded_at,
        lease_duration=timedelta(hours=1),
    )
    assert isinstance(claimed, ClaimedSlot)
    lease = claimed.lease

    plan_file = artifact_file(
        artifact_id=identity.edition_id,
        artifact_kind="video_digest_plan",
        title=f"Edition plan {seed}",
        content=f"plan {seed}".encode(),
        r2_key=f"video-digest/{seed}/plan.json",
        media_type="application/json",
    )
    stories = tuple(
        PlannedStory(
            story_id=planned_story_id(
                identity.edition_id, position, _sha256_id(seed * 1000 + position)
            ),
            edition_id=identity.edition_id,
            position=position,
            report_subject_id=_sha256_id(seed * 1000 + position),
            title=f"Story {seed}.{position}",
            requested_duration_ms=15_000,
        )
        for position in range(story_count)
    )
    video_digest_catalog.checkpoint_plan(
        lease,
        DigestPlan(
            edition_id=identity.edition_id,
            artifact_version_id=plan_file.version_id,
            stories=stories,
        ),
        plan_file=plan_file,
        recorded_at=recorded_at,
    )

    assembled_file = artifact_file(
        artifact_id=f"{identity.edition_id}:assembled-video",
        artifact_kind="video_digest_assembled_video",
        title=f"Edition video {seed}",
        content=f"assembled video {seed}".encode(),
        r2_key=f"video-digest/{seed}/assembled.mp4",
        media_type="video/mp4",
    )
    subtitle_file = (
        artifact_file(
            artifact_id=f"{identity.edition_id}:subtitles",
            artifact_kind="video_digest_subtitles",
            title=f"Edition subtitles {seed}",
            content=f"subtitles {seed}".encode(),
            r2_key=f"video-digest/{seed}/subtitles.vtt",
            media_type="text/vtt",
        )
        if subtitle_available
        else None
    )
    intent = _publication_intent(
        identity,
        assembled_file,
        subtitle_file,
        expected_video_key=f"video-digest/{seed}/public.mp4",
        expected_subtitle_key=f"video-digest/{seed}/public.vtt",
    )
    run = PublicationRun(
        recorded_at=recorded_at,
        identity=identity,
        slot=slot,
        lease=lease,
        stories=stories,
        intent=intent,
        assembled_file=assembled_file,
        subtitle_file=subtitle_file,
        upload_file=_evidence(
            seed,
            f"{intent.publication_id}:upload",
            "video_digest_publication_upload",
            "publication-upload",
        ),
        verification_file=_evidence(
            seed,
            f"{intent.publication_id}:verification",
            "video_digest_publication_verification",
            "publication-verification",
        ),
        publication_failure_file=_evidence(
            seed,
            f"{intent.publication_id}:failure",
            "video_digest_publication_failure",
            "publication-failure",
        ),
        slot_failure_file=_evidence(
            seed, f"{slot.slot_id}:failure", "video_digest_failure", "slot-failure"
        ),
    )
    if stop_at == "planned":
        return run

    for position, story in enumerate(stories):
        verification = artifact_file(
            artifact_id=story.story_id,
            artifact_kind="video_digest_story_verification",
            title=f"Story verification {seed}.{position}",
            content=f"story verification {seed} {position}".encode(),
            r2_key=f"video-digest/{seed}/story-verification-{position}.json",
            media_type="application/json",
        )
        video_digest_catalog.checkpoint_story_verification(
            lease, story.story_id, evidence_file=verification, recorded_at=recorded_at
        )
        request_file = artifact_file(
            artifact_id=f"{identity.edition_id}:{position}:0:generation-request",
            artifact_kind="video_digest_generation_request",
            title=f"Generation request {seed}.{position}",
            content=f"generation request {seed} {position}".encode(),
            r2_key=f"video-digest/{seed}/generation-request-{position}.json",
            media_type="application/json",
        )
        request = GenerationRequestIdentity(
            request_id=generation_request_id(
                identity.edition_id, position, 0, request_file.version_id
            ),
            edition_id=identity.edition_id,
            story_position=position,
            attempt_index=0,
            request_artifact_version_id=request_file.version_id,
        )
        video_digest_catalog.checkpoint_generation_request(
            lease, request, request_file=request_file, recorded_at=recorded_at
        )
        if stop_at == "request_pending":
            return run
        receipt_id = f"provider-receipt-{seed}-{position}"
        receipt = artifact_file(
            artifact_id=receipt_id,
            artifact_kind="video_digest_provider_receipt",
            title=f"Provider receipt {seed}.{position}",
            content=f"provider receipt {seed} {position}".encode(),
            r2_key=f"video-digest/{seed}/provider-receipt-{position}.json",
            media_type="application/json",
        )
        video_digest_catalog.checkpoint_generation_submission(
            lease,
            request.request_id,
            provider_receipt_id=receipt_id,
            receipt_file=receipt,
            cost=EstimatedAttemptCost(usd=Decimal("1.25")),
            recorded_at=recorded_at,
        )
        response = artifact_file(
            artifact_id=f"{request.request_id}:response",
            artifact_kind="video_digest_generation_response",
            title=f"Generation response {seed}.{position}",
            content=f"generation response {seed} {position}".encode(),
            r2_key=f"video-digest/{seed}/generation-response-{position}.json",
            media_type="application/json",
        )
        video_digest_catalog.checkpoint_generation_response(
            lease, request.request_id, response_file=response, recorded_at=recorded_at
        )
        clip = artifact_file(
            artifact_id=f"{story.story_id}:accepted-clip",
            artifact_kind="video_digest_accepted_clip",
            title=f"Accepted clip {seed}.{position}",
            content=f"accepted clip {seed} {position}".encode(),
            r2_key=f"video-digest/{seed}/accepted-clip-{position}.mp4",
            media_type="video/mp4",
        )
        video_digest_catalog.checkpoint_generation_acceptance(
            lease,
            request.request_id,
            clip_file=clip,
            cost=MeasuredAttemptCost(usd=Decimal("1.10")),
            recorded_at=recorded_at,
        )

    video_digest_catalog.checkpoint_assembly_ready(lease, recorded_at=recorded_at)
    video_digest_catalog.checkpoint_assembled_video(
        lease, video_file=assembled_file, recorded_at=recorded_at
    )
    if subtitle_file is not None:
        video_digest_catalog.checkpoint_subtitles(
            lease,
            AvailableSubtitles(artifact_version_id=subtitle_file.version_id),
            artifact_file=subtitle_file,
            recorded_at=recorded_at,
        )
    else:
        subtitle_failure = artifact_file(
            artifact_id=f"{identity.edition_id}:subtitle-failure",
            artifact_kind="video_digest_subtitle_failure",
            title=f"Edition subtitle failure {seed}",
            content=f"subtitle failure {seed}".encode(),
            r2_key=f"video-digest/{seed}/subtitle-failure.json",
            media_type="application/json",
        )
        video_digest_catalog.checkpoint_subtitles(
            lease,
            FailedSubtitles(evidence_artifact_version_id=subtitle_failure.version_id),
            artifact_file=subtitle_failure,
            recorded_at=recorded_at,
        )
    if stop_at == "publishing":
        return run

    video_digest_catalog.record_publication_intent(lease, intent, recorded_at=recorded_at)
    if stop_at == "intent":
        return run
    video_digest_catalog.checkpoint_publication_progress(
        lease, intent.publication_id, UploadingPublication(), recorded_at=recorded_at
    )
    if stop_at == "uploading":
        return run
    video_digest_catalog.checkpoint_publication_progress(
        lease,
        intent.publication_id,
        UploadedPublication(evidence_artifact_version_id=run.upload_file.version_id),
        evidence_file=run.upload_file,
        recorded_at=recorded_at,
    )
    if stop_at == "uploaded":
        return run
    video_digest_catalog.checkpoint_publication_progress(
        lease,
        intent.publication_id,
        _verified_publication(run),
        evidence_file=run.verification_file,
        recorded_at=recorded_at,
    )
    if stop_at == "verified":
        return run
    video_digest_catalog.complete_publication(lease, intent.publication_id, recorded_at=recorded_at)
    return run


def test_publication_intent_inserts_once_and_replays_identically(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=10, stop_at="publishing")

    first = video_digest_catalog.record_publication_intent(
        run.lease, run.intent, recorded_at=run.recorded_at
    )
    assert first == PublicationStatus(
        publication_id=run.intent.publication_id,
        edition_id=run.identity.edition_id,
        stage=PublicationState.PENDING,
    )
    stored = _fetch_row(
        "SELECT stage, created_at, updated_at FROM video_digest_publication_intents "
        "WHERE edition_id = %s",
        (run.identity.edition_id,),
    )
    assert stored is not None

    replayed = video_digest_catalog.record_publication_intent(
        run.lease, run.intent, recorded_at=run.recorded_at + timedelta(minutes=5)
    )
    assert replayed == first
    assert (
        _fetch_row(
            "SELECT stage, created_at, updated_at FROM video_digest_publication_intents "
            "WHERE edition_id = %s",
            (run.identity.edition_id,),
        )
        == stored
    )
    assert _fetch_row(
        "SELECT count(*) FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == (1,)


def test_publication_intent_conflicts_when_a_different_intent_replays(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=20, stop_at="publishing")
    video_digest_catalog.record_publication_intent(
        run.lease, run.intent, recorded_at=run.recorded_at
    )
    divergent = _publication_intent(
        run.identity,
        run.assembled_file,
        run.subtitle_file,
        expected_video_key="video-digest/20/other.mp4",
        expected_subtitle_key="video-digest/20/other.vtt",
    )
    assert divergent.publication_id != run.intent.publication_id

    with pytest.raises(VideoDigestCheckpointConflictError, match="intent conflicts"):
        video_digest_catalog.record_publication_intent(
            run.lease, divergent, recorded_at=run.recorded_at
        )
    assert _fetch_row(
        "SELECT stage, expected_video_key FROM video_digest_publication_intents "
        "WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("pending", run.intent.expected_video_key)


def test_publication_intent_rejects_a_source_video_from_another_edition_output(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=30, stop_at="publishing")
    dsn = news_schema.NEWS_POSTGRES_DSN
    assert dsn is not None
    with psycopg.connect(dsn, autocommit=True) as connection:
        foreign_version = _record_artifact_versions(connection, 930, 1)[0]
    wrong_source = _publication_intent(
        run.identity,
        run.assembled_file,
        run.subtitle_file,
        expected_video_key="video-digest/30/public.mp4",
        expected_subtitle_key="video-digest/30/public.vtt",
        source_video_version_id=foreign_version,
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="sources conflict"):
        video_digest_catalog.record_publication_intent(
            run.lease, wrong_source, recorded_at=run.recorded_at
        )
    assert _fetch_row(
        "SELECT count(*) FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == (0,)


def test_publication_intent_rejects_a_subtitle_shape_that_disagrees_with_the_edition(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=40, stop_at="publishing", subtitle_available=True)
    without_subtitle = _publication_intent(
        run.identity,
        run.assembled_file,
        None,
        expected_video_key="video-digest/40/public.mp4",
        expected_subtitle_key="video-digest/40/public.vtt",
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="sources conflict"):
        video_digest_catalog.record_publication_intent(
            run.lease, without_subtitle, recorded_at=run.recorded_at
        )
    assert _fetch_row(
        "SELECT count(*) FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == (0,)


def test_publication_progress_persists_each_stage_with_its_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=50, stop_at="intent")
    evidence_row: LiteralString = (
        "SELECT stage, upload_evidence_artifact_version_id, "
        "verification_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s"
    )

    uploading = video_digest_catalog.checkpoint_publication_progress(
        run.lease, run.intent.publication_id, UploadingPublication(), recorded_at=run.recorded_at
    )
    assert uploading == PublicationStatus(
        publication_id=run.intent.publication_id,
        edition_id=run.identity.edition_id,
        stage=PublicationState.UPLOADING,
    )
    assert _fetch_row(evidence_row, (run.identity.edition_id,)) == ("uploading", None, None)

    uploaded = video_digest_catalog.checkpoint_publication_progress(
        run.lease,
        run.intent.publication_id,
        UploadedPublication(evidence_artifact_version_id=run.upload_file.version_id),
        evidence_file=run.upload_file,
        recorded_at=run.recorded_at,
    )
    assert uploaded.stage is PublicationState.UPLOADED
    assert _fetch_row(evidence_row, (run.identity.edition_id,)) == (
        "uploaded",
        run.upload_file.version_id,
        None,
    )

    verified = video_digest_catalog.checkpoint_publication_progress(
        run.lease,
        run.intent.publication_id,
        _verified_publication(run),
        evidence_file=run.verification_file,
        recorded_at=run.recorded_at,
    )
    assert verified.stage is PublicationState.VERIFIED
    assert _fetch_row(evidence_row, (run.identity.edition_id,)) == (
        "verified",
        run.upload_file.version_id,
        run.verification_file.version_id,
    )


def test_publication_progress_replays_completed_stages_idempotently(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=60, stop_at="verified")
    evidence_row: LiteralString = (
        "SELECT stage, upload_evidence_artifact_version_id, "
        "verification_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s"
    )
    stored = _fetch_row(evidence_row, (run.identity.edition_id,))
    assert stored == (
        "verified",
        run.upload_file.version_id,
        run.verification_file.version_id,
    )

    replayed_uploading = video_digest_catalog.checkpoint_publication_progress(
        run.lease, run.intent.publication_id, UploadingPublication(), recorded_at=run.recorded_at
    )
    assert replayed_uploading.stage is PublicationState.VERIFIED
    replayed_uploaded = video_digest_catalog.checkpoint_publication_progress(
        run.lease,
        run.intent.publication_id,
        UploadedPublication(evidence_artifact_version_id=run.upload_file.version_id),
        evidence_file=run.upload_file,
        recorded_at=run.recorded_at,
    )
    assert replayed_uploaded.stage is PublicationState.VERIFIED
    replayed_verified = video_digest_catalog.checkpoint_publication_progress(
        run.lease,
        run.intent.publication_id,
        _verified_publication(run),
        evidence_file=run.verification_file,
        recorded_at=run.recorded_at,
    )
    assert replayed_verified.stage is PublicationState.VERIFIED
    assert _fetch_row(evidence_row, (run.identity.edition_id,)) == stored


def test_publication_progress_rejects_a_skipped_stage(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=70, stop_at="intent")

    with pytest.raises(VideoDigestCheckpointConflictError, match="cannot skip a checkpoint"):
        video_digest_catalog.checkpoint_publication_progress(
            run.lease,
            run.intent.publication_id,
            _verified_publication(run),
            evidence_file=run.verification_file,
            recorded_at=run.recorded_at,
        )
    assert _fetch_row(
        "SELECT stage, upload_evidence_artifact_version_id, "
        "verification_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("pending", None, None)


def test_publication_progress_rejects_divergent_replay_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=80, stop_at="verified")
    divergent_evidence = artifact_file(
        artifact_id=f"{run.intent.publication_id}:verification",
        artifact_kind="video_digest_publication_verification",
        title="Divergent verification evidence",
        content=b"divergent verification evidence",
        r2_key="video-digest/80/divergent-verification.json",
        media_type="application/json",
    )
    divergent_progress = _verified_publication(run).model_copy(
        update={"evidence_artifact_version_id": divergent_evidence.version_id}
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="evidence conflicts"):
        video_digest_catalog.checkpoint_publication_progress(
            run.lease,
            run.intent.publication_id,
            divergent_progress,
            evidence_file=divergent_evidence,
            recorded_at=run.recorded_at,
        )
    assert _fetch_row(
        "SELECT stage, upload_evidence_artifact_version_id, "
        "verification_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("verified", run.upload_file.version_id, run.verification_file.version_id)


def test_publication_progress_replays_verified_after_completion(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=90, stop_at="published")

    status = video_digest_catalog.checkpoint_publication_progress(
        run.lease,
        run.intent.publication_id,
        _verified_publication(run),
        evidence_file=run.verification_file,
        recorded_at=run.recorded_at,
    )
    assert status == PublicationStatus(
        publication_id=run.intent.publication_id,
        edition_id=run.identity.edition_id,
        stage=PublicationState.PUBLISHED,
    )


def test_publication_completion_terminalizes_intent_and_slot_atomically(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=100, stop_at="verified")

    with _rejected_slot_stage("published"):
        with pytest.raises(VideoDigestCheckpointConflictError):
            video_digest_catalog.complete_publication(
                run.lease, run.intent.publication_id, recorded_at=run.recorded_at
            )
    assert _fetch_row(
        "SELECT stage FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("verified",)
    assert _fetch_row(
        "SELECT stage, lease_owner_token FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == ("publishing", run.lease.owner_token)

    published = video_digest_catalog.complete_publication(
        run.lease, run.intent.publication_id, recorded_at=run.recorded_at
    )
    assert published.publication_id == run.intent.publication_id
    assert published.edition_id == run.identity.edition_id
    assert _fetch_row(
        "SELECT stage FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == ("published",)
    reader_edition = video_digest_catalog.read_published_edition(
        run.identity.edition_id, public_media_base_url=PUBLIC_MEDIA_BASE_URL
    )
    assert reader_edition is not None
    assert reader_edition.edition_id == run.identity.edition_id


def test_publication_completion_replay_returns_the_original_timestamp(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=110, stop_at="verified")

    first = video_digest_catalog.complete_publication(
        run.lease, run.intent.publication_id, recorded_at=run.recorded_at
    )
    replayed = video_digest_catalog.complete_publication(
        run.lease, run.intent.publication_id, recorded_at=run.recorded_at + timedelta(hours=1)
    )
    assert replayed == first
    assert replayed.published_at == first.published_at


def test_publication_failure_terminalizes_intent_and_slot_atomically(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=120, stop_at="uploading")

    with _rejected_slot_stage("failed"):
        with pytest.raises(VideoDigestCheckpointConflictError):
            video_digest_catalog.fail_publication(
                run.lease,
                run.intent.publication_id,
                state=PublicationState.CONFLICT,
                evidence_file=run.publication_failure_file,
                recorded_at=run.recorded_at,
            )
    assert _fetch_row(
        "SELECT stage, failure_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("uploading", None)
    assert _fetch_row(
        "SELECT stage, lease_owner_token FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == ("publishing", run.lease.owner_token)

    terminal = video_digest_catalog.fail_publication(
        run.lease,
        run.intent.publication_id,
        state=PublicationState.CONFLICT,
        evidence_file=run.publication_failure_file,
        recorded_at=run.recorded_at,
    )
    assert terminal == TerminalSlot(state=TerminalSlotState.FAILED)
    assert _fetch_row(
        "SELECT stage, failure_evidence_artifact_version_id "
        "FROM video_digest_publication_intents WHERE edition_id = %s",
        (run.identity.edition_id,),
    ) == ("conflict", run.publication_failure_file.version_id)
    assert _fetch_row(
        "SELECT stage, lease_owner_token, terminal_lease_owner_token, terminal_claim_count, "
        "failure_evidence_artifact_version_id FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == (
        "failed",
        None,
        run.lease.owner_token,
        run.lease.claim_count,
        run.publication_failure_file.version_id,
    )
    assert (
        video_digest_catalog.read_published_edition(
            run.identity.edition_id, public_media_base_url=PUBLIC_MEDIA_BASE_URL
        )
        is None
    )


def test_publication_failure_replays_matching_terminal_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=130, stop_at="uploading")

    first = video_digest_catalog.fail_publication(
        run.lease,
        run.intent.publication_id,
        state=PublicationState.FAILED,
        evidence_file=run.publication_failure_file,
        recorded_at=run.recorded_at,
    )
    assert first == TerminalSlot(state=TerminalSlotState.FAILED)
    terminal_row: LiteralString = (
        "SELECT slot.stage, publication.stage, slot.failure_evidence_artifact_version_id, "
        "publication.failure_evidence_artifact_version_id "
        "FROM video_digest_slots AS slot "
        "JOIN video_digest_publication_intents AS publication "
        "  ON publication.edition_id = slot.edition_id "
        "WHERE slot.slot_id = %s"
    )
    stored = _fetch_row(terminal_row, (run.slot.slot_id,))

    replayed = video_digest_catalog.fail_publication(
        run.lease,
        run.intent.publication_id,
        state=PublicationState.FAILED,
        evidence_file=run.publication_failure_file,
        recorded_at=run.recorded_at + timedelta(minutes=5),
    )
    assert replayed == TerminalSlot(state=TerminalSlotState.FAILED)
    assert _fetch_row(terminal_row, (run.slot.slot_id,)) == stored


def test_publication_failure_replay_rejects_a_divergent_terminal_state(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=140, stop_at="uploading")
    video_digest_catalog.fail_publication(
        run.lease,
        run.intent.publication_id,
        state=PublicationState.CONFLICT,
        evidence_file=run.publication_failure_file,
        recorded_at=run.recorded_at,
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="publication failure conflicts"):
        video_digest_catalog.fail_publication(
            run.lease,
            run.intent.publication_id,
            state=PublicationState.FAILED,
            evidence_file=run.publication_failure_file,
            recorded_at=run.recorded_at,
        )


def test_generic_slot_failure_rejects_an_active_generation(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=150, stop_at="request_pending")

    with pytest.raises(VideoDigestCheckpointConflictError, match="Active generation"):
        video_digest_catalog.fail_slot(
            run.lease, evidence_file=run.slot_failure_file, recorded_at=run.recorded_at
        )
    assert _fetch_row(
        "SELECT stage FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == ("generating",)
    assert _fetch_row(
        "SELECT stage FROM video_digest_stories WHERE edition_id = %s AND position = 0",
        (run.identity.edition_id,),
    ) == ("generating",)


def test_generic_slot_failure_requires_publication_failure_once_an_intent_exists(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=160, stop_at="intent")

    with pytest.raises(VideoDigestCheckpointConflictError, match="requires publication failure"):
        video_digest_catalog.fail_slot(
            run.lease, evidence_file=run.slot_failure_file, recorded_at=run.recorded_at
        )
    assert _fetch_row(
        "SELECT stage, lease_owner_token FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == ("publishing", run.lease.owner_token)


def test_generic_slot_failure_fails_remaining_stories_and_the_slot_together(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=170, stop_at="planned", story_count=2)

    terminal = video_digest_catalog.fail_slot(
        run.lease, evidence_file=run.slot_failure_file, recorded_at=run.recorded_at
    )
    assert terminal == TerminalSlot(state=TerminalSlotState.FAILED)
    assert _fetch_row(
        "SELECT stage, failure_evidence_artifact_version_id FROM video_digest_stories "
        "WHERE edition_id = %s AND position = 0",
        (run.identity.edition_id,),
    ) == ("failed", run.slot_failure_file.version_id)
    assert _fetch_row(
        "SELECT stage, failure_evidence_artifact_version_id FROM video_digest_stories "
        "WHERE edition_id = %s AND position = 1",
        (run.identity.edition_id,),
    ) == ("failed", run.slot_failure_file.version_id)
    assert _fetch_row(
        "SELECT stage, lease_owner_token, terminal_lease_owner_token, terminal_claim_count, "
        "failure_evidence_artifact_version_id FROM video_digest_slots WHERE slot_id = %s",
        (run.slot.slot_id,),
    ) == (
        "failed",
        None,
        run.lease.owner_token,
        run.lease.claim_count,
        run.slot_failure_file.version_id,
    )


def test_generic_slot_failure_replays_terminal_evidence(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=180, stop_at="planned")

    first = video_digest_catalog.fail_slot(
        run.lease, evidence_file=run.slot_failure_file, recorded_at=run.recorded_at
    )
    assert first == TerminalSlot(state=TerminalSlotState.FAILED)
    stored = _fetch_row(
        "SELECT stage, failure_evidence_artifact_version_id FROM video_digest_slots "
        "WHERE slot_id = %s",
        (run.slot.slot_id,),
    )

    replayed = video_digest_catalog.fail_slot(
        run.lease,
        evidence_file=run.slot_failure_file,
        recorded_at=run.recorded_at + timedelta(minutes=5),
    )
    assert replayed == TerminalSlot(state=TerminalSlotState.FAILED)
    assert (
        _fetch_row(
            "SELECT stage, failure_evidence_artifact_version_id FROM video_digest_slots "
            "WHERE slot_id = %s",
            (run.slot.slot_id,),
        )
        == stored
    )


def test_read_published_edition_returns_none_for_unpublished_editions(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=190, stop_at="verified")

    assert (
        video_digest_catalog.read_published_edition(
            run.identity.edition_id, public_media_base_url=PUBLIC_MEDIA_BASE_URL
        )
        is None
    )
    assert (
        video_digest_catalog.read_published_edition(
            EditionId(_sha256_id(1901)), public_media_base_url=PUBLIC_MEDIA_BASE_URL
        )
        is None
    )


def test_read_published_edition_serves_public_media_and_hides_internal_keys(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(
        seed=200, stop_at="published", subtitle_available=True, story_count=2
    )
    assert run.subtitle_file is not None

    edition = video_digest_catalog.read_published_edition(
        run.identity.edition_id, public_media_base_url=PUBLIC_MEDIA_BASE_URL
    )
    assert edition is not None
    assert edition.edition_id == run.identity.edition_id
    assert edition.publication_id == run.intent.publication_id
    assert edition.slot_name is SlotName.MORNING
    assert edition.day == run.slot.bucharest_day
    assert str(edition.video.url) == "https://media.example.com/video-digest/200/public.mp4"
    assert edition.video.content_digest == run.assembled_file.content_digest
    assert edition.video.byte_size == len(run.assembled_file.content)
    assert edition.video.media_type == run.assembled_file.media_type
    assert isinstance(edition.subtitle, PublishedSubtitleAvailable)
    assert (
        str(edition.subtitle.media.url) == "https://media.example.com/video-digest/200/public.vtt"
    )
    assert edition.subtitle.media.content_digest == run.subtitle_file.content_digest
    assert edition.subtitle.media.byte_size == len(run.subtitle_file.content)
    assert edition.subtitle.media.media_type == run.subtitle_file.media_type
    assert tuple(story.position for story in edition.stories) == (0, 1)
    assert tuple(story.story_id for story in edition.stories) == tuple(
        story.story_id for story in run.stories
    )

    leaked = edition.model_dump(mode="json")
    assert set(leaked) == {
        "edition_id",
        "publication_id",
        "day",
        "slot_name",
        "scheduled_at",
        "published_at",
        "daily_report_version_id",
        "video",
        "subtitle",
        "stories",
    }
    assert set(leaked["video"]) == {"url", "content_digest", "byte_size", "media_type"}
    dumped = json.dumps(leaked)
    assert run.assembled_file.r2_key not in dumped
    assert run.assembled_file.version_id not in dumped
    assert run.subtitle_file.r2_key not in dumped
    assert run.subtitle_file.version_id not in dumped
    assert run.upload_file.version_id not in dumped
    assert run.verification_file.version_id not in dumped


@pytest.mark.parametrize(
    "base_url",
    [
        "https://media.example.com/video-digest",
        "https://media.example.com?media=1",
        "http://media.example.com",
    ],
)
def test_published_reads_reject_non_origin_media_base_urls(
    postgres_news_schema: str, base_url: str
) -> None:
    ensure_news_catalog_schema()

    with pytest.raises(ValueError, match="HTTPS origin"):
        video_digest_catalog.read_published_edition(
            EditionId(_sha256_id(1)), public_media_base_url=base_url
        )
    with pytest.raises(ValueError, match="HTTPS origin"):
        video_digest_catalog.list_published_editions(
            date(2026, 9, 20), public_media_base_url=base_url
        )


def test_read_published_edition_rejects_a_malformed_legacy_object_key(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(seed=210, stop_at="published")
    _corrupt_published_video_key(run.identity.edition_id)

    with pytest.raises(ResearchCatalogError, match="invalid object key"):
        video_digest_catalog.read_published_edition(
            run.identity.edition_id, public_media_base_url=PUBLIC_MEDIA_BASE_URL
        )


def test_list_published_editions_orders_most_recent_first_with_public_media_only(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    earlier = _walk_publication_pipeline(
        seed=220,
        stop_at="published",
        slot_name=SlotName.MORNING,
        scheduled_at=datetime(2026, 9, 20, 6, tzinfo=UTC),
    )
    later = _walk_publication_pipeline(
        seed=230,
        stop_at="published",
        subtitle_available=True,
        slot_name=SlotName.MIDDAY,
        scheduled_at=datetime(2026, 9, 20, 9, tzinfo=UTC),
    )

    summaries = video_digest_catalog.list_published_editions(
        date(2026, 9, 20), public_media_base_url=PUBLIC_MEDIA_BASE_URL
    )
    assert [summary.edition_id for summary in summaries] == [
        later.identity.edition_id,
        earlier.identity.edition_id,
    ]
    assert [summary.slot_name for summary in summaries] == [SlotName.MIDDAY, SlotName.MORNING]

    top = summaries[0]
    assert top.publication_id == later.intent.publication_id
    assert top.day == date(2026, 9, 20)
    assert str(top.video.url) == "https://media.example.com/video-digest/230/public.mp4"
    assert top.video.content_digest == later.assembled_file.content_digest
    assert top.video.byte_size == len(later.assembled_file.content)
    assert top.video.media_type == later.assembled_file.media_type
    assert isinstance(top.subtitle, PublishedSubtitleAvailable)
    assert str(top.subtitle.media.url) == "https://media.example.com/video-digest/230/public.vtt"
    leaked = top.model_dump(mode="json")
    assert set(leaked) == {
        "edition_id",
        "publication_id",
        "day",
        "slot_name",
        "scheduled_at",
        "published_at",
        "daily_report_version_id",
        "video",
        "subtitle",
    }
    dumped = json.dumps(leaked)
    assert later.assembled_file.r2_key not in dumped
    assert later.assembled_file.version_id not in dumped


def test_list_published_editions_scopes_to_the_requested_day(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    run = _walk_publication_pipeline(
        seed=250,
        stop_at="published",
        scheduled_at=datetime(2026, 9, 20, 6, tzinfo=UTC),
    )

    assert (
        video_digest_catalog.list_published_editions(
            date(2026, 9, 21), public_media_base_url=PUBLIC_MEDIA_BASE_URL
        )
        == ()
    )
    summaries = video_digest_catalog.list_published_editions(
        date(2026, 9, 20), public_media_base_url=PUBLIC_MEDIA_BASE_URL
    )
    assert [summary.edition_id for summary in summaries] == [run.identity.edition_id]


def test_list_published_editions_applies_the_requested_limit(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    _walk_publication_pipeline(
        seed=260,
        stop_at="published",
        slot_name=SlotName.MORNING,
        scheduled_at=datetime(2026, 9, 20, 6, tzinfo=UTC),
    )
    later = _walk_publication_pipeline(
        seed=270,
        stop_at="published",
        slot_name=SlotName.MIDDAY,
        scheduled_at=datetime(2026, 9, 20, 9, tzinfo=UTC),
    )

    summaries = video_digest_catalog.list_published_editions(
        date(2026, 9, 20), public_media_base_url=PUBLIC_MEDIA_BASE_URL, limit=1
    )
    assert [summary.edition_id for summary in summaries] == [later.identity.edition_id]


def test_list_published_editions_rejects_out_of_range_limits(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()

    with pytest.raises(ValueError, match="limit must be between"):
        video_digest_catalog.list_published_editions(
            date(2026, 9, 20), public_media_base_url=PUBLIC_MEDIA_BASE_URL, limit=0
        )
    with pytest.raises(ValueError, match="limit must be between"):
        video_digest_catalog.list_published_editions(
            date(2026, 9, 20), public_media_base_url=PUBLIC_MEDIA_BASE_URL, limit=101
        )
