from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from math import ceil
from typing import Annotated, Literal
from urllib.parse import urlsplit

import requests
from botocore.client import BaseClient
from pydantic import Field, StringConstraints, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog.video_digest import (
    begin_publication_attempt,
    checkpoint_publication_progress,
    checkpoint_publication_retry,
    complete_publication,
    fail_publication,
    record_publication_intent,
)
from romanian_news.config import (
    NEWS_PUBLIC_MEDIA_BASE_URL,
    NEWS_PUBLIC_MEDIA_R2_BUCKET,
    NEWS_R2_BUCKET,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import (
    PublicObjectConflict,
    PublicObjectUnavailable,
    PublicObjectVerification,
    PublicR2Object,
    ResearchObjectIntegrityError,
    ResearchObjectUnavailable,
    publish_immutable_r2_objects,
    publish_public_r2_object,
    r2_client,
    read_verified_r2_object,
    verify_public_object_url,
)
from romanian_news.video_digest.models import (
    PUBLIC_MEDIA_CACHE_CONTROL,
    PUBLIC_MEDIA_RETENTION,
    PUBLIC_MEDIA_VISIBILITY,
    EditionId,
    PublicationAttemptWaiting,
    PublicationCheckpointSuperseded,
    PublicationId,
    PublicationIntent,
    PublicationState,
    PublicationStatus,
    SubtitleObjectMetadata,
    UploadedPublication,
    UploadingPublication,
    VerifiedPublication,
    VerifiedPublicObject,
    publication_id,
)
from romanian_news.video_digest.orchestration import (
    MAX_RETRY_SECONDS,
    ActionAdvanced,
    ActionOutcome,
    ActionWaiting,
    AvailablePublicationSubtitles,
    PublicationHandoff,
    PublishAction,
)

_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PublicMediaConfiguration(NewsModel):
    private_bucket: _NonEmpty
    public_bucket: _NonEmpty
    base_url: _NonEmpty

    @model_validator(mode="after")
    def validate_boundary(self) -> PublicMediaConfiguration:
        if self.private_bucket == self.public_bucket:
            raise ValueError("Private and public R2 buckets must be distinct")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Public media base URL must be a bare HTTPS origin")
        return self


class PublicationObjectEvidence(NewsModel):
    key: str
    content_digest: Sha256
    byte_size: int
    content_type: Literal["video/mp4", "text/vtt"]
    cache_control: Literal["public,max-age=31536000,immutable"]
    visibility: Literal["public"]
    retention: Literal["permanent"]
    source_lineage: Sha256


class PublicationUploadEvidence(NewsModel):
    kind: Literal["upload"] = "upload"
    publication_id: Sha256
    objects: tuple[PublicationObjectEvidence, ...]


class PublicationVerificationEvidence(NewsModel):
    kind: Literal["verification"] = "verification"
    publication_id: Sha256
    objects: tuple[PublicObjectVerification, ...]


class PublicationRetryEvidence(NewsModel):
    kind: Literal["retryable_failure"] = "retryable_failure"
    publication_id: Sha256
    attempt_index: Annotated[int, Field(ge=0, le=3)]
    error_code: str


class PublicationTerminalFailureEvidence(NewsModel):
    kind: Literal["conflict", "failed"]
    publication_id: Sha256
    attempt_index: Annotated[int, Field(ge=0, le=4)]
    error_code: str


class R2PublicationPort:
    def __init__(
        self,
        configuration: PublicMediaConfiguration,
        *,
        client: BaseClient | None = None,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._configuration = configuration
        self._client = client or r2_client()
        self._session = session
        self._now = now or (lambda: datetime.now(UTC))

    @classmethod
    def from_environment(cls) -> R2PublicationPort:
        return cls(
            PublicMediaConfiguration(
                private_bucket=NEWS_R2_BUCKET,
                public_bucket=NEWS_PUBLIC_MEDIA_R2_BUCKET,
                base_url=NEWS_PUBLIC_MEDIA_BASE_URL,
            )
        )

    def execute(self, action: PublishAction) -> ActionOutcome:
        intent = publication_intent_from_handoff(action.handoff)
        status = record_publication_intent(action.lease, intent, recorded_at=self._now())
        if status.stage in {
            PublicationState.PUBLISHED,
            PublicationState.CONFLICT,
            PublicationState.FAILED,
        }:
            return ActionAdvanced()
        attempt = begin_publication_attempt(
            action.lease,
            intent.publication_id,
            recorded_at=self._now(),
        )
        if isinstance(attempt, PublicationAttemptWaiting):
            return _waiting(attempt.retry_at, self._now())
        if isinstance(attempt, PublicationCheckpointSuperseded):
            return _superseded_outcome(attempt, self._now())
        try:
            while True:
                if status.stage == PublicationState.PENDING:
                    result = checkpoint_publication_progress(
                        action.lease,
                        intent.publication_id,
                        attempt.attempt_index,
                        UploadingPublication(),
                        recorded_at=self._now(),
                    )
                elif status.stage == PublicationState.UPLOADING:
                    objects = self._objects(action.handoff, intent)
                    result = self._upload(action, intent, attempt.attempt_index, objects)
                elif status.stage == PublicationState.UPLOADED:
                    objects = self._objects(action.handoff, intent)
                    result = self._verify(action, intent, attempt.attempt_index, objects)
                elif status.stage == PublicationState.VERIFIED:
                    completed = complete_publication(
                        action.lease,
                        intent.publication_id,
                        attempt.attempt_index,
                        recorded_at=self._now(),
                    )
                    if isinstance(completed, PublicationCheckpointSuperseded):
                        return _superseded_outcome(completed, self._now())
                    return ActionAdvanced()
                else:
                    raise RuntimeError("Publication is not active")
                if isinstance(result, PublicationCheckpointSuperseded):
                    return _superseded_outcome(result, self._now())
                status = result
        except (PublicObjectConflict, ResearchObjectIntegrityError) as error:
            self._terminalize(
                action,
                intent.publication_id,
                attempt.attempt_index,
                status.stage,
                PublicationState.CONFLICT,
                error,
            )
            return ActionAdvanced()
        except (PublicObjectUnavailable, ResearchObjectUnavailable) as error:
            if attempt.attempt_index < 4:
                evidence = _attempt_failure_file(
                    intent.publication_id,
                    attempt.attempt_index,
                    error,
                )
                self._publish_evidence(evidence)
                waiting = checkpoint_publication_retry(
                    action.lease,
                    intent.publication_id,
                    attempt.attempt_index,
                    expected_stage=status.stage,
                    evidence_file=evidence,
                    recorded_at=self._now(),
                )
                if isinstance(waiting, PublicationCheckpointSuperseded):
                    return _superseded_outcome(waiting, self._now())
                return _waiting(waiting.retry_at, self._now())
            self._terminalize(
                action,
                intent.publication_id,
                attempt.attempt_index,
                status.stage,
                PublicationState.FAILED,
                error,
            )
            return ActionAdvanced()

    def _objects(
        self,
        handoff: PublicationHandoff,
        intent: PublicationIntent,
    ) -> tuple[PublicR2Object, ...]:
        video_content = read_verified_r2_object(
            handoff.video.r2_key,
            handoff.video.content_digest,
            client=self._client,
            bucket=self._configuration.private_bucket,
        )
        objects = [
            PublicR2Object(
                key=intent.expected_video_key,
                content=video_content,
                content_digest=intent.video_digest,
                byte_size=intent.video_byte_size,
                content_type=intent.video_media_type,
                cache_control=intent.cache_control,
                visibility=intent.visibility,
                retention=intent.retention,
                source_lineage=intent.source_video_version_id,
            )
        ]
        if isinstance(handoff.subtitles, AvailablePublicationSubtitles):
            if intent.subtitle is None:
                raise RuntimeError("Available subtitles are missing from publication intent")
            artifact = handoff.subtitles.artifact
            objects.append(
                PublicR2Object(
                    key=intent.subtitle.expected_key,
                    content=read_verified_r2_object(
                        artifact.r2_key,
                        artifact.content_digest,
                        client=self._client,
                        bucket=self._configuration.private_bucket,
                    ),
                    content_digest=intent.subtitle.content_digest,
                    byte_size=intent.subtitle.byte_size,
                    content_type=intent.subtitle.media_type,
                    cache_control=intent.cache_control,
                    visibility=intent.visibility,
                    retention=intent.retention,
                    source_lineage=artifact.version_id,
                )
            )
        return tuple(objects)

    def _upload(
        self,
        action: PublishAction,
        intent: PublicationIntent,
        attempt_index: int,
        objects: tuple[PublicR2Object, ...],
    ) -> PublicationStatus | PublicationCheckpointSuperseded:
        for value in objects:
            publish_public_r2_object(
                self._client,
                self._configuration.public_bucket,
                value,
            )
        evidence = _upload_evidence_file(intent.publication_id, objects)
        self._publish_evidence(evidence)
        return checkpoint_publication_progress(
            action.lease,
            intent.publication_id,
            attempt_index,
            UploadedPublication(evidence_artifact_version_id=evidence.version_id),
            evidence_file=evidence,
            recorded_at=self._now(),
        )

    def _verify(
        self,
        action: PublishAction,
        intent: PublicationIntent,
        attempt_index: int,
        objects: tuple[PublicR2Object, ...],
    ) -> PublicationStatus | PublicationCheckpointSuperseded:
        for value in objects:
            publish_public_r2_object(
                self._client,
                self._configuration.public_bucket,
                value,
            )
        verifications = tuple(
            verify_public_object_url(
                f"{self._configuration.base_url}/{value.key}",
                value,
                session=self._session,
            )
            for value in objects
        )
        evidence = _verification_evidence_file(intent.publication_id, verifications)
        self._publish_evidence(evidence)
        return checkpoint_publication_progress(
            action.lease,
            intent.publication_id,
            attempt_index,
            VerifiedPublication(
                evidence_artifact_version_id=evidence.version_id,
                video=_verified_object(objects[0]),
                subtitle=_verified_object(objects[1]) if len(objects) == 2 else None,
            ),
            evidence_file=evidence,
            recorded_at=self._now(),
        )

    def _terminalize(
        self,
        action: PublishAction,
        publication_id_value: PublicationId,
        attempt_index: int,
        expected_stage: PublicationState,
        state: Literal[PublicationState.CONFLICT, PublicationState.FAILED],
        error: Exception,
    ) -> None:
        evidence = _terminal_failure_file(
            publication_id_value,
            attempt_index,
            state,
            error,
        )
        self._publish_evidence(evidence)
        fail_publication(
            action.lease,
            publication_id_value,
            attempt_index,
            expected_stage=expected_stage,
            state=state,
            evidence_file=evidence,
            recorded_at=self._now(),
        )

    def _publish_evidence(self, file: ArtifactFile) -> None:
        publish_immutable_r2_objects(
            ((file.r2_key, file.content),),
            client=self._client,
            bucket=self._configuration.private_bucket,
        )


def publication_intent_from_handoff(handoff: PublicationHandoff) -> PublicationIntent:
    if handoff.video.media_type != "video/mp4":
        raise ValueError("Publication video must use video/mp4")
    edition_id = EditionId(handoff.edition_id)
    video_key = f"video-digests/{edition_id}/{handoff.video.content_digest}.mp4"
    subtitle: SubtitleObjectMetadata | None = None
    source_subtitle_version_id: Sha256 | None = None
    if isinstance(handoff.subtitles, AvailablePublicationSubtitles):
        artifact = handoff.subtitles.artifact
        if artifact.media_type != "text/vtt":
            raise ValueError("Publication subtitles must use text/vtt")
        subtitle = SubtitleObjectMetadata(
            expected_key=f"video-digests/{edition_id}/{artifact.content_digest}.vtt",
            content_digest=artifact.content_digest,
            byte_size=artifact.byte_size,
            media_type="text/vtt",
        )
        source_subtitle_version_id = artifact.version_id
    identity = publication_id(
        edition_id_value=edition_id,
        expected_video_key=video_key,
        video_digest=handoff.video.content_digest,
        video_byte_size=handoff.video.byte_size,
        video_media_type="video/mp4",
        subtitle=subtitle,
        source_video_version_id=handoff.video.version_id,
        source_subtitle_version_id=source_subtitle_version_id,
        cache_control=PUBLIC_MEDIA_CACHE_CONTROL,
        visibility=PUBLIC_MEDIA_VISIBILITY,
        retention=PUBLIC_MEDIA_RETENTION,
    )
    return PublicationIntent(
        publication_id=identity,
        edition_id=edition_id,
        expected_video_key=video_key,
        video_digest=handoff.video.content_digest,
        video_byte_size=handoff.video.byte_size,
        video_media_type="video/mp4",
        subtitle=subtitle,
        source_video_version_id=handoff.video.version_id,
        source_subtitle_version_id=source_subtitle_version_id,
    )


def _object_evidence(value: PublicR2Object) -> PublicationObjectEvidence:
    return PublicationObjectEvidence(
        **value.model_dump(exclude={"content"}),
    )


def _upload_evidence_file(
    publication_id_value: PublicationId,
    objects: tuple[PublicR2Object, ...],
) -> ArtifactFile:
    content = canonical_json(
        PublicationUploadEvidence(
            publication_id=publication_id_value,
            objects=tuple(_object_evidence(value) for value in objects),
        ).model_dump(mode="json")
    )
    return _publication_evidence_file(publication_id_value, "upload", content)


def _verification_evidence_file(
    publication_id_value: PublicationId,
    objects: tuple[PublicObjectVerification, ...],
) -> ArtifactFile:
    content = canonical_json(
        PublicationVerificationEvidence(
            publication_id=publication_id_value,
            objects=objects,
        ).model_dump(mode="json")
    )
    return _publication_evidence_file(publication_id_value, "verification", content)


def _attempt_failure_file(
    publication_id_value: PublicationId,
    attempt_index: int,
    error: Exception,
) -> ArtifactFile:
    content = canonical_json(
        PublicationRetryEvidence(
            publication_id=publication_id_value,
            attempt_index=attempt_index,
            error_code=type(error).__name__,
        ).model_dump(mode="json")
    )
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"{publication_id_value}:attempt-{attempt_index}-failure",
        artifact_kind="video_digest_publication_attempt_failure",
        title=f"Publication attempt {attempt_index} failure",
        content=content,
        r2_key=(
            f"news/video-digest/publication/{publication_id_value}/"
            f"attempt-{attempt_index}-failure-{digest}.json"
        ),
        media_type="application/json",
    )


def _terminal_failure_file(
    publication_id_value: PublicationId,
    attempt_index: int,
    state: Literal[PublicationState.CONFLICT, PublicationState.FAILED],
    error: Exception,
) -> ArtifactFile:
    content = canonical_json(
        PublicationTerminalFailureEvidence(
            kind=state.value,
            publication_id=publication_id_value,
            attempt_index=attempt_index,
            error_code=type(error).__name__,
        ).model_dump(mode="json")
    )
    return _publication_evidence_file(publication_id_value, "failure", content)


def _publication_evidence_file(
    publication_id_value: PublicationId,
    kind: Literal["upload", "verification", "failure"],
    content: bytes,
) -> ArtifactFile:
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"{publication_id_value}:{kind}",
        artifact_kind=f"video_digest_publication_{kind}",
        title=f"Publication {kind} evidence",
        content=content,
        r2_key=(f"news/video-digest/publication/{publication_id_value}/{kind}-{digest}.json"),
        media_type="application/json",
    )


def _verified_object(value: PublicR2Object) -> VerifiedPublicObject:
    return VerifiedPublicObject(
        content_digest=value.content_digest,
        byte_size=value.byte_size,
        media_type=value.content_type,
        source_artifact_version_id=value.source_lineage,
    )


def _waiting(retry_at: datetime, now: datetime) -> ActionWaiting:
    remaining = max(1, ceil((retry_at - now.astimezone(UTC)).total_seconds()))
    return ActionWaiting(
        reason="public media publication retry is not ready",
        retry_after_seconds=min(MAX_RETRY_SECONDS, remaining),
    )


def _superseded_outcome(
    result: PublicationCheckpointSuperseded,
    now: datetime,
) -> ActionOutcome:
    if result.retry_at is None:
        return ActionAdvanced()
    return _waiting(result.retry_at, now)
