from __future__ import annotations

from typing import Annotated, Literal

from openai.types.chat import ChatCompletion
from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelAttempt, model_attempt_from_payload
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.identity import canonical_json, sha256
from romanian_news.video_digest.models import DigestPlan, PlannedStory, planned_story_id
from romanian_news.video_digest.planning import (
    PlanningAttempt,
    PlanningFailure,
    ScreenplayStory,
    VerifiedDigestPlan,
    VideoDigestPolicyBundle,
    policy_bundle_digest,
)

PLANNING_OPERATION = "news.video_digest.plan"
VERIFICATION_OPERATION = "news.video_digest.verify_story"


class PlanningResponse(NewsModel):
    stories: Annotated[tuple[ScreenplayStory, ...], Field(min_length=1)]


class VerificationResponse(NewsModel):
    status: Literal["accepted", "rejected"]
    failures: tuple[PlanningFailure, ...]

    @model_validator(mode="after")
    def require_failures_for_rejection(self) -> VerificationResponse:
        if self.status == "accepted" and self.failures:
            raise ValueError("Accepted verification cannot contain failures")
        if self.status == "rejected" and not self.failures:
            raise ValueError("Rejected verification requires a structured failure")
        return self


class RecordedProviderResponse(NewsModel):
    attempt: ModelAttempt
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]

    @model_validator(mode="after")
    def require_exact_provider_response(self) -> RecordedProviderResponse:
        if self.response_content_digest != sha256(self.response_content.encode()):
            raise ValueError("Recorded provider response digest does not match its content")
        response = ChatCompletion.model_validate(self.provider_response, strict=True)
        if (
            response.id != self.attempt.response_id
            or response.model != self.attempt.model
            or (response.choices[0].message.content or "") != self.response_content
        ):
            raise ValueError("Recorded provider response does not match its model attempt")
        return self


class PlanningAttemptArtifact(NewsModel):
    attempt_index: int
    disposition: Literal["accepted", "rejected"]
    policy: VideoDigestPolicyBundle
    planning_response: RecordedProviderResponse
    verification_responses: tuple[RecordedProviderResponse, ...]
    attempt: PlanningAttempt | None
    failures: tuple[PlanningFailure, ...]

    @model_validator(mode="after")
    def require_consistent_attempt(self) -> PlanningAttemptArtifact:
        _validate_attempt_projection(self)
        _validate_response_content(self)
        _validate_disposition(self)
        return self


def planning_request_id(edition_id: Sha256, attempt_index: int, stage: str) -> Sha256:
    return sha256(
        canonical_json(
            {
                "edition_id": edition_id,
                "planning_attempt_index": attempt_index,
                "stage": stage,
            }
        )
    )


def _validate_attempt_projection(artifact: PlanningAttemptArtifact) -> None:
    attempt = artifact.attempt
    if (
        artifact.planning_response.attempt.status == "accepted"
        and artifact.planning_response.attempt.model != artifact.policy.planning_model
    ):
        raise ValueError("Planning response model does not match the planning policy")
    if any(
        response.attempt.status == "accepted"
        and response.attempt.model != artifact.policy.verification_model
        for response in artifact.verification_responses
    ):
        raise ValueError("Verification response model does not match the planning policy")
    if attempt is None:
        return
    if attempt.attempt_index != artifact.attempt_index:
        raise ValueError("Planning attempt artifact index does not match its attempt")
    if attempt.disposition != artifact.disposition:
        raise ValueError("Planning attempt artifact disposition does not match its attempt")
    if artifact.failures != attempt.failures:
        raise ValueError("Planning attempt artifact failures do not match its attempt")
    if policy_bundle_digest(artifact.policy) != attempt.plan.policy_bundle_digest:
        raise ValueError("Planning attempt policy does not match its screenplay")
    if len(artifact.verification_responses) != len(attempt.story_evidence):
        raise ValueError("Planning attempt artifact must retain every verifier response")


def _validate_response_content(artifact: PlanningAttemptArtifact) -> None:
    attempt = artifact.attempt
    if attempt is None:
        return
    planning = PlanningResponse.model_validate_json(
        artifact.planning_response.response_content, strict=True
    )
    if planning.stories != attempt.plan.stories:
        raise ValueError("Planning response does not match the recorded screenplay")
    for response, evidence in zip(
        artifact.verification_responses, attempt.story_evidence, strict=True
    ):
        if response.attempt.status == "rejected":
            if evidence.status != "rejected":
                raise ValueError("Rejected verifier response requires rejected evidence")
            continue
        verification = VerificationResponse.model_validate_json(
            response.response_content, strict=True
        )
        if (verification.status, verification.failures) != (
            evidence.status,
            evidence.failures,
        ):
            raise ValueError("Verifier response does not match its recorded evidence")


def _validate_disposition(artifact: PlanningAttemptArtifact) -> None:
    if artifact.disposition == "accepted" and (artifact.attempt is None or artifact.failures):
        raise ValueError("Accepted planning artifact requires its accepted attempt")
    if artifact.disposition == "rejected" and not artifact.failures:
        raise ValueError("Rejected planning artifact requires structured failures")


def planning_attempt_file(edition_id: Sha256, artifact: PlanningAttemptArtifact) -> ArtifactFile:
    content = canonical_json(artifact.model_dump(mode="json"))
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"{edition_id}:{artifact.attempt_index}:planning-attempt",
        artifact_kind="video_digest_planning_attempt",
        title=f"Video digest planning attempt {artifact.attempt_index}",
        content=content,
        r2_key=(
            f"news/video-digest/{edition_id}/planning/"
            f"attempt-{artifact.attempt_index}-{digest}.json"
        ),
        media_type="application/json",
    )


def parse_planning_attempt_file(edition_id: Sha256, file: ArtifactFile) -> PlanningAttemptArtifact:
    artifact = PlanningAttemptArtifact.model_validate_json(file.content, strict=True)
    if planning_attempt_file(edition_id, artifact) != file:
        raise ValueError("Planning attempt artifact is not canonical")
    _validate_response_identities(edition_id, artifact)
    return artifact


def _validate_response_identities(edition_id: Sha256, artifact: PlanningAttemptArtifact) -> None:
    _validate_recorded_attempt(
        artifact.planning_response,
        planning_request_id(edition_id, artifact.attempt_index, "planning"),
        PLANNING_OPERATION,
        artifact.attempt_index,
    )
    for position, response in enumerate(artifact.verification_responses):
        _validate_recorded_attempt(
            response,
            planning_request_id(edition_id, artifact.attempt_index, f"verification:{position}"),
            VERIFICATION_OPERATION,
            artifact.attempt_index,
        )


def _validate_recorded_attempt(
    response: RecordedProviderResponse,
    request_id: Sha256,
    operation: str,
    attempt_index: int,
) -> None:
    attempt = response.attempt
    expected = model_attempt_from_payload(
        response.provider_response,
        request_id=request_id,
        operation_key=operation,
        attempt_index=attempt_index,
        latency_ms=attempt.latency_ms,
        status=attempt.status,
        error=attempt.error,
        observed_at=attempt.observed_at,
    )
    if expected != attempt:
        raise ValueError("Recorded model attempt does not match its deterministic identity")


def verified_plan_file(verified: VerifiedDigestPlan) -> tuple[ArtifactFile, DigestPlan]:
    content = canonical_json(verified.model_dump(mode="json"))
    digest = sha256(content)
    file = artifact_file(
        artifact_id=verified.plan.edition_id,
        artifact_kind="video_digest_plan",
        title=f"Accepted video digest plan {verified.plan.edition_id}",
        content=content,
        r2_key=f"news/video-digest/{verified.plan.edition_id}/plans/{digest}.json",
        media_type="application/json",
    )
    stories = tuple(
        PlannedStory(
            story_id=planned_story_id(verified.plan.edition_id, position, story.report_subject_id),
            edition_id=verified.plan.edition_id,
            position=position,
            report_subject_id=story.report_subject_id,
            title=story.title,
            requested_duration_ms=story.requested_duration_ms,
        )
        for position, story in enumerate(verified.plan.stories)
    )
    return file, DigestPlan(
        edition_id=verified.plan.edition_id,
        artifact_version_id=file.version_id,
        stories=stories,
    )


def parse_verified_plan_file(file: ArtifactFile) -> tuple[VerifiedDigestPlan, DigestPlan]:
    verified = VerifiedDigestPlan.model_validate_json(file.content, strict=True)
    expected_file, plan = verified_plan_file(verified)
    if expected_file != file:
        raise ValueError("Verified digest plan artifact is not canonical")
    return verified, plan
