from __future__ import annotations

from datetime import UTC, datetime

from romanian_news import Sha256
from romanian_news.analysis.attempts import model_attempt_from_payload
from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.identity import canonical_json, sha256
from romanian_news.video_digest.models import DigestPlan, EditionIdentity
from romanian_news.video_digest.planning import (
    PlanningAttempt,
    PlanningFailure,
    ScreenplayPlan,
    ScreenplayStory,
    StoryVerificationEvidence,
    VerifiedDigestPlan,
    VideoDigestPolicyBundle,
    policy_bundle_digest,
    screenplay_plan_digest,
    screenplay_story_digest,
)
from romanian_news.video_digest.planning_artifacts import (
    PLANNING_OPERATION,
    VERIFICATION_OPERATION,
    PlanningAttemptArtifact,
    PlanningResponse,
    RecordedProviderResponse,
    VerificationResponse,
    planning_attempt_file,
    planning_request_id,
    verified_plan_file,
)


def accepted_planning_files(
    edition: EditionIdentity,
    stories: tuple[tuple[str, str, int], ...],
    *,
    attempt_index: int = 0,
    seed: str = "contract",
) -> tuple[DigestPlan, ArtifactFile, ArtifactFile]:
    policy = _policy(seed)
    screenplay_stories = tuple(
        ScreenplayStory(
            report_subject_id=subject_id,
            title=title,
            citation_article_version_ids=(sha256(f"{seed}:citation:{position}".encode()),),
            narration=" ".join(f"word{word}" for word in range(30)),
            visual_direction=f"Contract visual direction {position}",
            requested_duration_ms=duration_ms,
        )
        for position, (subject_id, title, duration_ms) in enumerate(stories)
    )
    screenplay = ScreenplayPlan(
        edition_id=edition.edition_id,
        daily_report_version_id=edition.daily_report_version_id,
        policy_bundle_version_id=edition.policy_bundle_version_id,
        policy_bundle_digest=policy_bundle_digest(policy),
        stories=screenplay_stories,
    )
    evidence = tuple(
        StoryVerificationEvidence(
            story_screenplay_digest=screenplay_story_digest(story),
            daily_report_version_id=screenplay.daily_report_version_id,
            policy_bundle_digest=screenplay.policy_bundle_digest,
            citation_article_version_ids=story.citation_article_version_ids,
            status="accepted",
        )
        for story in screenplay.stories
    )
    attempt = PlanningAttempt(
        attempt_index=attempt_index,
        plan=screenplay,
        story_evidence=evidence,
        disposition="accepted",
    )
    artifact = PlanningAttemptArtifact(
        attempt_index=attempt_index,
        disposition="accepted",
        policy=policy,
        planning_response=_response(
            seed,
            attempt_index,
            PLANNING_OPERATION,
            "google/contract-planner",
            canonical_json(
                PlanningResponse(stories=screenplay_stories).model_dump(mode="json")
            ).decode(),
            request_id=planning_request_id(edition.edition_id, attempt_index, "planning"),
            position=None,
        ),
        verification_responses=tuple(
            _response(
                seed,
                attempt_index,
                VERIFICATION_OPERATION,
                "openai/contract-verifier",
                canonical_json(
                    VerificationResponse(status=item.status, failures=item.failures).model_dump(
                        mode="json"
                    )
                ).decode(),
                request_id=planning_request_id(
                    edition.edition_id, attempt_index, f"verification:{position}"
                ),
                position=position,
            )
            for position, item in enumerate(evidence)
        ),
        attempt=attempt,
        failures=(),
    )
    verified = VerifiedDigestPlan(
        screenplay_plan_digest=screenplay_plan_digest(screenplay),
        plan=screenplay,
        report_subject_ids=tuple(story.report_subject_id for story in screenplay.stories),
        accepted_evidence=evidence,
    )
    plan_file, plan = verified_plan_file(verified)
    return plan, plan_file, planning_attempt_file(edition.edition_id, artifact)


def rejected_planning_file(
    edition_id: Sha256, attempt_index: int = 0, *, seed: str = "contract"
) -> ArtifactFile:
    failure = PlanningFailure(code="invalid_planning_response", message="Contract rejection")
    policy = _policy(seed)
    artifact = PlanningAttemptArtifact(
        attempt_index=attempt_index,
        disposition="rejected",
        policy=policy,
        planning_response=_response(
            seed,
            attempt_index,
            PLANNING_OPERATION,
            "google/contract-planner",
            "{}",
            request_id=planning_request_id(edition_id, attempt_index, "planning"),
            position=None,
            error="Contract rejection",
        ),
        verification_responses=(),
        attempt=None,
        failures=(failure,),
    )
    return planning_attempt_file(edition_id, artifact)


def _response(
    seed: str,
    attempt_index: int,
    operation: str,
    model: str,
    content: str,
    *,
    request_id: Sha256,
    position: int | None,
    error: str | None = None,
) -> RecordedProviderResponse:
    suffix = "planning" if position is None else f"verification-{position}"
    response_id = f"{seed}-{attempt_index}-{suffix}"
    payload: dict[str, object] = {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0,
        },
    }
    status = "rejected" if error is not None else "accepted"
    attempt = model_attempt_from_payload(
        payload,
        request_id=request_id,
        operation_key=operation,
        attempt_index=attempt_index,
        latency_ms=1,
        status=status,
        error=error,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return RecordedProviderResponse(
        attempt=attempt,
        response_content=content,
        response_content_digest=sha256(content.encode()),
        provider_response=payload,
    )


def _policy(seed: str) -> VideoDigestPolicyBundle:
    return VideoDigestPolicyBundle(
        policy_id=f"contract-{seed}",
        planning_model="google/contract-planner",
        verification_model="openai/contract-verifier",
        planning_prompt_digest=sha256(f"{seed}:planning-prompt".encode()),
        verification_prompt_digest=sha256(f"{seed}:verification-prompt".encode()),
    )
