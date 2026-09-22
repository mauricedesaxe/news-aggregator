from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from romanian_news.analysis.attempts import model_attempt_from_payload
from romanian_news.catalog.artifacts import canonical_json, sha256
from romanian_news.video_digest.models import edition_id
from romanian_news.video_digest.planning import (
    PlanningAttempt,
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
    parse_planning_attempt_file,
    parse_verified_plan_file,
    planning_attempt_file,
    planning_request_id,
    verified_plan_file,
)

REPORT_VERSION = "a" * 64
POLICY_BUNDLE_VERSION = "b" * 64
EDITION = edition_id(REPORT_VERSION, POLICY_BUNDLE_VERSION)
PLANNING_MODEL = "google/test-planner"
VERIFICATION_MODEL = "openai/test-verifier"
POLICY = VideoDigestPolicyBundle(
    policy_id="artifact-tests",
    planning_model=PLANNING_MODEL,
    verification_model=VERIFICATION_MODEL,
    planning_prompt_digest=sha256(b"planning-prompt"),
    verification_prompt_digest=sha256(b"verification-prompt"),
)


def _stories() -> tuple[ScreenplayStory, ...]:
    narration = " ".join(f"cuvant{index}" for index in range(30))
    return tuple(
        ScreenplayStory(
            report_subject_id=str(position + 1) * 64,
            title=f"Story {position}",
            citation_article_version_ids=(sha256(f"citation-{position}".encode()),),
            narration=narration,
            visual_direction=f"Visual direction {position}",
            requested_duration_ms=15_000,
        )
        for position in range(2)
    )


def _screenplay() -> ScreenplayPlan:
    return ScreenplayPlan(
        edition_id=EDITION,
        daily_report_version_id=REPORT_VERSION,
        policy_bundle_version_id=POLICY_BUNDLE_VERSION,
        policy_bundle_digest=policy_bundle_digest(POLICY),
        stories=_stories(),
    )


def _recorded(
    operation: str,
    request_id: str,
    attempt_index: int,
    model: str,
    content: str,
    *,
    status: str = "accepted",
    error: str | None = None,
) -> RecordedProviderResponse:
    payload: dict[str, Any] = {
        "id": f"{operation}-{attempt_index}",
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
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0},
    }
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


def _artifact_dump() -> dict[str, Any]:
    screenplay = _screenplay()
    planning_content = canonical_json(
        PlanningResponse(stories=screenplay.stories).model_dump(mode="json")
    ).decode()
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
        attempt_index=0,
        plan=screenplay,
        story_evidence=evidence,
        disposition="accepted",
    )
    verifications = tuple(
        _recorded(
            VERIFICATION_OPERATION,
            planning_request_id(EDITION, 0, f"verification:{position}"),
            0,
            VERIFICATION_MODEL,
            '{"status":"accepted","failures":[]}',
        )
        for position in range(len(screenplay.stories))
    )
    return PlanningAttemptArtifact(
        attempt_index=0,
        disposition="accepted",
        policy=POLICY,
        planning_response=_recorded(
            PLANNING_OPERATION,
            planning_request_id(EDITION, 0, "planning"),
            0,
            PLANNING_MODEL,
            planning_content,
        ),
        verification_responses=verifications,
        attempt=attempt,
        failures=(),
    ).model_dump(mode="json")


def _parse(dump: dict[str, Any]) -> PlanningAttemptArtifact:
    artifact = PlanningAttemptArtifact.model_validate_json(json.dumps(dump))
    return parse_planning_attempt_file(EDITION, planning_attempt_file(EDITION, artifact))


def _validate(dump: dict[str, Any]) -> PlanningAttemptArtifact:
    return PlanningAttemptArtifact.model_validate_json(json.dumps(dump))


def _replaced(dump: dict[str, Any], path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    mutated = json.loads(json.dumps(dump))
    node: Any = mutated
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return mutated


def test_valid_artifact_round_trips_through_its_file() -> None:
    artifact = _parse(_artifact_dump())

    assert artifact.disposition == "accepted"
    assert artifact.attempt is not None
    assert len(artifact.verification_responses) == len(artifact.attempt.story_evidence)


def test_correlated_model_evidence_is_rejected() -> None:
    dump = _artifact_dump()

    planner_on_verifier = _replaced(
        dump, ("planning_response", "provider_response", "model"), VERIFICATION_MODEL
    )
    planner_on_verifier["planning_response"]["attempt"]["model"] = VERIFICATION_MODEL
    with pytest.raises(ValueError, match="Planning response model does not match"):
        _validate(planner_on_verifier)

    verifier_on_planner = _replaced(
        dump, ("verification_responses", 0, "provider_response", "model"), PLANNING_MODEL
    )
    verifier_on_planner["verification_responses"][0]["attempt"]["model"] = PLANNING_MODEL
    with pytest.raises(ValueError, match="Verification response model does not match"):
        _validate(verifier_on_planner)


def test_recorded_response_digest_and_identity_are_enforced() -> None:
    dump = _artifact_dump()

    wrong_digest = _replaced(dump, ("planning_response", "response_content_digest"), "0" * 64)
    with pytest.raises(ValueError, match="digest does not match its content"):
        _validate(wrong_digest)

    divergent_content = _replaced(
        dump,
        ("planning_response", "provider_response", "choices", 0, "message", "content"),
        "different narration",
    )
    with pytest.raises(ValueError, match="does not match its model attempt"):
        _validate(divergent_content)

    forged_request = _replaced(dump, ("planning_response", "attempt", "request_id"), "0" * 64)
    with pytest.raises(ValueError, match="deterministic identity"):
        _parse(forged_request)


def test_planning_response_must_match_the_recorded_screenplay() -> None:
    dump = _artifact_dump()
    stories = json.loads(dump["planning_response"]["response_content"])["stories"]
    stories[0]["narration"] = "tampered narration"
    tampered = _replaced(
        dump,
        ("planning_response", "response_content"),
        json.dumps({"stories": stories}),
    )
    tampered["planning_response"]["response_content_digest"] = sha256(
        tampered["planning_response"]["response_content"].encode()
    )
    tampered["planning_response"]["provider_response"]["choices"][0]["message"]["content"] = (
        tampered["planning_response"]["response_content"]
    )

    with pytest.raises(
        ValueError, match="Planning response does not match the recorded screenplay"
    ):
        _validate(tampered)


def test_verifier_responses_must_match_their_recorded_evidence() -> None:
    dump = _artifact_dump()

    divergent_status = _replaced(
        dump,
        ("verification_responses", 1, "response_content"),
        json.dumps(
            {
                "status": "rejected",
                "failures": [{"code": "unsupported_claim", "message": "No evidence"}],
            }
        ),
    )
    divergent_status["verification_responses"][1]["response_content_digest"] = sha256(
        divergent_status["verification_responses"][1]["response_content"].encode()
    )
    divergent_status["verification_responses"][1]["provider_response"]["choices"][0]["message"][
        "content"
    ] = divergent_status["verification_responses"][1]["response_content"]

    with pytest.raises(ValueError, match="Verifier response does not match its recorded evidence"):
        _validate(divergent_status)


def test_planning_attempt_files_must_be_canonical() -> None:
    artifact = _parse(_artifact_dump())
    canonical = planning_attempt_file(EDITION, artifact)
    pretty = json.dumps(json.loads(canonical.content), indent=2).encode()
    non_canonical = canonical.model_copy(
        update={"content": pretty, "content_digest": sha256(pretty)}
    )

    with pytest.raises(ValueError, match="not canonical"):
        parse_planning_attempt_file(EDITION, non_canonical)


def test_verified_plan_files_must_be_canonical() -> None:
    screenplay = _screenplay()
    plan = VerifiedDigestPlan(
        screenplay_plan_digest=screenplay_plan_digest(screenplay),
        plan=screenplay,
        report_subject_ids=tuple(story.report_subject_id for story in screenplay.stories),
        accepted_evidence=tuple(
            StoryVerificationEvidence(
                story_screenplay_digest=screenplay_story_digest(story),
                daily_report_version_id=screenplay.daily_report_version_id,
                policy_bundle_digest=screenplay.policy_bundle_digest,
                citation_article_version_ids=story.citation_article_version_ids,
                status="accepted",
            )
            for story in screenplay.stories
        ),
    )
    file, _ = verified_plan_file(plan)
    parsed, _ = parse_verified_plan_file(file)
    assert parsed == plan

    pretty = json.dumps(json.loads(file.content), indent=2).encode()
    with pytest.raises(ValueError, match="not canonical"):
        parse_verified_plan_file(
            file.model_copy(update={"content": pretty, "content_digest": sha256(pretty)})
        )
