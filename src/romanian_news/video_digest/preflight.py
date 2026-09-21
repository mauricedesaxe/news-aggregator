from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol, TypeVar

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelAttempt, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.tracing import ProviderChatRequest, trace_provider_call
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, canonical_json, sha256
from romanian_news.catalog.video_digest import (
    PlanningAttemptReference,
    checkpoint_edition_verification,
    checkpoint_planning_attempt,
    checkpoint_story_verification,
    read_planning_attempts,
    record_policy_bundle,
)
from romanian_news.reports import DailyReport, DailyReportSection
from romanian_news.storage import publish_immutable_r2_objects, read_verified_r2_object
from romanian_news.video_digest.models import (
    DigestPlan,
    PlannedStory,
    SlotLease,
    edition_id,
    planned_story_id,
)
from romanian_news.video_digest.planning import (
    GenerationAuthorization,
    PlanningAttempt,
    PlanningFailure,
    ScreenplayStory,
    StoryVerificationEvidence,
    VerifiedDigestPlan,
    VideoDigestPolicyBundle,
    accept_planning_attempt,
    authorize_generation,
    create_screenplay_plan,
    record_planning_attempt,
    screenplay_story_digest,
)

PLANNING_OPERATION = "news.video_digest.plan"
VERIFICATION_OPERATION = "news.video_digest.verify_story"
PLANNING_PROMPT = (
    "Write one complete Romanian-language video screenplay for every supplied main story. "
    "The recurring lead scientist and three-eyed pear-shaped alien co-host deliver each story "
    "inside a narrow retro-futurist broadcast booth. Keep both hosts visually stable, but make "
    "the action fast, strange, physical, and specific to the news: props transform, diagrams "
    "move, and the booth reacts to the facts instead of showing generic presenter shots. "
    "Preserve report order and exact subject and citation IDs. Write 25 to 35 spoken words "
    "of concise dialogue per narration, request exactly 15000 milliseconds, and use only "
    "supplied report evidence. Never invent names, dates, places, numbers, or outcomes."
)
VERIFICATION_PROMPT = (
    "Verify one screenplay story independently against only the supplied matching report story. "
    "Reject unsupported claims, citation changes, misleading emphasis, or narration outside the "
    "supplied evidence. Return structured failures and no editorial rewrite."
)
PLANNING_MODEL = "google/gemini-3.8-flash"
VERIFICATION_MODEL = "openai/gpt-4.1-mini"
MAX_TOKENS = 16_000


class PlanningReport(NewsModel):
    version_id: Sha256
    report: DailyReport

    @model_validator(mode="after")
    def require_exact_artifact_version(self) -> PlanningReport:
        content = canonical_json(self.report.model_dump(mode="json"))
        expected = artifact_file(
            artifact_id=f"news:daily:{self.report.day.isoformat()}",
            artifact_kind="news_daily_report",
            title=f"Romanian news report for {self.report.day.isoformat()}",
            content=content,
            r2_key=f"news/reports/daily/{self.report.day.isoformat()}/{sha256(content)}.json",
            media_type="application/json",
        )
        if self.version_id != expected.version_id:
            raise ValueError("Planning report version does not match its content")
        return self


class _PlanningResponse(NewsModel):
    stories: Annotated[tuple[ScreenplayStory, ...], Field(min_length=1)]


class _VerificationResponse(NewsModel):
    status: Literal["accepted", "rejected"]
    failures: tuple[PlanningFailure, ...]

    @model_validator(mode="after")
    def require_failures_for_rejection(self) -> _VerificationResponse:
        if self.status == "accepted" and self.failures:
            raise ValueError("Accepted verification cannot contain failures")
        if self.status == "rejected" and not self.failures:
            raise ValueError("Rejected verification requires a structured failure")
        return self


def _schema_digest(model: type[NewsModel]) -> Sha256:
    return sha256(canonical_json(model.model_json_schema()))


class VideoDigestPolicyDefinition(NewsModel):
    policy: VideoDigestPolicyBundle
    planning_prompt: str
    verification_prompt: str
    planning_response_schema_digest: Sha256
    verification_response_schema_digest: Sha256

    @model_validator(mode="after")
    def require_exact_prompts_and_schemas(self) -> VideoDigestPolicyDefinition:
        if sha256(self.planning_prompt.encode()) != self.policy.planning_prompt_digest:
            raise ValueError("Planning prompt digest does not match its prompt")
        if sha256(self.verification_prompt.encode()) != self.policy.verification_prompt_digest:
            raise ValueError("Verification prompt digest does not match its prompt")
        if self.planning_response_schema_digest != _schema_digest(_PlanningResponse):
            raise ValueError("Planning response schema digest does not match its schema")
        if self.verification_response_schema_digest != _schema_digest(_VerificationResponse):
            raise ValueError("Verification response schema digest does not match its schema")
        return self


class PlanningPolicy(NewsModel):
    definition: VideoDigestPolicyDefinition
    artifact: ArtifactFile

    @model_validator(mode="after")
    def require_immutable_artifact(self) -> PlanningPolicy:
        content = canonical_json(self.definition.model_dump(mode="json"))
        expected_id = f"video-digest-policy:{self.definition.policy.policy_id}"
        expected_digest = sha256(content)
        if (
            self.artifact.artifact_id != expected_id
            or self.artifact.artifact_kind != "video_digest_policy"
            or self.artifact.content != content
            or self.artifact.content_digest != expected_digest
            or self.artifact.version_id != sha256(f"{expected_id}\0{expected_digest}".encode())
            or self.artifact.r2_key != f"news/video-digest/policies/{expected_digest}.json"
        ):
            raise ValueError("Planning policy artifact does not match its definition")
        return self


class RecordedProviderResponse(NewsModel):
    attempt: ModelAttempt
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]


class PlanningAttemptArtifact(NewsModel):
    attempt_index: int
    disposition: Literal["accepted", "rejected"]
    planning_response: RecordedProviderResponse
    verification_responses: tuple[RecordedProviderResponse, ...]
    attempt: PlanningAttempt | None
    failures: tuple[PlanningFailure, ...]

    @model_validator(mode="after")
    def require_consistent_attempt(self) -> PlanningAttemptArtifact:
        if self.attempt is not None:
            if self.attempt.attempt_index != self.attempt_index:
                raise ValueError("Planning attempt artifact index does not match its attempt")
            if self.attempt.disposition != self.disposition:
                raise ValueError("Planning attempt artifact disposition does not match its attempt")
            if len(self.verification_responses) != len(self.attempt.story_evidence):
                raise ValueError("Planning attempt artifact must retain every verifier response")
        if self.disposition == "accepted" and (self.attempt is None or self.failures):
            raise ValueError("Accepted planning artifact requires its accepted attempt")
        if self.disposition == "rejected" and not self.failures:
            raise ValueError("Rejected planning artifact requires structured failures")
        return self


class PreparedPaidGeneration(NewsModel):
    authorization: GenerationAuthorization
    plan: DigestPlan


class PlanningExhaustion(NewsModel):
    edition_id: Sha256
    attempts: Annotated[tuple[PlanningAttemptArtifact, ...], Field(min_length=3, max_length=3)]


class PlanningExhaustedError(ValueError):
    failure: PlanningExhaustion

    def __init__(self, failure: PlanningExhaustion) -> None:
        super().__init__("Video digest planning exhausted the initial attempt and two rewrites")
        self.failure = failure


class PlanningProvider(Protocol):
    def __call__(self, request: ProviderChatRequest) -> ChatCompletion: ...


PRODUCTION_POLICY_DEFINITION = VideoDigestPolicyDefinition(
    policy=VideoDigestPolicyBundle(
        policy_id="video-digest-production-v1",
        planning_model=PLANNING_MODEL,
        verification_model=VERIFICATION_MODEL,
        planning_prompt_digest=sha256(PLANNING_PROMPT.encode()),
        verification_prompt_digest=sha256(VERIFICATION_PROMPT.encode()),
    ),
    planning_prompt=PLANNING_PROMPT,
    verification_prompt=VERIFICATION_PROMPT,
    planning_response_schema_digest=_schema_digest(_PlanningResponse),
    verification_response_schema_digest=_schema_digest(_VerificationResponse),
)


def policy_artifact(
    definition: VideoDigestPolicyDefinition = PRODUCTION_POLICY_DEFINITION,
) -> PlanningPolicy:
    content = canonical_json(definition.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"video-digest-policy:{definition.policy.policy_id}",
        artifact_kind="video_digest_policy",
        title=f"Video digest policy {definition.policy.policy_id}",
        content=content,
        r2_key=f"news/video-digest/policies/{sha256(content)}.json",
        media_type="application/json",
    )
    return PlanningPolicy(definition=definition, artifact=file)


PRODUCTION_POLICY = policy_artifact()


def publish_policy(policy: PlanningPolicy = PRODUCTION_POLICY) -> Sha256:
    publish_immutable_r2_objects(((policy.artifact.r2_key, policy.artifact.content),))
    return record_policy_bundle(policy.artifact, recorded_at=datetime.now(UTC))


def prepare_paid_generation(
    lease: SlotLease,
    report: PlanningReport,
    policy: PlanningPolicy = PRODUCTION_POLICY,
    *,
    provider: PlanningProvider | None = None,
) -> PreparedPaidGeneration:
    if lease.edition_id != edition_id(report.version_id, policy.artifact.version_id):
        raise ValueError("Planning inputs do not match the claimed edition")
    references = read_planning_attempts(lease.edition_id)
    recorded = tuple(_read_attempt(reference) for reference in references)
    accepted = next((item for item in recorded if item.disposition == "accepted"), None)
    if accepted is not None:
        return _finish_accepted(
            lease,
            report,
            policy,
            accepted,
            references[accepted.attempt_index],
        )
    if len(recorded) >= 3:
        raise PlanningExhaustedError(
            PlanningExhaustion(edition_id=lease.edition_id, attempts=recorded)
        )

    call = provider or _openrouter_completion
    attempts = list(recorded)
    for attempt_index in range(len(recorded), 3):
        artifact = _run_attempt(lease, report, policy, attempt_index, tuple(attempts), call)
        attempt_file = _attempt_file(lease, artifact)
        if artifact.disposition == "rejected":
            publish_immutable_r2_objects(((attempt_file.r2_key, attempt_file.content),))
            checkpoint_planning_attempt(
                lease,
                attempt_index,
                "rejected",
                evidence_file=attempt_file,
                recorded_at=datetime.now(UTC),
            )
            attempts.append(artifact)
            continue
        return _finish_new_accepted(lease, report, policy, artifact, attempt_file)
    raise PlanningExhaustedError(
        PlanningExhaustion(edition_id=lease.edition_id, attempts=tuple(attempts))
    )


def _run_attempt(
    lease: SlotLease,
    report: PlanningReport,
    policy: PlanningPolicy,
    attempt_index: int,
    prior_attempts: tuple[PlanningAttemptArtifact, ...],
    provider: PlanningProvider,
) -> PlanningAttemptArtifact:
    planning_request = _planning_request(lease, report, policy, attempt_index, prior_attempts)
    parsed, planning_response, parse_failure = _call_model(
        PLANNING_OPERATION,
        _request_id(lease, attempt_index, "planning"),
        planning_request,
        attempt_index,
        provider,
        _PlanningResponse,
    )
    if parsed is None:
        assert parse_failure is not None
        failure = PlanningFailure(code="invalid_planning_response", message=parse_failure)
        return PlanningAttemptArtifact(
            attempt_index=attempt_index,
            disposition="rejected",
            planning_response=planning_response,
            verification_responses=(),
            attempt=None,
            failures=(failure,),
        )

    try:
        plan = create_screenplay_plan(
            report.report,
            report.version_id,
            policy.artifact.version_id,
            policy.definition.policy,
            parsed.stories,
        )
    except ValueError as error:
        failure = PlanningFailure(code="invalid_screenplay_plan", message=str(error))
        return PlanningAttemptArtifact(
            attempt_index=attempt_index,
            disposition="rejected",
            planning_response=planning_response,
            verification_responses=(),
            attempt=None,
            failures=(failure,),
        )
    sections = {
        section.theme_id: section for section in report.report.sections if section.tier == "main"
    }
    evidence: list[StoryVerificationEvidence] = []
    verifier_responses: list[RecordedProviderResponse] = []
    for position, story in enumerate(plan.stories):
        section = sections[story.report_subject_id]
        verification, response, parse_error = _call_model(
            VERIFICATION_OPERATION,
            _request_id(lease, attempt_index, f"verification:{position}"),
            _verification_request(policy, story, section),
            attempt_index,
            provider,
            _VerificationResponse,
        )
        verifier_responses.append(response)
        if verification is None:
            verification = _VerificationResponse(
                status="rejected",
                failures=(
                    PlanningFailure(
                        code="invalid_verification_response",
                        message=parse_error or "Verifier response was invalid",
                    ),
                ),
            )
        evidence.append(
            StoryVerificationEvidence(
                story_screenplay_digest=screenplay_story_digest(story),
                daily_report_version_id=report.version_id,
                policy_bundle_digest=plan.policy_bundle_digest,
                citation_article_version_ids=story.citation_article_version_ids,
                status=verification.status,
                failures=verification.failures,
            )
        )

    failures = [
        failure
        for position, item in enumerate(evidence)
        for failure in (
            PlanningFailure(code=f"story_{position}:{value.code}", message=value.message)
            for value in item.failures
        )
    ]
    disposition: Literal["accepted", "rejected"] = "rejected" if failures else "accepted"
    attempt = record_planning_attempt(
        attempt_index,
        plan,
        tuple(evidence),
        disposition,
        failures=tuple(failures),
    )
    return PlanningAttemptArtifact(
        attempt_index=attempt_index,
        disposition=disposition,
        planning_response=planning_response,
        verification_responses=tuple(verifier_responses),
        attempt=attempt,
        failures=tuple(failures),
    )


def _finish_new_accepted(
    lease: SlotLease,
    report: PlanningReport,
    policy: PlanningPolicy,
    artifact: PlanningAttemptArtifact,
    attempt_file: ArtifactFile,
) -> PreparedPaidGeneration:
    attempt = artifact.attempt
    assert attempt is not None
    verified = accept_planning_attempt(report.report, policy.definition.policy, attempt)
    plan_file, plan = _canonical_plan_file(verified)
    publish_immutable_r2_objects(
        ((attempt_file.r2_key, attempt_file.content), (plan_file.r2_key, plan_file.content))
    )
    checkpoint_planning_attempt(
        lease,
        artifact.attempt_index,
        "accepted",
        evidence_file=attempt_file,
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=datetime.now(UTC),
    )
    return _checkpoint_evidence_and_manifest(lease, verified, plan)


def _finish_accepted(
    lease: SlotLease,
    report: PlanningReport,
    policy: PlanningPolicy,
    artifact: PlanningAttemptArtifact,
    reference: PlanningAttemptReference,
) -> PreparedPaidGeneration:
    attempt = artifact.attempt
    assert attempt is not None
    verified = accept_planning_attempt(report.report, policy.definition.policy, attempt)
    _plan_file, plan = _canonical_plan_file(verified)
    if reference.accepted_plan_artifact_version_id != plan.artifact_version_id:
        raise ValueError("Recorded accepted plan does not match its planning artifact")
    return _checkpoint_evidence_and_manifest(lease, verified, plan)


def _checkpoint_evidence_and_manifest(
    lease: SlotLease,
    verified: VerifiedDigestPlan,
    plan: DigestPlan,
) -> PreparedPaidGeneration:
    for story, item in zip(plan.stories, verified.accepted_evidence, strict=True):
        content = canonical_json(item.model_dump(mode="json"))
        file = artifact_file(
            artifact_id=story.story_id,
            artifact_kind="video_digest_story_verification",
            title=f"Verification for video digest story {story.position}",
            content=content,
            r2_key=(
                f"news/video-digest/{plan.edition_id}/verification/"
                f"{story.position}-{sha256(content)}.json"
            ),
            media_type="application/json",
        )
        publish_immutable_r2_objects(((file.r2_key, file.content),))
        checkpoint_story_verification(
            lease, story.story_id, evidence_file=file, recorded_at=datetime.now(UTC)
        )
    authorization = authorize_generation(verified)
    content = canonical_json(authorization.model_dump(mode="json"))
    manifest = artifact_file(
        artifact_id=f"{lease.edition_id}:verification-manifest",
        artifact_kind="video_digest_verification_manifest",
        title=f"Verification manifest for video digest edition {lease.edition_id}",
        content=content,
        r2_key=f"news/video-digest/{lease.edition_id}/verification/{sha256(content)}.json",
        media_type="application/json",
    )
    publish_immutable_r2_objects(((manifest.r2_key, manifest.content),))
    checkpoint_edition_verification(lease, manifest_file=manifest, recorded_at=datetime.now(UTC))
    return PreparedPaidGeneration(authorization=authorization, plan=plan)


def _canonical_plan_file(verified: VerifiedDigestPlan) -> tuple[ArtifactFile, DigestPlan]:
    content = canonical_json(verified.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=verified.plan.edition_id,
        artifact_kind="video_digest_plan",
        title=f"Accepted video digest plan {verified.plan.edition_id}",
        content=content,
        r2_key=f"news/video-digest/{verified.plan.edition_id}/plans/{sha256(content)}.json",
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


def _attempt_file(lease: SlotLease, artifact: PlanningAttemptArtifact) -> ArtifactFile:
    content = canonical_json(artifact.model_dump(mode="json"))
    return artifact_file(
        artifact_id=f"{lease.edition_id}:{artifact.attempt_index}:planning-attempt",
        artifact_kind="video_digest_planning_attempt",
        title=f"Video digest planning attempt {artifact.attempt_index}",
        content=content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/planning/"
            f"attempt-{artifact.attempt_index}-{sha256(content)}.json"
        ),
        media_type="application/json",
    )


def _read_attempt(reference: PlanningAttemptReference) -> PlanningAttemptArtifact:
    artifact = PlanningAttemptArtifact.model_validate_json(
        read_verified_r2_object(reference.evidence.r2_key, reference.evidence.content_digest),
        strict=True,
    )
    if (artifact.attempt_index, artifact.disposition) != (
        reference.attempt_index,
        reference.disposition,
    ):
        raise ValueError("Recorded planning attempt does not match its catalog projection")
    return artifact


def _planning_request(
    lease: SlotLease,
    report: PlanningReport,
    policy: PlanningPolicy,
    attempt_index: int,
    prior_attempts: tuple[PlanningAttemptArtifact, ...],
) -> ProviderChatRequest:
    context = {
        "edition_id": lease.edition_id,
        "attempt_index": attempt_index,
        "main_stories": [
            section.model_dump(mode="json")
            for section in report.report.sections
            if section.tier == "main"
        ],
        "prior_failures": [
            {
                "attempt_index": item.attempt_index,
                "failures": [failure.model_dump(mode="json") for failure in item.failures],
            }
            for item in prior_attempts
        ],
    }
    return _provider_request(
        policy.definition.policy.planning_model,
        policy.definition.planning_prompt,
        context,
        _PlanningResponse.model_json_schema(),
        "romanian_news_video_digest_plan",
    )


def _verification_request(
    policy: PlanningPolicy,
    story: ScreenplayStory,
    section: DailyReportSection,
) -> ProviderChatRequest:
    return _provider_request(
        policy.definition.policy.verification_model,
        policy.definition.verification_prompt,
        {
            "story": story.model_dump(mode="json"),
            "report_evidence": section.model_dump(mode="json"),
        },
        _VerificationResponse.model_json_schema(),
        "romanian_news_video_digest_story_verification",
    )


def _provider_request(
    model: str,
    prompt: str,
    context: object,
    schema: dict[str, object],
    schema_name: str,
) -> ProviderChatRequest:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    ]
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        "extra_body": {
            "provider": {"require_parameters": True},
            "reasoning": {"effort": "low"},
        },
    }


_ResponseT = TypeVar("_ResponseT", bound=NewsModel)


def _call_model(
    operation: str,
    request_id: Sha256,
    request: ProviderChatRequest,
    attempt_index: int,
    provider: PlanningProvider,
    response_type: type[_ResponseT],
) -> tuple[_ResponseT | None, RecordedProviderResponse, str | None]:
    started = time.monotonic()
    call = trace_provider_call(
        operation,
        request_id,
        request,
        lambda: provider(request),
    )
    response = call.response
    content = response.choices[0].message.content or ""
    parsed = None
    error = None
    try:
        parsed = response_type.model_validate_json(content, strict=True)
    except ValueError as exc:
        error = str(exc)
    status: Literal["accepted", "rejected"] = "accepted" if error is None else "rejected"
    recorded = record_model_attempt(
        response,
        request_id=request_id,
        operation_key=operation,
        attempt_index=attempt_index,
        latency_ms=round((time.monotonic() - started) * 1000),
        status=status,
        error=error,
        fallback_response_id=str(call.call_id),
        trace=call.trace,
    )
    return (
        parsed,
        RecordedProviderResponse(
            attempt=recorded,
            response_content=content,
            response_content_digest=sha256(content.encode()),
            provider_response=response.model_dump(mode="json"),
        ),
        error,
    )


def _openrouter_completion(request: ProviderChatRequest) -> ChatCompletion:
    return openrouter_client().chat.completions.create(**request)


def _request_id(lease: SlotLease, attempt_index: int, stage: str) -> Sha256:
    return sha256(
        canonical_json(
            {
                "edition_id": lease.edition_id,
                "planning_attempt_index": attempt_index,
                "stage": stage,
            }
        )
    )
