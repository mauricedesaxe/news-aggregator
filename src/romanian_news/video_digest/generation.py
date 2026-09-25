from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Protocol, cast

import requests
from pydantic import Field, HttpUrl, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    canonical_json,
    sha256,
)
from romanian_news.catalog.video_digest import (
    GenerationAttemptReference,
    checkpoint_fal_queue_state,
    checkpoint_generation_failure,
    checkpoint_generation_request,
    checkpoint_generation_response,
    checkpoint_generation_submission,
    fail_slot,
    read_generation_attempts,
    read_generation_deadline,
    record_generation_policy,
)
from romanian_news.config import FAL_KEY
from romanian_news.storage import (
    presigned_r2_url,
    publish_immutable_r2_objects,
    publish_private_video_object,
    read_verified_r2_object,
)
from romanian_news.video_digest.models import (
    EditionIdField,
    EstimatedAttemptCost,
    GenerationAdmission,
    GenerationBudgetLimits,
    GenerationRequestIdentity,
    GenerationRequestIdField,
    GenerationStage,
    MeasuredAttemptCost,
    NonEmptyText,
    SlotLease,
    StoryId,
    StoryIdField,
    UnknownAttemptCost,
    generation_request_id,
)
from romanian_news.video_digest.planning import (
    GenerationAuthorization,
    ScreenplayStory,
    screenplay_story_digest,
)
from romanian_news.video_digest.preflight import PreparedPaidGeneration

FAL_H3_ENDPOINT = "minimax/h3-max/reference-to-video"
FAL_QUEUE_BASE_URL = "https://queue.fal.run"
FAL_ESTIMATED_ATTEMPT_USD = Decimal("3.25632")
FAL_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024


class GenerationPolicy(NewsModel):
    policy_id: Literal["h3-max-production-v1"] = "h3-max-production-v1"
    endpoint: Literal["minimax/h3-max/reference-to-video"] = FAL_H3_ENDPOINT
    estimated_attempt_usd: Annotated[Decimal, Field(gt=0)] = FAL_ESTIMATED_ATTEMPT_USD
    story_limit_usd: Annotated[Decimal, Field(gt=0)] = Decimal("7")
    edition_per_story_limit_usd: Annotated[Decimal, Field(gt=0)] = Decimal("7")
    bucharest_day_limit_usd: Annotated[Decimal, Field(gt=0)] = Decimal("150")
    calendar_month_limit_usd: Annotated[Decimal, Field(gt=0)] = Decimal("1000")
    deadline_minutes: Literal[90] = 90


class GenerationPolicyArtifact(NewsModel):
    policy: GenerationPolicy
    artifact: ArtifactFile

    @model_validator(mode="after")
    def require_matching_artifact(self) -> GenerationPolicyArtifact:
        content = canonical_json(self.policy.model_dump(mode="json"))
        expected_id = f"video-digest-generation-policy:{self.policy.policy_id}"
        expected = artifact_file(
            artifact_id=expected_id,
            artifact_kind="video_digest_generation_policy",
            title=f"Video digest generation policy {self.policy.policy_id}",
            content=content,
            r2_key=f"news/video-digest/policies/generation-{sha256(content)}.json",
            media_type="application/json",
        )
        if self.artifact != expected:
            raise ValueError("Generation policy artifact does not match its policy")
        return self


DEFAULT_GENERATION_POLICY = GenerationPolicy()


def generation_policy_artifact(
    policy: GenerationPolicy = DEFAULT_GENERATION_POLICY,
) -> GenerationPolicyArtifact:
    content = canonical_json(policy.model_dump(mode="json"))
    return GenerationPolicyArtifact(
        policy=policy,
        artifact=artifact_file(
            artifact_id=f"video-digest-generation-policy:{policy.policy_id}",
            artifact_kind="video_digest_generation_policy",
            title=f"Video digest generation policy {policy.policy_id}",
            content=content,
            r2_key=f"news/video-digest/policies/generation-{sha256(content)}.json",
            media_type="application/json",
        ),
    )


PRODUCTION_GENERATION_POLICY = generation_policy_artifact()


class H3ReferencePack(NewsModel):
    videos: Annotated[tuple[ArtifactReference, ...], Field(min_length=1)]
    audio: Annotated[tuple[ArtifactReference, ...], Field(min_length=1)]


class H3GenerationRequest(NewsModel):
    edition_id: EditionIdField
    story_id: StoryIdField
    story_position: Annotated[int, Field(ge=0)]
    attempt_index: Annotated[int, Field(ge=0, le=1)]
    authorization: GenerationAuthorization
    verified_plan_digest: Sha256
    screenplay_story_digest: Sha256
    story: ScreenplayStory
    references: H3ReferencePack
    generation_policy_artifact_version_id: Sha256
    endpoint: Literal["minimax/h3-max/reference-to-video"] = FAL_H3_ENDPOINT
    duration: Literal[15] = 15
    resolution: Literal["768P"] = "768P"
    aspect_ratio: Literal["16:9"] = "16:9"
    prompt_expansion_mode: Literal["disabled"] = "disabled"


class FalSubmissionReceipt(NewsModel):
    request_id: NonEmptyText
    status_url: HttpUrl
    response_url: HttpUrl
    cancel_url: HttpUrl | None = None
    queue_position: int | None = None


class FalQueueStatus(NewsModel):
    status: Literal["IN_QUEUE", "IN_PROGRESS", "COMPLETED"]
    request_id: NonEmptyText
    response_url: HttpUrl | None = None
    queue_position: int | None = None
    error: str | None = None
    error_type: str | None = None
    logs: tuple[dict[str, object], ...] = ()
    metrics: dict[str, object] | None = None


class FalVideo(NewsModel):
    url: HttpUrl
    content_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None


class FalH3Result(NewsModel):
    video: FalVideo


class CandidateReference(NewsModel):
    r2_key: NonEmptyText
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    media_type: Literal["video/mp4"] = "video/mp4"


class H3CompletionEvidence(NewsModel):
    request_id: GenerationRequestIdField
    provider_receipt_id: NonEmptyText
    provider_response: dict[str, object]
    candidate: CandidateReference
    cost: MeasuredAttemptCost | UnknownAttemptCost


class CandidateReady(NewsModel):
    kind: Literal["candidate_ready"] = "candidate_ready"
    request_id: GenerationRequestIdField
    story_id: StoryIdField
    story_position: Annotated[int, Field(ge=0)]
    attempt_index: Annotated[int, Field(ge=0, le=1)]
    response_artifact_version_id: Sha256
    candidate: CandidateReference
    cost: MeasuredAttemptCost | UnknownAttemptCost


class GenerationInProgress(NewsModel):
    kind: Literal["in_progress"] = "in_progress"
    request_id: GenerationRequestIdField
    provider_receipt_id: NonEmptyText
    provider_status: Literal["IN_QUEUE", "IN_PROGRESS"]


class GenerationComplete(NewsModel):
    kind: Literal["complete"] = "complete"
    edition_id: EditionIdField


class GenerationFailed(NewsModel):
    kind: Literal["failed"] = "failed"
    edition_id: EditionIdField
    reason: NonEmptyText


class GenerationRetryAvailable(NewsModel):
    kind: Literal["retry_available"] = "retry_available"
    edition_id: EditionIdField
    story_position: Annotated[int, Field(ge=0)]
    reason: NonEmptyText


GenerationOutcome = (
    CandidateReady
    | GenerationInProgress
    | GenerationComplete
    | GenerationRetryAvailable
    | GenerationFailed
)


class FalH3Provider(Protocol):
    def submit(self, arguments: dict[str, object]) -> FalSubmissionReceipt: ...

    def status(self, receipt: FalSubmissionReceipt) -> FalQueueStatus: ...

    def result(self, receipt: FalSubmissionReceipt) -> tuple[FalH3Result, dict[str, object]]: ...

    def download(self, url: str) -> bytes: ...


class FalSubmissionAmbiguousError(RuntimeError):
    pass


class FalSubmissionRetryableError(RuntimeError):
    pass


class FalH3Client:
    def __init__(self, api_key: str | None = None) -> None:
        key = (api_key or FAL_KEY or "").strip()
        if not key:
            raise RuntimeError("FAL_KEY is required")
        self._headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        self._queue_url = f"{FAL_QUEUE_BASE_URL}/{FAL_H3_ENDPOINT}"

    def submit(self, arguments: dict[str, object]) -> FalSubmissionReceipt:
        try:
            response = requests.post(
                self._queue_url,
                headers={**self._headers, "X-Fal-No-Retry": "1"},
                json=arguments,
                timeout=60,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else None
            if status_code in {400, 401, 403, 404, 422, 502, 503, 504}:
                raise FalSubmissionRetryableError(
                    f"Fal did not accept submission with HTTP {status_code}"
                ) from error
            raise FalSubmissionAmbiguousError("Fal submission outcome is unknown") from error
        except requests.ConnectTimeout as error:
            raise FalSubmissionRetryableError(
                "Fal could not be reached before submission"
            ) from error
        except requests.RequestException as error:
            raise FalSubmissionAmbiguousError("Fal submission outcome is unknown") from error
        try:
            payload = cast(dict[str, object], response.json())
            return FalSubmissionReceipt.model_validate(
                {key: payload[key] for key in FalSubmissionReceipt.model_fields if key in payload},
                strict=True,
            )
        except (TypeError, ValueError) as error:
            raise FalSubmissionAmbiguousError("Fal submission outcome is unknown") from error

    def status(self, receipt: FalSubmissionReceipt) -> FalQueueStatus:
        response = requests.get(
            _trusted_queue_url(receipt.status_url), headers=self._headers, timeout=60
        )
        response.raise_for_status()
        payload = cast(dict[str, object], response.json())
        return FalQueueStatus.model_validate(
            {key: payload[key] for key in FalQueueStatus.model_fields if key in payload},
            strict=True,
        )

    def result(self, receipt: FalSubmissionReceipt) -> tuple[FalH3Result, dict[str, object]]:
        response = requests.get(
            _trusted_queue_url(receipt.response_url), headers=self._headers, timeout=60
        )
        response.raise_for_status()
        payload = cast(dict[str, object], response.json())
        return FalH3Result.model_validate({"video": payload.get("video")}, strict=True), payload

    def download(self, url: str) -> bytes:
        response = requests.get(_trusted_media_url(url), timeout=600, stream=True)
        try:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > FAL_MAX_DOWNLOAD_BYTES:
                raise ValueError("Fal candidate exceeds the maximum byte size")
            chunks: list[bytes] = []
            byte_size = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                byte_size += len(chunk)
                if byte_size > FAL_MAX_DOWNLOAD_BYTES:
                    raise ValueError("Fal candidate exceeds the maximum byte size")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            response.close()


def publish_generation_policy(
    value: GenerationPolicyArtifact = PRODUCTION_GENERATION_POLICY,
) -> Sha256:
    publish_immutable_r2_objects(((value.artifact.r2_key, value.artifact.content),))
    return record_generation_policy(value.artifact, recorded_at=datetime.now(UTC))


def generate_next_candidate(
    lease: SlotLease,
    prepared: PreparedPaidGeneration,
    references: H3ReferencePack,
    *,
    provider: FalH3Provider | None = None,
    sign_reference: Callable[[ArtifactReference], str] | None = None,
    policy: GenerationPolicyArtifact = PRODUCTION_GENERATION_POLICY,
) -> GenerationOutcome:
    if lease.edition_id != prepared.plan.edition_id:
        raise ValueError("Generation inputs do not match the claimed edition")
    if prepared.verified_plan.plan.edition_id != lease.edition_id:
        raise ValueError("Verified screenplay does not match the claimed edition")
    deadline = read_generation_deadline(lease.slot_id)
    attempts = read_generation_attempts(lease.edition_id)
    active = next(
        (
            item
            for item in attempts
            if item.stage
            in {GenerationStage.PENDING, GenerationStage.SUBMITTED, GenerationStage.PROCESSING}
        ),
        None,
    )
    if active is not None:
        return _resume_active(lease, active, deadline, provider)

    next_work = _next_story(prepared, attempts)
    if not isinstance(next_work, int):
        return next_work
    if datetime.now(UTC) >= deadline:
        return _fail_without_request(lease, "Video digest generation deadline has passed")
    return _start_attempt(
        lease,
        prepared,
        references,
        policy,
        next_work,
        len(tuple(item for item in attempts if item.request.story_position == next_work)),
        provider,
        sign_reference,
        deadline,
    )


def _next_story(
    prepared: PreparedPaidGeneration,
    attempts: tuple[GenerationAttemptReference, ...],
) -> int | GenerationComplete | GenerationFailed:
    accepted_positions = {
        item.request.story_position for item in attempts if item.stage is GenerationStage.ACCEPTED
    }
    if len(accepted_positions) == len(prepared.plan.stories):
        return GenerationComplete(edition_id=prepared.plan.edition_id)
    position = next(
        story.position
        for story in prepared.plan.stories
        if story.position not in accepted_positions
    )
    story_attempts = tuple(item for item in attempts if item.request.story_position == position)
    if any(item.stage is not GenerationStage.FAILED for item in story_attempts):
        raise ValueError("Stored generation attempts contain an invalid inactive state")
    attempt_index = len(story_attempts)
    if attempt_index > 1:
        return GenerationFailed(
            edition_id=prepared.plan.edition_id,
            reason=f"Story {position} exhausted both generation attempts",
        )
    return position


def _start_attempt(
    lease: SlotLease,
    prepared: PreparedPaidGeneration,
    references: H3ReferencePack,
    policy: GenerationPolicyArtifact,
    position: int,
    attempt_index: int,
    provider: FalH3Provider | None,
    sign_reference: Callable[[ArtifactReference], str] | None,
    deadline: datetime,
) -> GenerationOutcome:
    publish_generation_policy(policy)
    request_value, request_file, identity = _generation_request(
        prepared, references, policy, position, attempt_index
    )
    publish_immutable_r2_objects(((request_file.r2_key, request_file.content),))
    admission = checkpoint_generation_request(
        lease,
        identity,
        request_file=request_file,
        admission=_admission(policy, len(prepared.plan.stories)),
        recorded_at=datetime.now(UTC),
    )
    if not admission.created:
        return _fail_ambiguous_submission(lease, identity)

    client = provider or FalH3Client()
    signer = sign_reference or (
        lambda value: presigned_r2_url(value.r2_key, expires_in=policy.policy.deadline_minutes * 60)
    )
    try:
        receipt = client.submit(_fal_arguments(request_value, signer))
    except FalSubmissionAmbiguousError:
        return _fail_ambiguous_submission(lease, identity)
    except FalSubmissionRetryableError as error:
        return _fail_request(lease, identity, None, str(error), terminal=False)
    receipt_file = _receipt_file(identity, receipt)
    publish_immutable_r2_objects(((receipt_file.r2_key, receipt_file.content),))
    checkpoint_generation_submission(
        lease,
        identity.request_id,
        provider_receipt_id=receipt.request_id,
        receipt_file=receipt_file,
        cost=EstimatedAttemptCost(usd=policy.policy.estimated_attempt_usd),
        recorded_at=datetime.now(UTC),
    )
    active = GenerationAttemptReference(
        request=identity,
        stage=GenerationStage.SUBMITTED,
        provider_receipt_id=receipt.request_id,
        cost=EstimatedAttemptCost(usd=policy.policy.estimated_attempt_usd),
        request_evidence=_reference(request_file),
        receipt_evidence=_reference(receipt_file),
        response_evidence=None,
    )
    return _poll_submitted(lease, active, receipt, deadline, client)


def _resume_active(
    lease: SlotLease,
    active: GenerationAttemptReference,
    deadline: datetime,
    provider: FalH3Provider | None,
) -> GenerationOutcome:
    if active.stage is GenerationStage.PENDING:
        return _fail_ambiguous_submission(lease, active.request)
    if active.stage is GenerationStage.PROCESSING:
        if active.response_evidence is None:
            raise ValueError("Processing generation request has no response evidence")
        return _candidate_from_response(active, active.response_evidence)
    receipt = _read_receipt(active)
    return _poll_submitted(lease, active, receipt, deadline, provider or FalH3Client())


def _poll_submitted(
    lease: SlotLease,
    active: GenerationAttemptReference,
    receipt: FalSubmissionReceipt,
    deadline: datetime,
    provider: FalH3Provider,
) -> GenerationOutcome:
    if datetime.now(UTC) >= deadline:
        return _fail_attempt(
            lease,
            active,
            "Video digest generation deadline passed while Fal was active",
            terminal=True,
        )
    try:
        status = provider.status(receipt)
    except requests.HTTPError as error:
        if _is_definitive_provider_failure(error):
            return _fail_attempt(lease, active, f"Fal status failed: {error}", terminal=False)
        raise
    if status.request_id != receipt.request_id:
        raise ValueError("Fal status receipt identity changed")
    checkpoint_fal_queue_state(
        lease,
        active.request.request_id,
        provider_receipt_id=receipt.request_id,
        provider_status=status.status,
        recorded_at=datetime.now(UTC),
    )
    if status.status != "COMPLETED":
        return GenerationInProgress(
            request_id=active.request.request_id,
            provider_receipt_id=receipt.request_id,
            provider_status=status.status,
        )
    if status.error:
        return _fail_attempt(lease, active, status.error, terminal=False)
    try:
        result, raw = provider.result(receipt)
    except requests.HTTPError as error:
        if _is_definitive_provider_failure(error):
            return _fail_attempt(lease, active, f"Fal result failed: {error}", terminal=False)
        raise
    except ValueError as error:
        return _fail_attempt(lease, active, str(error), terminal=False)
    try:
        candidate_bytes = provider.download(str(result.video.url))
    except ValueError as error:
        return _fail_attempt(lease, active, str(error), terminal=False)
    candidate = CandidateReference(
        r2_key=(
            f"news/video-digest/candidates/7d/{active.request.request_id}/"
            f"{sha256(candidate_bytes)}.mp4"
        ),
        content_digest=sha256(candidate_bytes),
        byte_size=len(candidate_bytes),
    )
    cost = UnknownAttemptCost(reason="Fal response did not include realized billing")
    evidence = H3CompletionEvidence(
        request_id=active.request.request_id,
        provider_receipt_id=receipt.request_id,
        provider_response=raw,
        candidate=candidate,
        cost=cost,
    )
    response_file = artifact_file(
        artifact_id=f"{active.request.request_id}:response",
        artifact_kind="video_digest_generation_response",
        title=f"Fal response for {active.request.request_id}",
        content=canonical_json(evidence.model_dump(mode="json")),
        r2_key=(
            f"news/video-digest/{active.request.edition_id}/generation/"
            f"response-{active.request.request_id}.json"
        ),
        media_type="application/json",
    )
    publish_private_video_object(
        candidate.r2_key,
        candidate_bytes,
        retention="candidate-7d",
        source_lineage=active.request.request_id,
    )
    publish_immutable_r2_objects(((response_file.r2_key, response_file.content),))
    checkpoint_generation_response(
        lease,
        active.request.request_id,
        response_file=response_file,
        recorded_at=datetime.now(UTC),
    )
    return _candidate_ready(active, response_file.version_id, evidence)


def _candidate_from_response(
    active: GenerationAttemptReference,
    response: ArtifactReference,
) -> CandidateReady:
    evidence = H3CompletionEvidence.model_validate_json(
        read_verified_r2_object(response.r2_key, response.content_digest), strict=True
    )
    if evidence.request_id != active.request.request_id:
        raise ValueError("Stored Fal response does not match its generation request")
    return _candidate_ready(active, response.version_id, evidence)


def _candidate_ready(
    active: GenerationAttemptReference,
    response_version_id: Sha256,
    evidence: H3CompletionEvidence,
) -> CandidateReady:
    return CandidateReady(
        request_id=active.request.request_id,
        story_id=_story_id_from_request(active),
        story_position=active.request.story_position,
        attempt_index=active.request.attempt_index,
        response_artifact_version_id=response_version_id,
        candidate=evidence.candidate,
        cost=evidence.cost,
    )


def _story_id_from_request(active: GenerationAttemptReference) -> StoryId:
    content = H3GenerationRequest.model_validate_json(
        read_verified_r2_object(
            active.request_evidence.r2_key,
            active.request_evidence.content_digest,
        ),
        strict=True,
    )
    return content.story_id


def _generation_request(
    prepared: PreparedPaidGeneration,
    references: H3ReferencePack,
    policy: GenerationPolicyArtifact,
    position: int,
    attempt_index: int,
) -> tuple[H3GenerationRequest, ArtifactFile, GenerationRequestIdentity]:
    screenplay = prepared.verified_plan.plan.stories[position]
    value = H3GenerationRequest(
        edition_id=prepared.plan.edition_id,
        story_id=prepared.plan.stories[position].story_id,
        story_position=position,
        attempt_index=attempt_index,
        authorization=prepared.authorization,
        verified_plan_digest=sha256(canonical_json(prepared.verified_plan.model_dump(mode="json"))),
        screenplay_story_digest=screenplay_story_digest(screenplay),
        story=screenplay,
        references=references,
        generation_policy_artifact_version_id=policy.artifact.version_id,
    )
    file = artifact_file(
        artifact_id=f"{prepared.plan.edition_id}:{position}:{attempt_index}:generation-request",
        artifact_kind="video_digest_generation_request",
        title=f"Video digest generation request {position}:{attempt_index}",
        content=canonical_json(value.model_dump(mode="json")),
        r2_key=(
            f"news/video-digest/{prepared.plan.edition_id}/generation/"
            f"request-{position}-{attempt_index}-{sha256(canonical_json(value.model_dump(mode='json')))}.json"
        ),
        media_type="application/json",
    )
    identity = GenerationRequestIdentity(
        request_id=generation_request_id(
            prepared.plan.edition_id, position, attempt_index, file.version_id
        ),
        edition_id=prepared.plan.edition_id,
        story_position=position,
        attempt_index=attempt_index,
        request_artifact_version_id=file.version_id,
    )
    return value, file, identity


def _fal_arguments(
    request: H3GenerationRequest,
    signer: Callable[[ArtifactReference], str],
) -> dict[str, object]:
    prompt = (
        f"{request.story.visual_direction}\n\n"
        f'The hosts say exactly once in Romanian: "{request.story.narration}" '
        "No other speech, captions, subtitles, logos, title cards, or readable text."
    )
    seed_source = sha256(
        canonical_json(
            {
                "edition_id": request.edition_id,
                "story_position": request.story_position,
                "attempt_index": request.attempt_index,
            }
        )
    )
    return {
        "prompt": prompt,
        "duration": request.duration,
        "resolution": request.resolution,
        "seed": int(seed_source[:8], 16),
        "enable_safety_checker": True,
        "prompt_expansion_mode": request.prompt_expansion_mode,
        "aspect_ratio": request.aspect_ratio,
        "reference_video_urls": [signer(value) for value in request.references.videos],
        "reference_audio_urls": [signer(value) for value in request.references.audio],
    }


def _trusted_queue_url(url: HttpUrl) -> str:
    if url.scheme != "https" or url.host != "queue.fal.run":
        raise ValueError("Fal receipt URL must use the trusted Fal queue host")
    return str(url)


def _trusted_media_url(url: str) -> str:
    parsed = HttpUrl(url)
    host = parsed.host
    if (
        parsed.scheme != "https"
        or host is None
        or not (host == "fal.media" or host.endswith(".fal.media"))
    ):
        raise ValueError("Fal candidate URL must use the trusted Fal media host")
    return str(parsed)


def _is_definitive_provider_failure(error: requests.HTTPError) -> bool:
    status_code = error.response.status_code if error.response is not None else None
    return status_code is not None and 400 <= status_code < 500


def _admission(policy: GenerationPolicyArtifact, story_count: int) -> GenerationAdmission:
    return GenerationAdmission(
        generation_policy_artifact_version_id=policy.artifact.version_id,
        reserved_usd=policy.policy.estimated_attempt_usd,
        limits=GenerationBudgetLimits(
            story_usd=policy.policy.story_limit_usd,
            edition_usd=policy.policy.edition_per_story_limit_usd * story_count,
            bucharest_day_usd=policy.policy.bucharest_day_limit_usd,
            calendar_month_usd=policy.policy.calendar_month_limit_usd,
        ),
    )


def _receipt_file(
    request: GenerationRequestIdentity,
    receipt: FalSubmissionReceipt,
) -> ArtifactFile:
    return artifact_file(
        artifact_id=receipt.request_id,
        artifact_kind="video_digest_provider_receipt",
        title=f"Fal receipt for {request.request_id}",
        content=canonical_json(receipt.model_dump(mode="json")),
        r2_key=(
            f"news/video-digest/{request.edition_id}/generation/"
            f"receipt-{request.request_id}.json"
        ),
        media_type="application/json",
    )


def _read_receipt(active: GenerationAttemptReference) -> FalSubmissionReceipt:
    if active.provider_receipt_id is None or active.receipt_evidence is None:
        raise ValueError("Submitted generation request has no provider receipt")
    receipt = FalSubmissionReceipt.model_validate_json(
        read_verified_r2_object(
            active.receipt_evidence.r2_key,
            active.receipt_evidence.content_digest,
        ),
        strict=True,
    )
    if receipt.request_id != active.provider_receipt_id:
        raise ValueError("Stored Fal receipt does not match the catalog")
    return receipt


def _fail_ambiguous_submission(
    lease: SlotLease,
    request: GenerationRequestIdentity,
) -> GenerationFailed:
    reason = "Fal submission may have succeeded without a stored receipt"
    outcome = _fail_request(
        lease,
        request,
        None,
        reason,
        terminal=True,
    )
    assert isinstance(outcome, GenerationFailed)
    return outcome


def _fail_attempt(
    lease: SlotLease,
    active: GenerationAttemptReference,
    reason: str,
    *,
    terminal: bool,
) -> GenerationFailed | GenerationRetryAvailable:
    return _fail_request(
        lease,
        active.request,
        active.provider_receipt_id,
        reason,
        terminal=terminal,
    )


def _fail_request(
    lease: SlotLease,
    request: GenerationRequestIdentity,
    provider_receipt_id: str | None,
    reason: str,
    *,
    terminal: bool,
) -> GenerationFailed | GenerationRetryAvailable:
    content = canonical_json(
        {
            "request_id": request.request_id,
            "reason": reason,
            "provider_receipt_id": provider_receipt_id,
        }
    )
    request_failure = artifact_file(
        artifact_id=f"{request.request_id}:failure",
        artifact_kind="video_digest_generation_failure",
        title=f"Generation failure for {request.request_id}",
        content=content,
        r2_key=(
            f"news/video-digest/{lease.edition_id}/generation/"
            f"failure-{request.request_id}-{sha256(content)}.json"
        ),
        media_type="application/json",
    )
    publish_immutable_r2_objects(((request_failure.r2_key, request_failure.content),))
    checkpoint_generation_failure(
        lease,
        request.request_id,
        evidence_file=request_failure,
        cost=UnknownAttemptCost(reason=reason),
        recorded_at=datetime.now(UTC),
    )
    if terminal and request.attempt_index == 0:
        return _fail_without_request(lease, reason)
    if terminal or request.attempt_index == 1:
        return GenerationFailed(edition_id=lease.edition_id, reason=reason)
    return GenerationRetryAvailable(
        edition_id=lease.edition_id,
        story_position=request.story_position,
        reason=reason,
    )


def _fail_without_request(lease: SlotLease, reason: str) -> GenerationFailed:
    content = canonical_json({"edition_id": lease.edition_id, "reason": reason})
    file = artifact_file(
        artifact_id=f"{lease.slot_id}:failure",
        artifact_kind="video_digest_failure",
        title=f"Video digest failure for {lease.slot_id}",
        content=content,
        r2_key=f"news/video-digest/{lease.edition_id}/failure-{sha256(content)}.json",
        media_type="application/json",
    )
    publish_immutable_r2_objects(((file.r2_key, file.content),))
    fail_slot(lease, evidence_file=file, recorded_at=datetime.now(UTC))
    return GenerationFailed(edition_id=lease.edition_id, reason=reason)


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )
