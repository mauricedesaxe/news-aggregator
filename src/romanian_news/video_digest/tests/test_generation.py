from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
import requests

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.video_digest import generation, preflight
from romanian_news.video_digest.errors import VideoDigestCheckpointConflictError
from romanian_news.video_digest.models import (
    EstimatedAttemptCost,
    GenerationAdmission,
    GenerationRequestCheckpoint,
    GenerationRequestState,
    GenerationStage,
    PendingAttemptCost,
    SlotId,
    SlotLease,
    UnknownAttemptCost,
)
from romanian_news.video_digest.planning import (
    ScreenplayStory,
    StoryVerificationEvidence,
    accept_planning_attempt,
    authorize_generation,
    create_screenplay_plan,
    record_planning_attempt,
    screenplay_story_digest,
)
from romanian_news.video_digest.planning_artifacts import verified_plan_file

REPORT_VERSION = "a" * 64
SUBJECTS = ("1" * 64, "2" * 64)
ARTICLES = ("3" * 64, "4" * 64)


def _prepared() -> preflight.PreparedPaidGeneration:
    sections = tuple(
        _section(subject, article, position)
        for position, (subject, article) in enumerate(zip(SUBJECTS, ARTICLES, strict=True))
    )
    report = DailyReport(
        day=date(2026, 9, 21),
        accepted_article_count=2,
        theme_count=2,
        group_count=2,
        sections=sections,
    )
    narration = " ".join(f"cuvant{index}" for index in range(30))
    stories = tuple(
        ScreenplayStory(
            report_subject_id=subject,
            title=f"Story {position}",
            citation_article_version_ids=(article,),
            narration=narration,
            visual_direction=f"Visual action {position}",
            requested_duration_ms=15_000,
        )
        for position, (subject, article) in enumerate(zip(SUBJECTS, ARTICLES, strict=True))
    )
    policy = preflight.PRODUCTION_POLICY
    plan = create_screenplay_plan(
        report,
        REPORT_VERSION,
        policy.artifact.version_id,
        policy.definition.policy,
        stories,
    )
    evidence = tuple(
        StoryVerificationEvidence(
            story_screenplay_digest=screenplay_story_digest(story),
            daily_report_version_id=REPORT_VERSION,
            policy_bundle_digest=plan.policy_bundle_digest,
            citation_article_version_ids=story.citation_article_version_ids,
            status="accepted",
            failures=(),
        )
        for story in stories
    )
    attempt = record_planning_attempt(0, plan, evidence, "accepted")
    verified = accept_planning_attempt(report, policy.definition.policy, attempt)
    _, digest_plan = verified_plan_file(verified)
    return preflight.PreparedPaidGeneration(
        authorization=authorize_generation(verified),
        plan=digest_plan,
        verified_plan=verified,
    )


def test_fal_prompt_requires_english_speech() -> None:
    prepared = _prepared()
    request, _, _ = generation._generation_request(
        prepared, _references(), generation.PRODUCTION_GENERATION_POLICY, 0, 0
    )

    prompt = generation._fal_arguments(request, lambda _: "https://r2.example/reference")["prompt"]

    assert isinstance(prompt, str)
    assert f'in English: "{request.story.narration}"' in prompt
    assert "in Romanian" not in prompt


def _section(subject: str, article: str, position: int) -> DailyReportSection:
    return DailyReportSection(
        theme_id=subject,
        title=f"Subject {position}",
        summary=f"Summary {position}",
        events=(
            ReportEvent(
                group_id=article,
                title_ro=f"Titlu {position}",
                summary_ro=f"Rezumat {position}",
                key_points_ro=(f"Punct {position}",),
                disagreements_ro=(),
                sentiment_label="neutral",
                sentiment_score=0,
                sentiment_rationale_ro="Neutru",
                articles=(
                    ReportArticle(
                        article_version_id=article,
                        outlet_id="example",
                        title=f"Article {position}",
                        canonical_url=f"https://example.com/{position}",
                        sentiment_label="neutral",
                        sentiment_score=0,
                    ),
                ),
            ),
        ),
        tier="main",
        semantic_rank=position + 1,
        consequence_rationale=f"Consequence {position}",
        citations=(
            ReportSubjectCitation(
                article_version_id=article,
                evidence_quote=f"Evidence {position}",
            ),
        ),
    )


def _lease(prepared: preflight.PreparedPaidGeneration) -> SlotLease:
    return SlotLease(
        slot_id=SlotId("5" * 64),
        edition_id=prepared.plan.edition_id,
        owner_token="owner",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        claim_count=1,
    )


def _references() -> generation.H3ReferencePack:
    return generation.H3ReferencePack(
        videos=(_artifact_reference("video", "6" * 64),),
        audio=(_artifact_reference("audio", "7" * 64),),
    )


def _artifact_reference(name: str, digest: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"reference-{name}",
        version_id=digest,
        content_digest=digest,
        r2_key=f"references/{name}",
    )


class _Provider:
    def __init__(self) -> None:
        self.submitted_positions: list[int] = []
        self.status_calls: list[str] = []

    def submit(self, arguments: dict[str, object]) -> generation.FalSubmissionReceipt:
        prompt = str(arguments["prompt"])
        position = next(index for index in range(2) if f"Visual action {index}" in prompt)
        self.submitted_positions.append(position)
        return _receipt(f"fal-{position}-{len(self.submitted_positions)}")

    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        self.status_calls.append(receipt.request_id)
        return generation.FalQueueStatus(status="COMPLETED", request_id=receipt.request_id)

    def result(
        self, receipt: generation.FalSubmissionReceipt
    ) -> tuple[generation.FalH3Result, dict[str, object]]:
        payload: dict[str, object] = {
            "video": {"url": f"https://media.example/{receipt.request_id}.mp4"}
        }
        return generation.FalH3Result.model_validate(payload, strict=True), payload

    def download(self, url: str) -> bytes:
        return f"candidate:{url}".encode()


class _TransportFailureProvider(_Provider):
    def result(
        self, receipt: generation.FalSubmissionReceipt
    ) -> tuple[generation.FalH3Result, dict[str, object]]:
        del receipt
        raise requests.Timeout("result timed out")


class _StatusTransportFailureProvider(_Provider):
    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        del receipt
        raise _http_error(503)


class _RejectedProvider(_Provider):
    def submit(self, arguments: dict[str, object]) -> generation.FalSubmissionReceipt:
        del arguments
        raise generation.FalSubmissionRetryableError("Fal did not accept submission with HTTP 400")


class _StatusFailureProvider(_Provider):
    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        del receipt
        raise _http_error(404)


class _ResultFailureProvider(_Provider):
    def result(
        self, receipt: generation.FalSubmissionReceipt
    ) -> tuple[generation.FalH3Result, dict[str, object]]:
        del receipt
        raise _http_error(422)


class _ReferenceRecordingProvider(_Provider):
    def __init__(self, provided_urls: list[list[str]]) -> None:
        super().__init__()
        self._provided_urls = provided_urls

    def submit(self, arguments: dict[str, Any]) -> generation.FalSubmissionReceipt:
        self._provided_urls.append(list(arguments["reference_video_urls"]))
        self._provided_urls.append(list(arguments["reference_audio_urls"]))
        return super().submit(arguments)


class _AmbiguousSubmissionProvider(_Provider):
    def submit(self, arguments: dict[str, object]) -> generation.FalSubmissionReceipt:
        del arguments
        raise generation.FalSubmissionAmbiguousError("Fal submission outcome is unknown")


class _InProgressProvider(_Provider):
    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        self.status_calls.append(receipt.request_id)
        return generation.FalQueueStatus(status="IN_PROGRESS", request_id=receipt.request_id)


class _ProviderErrorProvider(_Provider):
    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        self.status_calls.append(receipt.request_id)
        return generation.FalQueueStatus(
            status="COMPLETED", request_id=receipt.request_id, error="model refused the prompt"
        )


class _UntouchableProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def submit(self, arguments: dict[str, object]) -> generation.FalSubmissionReceipt:
        self.calls += 1
        return super().submit(arguments)

    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        self.calls += 1
        return super().status(receipt)

    def result(
        self, receipt: generation.FalSubmissionReceipt
    ) -> tuple[generation.FalH3Result, dict[str, object]]:
        self.calls += 1
        return super().result(receipt)

    def download(self, url: str) -> bytes:
        self.calls += 1
        return super().download(url)


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://queue.fal.run/request"
    return requests.HTTPError(f"HTTP {status_code}", response=response)


def _receipt(receipt_id: str) -> generation.FalSubmissionReceipt:
    return generation.FalSubmissionReceipt.model_validate(
        {
            "request_id": receipt_id,
            "status_url": f"https://queue.example/{receipt_id}/status",
            "response_url": f"https://queue.example/{receipt_id}/response",
        },
        strict=True,
    )


class _Harness:
    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, prepared: preflight.PreparedPaidGeneration
    ) -> None:
        self.prepared = prepared
        self.objects: dict[str, bytes] = {}
        self.classifications: list[tuple[str, str, str]] = []
        self.attempts: list[generation.GenerationAttemptReference] = []
        self.admissions: list[GenerationAdmission] = []
        self.queue_statuses: list[str] = []
        self.failed_slot = False
        self.deny_admission = False
        self.deadline = datetime(2027, 1, 1, tzinfo=UTC)
        self.policy_publications = 0

        monkeypatch.setattr(generation, "publish_generation_policy", self.publish_policy)
        monkeypatch.setattr(
            generation,
            "read_generation_deadline",
            lambda _slot: self.deadline,
        )
        monkeypatch.setattr(
            generation, "read_generation_attempts", lambda _edition: tuple(self.attempts)
        )
        monkeypatch.setattr(generation, "publish_immutable_r2_objects", self.publish)
        monkeypatch.setattr(
            generation,
            "publish_private_video_object",
            self.publish_private_video,
        )
        monkeypatch.setattr(generation, "read_verified_r2_object", self.read)
        monkeypatch.setattr(generation, "checkpoint_generation_request", self.checkpoint_request)
        monkeypatch.setattr(
            generation, "checkpoint_generation_submission", self.checkpoint_submission
        )
        monkeypatch.setattr(generation, "checkpoint_fal_queue_state", self.checkpoint_queue_state)
        monkeypatch.setattr(generation, "checkpoint_generation_response", self.checkpoint_response)
        monkeypatch.setattr(generation, "checkpoint_generation_failure", self.checkpoint_failure)
        monkeypatch.setattr(generation, "fail_slot", self.fail_slot)

    def publish(self, objects) -> None:
        self.objects.update(objects)

    def checkpoint_queue_state(
        self,
        _lease: SlotLease,
        _request_id: str,
        *,
        provider_receipt_id: str,
        provider_status: str,
        recorded_at: datetime,
    ) -> None:
        del provider_receipt_id, recorded_at
        self.queue_statuses.append(provider_status)

    def publish_policy(self, _policy: generation.GenerationPolicyArtifact) -> str:
        self.policy_publications += 1
        return "8" * 64

    def publish_private_video(
        self,
        key: str,
        content: bytes,
        *,
        retention: str,
        source_lineage: str,
    ) -> None:
        self.objects[key] = content
        self.classifications.append((key, retention, source_lineage))

    def read(self, key: str, digest: str) -> bytes:
        content = self.objects[key]
        assert generation.sha256(content) == digest
        return content

    def checkpoint_request(
        self,
        _lease: SlotLease,
        request,
        *,
        request_file: ArtifactFile,
        admission: GenerationAdmission,
        recorded_at: datetime,
    ) -> GenerationRequestCheckpoint:
        del recorded_at
        self.admissions.append(admission)
        if self.deny_admission:
            raise VideoDigestCheckpointConflictError("budget exceeded")
        existing = next((item for item in self.attempts if item.request == request), None)
        if existing is not None:
            return GenerationRequestCheckpoint(
                state=GenerationRequestState(
                    request_id=request.request_id,
                    stage=existing.stage,
                    provider_receipt_id=existing.provider_receipt_id,
                    cost=existing.cost,
                ),
                created=False,
            )
        self.attempts.append(
            generation.GenerationAttemptReference(
                request=request,
                stage=GenerationStage.PENDING,
                provider_receipt_id=None,
                cost=PendingAttemptCost(),
                request_evidence=_reference(request_file),
                receipt_evidence=None,
                response_evidence=None,
            )
        )
        return GenerationRequestCheckpoint(
            state=GenerationRequestState(
                request_id=request.request_id,
                stage=GenerationStage.PENDING,
                cost=PendingAttemptCost(),
            ),
            created=True,
        )

    def checkpoint_submission(
        self,
        _lease: SlotLease,
        request_id,
        *,
        provider_receipt_id: str,
        receipt_file: ArtifactFile,
        cost: EstimatedAttemptCost,
        recorded_at: datetime,
    ) -> GenerationRequestState:
        del recorded_at
        index = self._index(request_id)
        self.attempts[index] = self.attempts[index].model_copy(
            update={
                "stage": GenerationStage.SUBMITTED,
                "provider_receipt_id": provider_receipt_id,
                "cost": cost,
                "receipt_evidence": _reference(receipt_file),
            }
        )
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.SUBMITTED,
            provider_receipt_id=provider_receipt_id,
            cost=cost,
        )

    def checkpoint_response(
        self,
        _lease: SlotLease,
        request_id,
        *,
        response_file: ArtifactFile,
        recorded_at: datetime,
    ) -> GenerationRequestState:
        del recorded_at
        index = self._index(request_id)
        self.attempts[index] = self.attempts[index].model_copy(
            update={
                "stage": GenerationStage.PROCESSING,
                "response_evidence": _reference(response_file),
            }
        )
        current = self.attempts[index]
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.PROCESSING,
            provider_receipt_id=current.provider_receipt_id,
            cost=current.cost,
        )

    def checkpoint_failure(
        self,
        _lease: SlotLease,
        request_id,
        *,
        evidence_file: ArtifactFile,
        cost: UnknownAttemptCost,
        recorded_at: datetime,
    ) -> GenerationRequestState:
        del evidence_file, recorded_at
        index = self._index(request_id)
        self.attempts[index] = self.attempts[index].model_copy(
            update={"stage": GenerationStage.FAILED, "cost": cost}
        )
        return GenerationRequestState(
            request_id=request_id,
            stage=GenerationStage.FAILED,
            cost=cost,
        )

    def fail_slot(self, *_args: Any, **_kwargs: Any) -> None:
        self.failed_slot = True

    def accept_latest(self) -> None:
        self.attempts[-1] = self.attempts[-1].model_copy(update={"stage": GenerationStage.ACCEPTED})

    def _index(self, request_id) -> int:
        return next(
            index
            for index, item in enumerate(self.attempts)
            if item.request.request_id == request_id
        )


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def test_generation_is_ordered_and_records_unknown_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()

    first = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )
    assert isinstance(first, generation.CandidateReady)
    assert first.story_position == 0
    assert first.cost.kind == "unknown"
    harness.accept_latest()

    second = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )
    assert isinstance(second, generation.CandidateReady)
    assert second.story_position == 1
    assert provider.submitted_positions == [0, 1]
    assert all(item.request.attempt_index == 0 for item in harness.attempts)
    assert all("/candidates/7d/" in key for key, _retention, _lineage in harness.classifications)
    assert all(retention == "candidate-7d" for _key, retention, _lineage in harness.classifications)
    assert [lineage for _key, _retention, lineage in harness.classifications] == [
        first.request_id,
        second.request_id,
    ]


def test_generation_reuses_stored_receipt_without_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    first = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )
    assert isinstance(first, generation.CandidateReady)
    harness.attempts[0] = harness.attempts[0].model_copy(
        update={"stage": GenerationStage.SUBMITTED, "response_evidence": None}
    )
    harness.policy_publications = 0

    resumed = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(resumed, generation.CandidateReady)
    assert provider.submitted_positions == [0]
    assert provider.status_calls == ["fal-0-1", "fal-0-1"]
    assert harness.policy_publications == 0


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_TransportFailureProvider(), "result timed out"),
        (_StatusTransportFailureProvider(), "HTTP 503"),
    ],
)
def test_provider_transport_failure_preserves_receipt(
    monkeypatch: pytest.MonkeyPatch, provider: _Provider, message: str
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)

    with pytest.raises(requests.RequestException, match=message):
        generation.generate_next_candidate(
            _lease(prepared),
            prepared,
            _references(),
            provider=provider,
            sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
        )

    assert harness.attempts[0].stage is GenerationStage.SUBMITTED
    assert harness.failed_slot is False


def test_confirmed_submission_rejection_allows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=_RejectedProvider(),
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.GenerationRetryAvailable)
    assert harness.attempts[0].stage is GenerationStage.FAILED
    assert harness.failed_slot is False


@pytest.mark.parametrize("provider", [_StatusFailureProvider(), _ResultFailureProvider()])
def test_definitive_provider_failure_allows_retry(
    monkeypatch: pytest.MonkeyPatch, provider: _Provider
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.GenerationRetryAvailable)
    assert harness.attempts[0].stage is GenerationStage.FAILED
    assert harness.failed_slot is False


def test_default_reference_signature_covers_generation_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    _Harness(monkeypatch, prepared)
    expirations: list[int] = []
    provided_urls: list[list[str]] = []

    def sign(key: str, *, expires_in: int) -> str:
        expirations.append(expires_in)
        return f"https://r2.example/signed/{key}"

    monkeypatch.setattr(generation, "presigned_r2_url", sign)
    provider = _ReferenceRecordingProvider(provided_urls)

    outcome = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(outcome, generation.CandidateReady)
    assert expirations and all(
        expires_in >= generation.PRODUCTION_GENERATION_POLICY.policy.deadline_minutes * 60
        for expires_in in expirations
    )
    assert provided_urls == [
        ["https://r2.example/signed/references/video"],
        ["https://r2.example/signed/references/audio"],
    ]


def test_pending_restart_fails_closed_without_duplicate_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    request, request_file, identity = generation._generation_request(
        prepared, _references(), generation.PRODUCTION_GENERATION_POLICY, 0, 0
    )
    del request
    harness.publish(((request_file.r2_key, request_file.content),))
    harness.attempts.append(
        generation.GenerationAttemptReference(
            request=identity,
            stage=GenerationStage.PENDING,
            provider_receipt_id=None,
            cost=PendingAttemptCost(),
            request_evidence=_reference(request_file),
            receipt_evidence=None,
            response_evidence=None,
        )
    )

    outcome = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(outcome, generation.GenerationFailed)
    assert provider.submitted_positions == []
    assert harness.failed_slot is True
    assert harness.attempts[0].cost.kind == "unknown"


def test_failed_first_attempt_creates_only_attempt_one(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    _, request_file, identity = generation._generation_request(
        prepared, _references(), generation.PRODUCTION_GENERATION_POLICY, 0, 0
    )
    harness.attempts.append(
        generation.GenerationAttemptReference(
            request=identity,
            stage=GenerationStage.FAILED,
            provider_receipt_id="failed-receipt",
            cost=UnknownAttemptCost(reason="provider failed"),
            request_evidence=_reference(request_file),
            receipt_evidence=None,
            response_evidence=None,
        )
    )

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.CandidateReady)
    assert outcome.attempt_index == 1
    assert provider.submitted_positions == [0]
    assert [item.request.attempt_index for item in harness.attempts] == [0, 1]


def test_budget_denial_happens_before_fal_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    harness.deny_admission = True
    provider = _Provider()

    with pytest.raises(VideoDigestCheckpointConflictError, match="budget exceeded"):
        generation.generate_next_candidate(
            _lease(prepared), prepared, _references(), provider=provider
        )

    assert provider.submitted_positions == []
    assert len(harness.admissions) == 1


def test_ambiguous_submission_fails_closed_and_fails_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=_AmbiguousSubmissionProvider(),
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.GenerationFailed)
    assert outcome.reason == "Fal submission may have succeeded without a stored receipt"
    assert harness.attempts[0].stage is GenerationStage.FAILED
    assert harness.attempts[0].cost.kind == "unknown"
    assert harness.failed_slot is True


def test_resume_from_processing_replays_the_stored_response_without_provider_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    first = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=_Provider(),
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )
    assert isinstance(first, generation.CandidateReady)
    assert harness.attempts[0].stage is GenerationStage.PROCESSING

    resumed = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=_UntouchableProvider()
    )

    assert isinstance(resumed, generation.CandidateReady)
    assert resumed.request_id == first.request_id
    assert resumed.candidate.content_digest == first.candidate.content_digest
    assert resumed.response_artifact_version_id == first.response_artifact_version_id


def test_stored_response_for_a_different_request_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    for _ in range(2):
        outcome = generation.generate_next_candidate(
            _lease(prepared),
            prepared,
            _references(),
            provider=provider,
            sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
        )
        assert isinstance(outcome, generation.CandidateReady)
        harness.accept_latest()
    harness.attempts[0] = harness.attempts[0].model_copy(
        update={
            "stage": GenerationStage.PROCESSING,
            "response_evidence": harness.attempts[1].response_evidence,
        }
    )

    with pytest.raises(ValueError, match="does not match its generation request"):
        generation.generate_next_candidate(
            _lease(prepared), prepared, _references(), provider=_UntouchableProvider()
        )


def test_deadline_passing_while_fal_is_active_fails_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    assert isinstance(
        generation.generate_next_candidate(
            _lease(prepared),
            prepared,
            _references(),
            provider=provider,
            sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
        ),
        generation.CandidateReady,
    )
    harness.attempts[0] = harness.attempts[0].model_copy(
        update={"stage": GenerationStage.SUBMITTED, "response_evidence": None}
    )
    harness.deadline = datetime(2020, 1, 1, tzinfo=UTC)

    outcome = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=_UntouchableProvider()
    )

    assert isinstance(outcome, generation.GenerationFailed)
    assert "deadline passed while Fal was active" in outcome.reason
    assert harness.attempts[0].stage is GenerationStage.FAILED
    assert harness.failed_slot is True


def test_in_progress_status_reports_progress_without_new_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _InProgressProvider()

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=provider,
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.GenerationInProgress)
    assert outcome.provider_status == "IN_PROGRESS"
    assert harness.queue_statuses == ["IN_PROGRESS"]
    assert provider.submitted_positions == [0]
    assert harness.attempts[0].stage is GenerationStage.SUBMITTED
    assert harness.failed_slot is False


def test_completed_status_with_an_error_fails_the_attempt_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)

    outcome = generation.generate_next_candidate(
        _lease(prepared),
        prepared,
        _references(),
        provider=_ProviderErrorProvider(),
        sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
    )

    assert isinstance(outcome, generation.GenerationRetryAvailable)
    assert outcome.reason == "model refused the prompt"
    assert harness.queue_statuses == ["COMPLETED"]
    assert harness.attempts[0].stage is GenerationStage.FAILED
    assert harness.failed_slot is False


def test_exhausted_attempts_fail_the_edition_without_further_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    for attempt_index in range(2):
        _, request_file, identity = generation._generation_request(
            prepared, _references(), generation.PRODUCTION_GENERATION_POLICY, 0, attempt_index
        )
        harness.attempts.append(
            generation.GenerationAttemptReference(
                request=identity,
                stage=GenerationStage.FAILED,
                provider_receipt_id=f"failed-receipt-{attempt_index}",
                cost=UnknownAttemptCost(reason="provider failed"),
                request_evidence=_reference(request_file),
                receipt_evidence=None,
                response_evidence=None,
            )
        )
    provider = _UntouchableProvider()

    outcome = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(outcome, generation.GenerationFailed)
    assert "exhausted both generation attempts" in outcome.reason
    assert provider.calls == 0
    assert harness.failed_slot is False


def test_all_positions_accepted_completes_the_edition(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    provider = _Provider()
    for _ in range(2):
        outcome = generation.generate_next_candidate(
            _lease(prepared),
            prepared,
            _references(),
            provider=provider,
            sign_reference=lambda item: f"https://r2.example/{item.r2_key}",
        )
        assert isinstance(outcome, generation.CandidateReady)
        harness.accept_latest()

    completed = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=_UntouchableProvider()
    )

    assert isinstance(completed, generation.GenerationComplete)
    assert completed.edition_id == prepared.plan.edition_id


def test_deadline_stops_before_request_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared()
    harness = _Harness(monkeypatch, prepared)
    harness.deadline = datetime(2020, 1, 1, tzinfo=UTC)
    provider = _Provider()

    outcome = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(outcome, generation.GenerationFailed)
    assert harness.failed_slot is True
    assert harness.admissions == []
    assert provider.submitted_positions == []


@pytest.mark.parametrize("status_code", [408, 409, 429, 500])
def test_fal_submit_maps_uncertain_http_failures_to_ambiguous(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://queue.fal.run/minimax/h3-max/reference-to-video"
    monkeypatch.setattr(generation.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(generation.FalSubmissionAmbiguousError):
        generation.FalH3Client("secret").submit({})


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 502, 503, 504])
def test_fal_submit_distinguishes_retryable_failure(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://queue.fal.run/minimax/h3-max/reference-to-video"
    monkeypatch.setattr(generation.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(generation.FalSubmissionRetryableError):
        generation.FalH3Client("secret").submit({})


def test_fal_submit_treats_an_invalid_success_body_as_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 200
    response.url = "https://queue.fal.run/minimax/h3-max/reference-to-video"
    response._content = b"not-json"
    monkeypatch.setattr(generation.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(generation.FalSubmissionAmbiguousError):
        generation.FalH3Client("secret").submit({})


def test_fal_client_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "FAL_KEY", "   ")

    with pytest.raises(RuntimeError, match="FAL_KEY is required"):
        generation.FalH3Client()


def test_fal_submit_retries_a_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(*args, **kwargs):
        raise requests.ConnectTimeout("connect timed out")

    monkeypatch.setattr(generation.requests, "post", post)

    with pytest.raises(generation.FalSubmissionRetryableError):
        generation.FalH3Client("secret").submit({})


def test_fal_receipt_url_cannot_exfiltrate_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = False

    def get(*args, **kwargs):
        nonlocal requested
        requested = True
        raise AssertionError("untrusted URL was requested")

    monkeypatch.setattr(generation.requests, "get", get)
    receipt = generation.FalSubmissionReceipt.model_validate(
        {
            "request_id": "receipt",
            "status_url": "https://attacker.example/status",
            "response_url": "https://queue.fal.run/response",
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="trusted Fal queue host"):
        generation.FalH3Client("secret").status(receipt)

    assert requested is False


@pytest.mark.parametrize("logs", [None, [{"message": "queued"}]])
def test_fal_status_accepts_nullable_or_list_logs(
    monkeypatch: pytest.MonkeyPatch,
    logs: list[dict[str, object]] | None,
) -> None:
    response = requests.Response()
    response.status_code = 200
    response.url = "https://queue.fal.run/minimax/h3-max/reference-to-video/requests/receipt/status"
    response._content = json.dumps(
        {"status": "IN_QUEUE", "request_id": "receipt", "logs": logs}
    ).encode()
    monkeypatch.setattr(generation.requests, "get", lambda *args, **kwargs: response)
    receipt = generation.FalSubmissionReceipt.model_validate(
        {
            "request_id": "receipt",
            "status_url": response.url,
            "response_url": "https://queue.fal.run/response",
        },
        strict=True,
    )

    assert generation.FalH3Client("secret").status(receipt).logs == logs


def test_fal_download_rejects_oversized_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {"Content-Length": str(generation.FAL_MAX_DOWNLOAD_BYTES + 1)}

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, *, chunk_size: int):
            del chunk_size
            yield b"oversized"

        def close(self) -> None:
            pass

    monkeypatch.setattr(generation.requests, "get", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="maximum byte size"):
        generation.FalH3Client("secret").download("https://v3.fal.media/video.mp4")


def test_fal_download_rejects_an_untrusted_host(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = False

    def get(*args, **kwargs):
        nonlocal requested
        requested = True
        raise AssertionError("untrusted URL was requested")

    monkeypatch.setattr(generation.requests, "get", get)

    with pytest.raises(ValueError, match="trusted Fal media host"):
        generation.FalH3Client("secret").download("https://attacker.example/video.mp4")

    assert requested is False
