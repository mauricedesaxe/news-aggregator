from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from romanian_news.catalog.artifacts import ArtifactFile, CatalogArtifactReference
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
    _, digest_plan = preflight._canonical_plan_file(verified)
    return preflight.PreparedPaidGeneration(
        authorization=authorize_generation(verified),
        plan=digest_plan,
        verified_plan=verified,
    )


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


def _artifact_reference(name: str, digest: str) -> CatalogArtifactReference:
    return CatalogArtifactReference(
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
        self.attempts: list[generation.GenerationAttemptReference] = []
        self.admissions: list[GenerationAdmission] = []
        self.failed_slot = False
        self.deny_admission = False
        self.deadline = datetime(2027, 1, 1, tzinfo=UTC)

        monkeypatch.setattr(generation, "publish_generation_policy", lambda _policy: "8" * 64)
        monkeypatch.setattr(
            generation,
            "read_generation_deadline",
            lambda _slot: self.deadline,
        )
        monkeypatch.setattr(
            generation, "read_generation_attempts", lambda _edition: tuple(self.attempts)
        )
        monkeypatch.setattr(generation, "publish_immutable_r2_objects", self.publish)
        monkeypatch.setattr(generation, "read_verified_r2_object", self.read)
        monkeypatch.setattr(generation, "checkpoint_generation_request", self.checkpoint_request)
        monkeypatch.setattr(
            generation, "checkpoint_generation_submission", self.checkpoint_submission
        )
        monkeypatch.setattr(generation, "checkpoint_generation_response", self.checkpoint_response)
        monkeypatch.setattr(generation, "checkpoint_generation_failure", self.checkpoint_failure)
        monkeypatch.setattr(generation, "fail_slot", self.fail_slot)

    def publish(self, objects) -> None:
        self.objects.update(objects)

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


def _reference(file: ArtifactFile) -> CatalogArtifactReference:
    return CatalogArtifactReference(
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

    resumed = generation.generate_next_candidate(
        _lease(prepared), prepared, _references(), provider=provider
    )

    assert isinstance(resumed, generation.CandidateReady)
    assert provider.submitted_positions == [0]
    assert provider.status_calls == ["fal-0-1", "fal-0-1"]


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
    admission = harness.admissions[0]
    assert admission.limits.story_usd == 7
    assert admission.limits.edition_usd == 14
    assert admission.limits.bucharest_day_usd == 150
    assert admission.limits.calendar_month_usd == 1000


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
