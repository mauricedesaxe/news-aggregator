from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, cast

import pytest
from openai.types.chat import ChatCompletion

from romanian_news.analysis.attempts import model_attempt_from_payload
from romanian_news.analysis.tracing import ProviderChatRequest
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    canonical_json,
    sha256,
)
from romanian_news.catalog.video_digest import (
    AcceptedPlanReference,
    GenerationPreparationReference,
    PlanningAttemptReference,
)
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.video_digest import planning, preflight
from romanian_news.video_digest.models import (
    DigestPlan,
    SlotId,
    SlotLease,
    StoryId,
    edition_id,
)
from romanian_news.video_digest.selection import SlotSubjectSelection, SubjectDecision

SUBJECT_ONE = "1" * 64
SUBJECT_TWO = "2" * 64
ARTICLE_ONE = "3" * 64
ARTICLE_TWO = "4" * 64


def test_video_policy_requires_english_narration() -> None:
    definition = preflight.PRODUCTION_POLICY.definition

    assert "English-language video screenplay" in definition.planning_prompt
    assert "all spoken narration in natural English" in definition.planning_prompt
    assert "Reject non-English narration" in definition.verification_prompt


def _section(subject: str, article: str, rank: int) -> DailyReportSection:
    return DailyReportSection(
        theme_id=subject,
        title=f"Subject {rank}",
        summary=f"Summary {rank}",
        events=(
            ReportEvent(
                group_id=article,
                title_ro=f"Titlu {rank}",
                summary_ro=f"Rezumat {rank}",
                key_points_ro=(f"Punct {rank}",),
                disagreements_ro=(),
                sentiment_label="neutral",
                sentiment_score=0,
                sentiment_rationale_ro="Neutru",
                articles=(
                    ReportArticle(
                        article_version_id=article,
                        outlet_id="example",
                        title=f"Article {rank}",
                        canonical_url=f"https://example.com/{rank}",
                        sentiment_label="neutral",
                        sentiment_score=0,
                    ),
                ),
            ),
        ),
        tier="main",
        semantic_rank=rank,
        consequence_rationale=f"Consequence {rank}",
        citations=(
            ReportSubjectCitation(
                article_version_id=article,
                evidence_quote=f"Evidence {rank}",
            ),
        ),
    )


def _report() -> preflight.PlanningReport:
    report = DailyReport(
        day=date(2026, 9, 20),
        accepted_article_count=2,
        theme_count=2,
        group_count=2,
        sections=(
            _section(SUBJECT_ONE, ARTICLE_ONE, 1),
            _section(SUBJECT_TWO, ARTICLE_TWO, 2),
        ),
    )
    content = canonical_json(report.model_dump(mode="json"))
    file = artifact_file(
        artifact_id=f"news:daily:{report.day.isoformat()}",
        artifact_kind="news_daily_report",
        title=f"Romanian news report for {report.day.isoformat()}",
        content=content,
        r2_key=f"news/reports/daily/{report.day.isoformat()}/{sha256(content)}.json",
        media_type="application/json",
    )
    return preflight.PlanningReport(version_id=file.version_id, report=report)


def _lease() -> SlotLease:
    report = _report()
    return SlotLease(
        slot_id=SlotId("5" * 64),
        edition_id=edition_id(report.version_id, preflight.PRODUCTION_POLICY.artifact.version_id),
        owner_token="owner",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        claim_count=1,
    )


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def test_read_prepared_generation_rehydrates_immutable_plan_and_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _lease()
    report = _report()
    harness = _Harness(monkeypatch)
    provider_calls = 0

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        nonlocal provider_calls
        provider_calls += 1
        content = (
            _plan_content()
            if _system_prompt(request) == preflight.PLANNING_PROMPT
            else '{"status":"accepted","failures":[]}'
        )
        return _response(f"response-{provider_calls}", content, request["model"])

    prepared = preflight.prepare_paid_generation(lease, report, provider=provider)
    plan_file, _plan = preflight.verified_plan_file(prepared.verified_plan)
    manifest = harness.manifest_file
    reference = GenerationPreparationReference(
        daily_report_version_id=report.version_id,
        policy_bundle_version_id=preflight.PRODUCTION_POLICY.artifact.version_id,
        plan=_reference(plan_file),
        authorization=_reference(manifest),
    )
    monkeypatch.setattr(preflight, "read_generation_preparation", lambda _lease: reference)

    assert preflight.read_prepared_paid_generation(lease) == prepared

    accepted = AcceptedPlanReference(
        daily_report_version_id=reference.daily_report_version_id,
        policy_bundle_version_id=reference.policy_bundle_version_id,
        plan=reference.plan,
    )
    monkeypatch.setattr(preflight, "read_accepted_plan_reference", lambda _edition: accepted)
    assert preflight.read_accepted_digest_plan(lease.edition_id) == (
        prepared.verified_plan,
        prepared.plan,
    )

    wrong = reference.model_copy(update={"authorization": _reference(plan_file)})
    monkeypatch.setattr(preflight, "read_generation_preparation", lambda _lease: wrong)
    with pytest.raises(ValueError):
        preflight.read_prepared_paid_generation(lease)

    wrong = reference.model_copy(
        update={
            "authorization": reference.authorization.model_copy(update={"version_id": "b" * 64})
        }
    )
    monkeypatch.setattr(preflight, "read_generation_preparation", lambda _lease: wrong)
    with pytest.raises(ValueError, match="catalog reference"):
        preflight.read_prepared_paid_generation(lease)

    wrong = reference.model_copy(update={"daily_report_version_id": "a" * 64})
    monkeypatch.setattr(preflight, "read_generation_preparation", lambda _lease: wrong)
    with pytest.raises(ValueError, match="catalog edition"):
        preflight.read_prepared_paid_generation(lease)


def test_planning_report_rejects_a_version_for_different_content() -> None:
    report = _report()

    with pytest.raises(ValueError, match="version does not match its content"):
        preflight.PlanningReport(version_id="a" * 64, report=report.report)


def test_production_policy_uses_independent_model_families() -> None:
    definition = preflight.PRODUCTION_POLICY.definition
    policy = definition.policy

    assert policy.planning_model.partition("/")[0] != policy.verification_model.partition("/")[0]

    same_family = policy.model_copy(update={"verification_model": policy.planning_model})
    with pytest.raises(ValueError, match="independent model families"):
        preflight.VideoDigestPolicyDefinition(
            policy=same_family,
            planning_prompt=definition.planning_prompt,
            verification_prompt=definition.verification_prompt,
            planning_response_schema_digest=definition.planning_response_schema_digest,
            verification_response_schema_digest=definition.verification_response_schema_digest,
        )


def _plan_content() -> str:
    narration = " ".join(f"cuvant{index}" for index in range(30))
    return json.dumps(
        {
            "stories": [
                {
                    "report_subject_id": subject,
                    "title": f"Video {position}",
                    "citation_article_version_ids": [article],
                    "narration": narration,
                    "visual_direction": f"Visual {position}",
                    "requested_duration_ms": 15000,
                }
                for position, (subject, article) in enumerate(
                    ((SUBJECT_ONE, ARTICLE_ONE), (SUBJECT_TWO, ARTICLE_TWO)), start=1
                )
            ]
        }
    )


def _response(response_id: str, content: str, model: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


class _Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.objects: dict[str, bytes] = {}
        self.attempts: list[PlanningAttemptReference] = []
        self.events: list[str] = []
        self.story_inputs: list[dict[str, Any]] = []
        self.planning_inputs: list[dict[str, Any]] = []
        self.story_files: dict[StoryId, Any] = {}
        self.manifest_file: Any = None

        monkeypatch.setattr(
            preflight, "read_planning_attempts", lambda _edition_id: tuple(self.attempts)
        )
        monkeypatch.setattr(preflight, "publish_immutable_r2_objects", self.publish)
        monkeypatch.setattr(preflight, "read_verified_r2_object", self.read)
        monkeypatch.setattr(preflight, "checkpoint_planning_attempt", self.checkpoint_attempt)
        monkeypatch.setattr(preflight, "checkpoint_story_verification", self.checkpoint_story)
        monkeypatch.setattr(preflight, "checkpoint_edition_verification", self.checkpoint_manifest)
        monkeypatch.setattr(preflight, "record_model_attempt", self.record_model_attempt)

    def publish(self, objects: Iterable[tuple[str, bytes]]) -> None:
        for key, content in objects:
            self.objects[key] = content
        return None

    def read(self, key: str, _digest: str) -> bytes:
        return self.objects[key]

    def checkpoint_attempt(
        self,
        _lease: SlotLease,
        attempt_index: int,
        disposition: Literal["rejected", "accepted"],
        *,
        evidence_file: Any,
        accepted_plan: DigestPlan | None = None,
        plan_file: Any = None,
        recorded_at: datetime,
    ) -> DigestPlan | None:
        del recorded_at, plan_file
        assert self.objects[evidence_file.r2_key] == evidence_file.content
        self.events.append(f"attempt:{attempt_index}:{disposition}")
        self.attempts.append(
            PlanningAttemptReference(
                attempt_index=attempt_index,
                disposition=disposition,
                evidence=ArtifactReference(
                    artifact_id=evidence_file.artifact_id,
                    version_id=evidence_file.version_id,
                    content_digest=evidence_file.content_digest,
                    r2_key=evidence_file.r2_key,
                ),
                accepted_plan_artifact_version_id=(
                    accepted_plan.artifact_version_id if accepted_plan is not None else None
                ),
            )
        )
        return accepted_plan

    def checkpoint_story(
        self,
        _lease: SlotLease,
        story_id: StoryId,
        *,
        evidence_file: Any,
        recorded_at: datetime,
    ) -> None:
        del recorded_at
        assert self.objects[evidence_file.r2_key] == evidence_file.content
        self.story_files[story_id] = evidence_file
        self.events.append(f"story:{story_id}")

    def checkpoint_manifest(
        self,
        _lease: SlotLease,
        *,
        manifest_file: Any,
        recorded_at: datetime,
    ) -> None:
        del recorded_at
        assert self.objects[manifest_file.r2_key] == manifest_file.content
        self.manifest_file = manifest_file
        self.events.append("manifest")

    def record_model_attempt(self, response: ChatCompletion, **values: Any):
        values.pop("trace")
        payload = response.model_dump(mode="json")
        usage = cast(dict[str, object], payload["usage"])
        usage["cost"] = 0
        return model_attempt_from_payload(
            payload,
            observed_at=datetime.now(UTC),
            **values,
        )


def _request_context(request: ProviderChatRequest) -> dict[str, Any]:
    content = request["messages"][1].get("content")
    assert isinstance(content, str)
    value = json.loads(content)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _system_prompt(request: ProviderChatRequest) -> str:
    content = request["messages"][0].get("content")
    assert isinstance(content, str)
    return content


def test_preflight_uses_the_stored_later_slot_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    selection = SlotSubjectSelection(
        report_version_id=report.version_id,
        prior_slot_ids=("a" * 64,),
        decisions=(
            SubjectDecision(theme_id=SUBJECT_ONE, disposition="unchanged", reason="exact_match"),
            SubjectDecision(theme_id=SUBJECT_TWO, disposition="selected", reason="new_subject"),
        ),
        selected_sections=(report.report.sections[1],),
    )
    lease = _lease().model_copy(
        update={
            "edition_id": edition_id(
                report.version_id,
                preflight.PRODUCTION_POLICY.artifact.version_id,
                selection.digest,
            )
        }
    )
    monkeypatch.setattr(preflight, "read_edition_subject_selection", lambda _id: selection)
    harness = _Harness(monkeypatch)

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            assert [item["theme_id"] for item in _request_context(request)["main_stories"]] == [
                SUBJECT_TWO
            ]
            content = json.loads(_plan_content())
            content["stories"] = content["stories"][1:]
            return _response("selected-plan", json.dumps(content), request["model"])
        return _response(
            "selected-verification",
            '{"status":"accepted","failures":[]}',
            request["model"],
        )

    prepared = preflight.prepare_paid_generation(lease, report, provider=provider)
    assert prepared.verified_plan.plan.selection_digest == selection.digest
    assert tuple(story.report_subject_id for story in prepared.plan.stories) == (SUBJECT_TWO,)
    assert len(harness.story_files) == 1


def test_preflight_preserves_report_order_isolates_verifiers_and_replays_without_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    provider_calls = 0

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        nonlocal provider_calls
        provider_calls += 1
        context = _request_context(request)
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            harness.planning_inputs.append(context)
            return _response(f"plan-{provider_calls}", _plan_content(), request["model"])
        harness.story_inputs.append(context)
        return _response(
            f"verify-{provider_calls}",
            '{"status":"accepted","failures":[]}',
            request["model"],
        )

    prepared = preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert tuple(story.report_subject_id for story in prepared.plan.stories) == (
        SUBJECT_ONE,
        SUBJECT_TWO,
    )
    assert [item["theme_id"] for item in harness.planning_inputs[0]["main_stories"]] == [
        SUBJECT_ONE,
        SUBJECT_TWO,
    ]
    assert len(harness.story_inputs) == 2
    assert all(set(item) == {"story", "report_evidence"} for item in harness.story_inputs)
    assert all(
        item["story"]["report_subject_id"] == item["report_evidence"]["theme_id"]
        for item in harness.story_inputs
    )
    assert harness.events[0] == "attempt:0:accepted"
    assert harness.events[-1] == "manifest"
    assert all(event.startswith("story:") for event in harness.events[1:-1])
    assert provider_calls == 3

    authorization = planning.GenerationAuthorization.model_validate_json(
        harness.manifest_file.content, strict=True
    )
    assert authorization == prepared.authorization
    ordered_story_files = [harness.story_files[story.story_id] for story in prepared.plan.stories]
    assert authorization.ordered_verification_evidence_digests == tuple(
        planning.verification_evidence_digest(
            planning.StoryVerificationEvidence.model_validate_json(
                harness.objects[file.r2_key], strict=True
            )
        )
        for file in ordered_story_files
    )

    def fail_provider(request: ProviderChatRequest) -> ChatCompletion:
        del request
        pytest.fail("accepted replay called the provider")

    replayed = preflight.prepare_paid_generation(_lease(), _report(), provider=fail_provider)
    assert replayed == prepared
    assert provider_calls == 3


def test_preflight_renews_lease_before_every_provider_call_and_checkpoints_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    original = _lease()
    renewals: list[SlotLease] = []
    checkpoint_leases: list[SlotLease] = []
    checkpoint_attempt = harness.checkpoint_attempt

    def renew(lease: SlotLease) -> SlotLease:
        updated = lease.model_copy(update={"expires_at": lease.expires_at + timedelta(minutes=1)})
        renewals.append(updated)
        return updated

    def checkpoint(lease: SlotLease, *args, **kwargs):
        checkpoint_leases.append(lease)
        return checkpoint_attempt(lease, *args, **kwargs)

    monkeypatch.setattr(preflight, "checkpoint_planning_attempt", checkpoint)

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            return _response("plan", _plan_content(), request["model"])
        return _response("verify", '{"status":"accepted","failures":[]}', request["model"])

    preflight.prepare_paid_generation(original, _report(), provider=provider, renew_lease=renew)

    assert len(renewals) == 3
    assert checkpoint_leases == [renewals[-1]]
    assert checkpoint_leases[0].expires_at > original.expires_at


def test_preflight_bounds_whole_plan_rewrites_and_returns_structured_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    requests: list[ProviderChatRequest] = []

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        requests.append(request)
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            return _response(f"plan-{len(requests)}", _plan_content(), request["model"])
        return _response(
            f"verify-{len(requests)}",
            json.dumps(
                {
                    "status": "rejected",
                    "failures": [{"code": "unsupported_claim", "message": "No evidence"}],
                }
            ),
            request["model"],
        )

    with pytest.raises(preflight.PlanningExhaustedError) as raised:
        preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert len(requests) == 9
    assert len(raised.value.failure.attempts) == 3
    assert [item.attempt_index for item in raised.value.failure.attempts] == [0, 1, 2]
    planning_requests = [
        _request_context(item)
        for item in requests
        if _system_prompt(item) == preflight.PLANNING_PROMPT
    ]
    assert [len(item["prior_failures"]) for item in planning_requests] == [0, 1, 2]
    assert [failure.code for failure in raised.value.failure.attempts[0].failures] == [
        "story_0:unsupported_claim",
        "story_1:unsupported_claim",
    ]
    assert "manifest" not in harness.events
    assert not any(event.startswith("story:") for event in harness.events)


def test_invalid_plan_is_rejected_before_verifier_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(monkeypatch)
    requests: list[ProviderChatRequest] = []
    missing_story = json.loads(_plan_content())
    missing_story["stories"] = missing_story["stories"][:1]

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        requests.append(request)
        if _system_prompt(request) != preflight.PLANNING_PROMPT:
            pytest.fail("structurally invalid plan reached the verifier")
        return _response(f"plan-{len(requests)}", json.dumps(missing_story), request["model"])

    with pytest.raises(preflight.PlanningExhaustedError):
        preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert len(requests) == 3


def test_provider_model_mismatch_is_recorded_as_a_rejected_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        return _response("wrong-model", _plan_content(), "other/model")

    with pytest.raises(preflight.PlanningExhaustedError) as raised:
        preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert [item.disposition for item in harness.attempts] == ["rejected"] * 3
    assert all(
        "does not match the requested policy model" in item.failures[0].message
        for item in raised.value.failure.attempts
    )


def test_policy_publication_records_durable_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    published: dict[str, bytes] = {}

    def publish(objects: Iterable[tuple[str, bytes]]) -> None:
        for key, content in objects:
            published[key] = content

    def record(file: Any, *, recorded_at: datetime) -> str:
        del recorded_at
        assert published[file.r2_key] == file.content
        return file.version_id

    monkeypatch.setattr(preflight, "publish_immutable_r2_objects", publish)
    monkeypatch.setattr(preflight, "record_policy_bundle", record)

    version_id = preflight.publish_policy()

    assert version_id == preflight.PRODUCTION_POLICY.artifact.version_id


def _rejecting_provider(requests: list[ProviderChatRequest]) -> Any:
    def provider(request: ProviderChatRequest) -> ChatCompletion:
        requests.append(request)
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            return _response(f"plan-{len(requests)}", _plan_content(), request["model"])
        return _response(
            f"verify-{len(requests)}",
            json.dumps(
                {
                    "status": "rejected",
                    "failures": [{"code": "unsupported_claim", "message": "No evidence"}],
                }
            ),
            request["model"],
        )

    return provider


def test_prepare_paid_generation_rejects_a_lease_for_a_different_edition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(monkeypatch)
    monkeypatch.setattr(preflight, "read_edition_subject_selection", lambda _edition_id: None)
    report = _report()
    wrong_edition = SlotLease(
        slot_id=SlotId("5" * 64),
        edition_id=edition_id("b" * 64, "c" * 64),
        owner_token="owner",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        claim_count=1,
    )

    with pytest.raises(ValueError, match="Planning inputs do not match the claimed edition"):
        preflight.prepare_paid_generation(wrong_edition, report)


def test_recorded_accepted_plan_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            return _response("plan-1", _plan_content(), request["model"])
        return _response("verify-1", '{"status":"accepted","failures":[]}', request["model"])

    preflight.prepare_paid_generation(_lease(), _report(), provider=provider)
    harness.attempts[0] = harness.attempts[0].model_copy(
        update={"accepted_plan_artifact_version_id": "0" * 64}
    )

    with pytest.raises(ValueError, match="Recorded accepted plan does not match"):
        preflight.prepare_paid_generation(_lease(), _report(), provider=provider)


def test_pre_recorded_rejections_exhaust_without_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(monkeypatch)
    first_requests: list[ProviderChatRequest] = []
    with pytest.raises(preflight.PlanningExhaustedError):
        preflight.prepare_paid_generation(
            _lease(), _report(), provider=_rejecting_provider(first_requests)
        )

    def fail_provider(request: ProviderChatRequest) -> ChatCompletion:
        del request
        pytest.fail("an exhausted edition must not call the provider")

    with pytest.raises(preflight.PlanningExhaustedError) as raised:
        preflight.prepare_paid_generation(_lease(), _report(), provider=fail_provider)

    assert len(raised.value.failure.attempts) == 3
    assert len(first_requests) == 9


def test_resume_feeds_recorded_rejections_to_the_next_planning_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    exhausting_requests: list[ProviderChatRequest] = []
    with pytest.raises(preflight.PlanningExhaustedError):
        preflight.prepare_paid_generation(
            _lease(), _report(), provider=_rejecting_provider(exhausting_requests)
        )
    harness.attempts[:] = harness.attempts[:1]
    harness.events.clear()
    harness.planning_inputs.clear()

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            harness.planning_inputs.append(_request_context(request))
            return _response("plan-resumed", _plan_content(), request["model"])
        return _response("verify-resumed", '{"status":"accepted","failures":[]}', request["model"])

    prepared = preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert harness.events[0] == "attempt:1:accepted"
    prior_failures = harness.planning_inputs[0]["prior_failures"]
    assert [item["attempt_index"] for item in prior_failures] == [0]
    assert [failure["code"] for item in prior_failures for failure in item["failures"]] == [
        "story_0:unsupported_claim",
        "story_1:unsupported_claim",
    ]
    assert tuple(story.report_subject_id for story in prepared.plan.stories) == (
        SUBJECT_ONE,
        SUBJECT_TWO,
    )
