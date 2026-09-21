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
    artifact_file,
    canonical_json,
    sha256,
)
from romanian_news.catalog.video_digest import PlanningAttemptReference
from romanian_news.reports import (
    DailyReport,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
)
from romanian_news.video_digest import preflight
from romanian_news.video_digest.models import (
    DigestPlan,
    SlotId,
    SlotLease,
    StoryId,
    edition_id,
)

SUBJECT_ONE = "1" * 64
SUBJECT_TWO = "2" * 64
ARTICLE_ONE = "3" * 64
ARTICLE_TWO = "4" * 64


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


def test_planning_report_rejects_a_version_for_different_content() -> None:
    report = _report()

    with pytest.raises(ValueError, match="version does not match its content"):
        preflight.PlanningReport(version_id="a" * 64, report=report.report)


def test_production_policy_uses_independent_model_families() -> None:
    policy = preflight.PRODUCTION_POLICY.definition.policy

    assert policy.planning_model.partition("/")[0] != policy.verification_model.partition("/")[0]


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


def _response(response_id: str, content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "fake/model",
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

        monkeypatch.setattr(
            preflight, "read_planning_attempts", lambda _edition_id: tuple(self.attempts)
        )
        monkeypatch.setattr(preflight, "publish_immutable_r2_objects", self.publish)
        monkeypatch.setattr(preflight, "read_verified_r2_object", self.read)
        monkeypatch.setattr(preflight, "checkpoint_planning_attempt", self.checkpoint_attempt)
        monkeypatch.setattr(preflight, "checkpoint_story_verification", self.checkpoint_story)
        monkeypatch.setattr(preflight, "checkpoint_edition_verification", self.checkpoint_manifest)
        monkeypatch.setattr(preflight, "record_model_attempt", self.record_model_attempt)
        authorize = preflight.authorize_generation

        def record_authorization(verified):
            self.events.append("authorize")
            return authorize(verified)

        monkeypatch.setattr(preflight, "authorize_generation", record_authorization)

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
            return _response(f"plan-{provider_calls}", _plan_content())
        harness.story_inputs.append(context)
        return _response(f"verify-{provider_calls}", '{"status":"accepted","failures":[]}')

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
    assert harness.events[-2:] == ["authorize", "manifest"]
    assert all(event.startswith("story:") for event in harness.events[1:-2])
    assert provider_calls == 3

    def fail_provider(request: ProviderChatRequest) -> ChatCompletion:
        del request
        pytest.fail("accepted replay called the provider")

    replayed = preflight.prepare_paid_generation(_lease(), _report(), provider=fail_provider)
    assert replayed == prepared
    assert provider_calls == 3


def test_preflight_bounds_whole_plan_rewrites_and_returns_structured_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)
    requests: list[ProviderChatRequest] = []

    def provider(request: ProviderChatRequest) -> ChatCompletion:
        requests.append(request)
        if _system_prompt(request) == preflight.PLANNING_PROMPT:
            return _response(f"plan-{len(requests)}", _plan_content())
        return _response(
            f"verify-{len(requests)}",
            json.dumps(
                {
                    "status": "rejected",
                    "failures": [{"code": "unsupported_claim", "message": "No evidence"}],
                }
            ),
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
    assert "manifest" not in harness.events
    assert "authorize" not in harness.events
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
        return _response(f"plan-{len(requests)}", json.dumps(missing_story))

    with pytest.raises(preflight.PlanningExhaustedError):
        preflight.prepare_paid_generation(_lease(), _report(), provider=provider)

    assert len(requests) == 3


def test_policy_publication_writes_r2_before_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "publish_immutable_r2_objects",
        lambda _objects: calls.append("r2"),
    )
    monkeypatch.setattr(
        preflight,
        "record_policy_bundle",
        lambda file, *, recorded_at: calls.append("catalog") or file.version_id,
    )

    version_id = preflight.publish_policy()

    assert version_id == preflight.PRODUCTION_POLICY.artifact.version_id
    assert calls == ["r2", "catalog"]
