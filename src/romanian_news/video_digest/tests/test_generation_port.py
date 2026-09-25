from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from romanian_news.artifacts import ArtifactReference
from romanian_news.video_digest import generation, generation_port
from romanian_news.video_digest.generation import (
    CandidateReady,
    CandidateReference,
    GenerationComplete,
    GenerationFailed,
    GenerationInProgress,
    GenerationRetryAvailable,
    H3ReferencePack,
)
from romanian_news.video_digest.models import (
    EditionId,
    GenerationRequestId,
    ScheduledSlot,
    SlotFailureReason,
    SlotId,
    SlotLease,
    SlotName,
    StoryId,
    UnknownAttemptCost,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionWaiting,
    FailedResume,
    GenerateAction,
)
from romanian_news.video_digest.tests.test_generation import _prepared, _references


def _action() -> GenerateAction:
    return GenerateAction(
        lease=SlotLease(
            slot_id=SlotId("1" * 64),
            edition_id=EditionId("2" * 64),
            owner_token="owner",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            claim_count=1,
        )
    )


def _pack() -> H3ReferencePack:
    video = ArtifactReference(
        artifact_id="video",
        version_id="3" * 64,
        content_digest="4" * 64,
        r2_key="references/video",
    )
    audio = ArtifactReference(
        artifact_id="audio",
        version_id="5" * 64,
        content_digest="6" * 64,
        r2_key="references/audio",
    )
    return H3ReferencePack(videos=(video,), audio=(audio,))


def _configured(monkeypatch: pytest.MonkeyPatch, outcome: object) -> tuple[Mock, Mock]:
    candidate_acceptance = Mock()
    assembly_ready = Mock()
    monkeypatch.setattr(generation_port, "read_h3_reference_pack", lambda _pack_id: _pack())
    monkeypatch.setattr(
        generation_port,
        "read_prepared_paid_generation",
        lambda _lease: SimpleNamespace(
            plan=SimpleNamespace(stories=("story",)),
            verified_plan=SimpleNamespace(
                plan=SimpleNamespace(stories=(SimpleNamespace(narration="English narration"),))
            ),
        ),
    )
    monkeypatch.setattr(generation_port, "read_generation_attempts", lambda _edition: ())
    monkeypatch.setattr(generation_port, "generate_next_candidate", lambda *_args, **_kw: outcome)
    monkeypatch.setattr(generation_port, "accept_candidate", candidate_acceptance)
    monkeypatch.setattr(generation_port, "checkpoint_assembly_ready", assembly_ready)
    return candidate_acceptance, assembly_ready


def test_missing_configuration_fails_before_provider_or_catalog_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation_port, "NEWS_H3_REFERENCE_PACK_ID", None)
    monkeypatch.setattr(
        generation_port,
        "read_h3_reference_pack",
        lambda _pack_id: pytest.fail("missing pack configuration reached the catalog"),
    )
    with pytest.raises(ValueError, match="NEWS_H3_REFERENCE_PACK_ID"):
        generation_port.ProductionH3GenerationPort().execute(_action())


def test_provider_credentials_are_checked_before_pack_or_request_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        generation_port,
        "FalH3Client",
        lambda: (_ for _ in ()).throw(ValueError("FAL_KEY is required")),
    )
    monkeypatch.setattr(
        generation_port,
        "read_h3_reference_pack",
        lambda _pack_id: pytest.fail("missing Fal credentials reached the catalog"),
    )
    with pytest.raises(ValueError, match="FAL_KEY"):
        generation_port.ProductionH3GenerationPort(pack_id="7" * 64).execute(_action())


def test_generation_port_validates_candidate_then_advances(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = CandidateReady(
        request_id=GenerationRequestId("8" * 64),
        story_id=StoryId("9" * 64),
        story_position=0,
        attempt_index=0,
        response_artifact_version_id="a" * 64,
        candidate=CandidateReference(r2_key="candidate.mp4", content_digest="b" * 64, byte_size=10),
        cost=UnknownAttemptCost(reason="provider did not report cost"),
    )
    accepted, assembly = _configured(monkeypatch, candidate)
    action = _action()
    provider = Mock()

    outcome = generation_port.ProductionH3GenerationPort(
        provider=provider, pack_id="7" * 64
    ).execute(action)

    assert isinstance(outcome, ActionAdvanced)
    accepted.assert_called_once_with(
        action.lease, candidate, "story", approved_narration="English narration"
    )
    assembly.assert_not_called()


def test_generation_port_waits_for_fal_and_advances_to_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action()
    provider = Mock()
    accepted, assembly = _configured(
        monkeypatch,
        GenerationInProgress(
            request_id=GenerationRequestId("8" * 64),
            provider_receipt_id="fal-1",
            provider_status="IN_QUEUE",
        ),
    )
    port = generation_port.ProductionH3GenerationPort(provider=provider, pack_id="7" * 64)
    assert port.execute(action) == ActionWaiting(
        reason="Fal H3 request IN_QUEUE", retry_after_seconds=60
    )
    accepted.assert_not_called()
    assembly.assert_not_called()

    _, assembly = _configured(monkeypatch, GenerationComplete(edition_id=action.lease.edition_id))
    assert isinstance(port.execute(action), ActionAdvanced)
    assembly.assert_called_once()


def test_terminal_generation_failure_requires_failed_catalog_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action()
    _configured(
        monkeypatch,
        GenerationFailed(edition_id=action.lease.edition_id, reason="both attempts failed"),
    )
    scheduled_at = datetime(2099, 9, 20, 5, tzinfo=UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, scheduled_at),
        name=SlotName.MORNING,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.date(),
    )
    monkeypatch.setattr(
        generation_port,
        "read_slot_resume_state",
        lambda _slot_id: FailedResume(slot=slot, reason=SlotFailureReason.TERMINAL_FAILURE),
    )
    port = generation_port.ProductionH3GenerationPort(provider=Mock(), pack_id="7" * 64)
    assert isinstance(port.execute(action), ActionAdvanced)

    monkeypatch.setattr(generation_port, "read_slot_resume_state", lambda _slot_id: object())
    with pytest.raises(ValueError, match="did not fail its slot"):
        port.execute(action)


def test_retry_available_advances_from_durable_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = _action()
    _configured(
        monkeypatch,
        GenerationRetryAvailable(
            edition_id=action.lease.edition_id,
            story_position=0,
            reason="technical validation failed",
        ),
    )
    assert isinstance(
        generation_port.ProductionH3GenerationPort(provider=Mock(), pack_id="7" * 64).execute(
            action
        ),
        ActionAdvanced,
    )


def test_generation_port_rejects_reference_pack_drift_from_prior_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    old_pack = _references()
    _request, file, _identity = generation._generation_request(
        prepared, old_pack, generation.PRODUCTION_GENERATION_POLICY, 0, 0
    )
    monkeypatch.setattr(
        generation_port,
        "read_generation_attempts",
        lambda _edition: (SimpleNamespace(request_evidence=generation._reference(file)),),
    )
    monkeypatch.setattr(
        generation_port,
        "read_verified_r2_object",
        lambda _key, _digest: file.content,
    )

    generation_port._require_same_references(prepared.plan.edition_id, old_pack)
    with pytest.raises(ValueError, match="differs from stored generation work"):
        generation_port._require_same_references(prepared.plan.edition_id, _pack())
