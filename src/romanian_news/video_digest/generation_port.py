from __future__ import annotations

from datetime import UTC, datetime

from pydantic import TypeAdapter

from romanian_news import Sha256
from romanian_news.catalog.video_digest import (
    checkpoint_assembly_ready,
    read_generation_attempts,
    read_slot_resume_state,
)
from romanian_news.config import NEWS_H3_REFERENCE_PACK_ID
from romanian_news.storage import read_verified_r2_object
from romanian_news.video_digest.generation import (
    CandidateReady,
    FalH3Client,
    FalH3Provider,
    GenerationComplete,
    GenerationFailed,
    GenerationInProgress,
    H3GenerationRequest,
    H3ReferencePack,
    generate_next_candidate,
)
from romanian_news.video_digest.media import accept_candidate
from romanian_news.video_digest.models import EditionId
from romanian_news.video_digest.orchestration import (
    ActionAdvanced,
    ActionOutcome,
    ActionWaiting,
    FailedResume,
    GenerateAction,
    GenerationPort,
)
from romanian_news.video_digest.preflight import read_prepared_paid_generation
from romanian_news.video_digest.references import read_h3_reference_pack


class ProductionH3GenerationPort(GenerationPort):
    def __init__(
        self,
        *,
        provider: FalH3Provider | None = None,
        pack_id: Sha256 | None = None,
    ) -> None:
        self._provider = provider
        self._pack_id = pack_id

    def execute(self, action: GenerateAction) -> ActionOutcome:
        configured = self._pack_id or NEWS_H3_REFERENCE_PACK_ID
        if not configured:
            raise ValueError("NEWS_H3_REFERENCE_PACK_ID is required for H3 generation")
        pack_id = TypeAdapter(Sha256).validate_python(configured)
        provider = self._provider or FalH3Client()
        references = read_h3_reference_pack(pack_id)
        prepared = read_prepared_paid_generation(action.lease)
        _require_same_references(action.lease.edition_id, references)
        result = generate_next_candidate(
            action.lease,
            prepared,
            references,
            provider=provider,
        )
        if isinstance(result, GenerationInProgress):
            return ActionWaiting(
                reason=f"Fal H3 request {result.provider_status}", retry_after_seconds=60
            )
        if isinstance(result, CandidateReady):
            story = prepared.plan.stories[result.story_position]
            accept_candidate(action.lease, result, story)
        elif isinstance(result, GenerationComplete):
            checkpoint_assembly_ready(action.lease, recorded_at=datetime.now(UTC))
        elif isinstance(result, GenerationFailed):
            if not isinstance(read_slot_resume_state(action.lease.slot_id), FailedResume):
                raise ValueError("Terminal H3 generation failure did not fail its slot")
        return ActionAdvanced()


def _require_same_references(edition_id: EditionId, references: H3ReferencePack) -> None:
    for attempt in read_generation_attempts(edition_id):
        request = H3GenerationRequest.model_validate_json(
            read_verified_r2_object(
                attempt.request_evidence.r2_key,
                attempt.request_evidence.content_digest,
            ),
            strict=True,
        )
        if request.edition_id != edition_id or request.references != references:
            raise ValueError("Configured H3 reference pack differs from stored generation work")
