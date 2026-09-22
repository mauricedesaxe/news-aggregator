from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, LiteralString, cast

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
import romanian_news.catalog.video_digest as video_digest_catalog
from romanian_news import BUCHAREST
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.video_digest.errors import (
    VideoDigestCheckpointConflictError,
    VideoDigestLeaseLostError,
)
from romanian_news.video_digest.models import (
    AssembledVideo,
    AvailableSubtitles,
    ClaimedSlot,
    DigestPlan,
    EditionIdentity,
    EstimatedAttemptCost,
    FailedSubtitles,
    GenerationAdmission,
    GenerationBudgetLimits,
    GenerationRequestId,
    GenerationRequestIdentity,
    GenerationRequestState,
    GenerationStage,
    MeasuredAttemptCost,
    PendingAttemptCost,
    PlannedStory,
    PublicationIntent,
    PublicationState,
    ScheduledSlot,
    SlotLease,
    SlotName,
    StoryId,
    SubtitleOutcome,
    UnknownAttemptCost,
    edition_id,
    generation_request_id,
    planned_story_id,
    publication_id,
    scheduled_slot_id,
)
from romanian_news.video_digest.planning_artifacts import (
    PlanningAttemptArtifact,
)
from romanian_news.video_digest.planning_artifacts import (
    planning_attempt_file as canonical_planning_attempt_file,
)
from tests.contracts.video_digest_planning_fixtures import (
    accepted_planning_files,
    rejected_planning_file,
)

SCHEDULED_AT = datetime(2099, 9, 20, 6, tzinfo=UTC)


def _sha256_id(value: int) -> str:
    return f"{value:064x}"


def _record_artifact_versions(
    connection: psycopg.Connection[Any], start: int, count: int
) -> tuple[str, ...]:
    version_ids = tuple(_sha256_id(value) for value in range(start, start + count))
    for version_id in version_ids:
        artifact_id = f"artifact-{version_id}"
        connection.execute(
            "INSERT INTO artifacts "
            "(id, kind, title, authority_class, lifecycle_state, visibility, created_at) "
            "VALUES (%s, 'test', 'Test artifact', 'test', 'current', 'private', "
            "CURRENT_TIMESTAMP)",
            (artifact_id,),
        )
        connection.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, schema_version, content_digest, created_at) "
            "VALUES (%s, %s, 1, %s, CURRENT_TIMESTAMP)",
            (version_id, artifact_id, version_id),
        )
    return version_ids


def _query_one(statement: str, parameters: Sequence[object] = ()) -> tuple[Any, ...] | None:
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        row = connection.execute(cast(LiteralString, statement), parameters).fetchone()
    return tuple(row) if row is not None else None


def _arrange(statement: str, parameters: Sequence[object] = ()) -> None:
    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(cast(LiteralString, statement), parameters)


@contextmanager
def _rejecting_update(table: str, label: str, condition: str) -> Iterator[None]:
    assert news_schema.NEWS_POSTGRES_DSN is not None
    function_name = f"reject_{label}"
    trigger_name = f"reject_{label}_trigger"
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(
            cast(
                LiteralString,
                f"CREATE FUNCTION {function_name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
                f"BEGIN IF {condition} THEN RAISE EXCEPTION 'rejected by contract' "
                "USING ERRCODE = '23000'; END IF; RETURN NEW; END; $$",
            )
        )
        connection.execute(
            cast(
                LiteralString,
                f"CREATE TRIGGER {trigger_name} BEFORE UPDATE ON {table} "
                f"FOR EACH ROW EXECUTE FUNCTION {function_name}()",
            )
        )
    try:
        yield
    finally:
        assert news_schema.NEWS_POSTGRES_DSN is not None
        with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(cast(LiteralString, f"DROP TRIGGER {trigger_name} ON {table}"))
            connection.execute(cast(LiteralString, f"DROP FUNCTION {function_name}"))


def _expire_lease(pipeline: _GenerationPipeline) -> None:
    _arrange(
        "UPDATE video_digest_slots SET lease_expires_at = CURRENT_TIMESTAMP "
        "- INTERVAL '1 second', updated_at = CURRENT_TIMESTAMP WHERE slot_id = %s",
        (pipeline.lease.slot_id,),
    )


def _recover_lease(pipeline: _GenerationPipeline) -> ClaimedSlot:
    recovered = video_digest_catalog.claim_slot(
        pipeline.lease.slot_id,
        pipeline.edition,
        owner_token=f"recovery-{pipeline.seed}",
        now=datetime.now(UTC),
        lease_duration=timedelta(hours=1),
    )
    assert isinstance(recovered, ClaimedSlot)
    return recovered


class _GenerationPipeline:
    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self.story_count = 0
        self.plan_stories: tuple[tuple[str, str, int], ...] | None = None
        self.recorded_at = datetime.now(UTC)
        assert news_schema.NEWS_POSTGRES_DSN is not None
        with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
            report, policy = _record_artifact_versions(connection, seed * 1000, 2)
        self.edition = EditionIdentity(
            edition_id=edition_id(report, policy),
            daily_report_version_id=report,
            policy_bundle_version_id=policy,
        )
        self.slot = ScheduledSlot(
            slot_id=scheduled_slot_id(SlotName.MORNING, SCHEDULED_AT),
            name=SlotName.MORNING,
            scheduled_at=SCHEDULED_AT,
            bucharest_day=SCHEDULED_AT.astimezone(BUCHAREST).date(),
        )
        video_digest_catalog.schedule_slot(self.slot, recorded_at=self.recorded_at)
        claimed = video_digest_catalog.claim_slot(
            self.slot.slot_id,
            self.edition,
            owner_token=f"owner-{seed}",
            now=self.recorded_at,
            lease_duration=timedelta(hours=1),
        )
        assert isinstance(claimed, ClaimedSlot)
        self.lease: SlotLease = claimed.lease

    def _file(
        self,
        artifact_id: str,
        artifact_kind: str,
        *,
        title: str,
        media_type: str = "application/json",
        content: bytes | None = None,
    ) -> ArtifactFile:
        payload = content if content is not None else f"{self.seed}:{artifact_id}".encode()
        return artifact_file(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            title=title,
            content=payload,
            r2_key=f"video-digest/{self.seed}/{artifact_kind}/{artifact_id}",
            media_type=media_type,
        )

    def _subject_id(self, position: int) -> str:
        return _sha256_id(self.seed * 100 + position)

    def story_id(self, position: int) -> StoryId:
        return planned_story_id(self.edition.edition_id, position, self._subject_id(position))

    def build_plan(self, story_count: int = 1) -> tuple[DigestPlan, ArtifactFile]:
        self.story_count = story_count
        self.plan_stories = tuple(
            (self._subject_id(position), f"Story {position}", 15_000)
            for position in range(story_count)
        )
        plan, plan_file, _attempt_file = accepted_planning_files(
            self.edition,
            self.plan_stories,
            seed=str(self.seed),
        )
        return plan, plan_file

    def checkpoint_plan(self, story_count: int = 1) -> DigestPlan:
        plan, plan_file = self.build_plan(story_count)
        stored = video_digest_catalog.checkpoint_planning_attempt(
            self.lease,
            0,
            "accepted",
            evidence_file=self.planning_attempt_file(),
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=self.recorded_at,
        )
        assert stored is not None
        return stored

    def planning_attempt_file(
        self, attempt: int = 0, *, accepted: bool | None = None
    ) -> ArtifactFile:
        use_accepted = self.plan_stories is not None if accepted is None else accepted
        if not use_accepted:
            return rejected_planning_file(self.edition.edition_id, attempt, seed=str(self.seed))
        assert self.plan_stories is not None
        _plan, _plan_file, attempt_file = accepted_planning_files(
            self.edition,
            self.plan_stories,
            attempt_index=attempt,
            seed=str(self.seed),
        )
        return attempt_file

    def evidence_file(self, position: int, *, content: bytes | None = None) -> ArtifactFile:
        return self._file(
            self.story_id(position),
            "video_digest_story_verification",
            title=f"Story verification {position}",
            content=content,
        )

    def verify_story(self, position: int) -> ArtifactFile:
        evidence = self.evidence_file(position)
        video_digest_catalog.checkpoint_story_verification(
            self.lease,
            self.story_id(position),
            evidence_file=evidence,
            recorded_at=self.recorded_at,
        )
        return evidence

    def manifest_file(self, *, content: bytes | None = None) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:verification-manifest",
            "video_digest_verification_manifest",
            title="Edition verification manifest",
            content=content,
        )

    def verify_edition(self) -> ArtifactFile:
        manifest = self.manifest_file()
        video_digest_catalog.checkpoint_edition_verification(
            self.lease, manifest_file=manifest, recorded_at=self.recorded_at
        )
        return manifest

    def verify_all_stories(self) -> None:
        for position in range(self.story_count):
            self.verify_story(position)
        self.verify_edition()

    def _request_file(self, position: int, attempt: int = 0) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:{position}:{attempt}:generation-request",
            "video_digest_generation_request",
            title=f"Generation request {position}/{attempt}",
        )

    def request_id(self, position: int, attempt: int = 0) -> GenerationRequestId:
        return generation_request_id(
            self.edition.edition_id,
            position,
            attempt,
            self._request_file(position, attempt).version_id,
        )

    def request_identity(self, position: int, attempt: int = 0) -> GenerationRequestIdentity:
        request_file = self._request_file(position, attempt)
        return GenerationRequestIdentity(
            request_id=generation_request_id(
                self.edition.edition_id, position, attempt, request_file.version_id
            ),
            edition_id=self.edition.edition_id,
            story_position=position,
            attempt_index=attempt,
            request_artifact_version_id=request_file.version_id,
        )

    def start_request(
        self,
        position: int,
        attempt: int = 0,
        *,
        admission: GenerationAdmission | None = None,
    ) -> GenerationRequestState:
        self.verify_edition()
        request_file = self._request_file(position, attempt)
        return video_digest_catalog.checkpoint_generation_request(
            self.lease,
            self.request_identity(position, attempt),
            request_file=request_file,
            admission=admission or self.generation_admission(),
            recorded_at=self.recorded_at,
        ).state

    def generation_admission(
        self, *, story_limit_usd: Decimal = Decimal("7")
    ) -> GenerationAdmission:
        return GenerationAdmission(
            generation_policy_artifact_version_id=self.edition.policy_bundle_version_id,
            reserved_usd=Decimal("3.25632"),
            limits=GenerationBudgetLimits(
                story_usd=story_limit_usd,
                edition_usd=Decimal("7") * self.story_count,
                bucharest_day_usd=Decimal("150"),
                calendar_month_usd=Decimal("1000"),
            ),
        )

    def receipt_file(self, position: int, attempt: int = 0) -> ArtifactFile:
        receipt_id = f"fal-{self.request_id(position, attempt)}"
        return self._file(
            receipt_id,
            "video_digest_provider_receipt",
            title=f"Provider receipt {position}/{attempt}",
        )

    def submit(
        self, position: int, attempt: int = 0, *, usd: str = "1.2500"
    ) -> GenerationRequestState:
        return video_digest_catalog.checkpoint_generation_submission(
            self.lease,
            self.request_id(position, attempt),
            provider_receipt_id=f"fal-{self.request_id(position, attempt)}",
            receipt_file=self.receipt_file(position, attempt),
            cost=EstimatedAttemptCost(usd=Decimal(usd)),
            recorded_at=self.recorded_at,
        )

    def response_file(self, position: int, attempt: int = 0) -> ArtifactFile:
        request_id = self.request_id(position, attempt)
        return self._file(
            f"{request_id}:response",
            "video_digest_generation_response",
            title=f"Generation response {position}/{attempt}",
        )

    def respond(self, position: int, attempt: int = 0) -> GenerationRequestState:
        return video_digest_catalog.checkpoint_generation_response(
            self.lease,
            self.request_id(position, attempt),
            response_file=self.response_file(position, attempt),
            recorded_at=self.recorded_at,
        )

    def clip_file(self, position: int) -> ArtifactFile:
        return self._file(
            f"{self.story_id(position)}:accepted-clip",
            "video_digest_accepted_clip",
            title=f"Accepted clip {position}",
            media_type="video/mp4",
        )

    def validation_file(self, position: int) -> ArtifactFile:
        request_id = self.request_id(position)
        return self._file(
            f"{request_id}:validation",
            "video_digest_candidate_validation",
            title=f"Candidate validation {position}",
        )

    def accept(self, position: int, *, usd: str = "1.10") -> GenerationRequestState:
        return video_digest_catalog.checkpoint_generation_acceptance(
            self.lease,
            self.request_id(position),
            clip_file=self.clip_file(position),
            validation_file=self.validation_file(position),
            cost=MeasuredAttemptCost(usd=Decimal(usd)),
            recorded_at=self.recorded_at,
        )

    def failure_file(self, position: int, attempt: int = 0) -> ArtifactFile:
        request_id = self.request_id(position, attempt)
        return self._file(
            f"{request_id}:failure",
            "video_digest_generation_failure",
            title=f"Generation failure {position}/{attempt}",
        )

    def fail_attempt(
        self,
        position: int,
        attempt: int = 0,
        *,
        measured: str | None = None,
        unknown_reason: str | None = None,
    ) -> GenerationRequestState:
        cost: MeasuredAttemptCost | UnknownAttemptCost = (
            MeasuredAttemptCost(usd=Decimal(measured))
            if measured is not None
            else UnknownAttemptCost(reason=unknown_reason or "Provider omitted billing data")
        )
        return video_digest_catalog.checkpoint_generation_failure(
            self.lease,
            self.request_id(position, attempt),
            evidence_file=self.failure_file(position, attempt),
            cost=cost,
            recorded_at=self.recorded_at,
        )

    def generate(self, position: int, *, usd: str = "1.2500") -> None:
        self.verify_all_stories()
        self.start_request(position)
        self.submit(position, usd=usd)
        self.respond(position)

    def assembly_ready(self) -> None:
        video_digest_catalog.checkpoint_assembly_ready(self.lease, recorded_at=self.recorded_at)

    def assemble_file(self) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:assembled-video",
            "video_digest_assembled_video",
            title="Assembled video digest",
            media_type="video/mp4",
        )

    def assembly_manifest_file(self) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:assembly-manifest",
            "video_digest_assembly_manifest",
            title="Video digest assembly manifest",
        )

    def assemble(self) -> AssembledVideo:
        return video_digest_catalog.checkpoint_assembled_video(
            self.lease,
            video_file=self.assemble_file(),
            manifest_file=self.assembly_manifest_file(),
            recorded_at=self.recorded_at,
        )

    def subtitles_file(self) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:subtitles",
            "video_digest_subtitles",
            title="Edition subtitles",
            media_type="text/vtt",
        )

    def subtitle_failure_file(self) -> ArtifactFile:
        return self._file(
            f"{self.edition.edition_id}:subtitle-failure",
            "video_digest_subtitle_failure",
            title="Edition subtitle failure",
        )

    def record_available_subtitles(self) -> SubtitleOutcome:
        outcome = AvailableSubtitles(artifact_version_id=self.subtitles_file().version_id)
        return video_digest_catalog.checkpoint_subtitles(
            self.lease, outcome, artifact_file=self.subtitles_file(), recorded_at=self.recorded_at
        )

    def record_failed_subtitles(self) -> SubtitleOutcome:
        outcome = FailedSubtitles(
            evidence_artifact_version_id=self.subtitle_failure_file().version_id
        )
        return video_digest_catalog.checkpoint_subtitles(
            self.lease,
            outcome,
            artifact_file=self.subtitle_failure_file(),
            recorded_at=self.recorded_at,
        )

    def slot_stage(self) -> tuple[str] | None:
        return _query_one(
            "SELECT stage FROM video_digest_slots WHERE slot_id = %s", (self.slot.slot_id,)
        )


def test_rejected_planning_attempt_records_only_immutable_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=31)
    evidence = pipeline.planning_attempt_file()

    result = video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        0,
        "rejected",
        evidence_file=evidence,
        recorded_at=pipeline.recorded_at,
    )

    assert result is None
    assert pipeline.slot_stage() == ("claimed",)
    assert _query_one(
        "SELECT disposition, attempt_evidence_artifact_version_id, "
        "accepted_plan_artifact_version_id FROM video_digest_planning_attempts "
        "WHERE edition_id = %s AND attempt_index = 0",
        (pipeline.edition.edition_id,),
    ) == ("rejected", evidence.version_id, None)
    assert _query_one(
        "SELECT count(*) FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)


def test_planning_attempt_validates_disposition_shape_before_writing(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=32)
    plan, plan_file = pipeline.build_plan()

    with pytest.raises(ValueError, match="index"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            3,
            "rejected",
            evidence_file=pipeline.planning_attempt_file(3),
            recorded_at=pipeline.recorded_at,
        )
    with pytest.raises(ValueError, match="cannot register"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "rejected",
            evidence_file=pipeline.planning_attempt_file(accepted=False),
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one(
        "SELECT count(*) FROM video_digest_planning_attempts WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)


def test_planning_attempt_rejects_unbound_artifact_content_before_writing(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=33)
    plan, plan_file = pipeline.build_plan()
    arbitrary_evidence = pipeline._file(
        f"{pipeline.edition.edition_id}:0:planning-attempt",
        "video_digest_planning_attempt",
        title="Video digest planning attempt 0",
        content=b"arbitrary evidence",
    )

    with pytest.raises(ValueError):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=arbitrary_evidence,
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=pipeline.recorded_at,
        )

    assert pipeline.slot_stage() == ("claimed",)
    assert _query_one(
        "SELECT count(*) FROM video_digest_planning_attempts WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)


def test_planning_attempt_rejects_evidence_for_a_different_canonical_plan(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=34)
    _first_plan, _first_file = pipeline.build_plan()
    first_evidence = pipeline.planning_attempt_file()
    second_plan, second_file = pipeline.build_plan(story_count=2)

    with pytest.raises(ValueError, match="does not match its planning attempt evidence"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=first_evidence,
            accepted_plan=second_plan,
            plan_file=second_file,
            recorded_at=pipeline.recorded_at,
        )

    assert pipeline.slot_stage() == ("claimed",)


def test_planning_attempt_rejects_a_duplicated_verifier_response(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=35)
    plan, plan_file = pipeline.build_plan(story_count=2)
    evidence_file = pipeline.planning_attempt_file()
    artifact = PlanningAttemptArtifact.model_validate_json(evidence_file.content, strict=True)
    duplicated = artifact.model_copy(
        update={
            "verification_responses": (
                artifact.verification_responses[0],
                artifact.verification_responses[0],
            )
        }
    )

    with pytest.raises(ValueError, match="deterministic identity"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=canonical_planning_attempt_file(pipeline.edition.edition_id, duplicated),
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=pipeline.recorded_at,
        )

    assert pipeline.slot_stage() == ("claimed",)


def test_read_planning_attempts_returns_ordered_immutable_artifact_projection(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=36)
    rejected = pipeline.planning_attempt_file()
    video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        0,
        "rejected",
        evidence_file=rejected,
        recorded_at=pipeline.recorded_at,
    )
    plan, plan_file = pipeline.build_plan()
    accepted = pipeline.planning_attempt_file(1)
    video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        1,
        "accepted",
        evidence_file=accepted,
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=pipeline.recorded_at,
    )

    attempts = video_digest_catalog.read_planning_attempts(pipeline.edition.edition_id)

    assert tuple(item.attempt_index for item in attempts) == (0, 1)
    assert tuple(item.disposition for item in attempts) == ("rejected", "accepted")
    assert attempts[0].evidence.version_id == rejected.version_id
    assert attempts[0].accepted_plan_artifact_version_id is None
    assert attempts[1].evidence.version_id == accepted.version_id
    assert attempts[1].accepted_plan_artifact_version_id == plan.artifact_version_id


def test_record_policy_bundle_validates_and_registers_exact_artifact(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    policy = artifact_file(
        artifact_id="video-digest-policy:contract",
        artifact_kind="video_digest_policy",
        title="Video digest policy contract",
        content=b"policy contract",
        r2_key="video-digest/policies/contract.json",
        media_type="application/json",
    )

    assert (
        video_digest_catalog.record_policy_bundle(policy, recorded_at=datetime.now(UTC))
        == policy.version_id
    )
    assert (
        video_digest_catalog.record_policy_bundle(policy, recorded_at=datetime.now(UTC))
        == policy.version_id
    )
    assert _query_one(
        "SELECT kind, current_version_id FROM artifacts WHERE id = %s", (policy.artifact_id,)
    ) == ("video_digest_policy", policy.version_id)

    wrong_kind = policy.model_copy(update={"artifact_kind": "test"})
    with pytest.raises(ValueError, match="policy artifact identity"):
        video_digest_catalog.record_policy_bundle(wrong_kind, recorded_at=datetime.now(UTC))


def test_record_generation_policy_validates_and_registers_exact_artifact(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    policy = artifact_file(
        artifact_id="video-digest-generation-policy:contract",
        artifact_kind="video_digest_generation_policy",
        title="Video digest generation policy contract",
        content=b"generation policy contract",
        r2_key="video-digest/policies/generation-contract.json",
        media_type="application/json",
    )

    assert (
        video_digest_catalog.record_generation_policy(policy, recorded_at=datetime.now(UTC))
        == policy.version_id
    )
    assert (
        video_digest_catalog.record_generation_policy(policy, recorded_at=datetime.now(UTC))
        == policy.version_id
    )
    assert _query_one(
        "SELECT kind, current_version_id FROM artifacts WHERE id = %s", (policy.artifact_id,)
    ) == ("video_digest_generation_policy", policy.version_id)

    wrong_kind = policy.model_copy(update={"artifact_kind": "test"})
    with pytest.raises(ValueError, match="generation policy artifact identity"):
        video_digest_catalog.record_generation_policy(wrong_kind, recorded_at=datetime.now(UTC))


def test_accepted_planning_attempt_persists_plan_stories_and_artifacts_atomically(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=1)
    plan, plan_file = pipeline.build_plan()

    with _rejecting_update(
        "video_digest_editions", "plan", "NEW.plan_artifact_version_id IS NOT NULL"
    ):
        with pytest.raises(VideoDigestCheckpointConflictError):
            video_digest_catalog.checkpoint_planning_attempt(
                pipeline.lease,
                0,
                "accepted",
                evidence_file=pipeline.planning_attempt_file(),
                accepted_plan=plan,
                plan_file=plan_file,
                recorded_at=pipeline.recorded_at,
            )
    assert pipeline.slot_stage() == ("claimed",)
    assert _query_one(
        "SELECT count(*) FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)
    assert _query_one("SELECT count(*) FROM artifacts WHERE id = %s", (plan_file.artifact_id,)) == (
        0,
    )
    assert _query_one(
        "SELECT plan_artifact_version_id FROM video_digest_editions WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (None,)

    stored = video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        0,
        "accepted",
        evidence_file=pipeline.planning_attempt_file(),
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=pipeline.recorded_at,
    )
    assert stored == plan
    assert pipeline.slot_stage() == ("generating",)
    assert _query_one(
        "SELECT story_id, position, report_subject_id, stage "
        "FROM video_digest_stories WHERE edition_id = %s ORDER BY position",
        (pipeline.edition.edition_id,),
    ) == (pipeline.story_id(0), 0, pipeline._subject_id(0), "planned")
    assert _query_one(
        "SELECT plan_artifact_version_id FROM video_digest_editions WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (plan.artifact_version_id,)
    assert _query_one(
        "SELECT current_version_id FROM artifacts WHERE id = %s", (plan_file.artifact_id,)
    ) == (plan.artifact_version_id,)


def test_accepted_planning_attempt_replays_the_exact_stored_plan(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=2)
    plan, plan_file = pipeline.build_plan()

    stored = video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        0,
        "accepted",
        evidence_file=pipeline.planning_attempt_file(),
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=pipeline.recorded_at,
    )
    replayed = video_digest_catalog.checkpoint_planning_attempt(
        pipeline.lease,
        0,
        "accepted",
        evidence_file=pipeline.planning_attempt_file(),
        accepted_plan=plan,
        plan_file=plan_file,
        recorded_at=pipeline.recorded_at,
    )

    assert replayed == plan
    assert replayed == stored
    assert _query_one(
        "SELECT count(*) FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (1,)
    assert _query_one(
        "SELECT count(*) FROM artifact_versions WHERE id = %s", (plan_file.version_id,)
    ) == (1,)


def test_planning_attempt_rejects_a_different_plan_for_a_planned_edition(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=3)
    pipeline.checkpoint_plan()
    revised_plan, revised_file, revised_evidence = accepted_planning_files(
        pipeline.edition,
        ((pipeline._subject_id(0), "Revised story", 20_000),),
        seed=str(pipeline.seed),
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="planning attempt conflicts"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=revised_evidence,
            accepted_plan=revised_plan,
            plan_file=revised_file,
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one(
        "SELECT title, requested_duration_ms FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == ("Story 0", 15_000)


def test_planning_attempt_rejects_a_slot_that_is_already_planning(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=4)
    _arrange(
        "UPDATE video_digest_slots SET stage = 'planning', updated_at = CURRENT_TIMESTAMP "
        "WHERE slot_id = %s",
        (pipeline.slot.slot_id,),
    )
    plan, plan_file = pipeline.build_plan()

    with pytest.raises(VideoDigestCheckpointConflictError, match="planning attempt"):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=pipeline.planning_attempt_file(),
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=pipeline.recorded_at,
        )

    assert pipeline.slot_stage() == ("planning",)
    assert _query_one(
        "SELECT count(*) FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)


def test_planning_attempt_raises_lease_lost_after_the_fence_was_recovered(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=5)
    pipeline.checkpoint_plan()

    _expire_lease(pipeline)
    recovered = _recover_lease(pipeline)
    assert recovered.lease.claim_count == pipeline.lease.claim_count + 1

    plan, plan_file = pipeline.build_plan()
    with pytest.raises(VideoDigestLeaseLostError):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=pipeline.planning_attempt_file(),
            accepted_plan=plan,
            plan_file=plan_file,
            recorded_at=pipeline.recorded_at,
        )

    assert pipeline.slot_stage() == ("generating",)
    assert _query_one(
        "SELECT count(*) FROM video_digest_stories WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (1,)


def test_planning_attempt_rejects_invalid_boundary_identity_before_writing(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=6)
    other_edition = EditionIdentity(
        edition_id=edition_id(_sha256_id(60), _sha256_id(61)),
        daily_report_version_id=_sha256_id(60),
        policy_bundle_version_id=_sha256_id(61),
    )
    other_file = artifact_file(
        artifact_id=other_edition.edition_id,
        artifact_kind="video_digest_plan",
        title="Other edition plan",
        content=b"other edition plan",
        r2_key="video-digest/other/plan.json",
        media_type="application/json",
    )
    other_story = PlannedStory(
        story_id=planned_story_id(other_edition.edition_id, 0, _sha256_id(62)),
        edition_id=other_edition.edition_id,
        position=0,
        report_subject_id=_sha256_id(62),
        title="Other edition story",
        requested_duration_ms=15_000,
    )
    other_plan = DigestPlan(
        edition_id=other_edition.edition_id,
        artifact_version_id=other_file.version_id,
        stories=(other_story,),
    )
    plan, plan_file = pipeline.build_plan()

    with pytest.raises(ValueError):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=pipeline.planning_attempt_file(),
            accepted_plan=other_plan,
            plan_file=other_file,
            recorded_at=pipeline.recorded_at,
        )
    with pytest.raises(ValueError):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=pipeline.planning_attempt_file(),
            accepted_plan=plan,
            plan_file=plan_file.model_copy(update={"version_id": _sha256_id(63)}),
            recorded_at=pipeline.recorded_at,
        )
    with pytest.raises(ValueError):
        video_digest_catalog.checkpoint_planning_attempt(
            pipeline.lease,
            0,
            "accepted",
            evidence_file=pipeline.planning_attempt_file(),
            accepted_plan=plan,
            plan_file=plan_file.model_copy(update={"artifact_id": "another-edition"}),
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one("SELECT count(*) FROM video_digest_stories", ()) == (0,)
    assert _query_one(
        "SELECT count(*) FROM artifacts WHERE id IN (%s, %s)",
        (plan_file.artifact_id, other_file.artifact_id),
    ) == (0,)
    assert _query_one(
        "SELECT plan_artifact_version_id FROM video_digest_editions WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (None,)


def test_story_verification_records_evidence_and_advances_the_story(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=7)
    pipeline.checkpoint_plan()

    evidence = pipeline.verify_story(0)

    assert _query_one(
        "SELECT stage, verification_evidence_artifact_version_id "
        "FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("generating", evidence.version_id)
    assert _query_one(
        "SELECT count(*) FROM artifact_versions WHERE id = %s", (evidence.version_id,)
    ) == (1,)


def test_story_verification_replays_the_exact_evidence(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=8)
    pipeline.checkpoint_plan()
    evidence = pipeline.verify_story(0)

    pipeline.verify_story(0)

    assert _query_one(
        "SELECT stage, verification_evidence_artifact_version_id "
        "FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("generating", evidence.version_id)
    assert _query_one(
        "SELECT count(*) FROM artifact_versions WHERE id = %s", (evidence.version_id,)
    ) == (1,)


def test_story_verification_rejects_different_evidence_for_a_verified_story(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=9)
    pipeline.checkpoint_plan()
    original = pipeline.verify_story(0)
    changed = pipeline.evidence_file(0, content=b"changed verification evidence")

    with pytest.raises(VideoDigestCheckpointConflictError, match="verification request"):
        video_digest_catalog.checkpoint_story_verification(
            pipeline.lease,
            pipeline.story_id(0),
            evidence_file=changed,
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one(
        "SELECT verification_evidence_artifact_version_id FROM video_digest_stories "
        "WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == (original.version_id,)


def test_story_verification_rejects_an_unknown_story_identity(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=10)
    pipeline.checkpoint_plan()
    unknown_story = StoryId(_sha256_id(1000))
    evidence = pipeline._file(
        unknown_story,
        "video_digest_story_verification",
        title="Verification for an unknown story",
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="story identity"):
        video_digest_catalog.checkpoint_story_verification(
            pipeline.lease, unknown_story, evidence_file=evidence, recorded_at=pipeline.recorded_at
        )

    assert _query_one(
        "SELECT stage FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("planned",)


def test_edition_verification_records_and_replays_an_exact_manifest(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=33)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)

    manifest = pipeline.verify_edition()
    pipeline.verify_edition()

    assert _query_one(
        "SELECT verification_manifest_artifact_version_id FROM video_digest_editions "
        "WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (manifest.version_id,)
    assert _query_one(
        "SELECT count(*) FROM artifact_versions WHERE id = %s", (manifest.version_id,)
    ) == (1,)

    changed = pipeline.manifest_file(content=b"changed verification manifest")
    with pytest.raises(VideoDigestCheckpointConflictError, match="manifest conflicts"):
        video_digest_catalog.checkpoint_edition_verification(
            pipeline.lease, manifest_file=changed, recorded_at=pipeline.recorded_at
        )


def test_edition_verification_requires_every_mandatory_story(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=34)
    pipeline.checkpoint_plan(story_count=2)
    pipeline.verify_story(0)
    manifest = pipeline.manifest_file()

    with pytest.raises(VideoDigestCheckpointConflictError, match="every mandatory story"):
        video_digest_catalog.checkpoint_edition_verification(
            pipeline.lease, manifest_file=manifest, recorded_at=pipeline.recorded_at
        )

    assert _query_one(
        "SELECT verification_manifest_artifact_version_id FROM video_digest_editions "
        "WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (None,)
    assert _query_one("SELECT count(*) FROM artifacts WHERE id = %s", (manifest.artifact_id,)) == (
        0,
    )


def test_generation_request_requires_manifest_before_artifact_registration(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=35)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    request_file = pipeline._request_file(0)

    with pytest.raises(VideoDigestCheckpointConflictError, match="verification manifest"):
        video_digest_catalog.checkpoint_generation_request(
            pipeline.lease,
            pipeline.request_identity(0),
            request_file=request_file,
            admission=pipeline.generation_admission(),
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one(
        "SELECT count(*) FROM video_digest_generation_requests WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (0,)
    assert _query_one(
        "SELECT count(*) FROM artifacts WHERE id = %s", (request_file.artifact_id,)
    ) == (0,)


def test_generation_request_starts_pending_and_replays_idempotently(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=11)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)

    state = pipeline.start_request(0)
    replayed = pipeline.start_request(0)

    assert state.request_id == pipeline.request_id(0)
    assert state.stage is GenerationStage.PENDING
    assert state.cost == PendingAttemptCost()
    assert replayed == state
    assert _query_one(
        "SELECT count(*) FROM video_digest_generation_requests WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (1,)


def test_second_generation_attempt_requires_the_first_attempt_to_have_failed(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=12)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)

    with pytest.raises(VideoDigestCheckpointConflictError, match="failed first attempt"):
        pipeline.start_request(0, attempt=1)

    failed = pipeline.fail_attempt(0, measured="0.75")
    assert failed.stage is GenerationStage.FAILED

    retried = pipeline.start_request(0, attempt=1)
    assert retried.stage is GenerationStage.PENDING
    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests "
        "WHERE edition_id = %s AND attempt_index = 0",
        (pipeline.edition.edition_id,),
    ) == ("failed",)
    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests "
        "WHERE edition_id = %s AND attempt_index = 1",
        (pipeline.edition.edition_id,),
    ) == ("pending",)
    assert _query_one(
        "SELECT stage FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("generating",)


def test_generation_request_rejects_parallel_paid_work(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=13)
    pipeline.checkpoint_plan(story_count=2)
    pipeline.verify_story(0)
    pipeline.verify_story(1)
    pipeline.start_request(0)

    with pytest.raises(VideoDigestCheckpointConflictError, match="attempt is active"):
        pipeline.start_request(1)

    assert _query_one(
        "SELECT count(*) FROM video_digest_generation_requests WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (1,)


def test_generation_request_rejects_story_budget_exhaustion(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=37)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    admission = pipeline.generation_admission(story_limit_usd=Decimal("6"))
    pipeline.start_request(0, admission=admission)
    pipeline.fail_attempt(0, measured="0.10")

    with pytest.raises(VideoDigestCheckpointConflictError):
        pipeline.start_request(0, attempt=1, admission=admission)

    assert _query_one(
        "SELECT count(*) FROM video_digest_generation_requests WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (1,)


def test_generation_request_rejects_a_changed_scope_limit(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=38)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)
    pipeline.fail_attempt(0, measured="0.10")

    with pytest.raises(VideoDigestCheckpointConflictError):
        pipeline.start_request(
            0,
            attempt=1,
            admission=pipeline.generation_admission(story_limit_usd=Decimal("8")),
        )


def test_generation_request_admission_is_immutable(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=39)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)

    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_generation_requests SET reserved_cost_usd = 1 "
                "WHERE request_id = %s",
                (pipeline.request_id(0),),
            )


def test_generation_submission_persists_the_provider_receipt_and_estimate(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=14)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)

    state = pipeline.submit(0, usd="1.2500")

    assert state.stage is GenerationStage.SUBMITTED
    assert state.provider_receipt_id == f"fal-{pipeline.request_id(0)}"
    assert state.cost == EstimatedAttemptCost(usd=Decimal("1.2500"))
    assert _query_one(
        "SELECT stage, provider_receipt_id, cost_kind, cost_usd::TEXT "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("submitted", f"fal-{pipeline.request_id(0)}", "estimated", "1.2500")


def test_generation_submission_replays_the_stored_receipt_after_processing(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=15)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.respond(0)

    replayed = pipeline.submit(0)

    assert replayed.stage is GenerationStage.PROCESSING
    assert replayed.provider_receipt_id == f"fal-{pipeline.request_id(0)}"
    assert replayed.cost == EstimatedAttemptCost(usd=Decimal("1.2500"))
    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("processing",)


def test_generation_submission_rejects_a_changed_active_estimate(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=16)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.submit(0, usd="1.2500")

    with pytest.raises(VideoDigestCheckpointConflictError, match="receipt conflicts"):
        pipeline.submit(0, usd="2.00")

    assert _query_one(
        "SELECT cost_usd FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == (Decimal("1.2500"),)


def test_generation_response_is_durable_before_acceptance(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=17)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)
    pipeline.submit(0)

    state = pipeline.respond(0)
    response = pipeline.response_file(0)

    assert state.stage is GenerationStage.PROCESSING
    assert _query_one(
        "SELECT stage, response_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("processing", response.version_id)
    assert _query_one(
        "SELECT current_version_id FROM artifacts WHERE id = %s", (response.artifact_id,)
    ) == (response.version_id,)


def test_generation_response_rejects_a_wrong_artifact_identity_before_writing(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=18)
    pipeline.checkpoint_plan()
    pipeline.verify_story(0)
    pipeline.start_request(0)
    pipeline.submit(0)
    wrong_response = artifact_file(
        artifact_id="another-request:response",
        artifact_kind="video_digest_generation_response",
        title="Response for another request",
        content=b"wrong response",
        r2_key="video-digest/wrong/response.json",
        media_type="application/json",
    )

    with pytest.raises(ValueError, match="Generation response artifact"):
        video_digest_catalog.checkpoint_generation_response(
            pipeline.lease,
            pipeline.request_id(0),
            response_file=wrong_response,
            recorded_at=pipeline.recorded_at,
        )

    assert _query_one(
        "SELECT stage, response_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("submitted", None)
    assert _query_one(
        "SELECT count(*) FROM artifacts WHERE id = %s", (wrong_response.artifact_id,)
    ) == (0,)


def test_generation_acceptance_completes_atomically_with_the_story(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=19)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    clip = pipeline.clip_file(0)

    with _rejecting_update("video_digest_stories", "acceptance", "NEW.stage = 'accepted'"):
        with pytest.raises(VideoDigestCheckpointConflictError):
            pipeline.accept(0)
    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("processing",)
    assert _query_one(
        "SELECT stage FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("generating",)
    assert _query_one("SELECT count(*) FROM artifacts WHERE id = %s", (clip.artifact_id,)) == (0,)

    state = pipeline.accept(0, usd="1.10")
    assert state.stage is GenerationStage.ACCEPTED
    assert state.cost == MeasuredAttemptCost(usd=Decimal("1.10"))
    assert _query_one(
        "SELECT stage, accepted_clip_artifact_version_id, cost_kind, cost_usd "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("accepted", clip.version_id, "measured", Decimal("1.10"))
    assert _query_one(
        "SELECT stage, accepted_clip_artifact_version_id FROM video_digest_stories "
        "WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("accepted", clip.version_id)


def test_read_generation_deadline_returns_the_slot_utc_deadline(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=31)

    deadline = video_digest_catalog.read_generation_deadline(pipeline.slot.slot_id)

    assert deadline.tzinfo is not None
    assert deadline.utcoffset() == timedelta(0)
    assert deadline == SCHEDULED_AT + timedelta(minutes=90)


def test_read_generation_attempts_projects_accepted_media_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=32)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    state = pipeline.accept(0)

    attempts = video_digest_catalog.read_generation_attempts(pipeline.edition.edition_id)

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.request.request_id == pipeline.request_id(0)
    assert attempt.stage is state.stage
    assert attempt.accepted_clip is not None
    assert attempt.accepted_clip.version_id == pipeline.clip_file(0).version_id
    assert attempt.accepted_clip.byte_size == len(pipeline.clip_file(0).content)
    assert attempt.validation_evidence is not None
    assert attempt.validation_evidence.version_id == pipeline.validation_file(0).version_id
    assert attempt.receipt_evidence is not None
    assert attempt.receipt_evidence.version_id == pipeline.receipt_file(0).version_id
    assert attempt.response_evidence is not None
    assert attempt.response_evidence.version_id == pipeline.response_file(0).version_id
    assert attempt.cost == MeasuredAttemptCost(usd=Decimal("1.10"))


def test_generation_acceptance_replays_exactly_and_rejects_divergence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=33)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    state = pipeline.accept(0)

    replayed = pipeline.accept(0)

    assert replayed == state
    assert _query_one(
        "SELECT validation_evidence_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == (pipeline.validation_file(0).version_id,)

    with pytest.raises(VideoDigestCheckpointConflictError):
        pipeline.accept(0, usd="9.99")
    divergent = pipeline._file(
        f"{pipeline.request_id(0)}:validation",
        "video_digest_candidate_validation",
        title="Divergent validation",
        content=b"divergent validation evidence",
    )
    with pytest.raises(VideoDigestCheckpointConflictError):
        video_digest_catalog.checkpoint_generation_acceptance(
            pipeline.lease,
            pipeline.request_id(0),
            clip_file=pipeline.clip_file(0),
            validation_file=divergent,
            cost=MeasuredAttemptCost(usd=Decimal("1.10")),
            recorded_at=pipeline.recorded_at,
        )
    assert _query_one(
        "SELECT validation_evidence_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == (pipeline.validation_file(0).version_id,)


def test_first_generation_failure_records_evidence_and_permits_one_retry(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=20)
    pipeline.checkpoint_plan()
    pipeline.generate(0)

    state = pipeline.fail_attempt(0, measured="0.75")

    assert state.stage is GenerationStage.FAILED
    assert state.cost == MeasuredAttemptCost(usd=Decimal("0.75"))
    assert _query_one(
        "SELECT stage, failure_evidence_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("failed", pipeline.failure_file(0).version_id)
    assert _query_one(
        "SELECT stage FROM video_digest_stories WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("generating",)
    assert pipeline.slot_stage() == ("generating",)

    retried = pipeline.start_request(0, attempt=1)
    assert retried.stage is GenerationStage.PENDING


def test_second_generation_failure_terminates_the_story_and_the_slot(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=21)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.fail_attempt(0, measured="0.75")
    pipeline.start_request(0, attempt=1)
    pipeline.submit(0, attempt=1)
    pipeline.respond(0, attempt=1)

    state = pipeline.fail_attempt(0, attempt=1, unknown_reason="Provider omitted billing data")

    assert state.stage is GenerationStage.FAILED
    assert state.cost == UnknownAttemptCost(reason="Provider omitted billing data")
    assert _query_one(
        "SELECT stage, failure_evidence_artifact_version_id "
        "FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0, 1),),
    ) == ("failed", pipeline.failure_file(0, 1).version_id)
    assert _query_one(
        "SELECT stage, failure_evidence_artifact_version_id FROM video_digest_stories "
        "WHERE story_id = %s",
        (pipeline.story_id(0),),
    ) == ("failed", pipeline.failure_file(0, 1).version_id)
    assert _query_one(
        "SELECT stage, terminal_lease_owner_token, terminal_claim_count, lease_owner_token "
        "FROM video_digest_slots WHERE slot_id = %s",
        (pipeline.slot.slot_id,),
    ) == (
        "failed",
        pipeline.lease.owner_token,
        pipeline.lease.claim_count,
        None,
    )


def test_generation_failure_replay_does_not_bypass_a_recovered_fence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=22)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.fail_attempt(0, measured="0.75")

    _expire_lease(pipeline)
    recovered = _recover_lease(pipeline)
    assert recovered.lease.claim_count == pipeline.lease.claim_count + 1

    with pytest.raises(VideoDigestLeaseLostError):
        pipeline.fail_attempt(0, measured="0.75")

    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0),),
    ) == ("failed",)
    assert pipeline.slot_stage() == ("generating",)


def test_terminal_generation_failure_replays_only_its_final_evidence(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=23)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.fail_attempt(0, measured="0.75")
    pipeline.start_request(0, attempt=1)
    pipeline.submit(0, attempt=1)
    pipeline.respond(0, attempt=1)
    pipeline.fail_attempt(0, attempt=1, unknown_reason="Provider omitted billing data")

    replayed = pipeline.fail_attempt(0, attempt=1, unknown_reason="Provider omitted billing data")

    assert replayed.stage is GenerationStage.FAILED
    with pytest.raises(VideoDigestCheckpointConflictError, match="failure conflicts"):
        pipeline.fail_attempt(0, attempt=1, unknown_reason="A different billing problem")
    assert pipeline.slot_stage() == ("failed",)
    assert _query_one(
        "SELECT stage FROM video_digest_generation_requests WHERE request_id = %s",
        (pipeline.request_id(0, 1),),
    ) == ("failed",)


def test_assembly_ready_requires_every_story_accepted(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=24)
    pipeline.checkpoint_plan(story_count=2)
    pipeline.generate(0)
    pipeline.accept(0)

    with pytest.raises(VideoDigestCheckpointConflictError, match="incomplete mandatory"):
        pipeline.assembly_ready()
    assert pipeline.slot_stage() == ("generating",)

    pipeline.generate(1)
    pipeline.accept(1)
    pipeline.assembly_ready()
    assert pipeline.slot_stage() == ("assembling",)


def test_assembled_video_records_the_output_and_advances_the_edition(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=25)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.accept(0)
    pipeline.assembly_ready()

    assembled = pipeline.assemble()
    video = pipeline.assemble_file()

    assert assembled.edition_id == pipeline.edition.edition_id
    assert assembled.artifact_version_id == video.version_id
    assert _query_one(
        "SELECT assembled_video_artifact_version_id FROM video_digest_editions "
        "WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (video.version_id,)
    assert _query_one(
        "SELECT current_version_id FROM artifacts WHERE id = %s", (video.artifact_id,)
    ) == (video.version_id,)
    assert pipeline.slot_stage() == ("subtitling",)


def test_assembled_video_requires_accepted_stories(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=26)
    pipeline.checkpoint_plan(story_count=2)
    pipeline.generate(0)
    pipeline.accept(0)
    pipeline.verify_story(1)
    _arrange(
        "UPDATE video_digest_slots SET stage = 'assembling', updated_at = CURRENT_TIMESTAMP "
        "WHERE slot_id = %s",
        (pipeline.slot.slot_id,),
    )

    with pytest.raises(VideoDigestCheckpointConflictError, match="incomplete mandatory stories"):
        pipeline.assemble()

    assert _query_one(
        "SELECT assembled_video_artifact_version_id FROM video_digest_editions "
        "WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == (None,)


def test_assembled_video_replays_after_slot_progress(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=27)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.accept(0)
    pipeline.assembly_ready()
    assembled = pipeline.assemble()
    pipeline.record_available_subtitles()

    replayed = pipeline.assemble()

    assert replayed == assembled
    assert pipeline.slot_stage() == ("publishing",)


def test_available_subtitles_persist_and_replay_safely(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=28)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.accept(0)
    pipeline.assembly_ready()
    pipeline.assemble()

    outcome = pipeline.record_available_subtitles()
    subtitles = pipeline.subtitles_file()

    assert outcome == AvailableSubtitles(artifact_version_id=subtitles.version_id)
    assert _query_one(
        "SELECT subtitle_state, subtitle_artifact_version_id FROM video_digest_editions "
        "WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == ("available", subtitles.version_id)
    assert pipeline.slot_stage() == ("publishing",)

    replayed = pipeline.record_available_subtitles()
    assert replayed == outcome


def test_failed_subtitles_leave_the_edition_publishable(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=29)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.accept(0)
    pipeline.assembly_ready()
    pipeline.assemble()

    outcome = pipeline.record_failed_subtitles()
    failure = pipeline.subtitle_failure_file()

    assert outcome == FailedSubtitles(evidence_artifact_version_id=failure.version_id)
    assert _query_one(
        "SELECT subtitle_state, subtitle_failure_evidence_artifact_version_id "
        "FROM video_digest_editions WHERE edition_id = %s",
        (pipeline.edition.edition_id,),
    ) == ("failed", failure.version_id)
    assert pipeline.slot_stage() == ("publishing",)

    video = pipeline.assemble_file()
    intent = PublicationIntent(
        publication_id=publication_id(
            edition_id_value=pipeline.edition.edition_id,
            expected_video_key="video-digests/edition.mp4",
            video_digest=video.content_digest,
            video_byte_size=len(video.content),
            video_media_type=video.media_type,
            subtitle=None,
            source_video_version_id=video.version_id,
            source_subtitle_version_id=None,
        ),
        edition_id=pipeline.edition.edition_id,
        expected_video_key="video-digests/edition.mp4",
        video_digest=video.content_digest,
        video_byte_size=len(video.content),
        video_media_type=video.media_type,
        source_video_version_id=video.version_id,
    )
    status = video_digest_catalog.record_publication_intent(
        pipeline.lease, intent, recorded_at=pipeline.recorded_at
    )
    assert status.stage is PublicationState.PENDING

    replayed = pipeline.record_failed_subtitles()
    assert replayed == outcome


def test_generation_spend_preserves_measured_decimal_precision(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=30)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.accept(0, usd="1.10")

    spend = video_digest_catalog.read_generation_spend(pipeline.edition.edition_id)

    assert str(spend.measured_usd) == "1.10"
    assert spend.pending_requests == 0


def test_generation_spend_preserves_estimated_decimal_precision(
    postgres_news_schema: str,
) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=30)
    pipeline.checkpoint_plan()
    pipeline.generate(0, usd="2.2500")

    spend = video_digest_catalog.read_generation_spend(pipeline.edition.edition_id)

    assert str(spend.estimated_usd) == "2.2500"
    assert spend.pending_requests == 0


def test_generation_spend_counts_unknown_costs(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=30)
    pipeline.checkpoint_plan()
    pipeline.generate(0)
    pipeline.fail_attempt(0, unknown_reason="Provider omitted billing data")

    spend = video_digest_catalog.read_generation_spend(pipeline.edition.edition_id)

    assert spend.unknown_requests == 1
