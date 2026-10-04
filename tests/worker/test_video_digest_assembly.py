from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import (
    checkpoint_assembly_ready,
    read_assembly_attempts,
    read_slot_resume_state,
    record_assembly_attempt,
    renew_slot,
)
from romanian_news.identity import canonical_json
from romanian_news.storage import read_verified_r2_object
from romanian_news.video_digest import media
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.orchestration import ActionAdvanced, AssembleAction
from romanian_news.worker import video_digest_assembly as assembly
from tests.contracts.postgres_video_digest_generation import _GenerationPipeline
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog
from tests.worker.conftest import FakeR2Client

requires_postgres = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)


def _seed_accepted_edition(
    postgres_catalog: PostgresCatalog, fake_r2: FakeR2Client, *, seed: int
) -> _GenerationPipeline:
    ensure_news_catalog_schema()
    pipeline = _GenerationPipeline(seed=seed)
    plan, plan_file = pipeline.build_plan()
    pipeline.checkpoint_plan()
    fake_r2.objects[plan_file.r2_key] = plan_file.content
    pipeline.generate(0)
    pipeline.accept(0)
    clip = pipeline.clip_file(0)
    fake_r2.objects[clip.r2_key] = clip.content
    return pipeline


def _lease_expires_at(postgres_catalog: PostgresCatalog, pipeline: _GenerationPipeline):
    row = postgres_catalog.execute(
        "SELECT lease_expires_at FROM video_digest_slots WHERE slot_id = %s",
        (pipeline.slot.slot_id,),
    ).fetchone()
    assert row is not None
    return row["lease_expires_at"]


def _record_failed_attempt(pipeline: _GenerationPipeline, lease: SlotLease, index: int) -> None:
    content = canonical_json(
        {
            "kind": "assembly_attempt_failure",
            "edition_id": pipeline.edition.edition_id,
            "slot_id": pipeline.slot.slot_id,
            "attempt_index": index,
            "code": "seeded_prior_failure",
        }
    )
    evidence = pipeline._file(
        f"{pipeline.edition.edition_id}:{index}:assembly-attempt",
        "video_digest_assembly_attempt",
        title=f"Seeded assembly attempt {index}",
        content=content,
    )
    record_assembly_attempt(
        lease,
        index,
        "failed",
        evidence_file=evidence,
        recorded_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("attempt_index", "expected_code"),
    ((0, "probe_failed"), (2, "media_tool_missing")),
)
@requires_postgres
def test_assembly_failure_records_durable_evidence_and_renews_the_lease_fence(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attempt_index: int,
    expected_code: str,
) -> None:
    pipeline = _seed_accepted_edition(postgres_catalog, fake_r2, seed=41 + attempt_index)
    lease = renew_slot(pipeline.lease, now=datetime.now(UTC), lease_duration=timedelta(minutes=2))
    if attempt_index == 2:
        checkpoint_assembly_ready(lease, recorded_at=datetime.now(UTC))
        _record_failed_attempt(pipeline, lease, 0)
        _record_failed_attempt(pipeline, lease, 1)
    expires_before = _lease_expires_at(postgres_catalog, pipeline)
    if expected_code == "media_tool_missing":
        monkeypatch.setenv("PATH", str(tmp_path))
    else:

        def fail_probe(arguments: tuple[str, ...], *, capture_output: bool) -> None:
            assert arguments[0] == "ffprobe"
            assert capture_output
            raise subprocess.CalledProcessError(1, arguments)

        monkeypatch.setattr(media, "_run", fail_probe)

    outcome = assembly.ProductionAssemblyPort().execute(
        AssembleAction(lease=lease, attempt_index=attempt_index)
    )

    assert outcome == ActionAdvanced()
    attempts = read_assembly_attempts(pipeline.edition.edition_id)
    assert [attempt.attempt_index for attempt in attempts] == list(range(attempt_index + 1))
    assert attempts[-1].disposition == "failed"
    evidence = attempts[-1].evidence
    payload = json.loads(read_verified_r2_object(evidence.r2_key, evidence.content_digest))
    assert payload["kind"] == "assembly_attempt_failure"
    assert payload["edition_id"] == pipeline.edition.edition_id
    assert payload["slot_id"] == pipeline.slot.slot_id
    assert payload["attempt_index"] == attempt_index
    assert payload["code"] == expected_code
    assert evidence.r2_key in fake_r2.objects
    expected_kind = (
        "video_digest_failure" if attempt_index == 2 else "video_digest_assembly_attempt"
    )
    stored = postgres_catalog.execute(
        "SELECT kind FROM artifacts WHERE id = %s", (evidence.artifact_id,)
    ).fetchone()
    assert stored is not None
    assert stored["kind"] == expected_kind
    if attempt_index == 2:
        assert _lease_expires_at(postgres_catalog, pipeline) is None
        assert read_slot_resume_state(pipeline.slot.slot_id).kind == "failed"
    else:
        assert _lease_expires_at(postgres_catalog, pipeline) > expires_before
