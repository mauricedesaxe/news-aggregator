from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import pytest
from botocore.exceptions import ClientError
from openai.types.chat import ChatCompletion

from romanian_news import BUCHAREST, storage
from romanian_news.analysis.tracing import ProviderChatRequest
from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.video_digest import (
    read_generation_attempts,
    read_planning_attempts,
    read_slot_resume_state,
    schedule_slot,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.video_digest import generation, media, preflight, references
from romanian_news.video_digest.generation_port import ProductionH3GenerationPort
from romanian_news.video_digest.models import (
    GenerationStage,
    ScheduledSlot,
    SlotName,
    scheduled_slot_id,
)
from romanian_news.video_digest.orchestration import GenerationResume, RunDeferred
from romanian_news.video_digest.tests.test_media import _probe_payload
from romanian_news.video_digest.tests.test_preflight import (
    _plan_content,
    _report,
    _response,
    _system_prompt,
)
from romanian_news.worker.video_digest_runtime import ProductionVideoDigestRuntime
from tests import daily_report_catalog
from tests.postgres_catalog import PostgresCatalog, postgres_catalog_fixture
from tests.worker.conftest import FakeR2Client

postgres_catalog = postgres_catalog_fixture("news_runtime_adapter_contract")


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeR2Client:
    client = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    return client


@final
class _QueuedFal:
    def __init__(self, clip: bytes) -> None:
        self.clip = clip
        self.receipts: list[str] = []
        self.polls: dict[str, int] = {}

    def submit(self, arguments: dict[str, object]) -> generation.FalSubmissionReceipt:
        assert arguments["reference_video_urls"]
        assert arguments["reference_audio_urls"]
        receipt_id = f"fixture-{len(self.receipts)}"
        self.receipts.append(receipt_id)
        return generation.FalSubmissionReceipt.model_validate(
            {
                "request_id": receipt_id,
                "status_url": f"https://queue.fal.run/{receipt_id}/status",
                "response_url": f"https://queue.fal.run/{receipt_id}/response",
            },
            strict=True,
        )

    def status(self, receipt: generation.FalSubmissionReceipt) -> generation.FalQueueStatus:
        count = self.polls.get(receipt.request_id, 0)
        self.polls[receipt.request_id] = count + 1
        status = "COMPLETED" if receipt.request_id == "fixture-0" and count > 0 else "IN_QUEUE"
        return generation.FalQueueStatus(status=status, request_id=receipt.request_id)

    def result(
        self, receipt: generation.FalSubmissionReceipt
    ) -> tuple[generation.FalH3Result, dict[str, object]]:
        payload: dict[str, object] = {
            "video": {"url": f"https://v3.fal.media/{receipt.request_id}.mp4"}
        }
        return generation.FalH3Result.model_validate(payload, strict=True), payload

    def download(self, url: str) -> bytes:
        del url
        return self.clip


def test_production_runtime_reacquires_queued_generation_and_accepts_candidate(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from romanian_news.video_digest import publication

    headers: dict[str, dict[str, str]] = {}

    def put_object(
        *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str], **options: str
    ) -> None:
        del Bucket
        fake_r2.objects[Key] = Body
        fake_r2.metadata[Key] = dict(Metadata)
        headers[Key] = options

    def head_object(*, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        if Key not in fake_r2.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        content = fake_r2.objects[Key]
        return {
            "ContentLength": len(content),
            "ContentType": headers.get(Key, {}).get("ContentType", "application/octet-stream"),
            "CacheControl": headers.get(Key, {}).get("CacheControl", "private,no-store"),
            "Metadata": fake_r2.metadata[Key],
            "ETag": f'"{sha256(content)}"',
        }

    monkeypatch.setattr(fake_r2, "put_object", put_object)
    monkeypatch.setattr(fake_r2, "head_object", head_object)
    monkeypatch.setattr(
        fake_r2,
        "generate_presigned_url",
        lambda _operation, *, Params, ExpiresIn: (
            f"https://private.example/{Params['Key']}?expires={ExpiresIn}"
        ),
        raising=False,
    )
    monkeypatch.setattr(publication, "NEWS_R2_BUCKET", "private-fixture")
    monkeypatch.setattr(publication, "NEWS_PUBLIC_MEDIA_R2_BUCKET", "public-fixture")
    monkeypatch.setattr(publication, "NEWS_PUBLIC_MEDIA_BASE_URL", "https://media.example")
    monkeypatch.setattr(publication, "r2_client", lambda: fake_r2)

    scheduled_at = datetime.now(UTC)
    slot = ScheduledSlot(
        slot_id=scheduled_slot_id(SlotName.MORNING, scheduled_at),
        name=SlotName.MORNING,
        scheduled_at=scheduled_at,
        bucharest_day=scheduled_at.astimezone(BUCHAREST).date(),
    )
    report = _report().report.model_copy(update={"day": slot.bucharest_day})
    canonical = canonical_json(report.model_dump(mode="json"))
    report_file = artifact_file(
        artifact_id=f"news:daily:{report.day.isoformat()}",
        artifact_kind="news_daily_report",
        title=f"Romanian news report for {report.day.isoformat()}",
        content=canonical,
        r2_key=f"news/reports/daily/{report.day.isoformat()}/{sha256(canonical)}.json",
        media_type="application/json",
    )
    monkeypatch.setattr(daily_report_catalog, "REPORT_VERSION", report_file.version_id)
    payload = daily_report_catalog.seed_daily_report(postgres_catalog, report)
    payload_key = (
        f"news/reports/daily/{report.day.isoformat()}/{hashlib.sha256(payload).hexdigest()}.json"
    )
    fake_r2.objects[payload_key] = payload
    fake_r2.metadata[payload_key] = {"sha256": sha256(payload)}
    _ = schedule_slot(slot, recorded_at=scheduled_at)

    video_path = tmp_path / "reference.mp4"
    _ = video_path.write_bytes(b"fixture-reference-video")
    audio_path = tmp_path / "reference.wav"
    _ = audio_path.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    pack_id = references.import_h3_reference_pack(
        (video_path,), (audio_path,), approval_ref="fixture-approved-media"
    )

    def media_process(
        arguments: tuple[str, ...], *, capture_output: bool
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert arguments[0] in {"ffmpeg", "ffprobe"}
        output = json.dumps(_probe_payload(duration="15.0")) if arguments[0] == "ffprobe" else ""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr(media, "_run", media_process)
    monkeypatch.setattr(media, "clip_narration_matches", lambda _path, text: bool(text))

    calls = 0

    def completion(request: ProviderChatRequest) -> ChatCompletion:
        nonlocal calls
        calls += 1
        content = (
            _plan_content()
            if _system_prompt(request) == preflight.PLANNING_PROMPT
            else '{"status":"accepted","failures":[]}'
        )
        response = _response(f"fixture-{calls}", content, request["model"])
        payload = response.model_dump(mode="json")
        payload["usage"]["cost"] = 0.001
        return ChatCompletion.model_validate(payload)

    monkeypatch.setattr(preflight, "_openrouter_completion", completion)
    fal = _QueuedFal(b"fixture-generated-video")
    generation_port = ProductionH3GenerationPort(provider=fal, pack_id=pack_id)

    first = ProductionVideoDigestRuntime(generation_port)
    outcome, _alert = first.run(slot.slot_id, owner_token="integration-run")
    assert isinstance(outcome, RunDeferred)
    initial_state = read_slot_resume_state(slot.slot_id)
    assert isinstance(initial_state, GenerationResume)
    assert len(read_planning_attempts(initial_state.lease.edition_id)) == 1
    assert fal.receipts == ["fixture-0"]

    second = ProductionVideoDigestRuntime(generation_port)
    outcome, _alert = second.run(slot.slot_id, owner_token="integration-run")
    assert isinstance(outcome, RunDeferred)
    assert outcome.reason == "Fal H3 request IN_QUEUE"
    state = read_slot_resume_state(slot.slot_id)
    assert isinstance(state, GenerationResume)
    attempts = read_generation_attempts(state.lease.edition_id)
    assert [attempt.stage for attempt in attempts] == [
        GenerationStage.ACCEPTED,
        GenerationStage.SUBMITTED,
    ]
    assert fal.receipts == ["fixture-0", "fixture-1"]
    assert len(read_planning_attempts(state.lease.edition_id)) == 1
