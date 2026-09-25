from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
from romanian_news import storage
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file, sha256
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import (
    H3ReferenceMediaArtifact,
    read_h3_reference_pack_projection,
    record_h3_reference_pack,
)
from romanian_news.storage import ResearchObjectIntegrityError
from romanian_news.video_digest import references
from tests.worker.conftest import FakeR2Client


def test_h3_reference_pack_registry_is_exact_and_immutable(postgres_news_schema: str) -> None:
    ensure_news_catalog_schema()
    video = artifact_file(
        artifact_id="video-digest-h3-reference-video:contract",
        artifact_kind="video_digest_h3_reference_video",
        title="Contract video reference",
        content=b"contract-video",
        r2_key="news/video-digest/references/video/contract.mp4",
        media_type="video/mp4",
    )
    audio = artifact_file(
        artifact_id="video-digest-h3-reference-audio:contract",
        artifact_kind="video_digest_h3_reference_audio",
        title="Contract audio reference",
        content=b"contract-audio",
        r2_key="news/video-digest/references/audio/contract.wav",
        media_type="audio/wav",
    )
    approval_ref = "issue-11-approved-media"
    manifest = references.H3ReferenceManifest(
        approval_ref=approval_ref,
        videos=(references._asset(video),),
        audio=(references._asset(audio),),
    )
    pack_id, manifest_file = references._manifest_file(manifest)
    media = (
        H3ReferenceMediaArtifact(role="video", position=0, file=video),
        H3ReferenceMediaArtifact(role="audio", position=0, file=audio),
    )

    record_h3_reference_pack(
        pack_id, approval_ref, manifest_file, media, recorded_at=datetime.now(UTC)
    )
    record_h3_reference_pack(
        pack_id, approval_ref, manifest_file, media, recorded_at=datetime.now(UTC)
    )
    stored = read_h3_reference_pack_projection(pack_id)
    assert stored.manifest.version_id == manifest_file.version_id
    assert stored.approval_ref == approval_ref
    assert [(item.role, item.position) for item in stored.media] == [
        ("audio", 0),
        ("video", 0),
    ]

    assert news_schema.NEWS_POSTGRES_DSN is not None
    with psycopg.connect(news_schema.NEWS_POSTGRES_DSN, autocommit=True) as connection:
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "UPDATE video_digest_h3_reference_packs SET approval_ref = 'changed' "
                "WHERE pack_id = %s",
                (pack_id,),
            )
        with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
            connection.execute(
                "DELETE FROM video_digest_h3_reference_media WHERE pack_id = %s",
                (pack_id,),
            )


def _tamper_media_file(seed: int, role: str) -> ArtifactFile:
    content = f"tampered-reference-{seed}-{role}".encode()
    digest = sha256(content)
    suffix, media_type = (".mp4", "video/mp4") if role == "video" else (".wav", "audio/wav")
    return artifact_file(
        artifact_id=f"video-digest-h3-reference-{role}:{digest}",
        artifact_kind=f"video_digest_h3_reference_{role}",
        title=f"H3 reference {role} {digest}",
        content=content,
        r2_key=f"news/video-digest/references/{role}/{digest}{suffix}",
        media_type=media_type,
    )


@pytest.mark.parametrize(
    ("mutation", "seed"),
    (
        ("approval_ref_flip", 91),
        ("pack_identity", 92),
        ("registry_media_swap", 93),
        ("truncated_media_bytes", 94),
    ),
)
def test_read_h3_reference_pack_rejects_each_tampered_dimension(
    postgres_news_schema: str,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    seed: int,
) -> None:
    ensure_news_catalog_schema()
    r2 = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: r2)
    video_one = _tamper_media_file(seed, "video")
    audio = _tamper_media_file(seed + 10, "audio")
    manifest_videos = (video_one,)
    recorded_media = (
        H3ReferenceMediaArtifact(role="video", position=0, file=video_one),
        H3ReferenceMediaArtifact(role="audio", position=0, file=audio),
    )
    recorded_approval = "issue-11-approved-media"
    recorded_pack_id: str | None = None
    if mutation == "registry_media_swap":
        video_two = _tamper_media_file(seed + 20, "video")
        manifest_videos = (video_one, video_two)
        recorded_media = (
            H3ReferenceMediaArtifact(role="video", position=0, file=video_two),
            H3ReferenceMediaArtifact(role="video", position=1, file=video_one),
            H3ReferenceMediaArtifact(role="audio", position=0, file=audio),
        )
    elif mutation == "approval_ref_flip":
        recorded_approval = "a-different-approval"
    elif mutation == "pack_identity":
        recorded_pack_id = "e" * 64
    manifest = references.H3ReferenceManifest(
        approval_ref="issue-11-approved-media",
        videos=tuple(references._asset(file) for file in manifest_videos),
        audio=(references._asset(audio),),
    )
    pack_id, manifest_file = references._manifest_file(manifest)
    for file in (*manifest_videos, audio, manifest_file):
        r2.objects[file.r2_key] = file.content
    if mutation == "truncated_media_bytes":
        r2.objects[video_one.r2_key] = video_one.content[:-1]
    record_h3_reference_pack(
        recorded_pack_id or pack_id,
        recorded_approval,
        manifest_file,
        recorded_media,
        recorded_at=datetime.now(UTC),
    )

    if mutation == "truncated_media_bytes":
        with pytest.raises(ResearchObjectIntegrityError):
            references.read_h3_reference_pack(pack_id)
    elif mutation == "approval_ref_flip":
        with pytest.raises(ValueError, match="approval differs from its catalog record"):
            references.read_h3_reference_pack(pack_id)
    elif mutation == "pack_identity":
        with pytest.raises(ValueError, match="manifest differs from its catalog identity"):
            references.read_h3_reference_pack(recorded_pack_id or pack_id)
    else:
        with pytest.raises(ValueError, match="media differs from its catalog record"):
            references.read_h3_reference_pack(pack_id)
