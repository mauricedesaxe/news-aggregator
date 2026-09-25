from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

import romanian_news.catalog.schema as news_schema
from romanian_news.catalog.artifacts import artifact_file
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import (
    H3ReferenceMediaArtifact,
    read_h3_reference_pack_projection,
    record_h3_reference_pack,
)
from romanian_news.video_digest import references


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
