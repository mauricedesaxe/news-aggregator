from __future__ import annotations

from pathlib import Path

import pytest

from romanian_news.catalog.artifacts import ArtifactFile
from romanian_news.catalog.video_digest import (
    H3ReferenceMediaArtifact,
    H3ReferenceMediaProjection,
    H3ReferencePackProjection,
)
from romanian_news.video_digest import references


def test_import_and_read_exact_approved_h3_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "host.mp4"
    audio = tmp_path / "voice.wav"
    video.write_bytes(b"\0\0\0\x18ftypisomvideo")
    audio.write_bytes(b"RIFF\x08\0\0\0WAVEaudio")
    objects: dict[str, bytes] = {}
    records: dict[str, H3ReferencePackProjection] = {}
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(references, "ensure_news_catalog_schema", lambda: None)

    def publish_media(
        key: str, content: bytes, *, content_type: str, retention: str, source_lineage: str
    ) -> None:
        assert retention == "permanent"
        assert source_lineage
        calls.append((key, content_type, source_lineage))
        assert key not in objects or objects[key] == content
        objects[key] = content

    def publish_manifest(items: tuple[tuple[str, bytes], ...]) -> None:
        objects.update(items)

    def record(
        pack_id: str,
        approval_ref: str,
        manifest: ArtifactFile,
        media: tuple[H3ReferenceMediaArtifact, ...],
        **_: object,
    ) -> None:
        manifest_file = manifest
        media_items = media
        projection = H3ReferencePackProjection(
            pack_id=pack_id,
            approval_ref=approval_ref,
            manifest=references._reference(manifest_file),
            media=tuple(
                H3ReferenceMediaProjection(
                    role=item.role,
                    position=item.position,
                    reference=references._reference(item.file),
                    media_type=item.file.media_type,
                    byte_size=len(item.file.content),
                )
                for item in sorted(media_items, key=lambda item: (item.role, item.position))
            ),
        )
        assert pack_id not in records or records[pack_id] == projection
        records[pack_id] = projection

    monkeypatch.setattr(references, "publish_private_reference_media_object", publish_media)
    monkeypatch.setattr(references, "publish_immutable_r2_objects", publish_manifest)
    monkeypatch.setattr(references, "record_h3_reference_pack", record)
    monkeypatch.setattr(references, "read_h3_reference_pack_projection", records.__getitem__)
    monkeypatch.setattr(
        references,
        "read_verified_r2_object",
        lambda key, digest: _read(objects, key, digest),
    )

    pack_id = references.import_h3_reference_pack(
        (video,), (audio,), approval_ref="issue-11-approved-host-assets"
    )
    assert (
        references.import_h3_reference_pack(
            (video,), (audio,), approval_ref="issue-11-approved-host-assets"
        )
        == pack_id
    )
    pack = references.read_h3_reference_pack(pack_id)

    assert len(pack.videos) == len(pack.audio) == 1
    assert {content_type for _, content_type, _ in calls} == {"video/mp4", "audio/wav"}
    assert pack.videos[0].r2_key.startswith("news/video-digest/references/video/")
    assert pack.audio[0].r2_key.startswith("news/video-digest/references/audio/")

    records[pack_id] = records[pack_id].model_copy(update={"media": records[pack_id].media[:1]})
    with pytest.raises(ValueError, match="media count"):
        references.read_h3_reference_pack(pack_id)


def _read(objects: dict[str, bytes], key: str, digest: str) -> bytes:
    content = objects[key]
    assert references.sha256(content) == digest
    return content


def test_reference_import_rejects_missing_roles_and_unsupported_files(tmp_path: Path) -> None:
    video = tmp_path / "host.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="at least one video and one audio"):
        references.import_h3_reference_pack((video,), (), approval_ref="approval")
    with pytest.raises(ValueError, match="Unsupported H3 audio"):
        references.import_h3_reference_pack((video,), (video,), approval_ref="approval")
