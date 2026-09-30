from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import ArtifactFile, artifact_file
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.catalog.video_digest import (
    H3ReferenceMediaArtifact,
    H3ReferenceMediaProjection,
    read_h3_reference_pack_projection,
    record_h3_reference_pack,
)
from romanian_news.identity import canonical_json, sha256
from romanian_news.storage import (
    publish_immutable_r2_objects,
    publish_private_reference_media_object,
    read_verified_r2_object,
)
from romanian_news.video_digest.generation import H3ReferencePack

MAX_REFERENCE_BYTES = 256 * 1024 * 1024
MediaType = Literal["video/mp4", "audio/wav", "audio/mpeg"]
_SUFFIX_MEDIA_TYPES: dict[str, MediaType] = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


class H3ReferenceAsset(NewsModel):
    reference: ArtifactReference
    media_type: MediaType
    byte_size: Annotated[int, Field(gt=0, le=MAX_REFERENCE_BYTES)]


class H3ReferenceManifest(NewsModel):
    approval_ref: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    videos: Annotated[tuple[H3ReferenceAsset, ...], Field(min_length=1)]
    audio: Annotated[tuple[H3ReferenceAsset, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_media_roles(self) -> H3ReferenceManifest:
        if any(asset.media_type != "video/mp4" for asset in self.videos):
            raise ValueError("H3 video references must be MP4")
        if any(asset.media_type not in {"audio/wav", "audio/mpeg"} for asset in self.audio):
            raise ValueError("H3 audio references must be WAV or MP3")
        return self


def _reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def _media_file(path: Path, role: Literal["video", "audio"]) -> ArtifactFile:
    media_type = _SUFFIX_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None or (role == "video") != (media_type == "video/mp4"):
        raise ValueError(f"Unsupported H3 {role} reference file: {path}")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_REFERENCE_BYTES:
        raise ValueError(f"H3 reference file is missing, empty, or too large: {path}")
    content = path.read_bytes()
    if len(content) != path.stat().st_size:
        raise ValueError(f"H3 reference file changed while being read: {path}")
    digest = sha256(content)
    return artifact_file(
        artifact_id=f"video-digest-h3-reference-{role}:{digest}",
        artifact_kind=f"video_digest_h3_reference_{role}",
        title=f"H3 reference {role} {digest}",
        content=content,
        r2_key=f"news/video-digest/references/{role}/{digest}{path.suffix.lower()}",
        media_type=media_type,
    )


def _asset(file: ArtifactFile) -> H3ReferenceAsset:
    return H3ReferenceAsset(
        reference=_reference(file),
        media_type=cast(MediaType, file.media_type),
        byte_size=len(file.content),
    )


def _manifest_file(manifest: H3ReferenceManifest) -> tuple[Sha256, ArtifactFile]:
    content = canonical_json(manifest.model_dump(mode="json"))
    pack_id = sha256(content)
    return pack_id, artifact_file(
        artifact_id=f"video-digest-h3-reference-pack:{pack_id}",
        artifact_kind="video_digest_h3_reference_pack",
        title=f"H3 reference pack {pack_id}",
        content=content,
        r2_key=f"news/video-digest/references/packs/{pack_id}.json",
        media_type="application/json",
    )


def import_h3_reference_pack(
    videos: tuple[Path, ...],
    audio: tuple[Path, ...],
    *,
    approval_ref: str,
) -> Sha256:
    if not videos or not audio:
        raise ValueError("H3 reference pack needs at least one video and one audio file")
    video_files = tuple(_media_file(path, "video") for path in videos)
    audio_files = tuple(_media_file(path, "audio") for path in audio)
    manifest = H3ReferenceManifest(
        approval_ref=approval_ref,
        videos=tuple(_asset(file) for file in video_files),
        audio=tuple(_asset(file) for file in audio_files),
    )
    pack_id, manifest_file = _manifest_file(manifest)
    media = tuple(
        H3ReferenceMediaArtifact(role="video", position=position, file=file)
        for position, file in enumerate(video_files)
    ) + tuple(
        H3ReferenceMediaArtifact(role="audio", position=position, file=file)
        for position, file in enumerate(audio_files)
    )
    ensure_news_catalog_schema()
    for item in media:
        publish_private_reference_media_object(
            item.file.r2_key,
            item.file.content,
            content_type=cast(MediaType, item.file.media_type),
            retention="permanent",
            source_lineage=item.file.version_id,
        )
    publish_immutable_r2_objects(((manifest_file.r2_key, manifest_file.content),))
    record_h3_reference_pack(
        pack_id, manifest.approval_ref, manifest_file, media, recorded_at=datetime.now(UTC)
    )
    return pack_id


def read_h3_reference_pack(pack_id: Sha256) -> H3ReferencePack:
    stored = read_h3_reference_pack_projection(pack_id)
    manifest = H3ReferenceManifest.model_validate_json(
        read_verified_r2_object(stored.manifest.r2_key, stored.manifest.content_digest),
        strict=True,
    )
    expected_id, file = _manifest_file(manifest)
    if expected_id != pack_id or _reference(file) != stored.manifest:
        raise ValueError("H3 reference pack manifest differs from its catalog identity")
    if manifest.approval_ref != stored.approval_ref:
        raise ValueError("H3 reference pack approval differs from its catalog record")
    expected = tuple(
        (role, position, asset)
        for role, assets in (("audio", manifest.audio), ("video", manifest.videos))
        for position, asset in enumerate(assets)
    )
    if len(expected) != len(stored.media):
        raise ValueError("H3 reference pack media count differs from its manifest")
    for (role, position, asset), item in zip(expected, stored.media, strict=True):
        _verify_media_row(role, position, asset, item)
        content = read_verified_r2_object(asset.reference.r2_key, asset.reference.content_digest)
        if len(content) != asset.byte_size:
            raise ValueError("H3 reference media size differs from its manifest")
        digest = sha256(content)
        suffix = next(
            key for key, value in _SUFFIX_MEDIA_TYPES.items() if value == asset.media_type
        )
        canonical = artifact_file(
            artifact_id=f"video-digest-h3-reference-{role}:{digest}",
            artifact_kind=f"video_digest_h3_reference_{role}",
            title=f"H3 reference {role} {digest}",
            content=content,
            r2_key=f"news/video-digest/references/{role}/{digest}{suffix}",
            media_type=asset.media_type,
        )
        if _reference(canonical) != asset.reference:
            raise ValueError("H3 reference media identity differs from its bytes")
    return H3ReferencePack(
        videos=tuple(asset.reference for asset in manifest.videos),
        audio=tuple(asset.reference for asset in manifest.audio),
    )


def _verify_media_row(
    role: str,
    position: int,
    asset: H3ReferenceAsset,
    row: H3ReferenceMediaProjection,
) -> None:
    if (role, position, asset.reference, asset.media_type, asset.byte_size) != (
        row.role,
        row.position,
        row.reference,
        row.media_type,
        row.byte_size,
    ):
        raise ValueError("H3 reference media differs from its catalog record")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import or verify an approved H3 reference pack")
    commands = parser.add_subparsers(dest="command", required=True)
    importing = commands.add_parser("import")
    importing.add_argument("--video", type=Path, action="append", required=True)
    importing.add_argument("--audio", type=Path, action="append", required=True)
    importing.add_argument("--approval-ref", required=True)
    verifying = commands.add_parser("verify")
    verifying.add_argument("pack_id")
    args = parser.parse_args(argv)
    if args.command == "import":
        print(
            import_h3_reference_pack(
                tuple(args.video), tuple(args.audio), approval_ref=args.approval_ref
            )
        )
    else:
        pack_id = TypeAdapter(Sha256).validate_python(args.pack_id)
        read_h3_reference_pack(pack_id)
        print(pack_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
