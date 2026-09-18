from typing import TypedDict

from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
)


class _ArtifactFileArguments(TypedDict):
    artifact_id: str
    artifact_kind: str
    title: str
    content: bytes
    r2_key: str
    media_type: str


_ARGUMENTS: _ArtifactFileArguments = {
    "artifact_id": "news:daily:2026-08-31",
    "artifact_kind": "news_daily_report",
    "title": "Daily report",
    "content": b"same bytes",
    "r2_key": "news/reports/daily/2026-08-31/digest.json",
    "media_type": "application/json",
}


def test_generic_artifact_file_has_no_feed_metadata() -> None:
    assert set(ArtifactFile.model_fields) == {
        "artifact_id",
        "artifact_kind",
        "title",
        "version_id",
        "content_digest",
        "r2_key",
        "media_type",
        "content",
    }


def test_content_version_identity_is_stable_across_computations() -> None:
    first = artifact_file(**_ARGUMENTS)
    second = artifact_file(**_ARGUMENTS)

    assert first.version_id == second.version_id
    assert first.content_digest == second.content_digest
    assert first.r2_key == second.r2_key == _ARGUMENTS["r2_key"]
