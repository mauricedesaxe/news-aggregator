from typing import TypedDict

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import artifacts
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


def test_content_version_identity_is_stable_across_computations() -> None:
    first = artifact_file(**_ARGUMENTS)
    second = artifact_file(**_ARGUMENTS)

    assert first == second


def test_catalog_returns_the_shared_artifact_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts,
        "catalog_query",
        lambda *_args: [
            {
                "artifact_id": "news:daily:2026-08-31",
                "version_id": "a" * 64,
                "content_digest": "b" * 64,
                "r2_key": "news/reports/daily/2026-08-31/report.json",
            }
        ],
    )

    reference = artifacts.current_artifact_reference("news:daily:2026-08-31", "news_daily_report")

    assert type(reference) is ArtifactReference
