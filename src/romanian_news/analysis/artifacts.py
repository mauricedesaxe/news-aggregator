from romanian_news import NewsModel, Sha256


class ArtifactReference(NewsModel):
    artifact_id: str
    version_id: Sha256
    content_digest: Sha256
    r2_key: str


def existing_current_artifact_ids(artifact_ids: tuple[str, ...]) -> set[str]:
    """Return artifact IDs that have a current version."""
    from romanian_news.catalog.artifacts import existing_current_artifact_ids as read_existing

    return read_existing(artifact_ids)
