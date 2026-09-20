from romanian_news import NewsModel, Sha256


class ArtifactReference(NewsModel):
    artifact_id: str
    version_id: Sha256
    content_digest: Sha256
    r2_key: str
