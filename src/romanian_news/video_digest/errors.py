class VideoDigestCatalogError(RuntimeError):
    """The video digest catalog rejected a request."""


class VideoDigestLeaseLostError(VideoDigestCatalogError):
    """The video digest slot lease no longer matches the stored fence."""


class VideoDigestCheckpointConflictError(VideoDigestCatalogError):
    """Stored video digest state conflicts with an idempotent request."""
