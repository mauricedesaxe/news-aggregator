class YouTubeDeterministicError(ValueError):
    pass


class YouTubeConfigurationError(YouTubeDeterministicError):
    pass


class YouTubeRateCardError(YouTubeDeterministicError):
    pass


class YouTubeBudgetError(YouTubeDeterministicError):
    pass


class YouTubeProviderPayloadError(YouTubeDeterministicError):
    pass


class YouTubeEvidenceError(YouTubeDeterministicError):
    pass


class YouTubeMergeError(YouTubeDeterministicError):
    pass


class YouTubeInfrastructureError(RuntimeError):
    pass


class YouTubeLeaseLostError(RuntimeError):
    pass
