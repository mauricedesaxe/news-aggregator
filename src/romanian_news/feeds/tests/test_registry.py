from romanian_news.feeds.models import FeedSpec
from romanian_news.feeds.registry import feed_registry


def _feed(feed_id: str) -> FeedSpec:
    return {feed.id: feed for feed in feed_registry().feeds}[feed_id]


def test_registry_has_only_unique_verified_feeds() -> None:
    registry = feed_registry()

    assert len({feed.id for feed in registry.feeds}) == len(registry.feeds)
    assert len({str(feed.url) for feed in registry.feeds}) == len(registry.feeds)
    assert registry.version_id == "f86e8c5066242ab8c13e2296b98f34609e5c5497d13ed7aca1d08a01b23a8d0a"


def test_excluded_article_path_prefixes_reject_matching_article_urls() -> None:
    monitorul_botosani = _feed("monitorul-botosani")

    assert monitorul_botosani.accepts_article_url("https://www.monitorulbt.ro/national/x") is False
    assert monitorul_botosani.accepts_article_url("https://www.monitorulbt.ro/politica/x") is True

    observatorul_prahovean = _feed("observatorul-prahovean")
    assert (
        observatorul_prahovean.accepts_article_url("https://www.observatorulph.ro/advertorial/x")
        is False
    )
    assert (
        observatorul_prahovean.accepts_article_url("https://www.observatorulph.ro/stiri/x") is True
    )
