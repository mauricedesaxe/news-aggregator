from datetime import UTC, datetime

from romanian_news.catalog.feeds import read_current_feed_snapshot_files
from romanian_news.feeds.models import (
    CurrentFeedState,
    FeedAcquisitionResult,
    FeedCapture,
    FeedRegistry,
    FeedValidator,
    feed_schedule_slot,
)
from romanian_news.feeds.parsing import parse_feed
from romanian_news.storage import read_verified_r2_object


def read_current_feeds(registry: FeedRegistry, observed_at: datetime) -> CurrentFeedState:
    """Read every current verified feed snapshot for pending article work."""
    snapshots = read_current_feed_snapshot_files()
    feeds = {feed.id: feed for feed in registry.feeds}
    captures = []
    versions = {}
    for snapshot in snapshots:
        feed_id = snapshot.feed_id
        if feed_id not in feeds:
            continue
        content = read_verified_r2_object(snapshot.r2_key, snapshot.content_digest)
        entries, rejected = parse_feed(feeds[feed_id], content)
        captures.append(
            FeedCapture(
                feed_id=feed_id,
                observed_at=observed_at.astimezone(UTC),
                scheduled_slot=feed_schedule_slot(observed_at),
                status="ok",
                http_status=200,
                latency_ms=0,
                content=content,
                content_digest=snapshot.content_digest,
                entries=entries,
                rejected_entries=rejected,
                validator=FeedValidator(),
                error=None,
            )
        )
        versions[feed_id] = snapshot.version_id
    missing = tuple(sorted(set(feeds) - set(versions)))
    return CurrentFeedState(
        acquisition=FeedAcquisitionResult(load_ids=(), captures=tuple(captures)),
        snapshot_versions=versions,
        missing_feed_ids=missing,
    )
