from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from romanian_news import Sha256
from romanian_news.catalog.feeds import FeedSnapshotFile, read_feed_snapshot_files
from romanian_news.feeds.acquisition import feed_entry_event_from_payload
from romanian_news.feeds.models import (
    CatalogedFeedEntry,
    CatalogedFeedEntryReference,
    FeedEntry,
    FeedRegistry,
    FeedSpec,
    legacy_feed_entry_event_id,
)
from romanian_news.feeds.parsing import parse_feed
from romanian_news.storage import read_verified_r2_object


def materialize_cataloged_feed_entries(
    references: tuple[CatalogedFeedEntryReference, ...],
    registry: FeedRegistry,
) -> tuple[CatalogedFeedEntry, ...]:
    """Materialize selected entries from their exact verified feed snapshots."""
    if not references:
        return ()
    feeds = {feed.id: feed for feed in registry.feeds}
    references_by_version: dict[Sha256, list[CatalogedFeedEntryReference]] = defaultdict(list)
    for reference in references:
        if reference.feed_id not in feeds:
            raise ValueError(f"Cataloged feed entry uses an unknown feed: {reference.feed_id}")
        references_by_version[reference.feed_snapshot_version_id].append(reference)
    snapshots = read_feed_snapshot_files(tuple(sorted(references_by_version)))
    materialized: dict[Sha256, CatalogedFeedEntry] = {}
    for version_id, selected in references_by_version.items():
        materialized.update(
            _materialize_snapshot_entries(version_id, selected, snapshots[version_id], feeds)
        )
    return tuple(materialized[reference.event_id] for reference in references)


def _materialize_snapshot_entries(
    version_id: Sha256,
    selected: list[CatalogedFeedEntryReference],
    snapshot: FeedSnapshotFile,
    feeds: Mapping[str, FeedSpec],
) -> dict[Sha256, CatalogedFeedEntry]:
    if snapshot.feed_id != selected[0].feed_id or any(
        reference.feed_id != snapshot.feed_id for reference in selected
    ):
        raise ValueError(f"Feed snapshot identity mismatch: {version_id}")
    content = read_verified_r2_object(snapshot.r2_key, snapshot.content_digest)
    entries, _rejected = parse_feed(feeds[snapshot.feed_id], content)
    return {
        reference.event_id: CatalogedFeedEntry(
            **reference.model_dump(),
            entry=_matching_entry(reference, entries, snapshot.content_digest, version_id),
        )
        for reference in selected
    }


def _matching_entry(
    reference: CatalogedFeedEntryReference,
    entries: tuple[FeedEntry, ...],
    content_digest: Sha256,
    version_id: Sha256,
) -> FeedEntry:
    matches = tuple(
        entry
        for entry in entries
        if reference.event_id
        in {
            feed_entry_event_from_payload(
                entry,
                content_digest,
                reference.registry_version_id,
                reference.observed_at,
            ).event_id,
            legacy_feed_entry_event_id(entry, content_digest),
        }
    )
    if not matches:
        raise ValueError(
            f"Expected one feed entry {reference.event_id} in snapshot {version_id}, "
            f"received {len(matches)}"
        )
    return matches[0]
