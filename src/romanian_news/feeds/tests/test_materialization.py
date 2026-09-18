import hashlib
from datetime import datetime

import pytest
from pydantic import HttpUrl

from romanian_news.catalog.feeds import FeedSnapshotFile
from romanian_news.feeds.acquisition import feed_entry_event_from_payload
from romanian_news.feeds.materialization import materialize_cataloged_feed_entries
from romanian_news.feeds.models import CatalogedFeedEntryReference, legacy_feed_entry_event_id
from romanian_news.feeds.parsing import parse_feed
from romanian_news.feeds.registry import feed_registry


@pytest.mark.parametrize("legacy_identity", [False, True])
def test_materialization_reads_only_selected_snapshots_and_revalidates_event_id(
    monkeypatch,
    legacy_identity,
) -> None:
    registry = feed_registry()
    feed = next(value for value in registry.feeds if value.id == "hotnews")
    content = _feed_xml("source-1", repeats=4)
    digest = hashlib.sha256(content).hexdigest()
    entry = parse_feed(feed, content)[0][0]
    event = feed_entry_event_from_payload(
        entry,
        digest,
        registry.version_id,
        datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
    )
    reference = CatalogedFeedEntryReference(
        event_id=(legacy_feed_entry_event_id(entry, digest) if legacy_identity else event.event_id),
        dlt_load_id="load-1",
        registry_version_id=event.registry_version_id,
        feed_snapshot_version_id="a" * 64,
        observed_at=event.observed_at,
        feed_id=entry.feed_id,
        source_id=entry.source_id,
        url=entry.url,
        published_at=entry.published_at,
        source_updated_at=entry.source_updated_at,
    )
    queries = []
    reads = []

    def read_snapshots(version_ids):
        queries.append(version_ids)
        return {
            reference.feed_snapshot_version_id: FeedSnapshotFile(
                version_id=reference.feed_snapshot_version_id,
                content_digest=digest,
                r2_key="news/feeds/hotnews/exact.xml",
                feed_id="hotnews",
            )
        }

    def read(key, expected_digest):
        reads.append((key, expected_digest))
        return content

    monkeypatch.setattr(
        "romanian_news.feeds.materialization.read_feed_snapshot_files", read_snapshots
    )
    monkeypatch.setattr("romanian_news.feeds.materialization.read_verified_r2_object", read)

    materialized = materialize_cataloged_feed_entries((reference,), registry)

    assert queries == [(reference.feed_snapshot_version_id,)]
    assert reads == [("news/feeds/hotnews/exact.xml", digest)]
    assert materialized[0].entry == entry
    assert materialized[0].event_id == reference.event_id


def test_materialization_rejects_an_event_absent_from_recorded_snapshot(monkeypatch) -> None:
    registry = feed_registry()
    feed = next(value for value in registry.feeds if value.id == "hotnews")
    content = _feed_xml("different-source")
    digest = hashlib.sha256(content).hexdigest()
    entry = parse_feed(feed, content)[0][0]
    reference = CatalogedFeedEntryReference(
        event_id="f" * 64,
        dlt_load_id="load-1",
        registry_version_id=registry.version_id,
        feed_snapshot_version_id="a" * 64,
        observed_at=datetime.fromisoformat("2026-08-31T13:00:00+03:00"),
        feed_id=entry.feed_id,
        source_id="missing-source",
        url=HttpUrl("https://hotnews.ro/missing-source"),
        published_at=entry.published_at,
        source_updated_at=entry.source_updated_at,
    )
    monkeypatch.setattr(
        "romanian_news.feeds.materialization.read_feed_snapshot_files",
        lambda _version_ids: {
            reference.feed_snapshot_version_id: FeedSnapshotFile(
                version_id=reference.feed_snapshot_version_id,
                content_digest=digest,
                r2_key="news/feeds/hotnews/exact.xml",
                feed_id="hotnews",
            )
        },
    )
    monkeypatch.setattr(
        "romanian_news.feeds.materialization.read_verified_r2_object",
        lambda *_args: content,
    )

    with pytest.raises(ValueError, match="Expected one feed entry"):
        materialize_cataloged_feed_entries((reference,), registry)


def _feed_xml(source_id: str, *, repeats: int = 1) -> bytes:
    item = f"""
      <item>
        <guid>{source_id}</guid>
        <link>https://hotnews.ro/{source_id}</link>
        <title>Selected title</title>
        <description>Selected summary text</description>
        <pubDate>Mon, 31 Aug 2026 09:00:00 GMT</pubDate>
      </item>
    """
    return f"""
    <rss version="2.0"><channel><title>HotNews</title>
      {item * repeats}
    </channel></rss>
    """.encode()
