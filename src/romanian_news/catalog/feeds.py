from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated

from pydantic import Field, HttpUrl

from romanian_news import FeedId, NewsModel, Sha256
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
    canonical_json,
    sha256,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.feeds.models import (
    CatalogedFeedEntryReference,
    FeedAcquisitionResult,
    FeedCapture,
    FeedEntryEventOccurrence,
    FeedRegistry,
    feed_entry_event,
    legacy_feed_entry_event_id,
)
from romanian_news.storage import publish_immutable_r2_objects

_CATALOG_EVENT_ROWS_PER_STATEMENT = 10


class NewsAcquisitionPublication(NewsModel):
    run_id: Sha256
    uploaded_objects: Annotated[int, Field(ge=0)]
    reused_objects: Annotated[int, Field(ge=0)]
    feed_observations: Annotated[int, Field(ge=0)]
    feed_snapshots: Annotated[int, Field(ge=0)]
    feed_entry_events: Annotated[int, Field(ge=0)]
    feed_snapshot_versions: dict[str, Sha256]


class FeedEntryCatalogResult(NewsModel):
    cataloged_events: Annotated[int, Field(ge=0)]
    existing_events: Annotated[int, Field(ge=0)]


class _CatalogedFeedEntryOccurrence(NewsModel):
    source_event_id: Sha256
    reference: CatalogedFeedEntryReference


class FeedArtifactFile(ArtifactFile):
    feed_id: str | None = None
    feed_snapshot_version_id: Sha256 | None = None
    capture: FeedCapture | None = None


class FeedSnapshotFile(NewsModel):
    version_id: Sha256
    content_digest: Sha256
    feed_id: FeedId
    r2_key: str


class FeedCatalogLoadState(NewsModel):
    projected_load_ids: frozenset[str]
    registered_at_by_load_id: dict[str, datetime]


class _FeedPublicationItem(NewsModel):
    observation_file: FeedArtifactFile
    entry_occurrences: tuple[FeedEntryEventOccurrence, ...]


def read_durable_feed_ids() -> frozenset[FeedId]:
    """Read feed IDs that have a current durable snapshot."""
    rows = catalog_query(
        "SELECT substr(id, length('news:feed:') + 1) AS feed_id "
        "FROM artifacts WHERE kind = 'news_feed' AND current_version_id IS NOT NULL"
    )
    return frozenset(str(row["feed_id"]) for row in rows)


def read_current_feed_snapshot_files() -> tuple[FeedSnapshotFile, ...]:
    """Read the current file for each durable feed snapshot."""
    rows = catalog_query(
        """
        SELECT substr(artifact.id, length('news:feed:') + 1) AS feed_id,
               artifact.current_version_id AS version_id,
               file.content_digest, file.r2_key
        FROM artifacts artifact
        JOIN artifact_files file ON file.artifact_version_id = artifact.current_version_id
        WHERE artifact.kind = 'news_feed'
        ORDER BY artifact.id
        """
    )
    return tuple(FeedSnapshotFile.model_validate(row, strict=True) for row in rows)


def read_feed_snapshot_files(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, FeedSnapshotFile]:
    """Read one immutable feed snapshot file for each requested version."""
    if not version_ids:
        return {}
    placeholders = ", ".join("%s" for _ in version_ids)
    rows = catalog_query(
        f"""
        SELECT version.id AS version_id, version.content_digest, file.r2_key,
               substr(artifact.id, length('news:feed:') + 1) AS feed_id
        FROM artifact_versions version
        JOIN artifacts artifact ON artifact.id = version.artifact_id
        JOIN artifact_files file ON file.artifact_version_id = version.id
        WHERE version.id IN ({placeholders}) AND artifact.kind = 'news_feed'
        ORDER BY version.id, file.r2_key
        """,
        list(version_ids),
    )
    snapshots: dict[Sha256, FeedSnapshotFile] = {}
    for row in rows:
        snapshot = FeedSnapshotFile.model_validate(row, strict=True)
        if snapshot.version_id in snapshots:
            raise ValueError(f"Feed snapshot has more than one file: {snapshot.version_id}")
        snapshots[snapshot.version_id] = snapshot
    if set(snapshots) != set(version_ids):
        raise ValueError("Selected feed entries reference unknown snapshots")
    return snapshots


def read_feed_catalog_load_state() -> FeedCatalogLoadState:
    """Read registered dlt loads and immutable projection markers."""
    projected = frozenset(
        str(row["load_id"])
        for row in catalog_query("SELECT load_id FROM news_feed_entry_projection_loads")
    )
    registered = {
        str(row["load_id"]): datetime.fromisoformat(str(row["registered_at"]))
        for row in catalog_query("SELECT load_id, registered_at FROM news_dlt_loads")
    }
    return FeedCatalogLoadState(
        projected_load_ids=projected,
        registered_at_by_load_id=registered,
    )


def register_recovered_dlt_load(
    load_id: str,
    snapshots: tuple[ArtifactFile, ...],
    manifest: ArtifactFile,
    registered_at: datetime,
) -> None:
    """Register recovered immutable feed artifacts and their dlt load."""
    timestamp = registered_at.astimezone(UTC).isoformat()
    statements: list[tuple[str, list[object]]] = []
    for snapshot in snapshots:
        statements.extend(artifact_statements(snapshot, timestamp, produced_by_run_id=None))
    statements.extend(artifact_statements(manifest, timestamp, produced_by_run_id=None))
    statements.append(
        (
            "INSERT INTO news_dlt_loads VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            [load_id, manifest.version_id, timestamp],
        )
    )
    catalog_batch(statements)


def mark_feed_entry_projection_load(load_id: str, projected_at: datetime) -> None:
    """Mark one dlt load as projected into the feed-entry catalog."""
    catalog_batch(
        [
            (
                "INSERT INTO news_feed_entry_projection_loads VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [load_id, projected_at.astimezone(UTC).isoformat()],
            )
        ]
    )


def read_successful_feed_observation_count(scheduled_slot: datetime) -> int:
    """Count feeds with a non-failed observation for one scheduled slot."""
    rows = catalog_query(
        "SELECT count(DISTINCT feed_id) AS feed_count FROM news_feed_observations "
        "WHERE scheduled_slot = %s AND status != 'failed'",
        [scheduled_slot.isoformat()],
    )
    return int(rows[0]["feed_count"])


def read_successful_feed_counts(days: tuple[date, ...]) -> dict[date, int]:
    """Read successful feed observation counts for selected calendar days."""
    if not days:
        return {}
    values = [day.isoformat() for day in days]
    placeholders = ", ".join("%s" for _ in values)
    rows = catalog_query(
        f"""
        SELECT (scheduled_slot AT TIME ZONE 'Europe/Bucharest')::date AS day,
               count(DISTINCT feed_id) AS successful_feeds
        FROM news_feed_observations
        WHERE status != 'failed'
          AND (scheduled_slot AT TIME ZONE 'Europe/Bucharest')::date IN ({placeholders})
        GROUP BY day
        """,
        values,
    )
    return {date.fromisoformat(str(row["day"])): int(row["successful_feeds"]) for row in rows}


def publish_feed_acquisition(
    result: FeedAcquisitionResult,
    registry: FeedRegistry,
    implementation_ref: str,
) -> NewsAcquisitionPublication:
    """Publish feed bytes first, then expose the completed acquisition in the catalog."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    if len(result.load_ids) != 1:
        raise ValueError(
            f"Expected one dlt load for one feed publication, received {len(result.load_ids)}"
        )
    _validate_successful_captures(result.captures)
    published_at = max(capture.observed_at for capture in result.captures).astimezone(UTC)
    registry_file = _registry_file(registry)
    snapshots = {
        capture.feed_id: _snapshot_file(capture)
        for capture in result.captures
        if capture.content is not None
    }
    current_snapshots = _current_feed_snapshots(
        tuple(capture.feed_id for capture in result.captures if capture.status == "not_modified")
    )
    observations = tuple(
        _observation_file(
            capture,
            snapshots.get(capture.feed_id) or current_snapshots.get(capture.feed_id),
        )
        for capture in result.captures
    )
    load_file = _load_file(result, registry, observations)
    items = tuple(
        _feed_publication_item(
            capture,
            observation,
            result.load_ids[0],
            registry.version_id,
        )
        for capture, observation in zip(result.captures, observations, strict=True)
    )
    entry_occurrences = tuple(occurrence for item in items for occurrence in item.entry_occurrences)
    cataloged_entries = _cataloged_feed_entry_occurrences(
        entry_occurrences,
        _publication_snapshot_versions(items),
    )
    aliases = _feed_entry_version_aliases(entry_occurrences)
    files = (registry_file, *snapshots.values(), *observations, load_file)
    publication = publish_immutable_r2_objects((value.r2_key, value.content) for value in files)
    run_id = _run_id(result, registry, implementation_ref, observations)
    catalog_batch(
        _catalog_statements(
            run_id,
            implementation_ref,
            published_at,
            registry,
            registry_file,
            tuple(snapshots.values()),
            observations,
            load_file,
            result.load_ids,
            cataloged_entries,
            aliases,
        )
    )
    return NewsAcquisitionPublication(
        run_id=run_id,
        uploaded_objects=publication.uploaded_objects,
        reused_objects=publication.reused_objects,
        feed_observations=len(items),
        feed_snapshots=len(snapshots),
        feed_entry_events=len(cataloged_entries),
        feed_snapshot_versions={
            item.observation_file.feed_id: item.observation_file.feed_snapshot_version_id
            for item in items
            if item.observation_file.feed_snapshot_version_id is not None
            and item.observation_file.feed_id is not None
        },
    )


def _validate_successful_captures(captures: tuple[FeedCapture, ...]) -> None:
    for capture in captures:
        if capture.status != "ok":
            continue
        if capture.content is None or capture.content_digest is None:
            raise ValueError(f"Successful feed capture has no new snapshot: {capture.feed_id}")
        if any(entry.feed_id != capture.feed_id for entry in capture.entries):
            raise ValueError(f"Feed capture {capture.feed_id} contains entries for another feed")


def _registry_file(registry: FeedRegistry) -> FeedArtifactFile:
    content = canonical_json(registry.model_dump(mode="json"))
    return _feed_artifact_file(
        artifact_id="news:feed-registry",
        artifact_kind="news_registry",
        title="Romanian news feed registry",
        content=content,
        r2_key=f"news/registry/{registry.version_id}.json",
        media_type="application/json",
    )


def _snapshot_file(capture: FeedCapture) -> FeedArtifactFile:
    if capture.content is None or capture.content_digest is None:
        raise ValueError("A feed snapshot requires response bytes")
    return _feed_artifact_file(
        artifact_id=f"news:feed:{capture.feed_id}",
        artifact_kind="news_feed",
        title=f"RSS feed: {capture.feed_id}",
        content=capture.content,
        r2_key=f"news/feeds/{capture.feed_id}/{capture.content_digest}.xml",
        media_type="application/xml",
        feed_id=capture.feed_id,
    )


def _observation_file(
    capture: FeedCapture,
    snapshot: FeedArtifactFile | Sha256 | None,
) -> FeedArtifactFile:
    snapshot_version_id = (
        snapshot.version_id if isinstance(snapshot, FeedArtifactFile) else snapshot
    )
    if capture.status in ("ok", "not_modified") and snapshot_version_id is None:
        raise ValueError(f"Successful feed observation has no snapshot: {capture.feed_id}")
    payload = capture.model_dump(mode="json", exclude={"content", "entries"})
    payload["entry_ids"] = [entry.source_id for entry in capture.entries]
    payload["feed_snapshot_version_id"] = snapshot_version_id
    content = canonical_json(payload)
    slot = capture.scheduled_slot.isoformat().replace(":", "-")
    return _feed_artifact_file(
        artifact_id=f"news:feed-observation:{capture.feed_id}:{capture.scheduled_slot.isoformat()}",
        artifact_kind="news_feed_observation",
        title=f"Feed observation: {capture.feed_id} at {capture.scheduled_slot.isoformat()}",
        content=content,
        r2_key=f"news/feed-observations/{capture.feed_id}/{slot}/{sha256(content)}.json",
        media_type="application/json",
        feed_id=capture.feed_id,
        feed_snapshot_version_id=snapshot_version_id,
        capture=capture,
    )


def _load_file(
    result: FeedAcquisitionResult,
    registry: FeedRegistry,
    observations: tuple[FeedArtifactFile, ...],
) -> FeedArtifactFile:
    content = canonical_json(
        {
            "load_ids": result.load_ids,
            "observation_version_ids": [value.version_id for value in observations],
            "registry_version_id": registry.version_id,
        }
    )
    digest = sha256(content)
    return _feed_artifact_file(
        artifact_id=f"news:dlt-load:{digest}",
        artifact_kind="news_dlt_load",
        title=f"Romanian news dlt load {', '.join(result.load_ids)}",
        content=content,
        r2_key=f"news/dlt-loads/{digest}.json",
        media_type="application/json",
    )


def _feed_publication_item(
    capture: FeedCapture,
    observation_file: FeedArtifactFile,
    load_id: str,
    registry_version_id: Sha256,
) -> _FeedPublicationItem:
    entry_occurrences = ()
    if capture.status == "ok":
        entry_occurrences = tuple(
            FeedEntryEventOccurrence(
                source_event_id=event.event_id,
                dlt_load_id=load_id,
                event=event,
            )
            for event in (
                feed_entry_event(entry, capture, registry_version_id) for entry in capture.entries
            )
        )
    return _FeedPublicationItem(
        observation_file=observation_file,
        entry_occurrences=entry_occurrences,
    )


def _publication_snapshot_versions(
    items: tuple[_FeedPublicationItem, ...],
) -> dict[tuple[str, Sha256], Sha256]:
    return {
        (
            occurrence.event.entry.feed_id,
            occurrence.event.feed_content_digest,
        ): item.observation_file.feed_snapshot_version_id
        for item in items
        if item.observation_file.feed_snapshot_version_id is not None
        for occurrence in item.entry_occurrences
    }


def _current_feed_snapshots(feed_ids: tuple[str, ...]) -> dict[str, Sha256]:
    if not feed_ids:
        return {}
    placeholders = ", ".join("%s" for _ in feed_ids)
    rows = catalog_query(
        f"""
        SELECT substr(id, length('news:feed:') + 1) AS feed_id, current_version_id
        FROM artifacts
        WHERE id IN ({placeholders}) AND kind = 'news_feed'
        """,
        [f"news:feed:{feed_id}" for feed_id in feed_ids],
    )
    return {str(row["feed_id"]): row["current_version_id"] for row in rows}


def _run_id(
    result: FeedAcquisitionResult,
    registry: FeedRegistry,
    implementation_ref: str,
    observations: tuple[FeedArtifactFile, ...],
) -> Sha256:
    return sha256(
        canonical_json(
            {
                "implementation_ref": implementation_ref,
                "load_ids": result.load_ids,
                "observation_version_ids": [value.version_id for value in observations],
                "operation_key": "news.acquire",
                "registry_version_id": registry.version_id,
            }
        )
    )


def _catalog_statements(
    run_id: Sha256,
    implementation_ref: str,
    published_at: datetime,
    registry: FeedRegistry,
    registry_file: FeedArtifactFile,
    snapshots: tuple[FeedArtifactFile, ...],
    observations: tuple[FeedArtifactFile, ...],
    load_file: FeedArtifactFile,
    load_ids: tuple[str, ...],
    feed_entries: tuple[_CatalogedFeedEntryOccurrence, ...],
    aliases: dict[Sha256, Sha256],
) -> list[tuple[str, list[object]]]:
    timestamp = published_at.isoformat()
    statements = artifact_statements(registry_file, timestamp, produced_by_run_id=None)
    statements.append(
        advance_artifact_current_version_statement(
            registry_file.artifact_id, registry_file.version_id
        )
    )
    statements.append(
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                "news.acquire",
                "dlt",
                implementation_ref,
                canonical_json({"registry_version_id": registry.version_id}).decode(),
                "chartly",
                "publishing",
                f"news.acquire:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    )
    statements.append(
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                0,
                registry_file.version_id,
                "registry",
                None,
                registry_file.content_digest,
                "explicit",
                None,
            ],
        )
    )
    for position, value in enumerate((*snapshots, *observations, load_file)):
        statements.extend(artifact_statements(value, timestamp, produced_by_run_id=run_id))
        statements.append(
            (
                "INSERT INTO run_outputs VALUES (%s, %s, %s, 'output') ON CONFLICT DO NOTHING",
                [run_id, position, value.version_id],
            )
        )
        statements.append(
            advance_artifact_current_version_statement(value.artifact_id, value.version_id)
        )
    for observation in observations:
        capture = observation.capture
        if capture is None:
            raise ValueError("Observation catalog metadata is missing")
        statements.append(
            (
                "INSERT INTO news_feed_observations VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    observation.version_id,
                    observation.feed_snapshot_version_id,
                    capture.feed_id,
                    capture.scheduled_slot.isoformat(),
                    capture.status,
                    capture.http_status,
                    len(capture.entries),
                    capture.observed_at.isoformat(),
                    capture.latency_ms,
                    capture.error,
                ],
            )
        )
    for load_id in load_ids:
        statements.append(
            (
                "INSERT INTO news_dlt_loads VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                [load_id, load_file.version_id, timestamp],
            )
        )
    statements.extend(_feed_entry_event_statements(feed_entries, aliases))
    for load_id in load_ids:
        statements.append(
            (
                "INSERT INTO news_feed_entry_projection_loads VALUES (%s, %s) ON CONFLICT DO NOTHING",
                [load_id, timestamp],
            )
        )
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, run_id],
        )
    )
    return statements


def catalog_feed_entry_events(
    occurrences: tuple[FeedEntryEventOccurrence, ...],
    *,
    batch_size: int = 50,
) -> FeedEntryCatalogResult:
    """Catalog every exact occurrence and resolve each event to one feed-entry version."""
    if batch_size < 1:
        raise ValueError("Catalog batch size must be positive")
    if not occurrences:
        return FeedEntryCatalogResult(cataloged_events=0, existing_events=0)
    occurrences = _unique_feed_entry_occurrences(occurrences)
    load_ids = tuple(sorted({value.dlt_load_id for value in occurrences}))
    registered = _registered_load_ids(load_ids)
    if registered != set(load_ids):
        missing = sorted(set(load_ids) - registered)
        raise ValueError(f"Feed entry events use unregistered dlt loads: {', '.join(missing)}")
    snapshot_versions = _resolve_feed_snapshot_versions(occurrences)
    entries = _cataloged_feed_entry_occurrences(occurrences, snapshot_versions)
    aliases = _feed_entry_version_aliases(occurrences)
    source_event_ids = tuple(sorted({value.source_event_id for value in entries}))
    existing = _existing_feed_entry_event_ids(source_event_ids)
    incoming_ids = {value.source_event_id for value in entries}
    for offset in range(0, len(entries), batch_size):
        selected = entries[offset : offset + batch_size]
        selected_ids = {value.source_event_id for value in selected}
        selected_aliases = {
            event_id: version_id
            for event_id, version_id in aliases.items()
            if event_id in selected_ids or event_id not in incoming_ids
        }
        catalog_batch(_feed_entry_event_statements(selected, selected_aliases))
    return FeedEntryCatalogResult(
        cataloged_events=len(set(source_event_ids) - existing),
        existing_events=len(existing),
    )


def read_cataloged_feed_entry_reference(event_id: Sha256) -> CatalogedFeedEntryReference:
    """Resolve an observed or canonical event identity to one exact occurrence."""
    rows = catalog_query(
        """
        SELECT requested.version_id AS event_id,
               occurrence.dlt_load_id,
               occurrence.registry_version_id,
               occurrence.feed_snapshot_version_id,
               occurrence.observed_at,
               event.feed_id,
               event.source_id,
               event.original_url,
               event.published_at,
               event.source_updated_at
        FROM news_feed_entry_event_versions AS requested
        JOIN news_feed_entry_event_versions AS identity
          ON identity.version_id = requested.version_id
        JOIN news_feed_entry_occurrences AS occurrence
          ON occurrence.event_id = identity.event_id
        JOIN news_feed_entry_events AS event
          ON event.event_id = occurrence.event_id
        WHERE requested.event_id = %s
        ORDER BY occurrence.event_id IS DISTINCT FROM %s, occurrence.observed_at,
                 occurrence.dlt_load_id, occurrence.feed_snapshot_version_id
        LIMIT 1
        """,
        [event_id, event_id],
    )
    if len(rows) != 1:
        raise ValueError(f"Expected one cataloged feed entry: {event_id}")
    return _cataloged_feed_entry_reference(rows[0])


def read_cataloged_feed_entry_references(
    *,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> tuple[CatalogedFeedEntryReference, ...]:
    """Read one canonical feed-entry version with its earliest exact occurrence."""
    for name, value in (("start", start_at), ("end", end_at)):
        if value is not None and value.tzinfo is None:
            raise ValueError(f"Catalog {name} time must include a UTC offset")
    conditions = []
    parameters: list[object] = []
    if start_at is not None:
        conditions.append("event.published_at >= %s")
        parameters.append(start_at.astimezone(UTC).isoformat())
    if end_at is not None:
        conditions.append("event.published_at < %s")
        parameters.append(end_at.astimezone(UTC).isoformat())
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = catalog_query(
        f"""
        SELECT event_id, dlt_load_id, registry_version_id, feed_snapshot_version_id,
               observed_at, feed_id, source_id, original_url, published_at,
               source_updated_at
        FROM (
            SELECT identity.version_id AS event_id,
                   occurrence.dlt_load_id,
                   occurrence.registry_version_id,
                   occurrence.feed_snapshot_version_id,
                   occurrence.observed_at,
                   event.feed_id,
                   event.source_id,
                   event.original_url,
                   event.published_at,
                   event.source_updated_at,
                   row_number() OVER (
                       PARTITION BY identity.version_id
                       ORDER BY occurrence.observed_at, occurrence.dlt_load_id,
                                occurrence.feed_snapshot_version_id, occurrence.event_id
                   ) AS occurrence_rank
            FROM news_feed_entry_occurrences AS occurrence
            JOIN news_feed_entry_event_versions AS identity
              ON identity.event_id = occurrence.event_id
            JOIN news_feed_entry_events AS event
              ON event.event_id = occurrence.event_id
            {where}
        ) AS ranked
        WHERE occurrence_rank = 1
        ORDER BY published_at, feed_id, event_id
        """,
        parameters,
    )
    return tuple(_cataloged_feed_entry_reference(row) for row in rows)


def _cataloged_feed_entry_reference(row: dict[str, object]) -> CatalogedFeedEntryReference:
    return CatalogedFeedEntryReference(
        event_id=str(row["event_id"]),
        dlt_load_id=str(row["dlt_load_id"]),
        registry_version_id=str(row["registry_version_id"]),
        feed_snapshot_version_id=str(row["feed_snapshot_version_id"]),
        observed_at=datetime.fromisoformat(str(row["observed_at"])),
        feed_id=str(row["feed_id"]),
        source_id=str(row["source_id"]),
        url=HttpUrl(str(row["original_url"])),
        published_at=datetime.fromisoformat(str(row["published_at"])),
        source_updated_at=(
            datetime.fromisoformat(str(row["source_updated_at"]))
            if row["source_updated_at"] is not None
            else None
        ),
    )


def _cataloged_feed_entry_occurrences(
    occurrences: tuple[FeedEntryEventOccurrence, ...],
    snapshot_versions: dict[tuple[str, Sha256], Sha256],
) -> tuple[_CatalogedFeedEntryOccurrence, ...]:
    return tuple(
        _CatalogedFeedEntryOccurrence(
            source_event_id=value.source_event_id,
            reference=CatalogedFeedEntryReference(
                event_id=value.event.event_id,
                dlt_load_id=value.dlt_load_id,
                registry_version_id=value.event.registry_version_id,
                feed_snapshot_version_id=snapshot_versions[
                    (value.event.entry.feed_id, value.event.feed_content_digest)
                ],
                observed_at=value.event.observed_at,
                feed_id=value.event.entry.feed_id,
                source_id=value.event.entry.source_id,
                url=value.event.entry.url,
                published_at=value.event.entry.published_at,
                source_updated_at=value.event.entry.source_updated_at,
            ),
        )
        for value in occurrences
    )


def _feed_entry_event_statements(
    entries: tuple[_CatalogedFeedEntryOccurrence, ...],
    aliases: dict[Sha256, Sha256],
) -> list[tuple[str, list[object]]]:
    statements = []
    for offset in range(0, len(entries), _CATALOG_EVENT_ROWS_PER_STATEMENT):
        values = entries[offset : offset + _CATALOG_EVENT_ROWS_PER_STATEMENT]
        placeholders = ", ".join("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" for _ in values)
        parameters = []
        for occurrence in values:
            value = occurrence.reference
            parameters.extend(
                [
                    occurrence.source_event_id,
                    value.dlt_load_id,
                    value.registry_version_id,
                    value.feed_snapshot_version_id,
                    value.feed_id,
                    value.source_id,
                    str(value.url),
                    value.published_at.astimezone(UTC).isoformat(),
                    value.source_updated_at.astimezone(UTC).isoformat()
                    if value.source_updated_at
                    else None,
                    value.observed_at.astimezone(UTC).isoformat(),
                ]
            )
        statements.append(
            (
                f"INSERT INTO news_feed_entry_events VALUES {placeholders} ON CONFLICT DO NOTHING",
                parameters,
            )
        )
    for offset in range(0, len(entries), 20):
        values = entries[offset : offset + 20]
        placeholders = ", ".join("(%s, %s, %s, %s, %s)" for _ in values)
        parameters = []
        for occurrence in values:
            value = occurrence.reference
            parameters.extend(
                [
                    occurrence.source_event_id,
                    value.dlt_load_id,
                    value.registry_version_id,
                    value.feed_snapshot_version_id,
                    value.observed_at.astimezone(UTC).isoformat(),
                ]
            )
        statements.append(
            (
                f"INSERT INTO news_feed_entry_occurrences VALUES {placeholders} ON CONFLICT DO NOTHING",
                parameters,
            )
        )
    aliases_by_event_id = sorted(aliases.items())
    for offset in range(0, len(aliases_by_event_id), 50):
        values = aliases_by_event_id[offset : offset + 50]
        placeholders = ", ".join("(%s, %s)" for _ in values)
        parameters = [item for value in values for item in value]
        statements.append(
            (
                f"INSERT INTO news_feed_entry_event_versions VALUES {placeholders} "
                "ON CONFLICT(event_id) DO UPDATE SET version_id = excluded.version_id",
                parameters,
            )
        )
    return statements


def _unique_feed_entry_occurrences(
    occurrences: tuple[FeedEntryEventOccurrence, ...],
) -> tuple[FeedEntryEventOccurrence, ...]:
    versions = {}
    unique = {}
    for value in occurrences:
        existing = versions.get(value.event.event_id)
        if existing is not None and existing != value.event.entry:
            raise ValueError(f"Feed entry event identity conflict: {value.event.event_id}")
        versions[value.event.event_id] = value.event.entry
        key = (
            value.source_event_id,
            value.dlt_load_id,
            value.event.registry_version_id,
            value.event.feed_content_digest,
            value.event.observed_at,
        )
        unique[key] = value
    return tuple(unique[key] for key in sorted(unique))


def _feed_entry_version_aliases(
    occurrences: tuple[FeedEntryEventOccurrence, ...],
) -> dict[Sha256, Sha256]:
    aliases = {value.source_event_id: value.event.event_id for value in occurrences}
    for value in occurrences:
        aliases[value.event.event_id] = value.event.event_id
        entry = value.event.entry
        rows = catalog_query(
            """
            SELECT event.event_id, version.content_digest
            FROM news_feed_entry_events AS event
            JOIN artifact_versions AS version
              ON version.id = event.feed_snapshot_version_id
            WHERE event.feed_id = %s AND event.source_id = %s
              AND event.original_url = %s AND event.published_at = %s
              AND event.source_updated_at IS NOT DISTINCT FROM %s
            """,
            [
                entry.feed_id,
                entry.source_id,
                str(entry.url),
                entry.published_at.astimezone(UTC).isoformat(),
                entry.source_updated_at.astimezone(UTC).isoformat()
                if entry.source_updated_at
                else None,
            ],
        )
        for row in rows:
            candidate_id = str(row["event_id"])
            if candidate_id in {
                value.event.event_id,
                legacy_feed_entry_event_id(entry, row["content_digest"]),
            }:
                aliases[candidate_id] = value.event.event_id
    return aliases


def _registered_load_ids(load_ids: tuple[str, ...]) -> set[str]:
    placeholders = ", ".join("%s" for _ in load_ids)
    rows = catalog_query(
        f"SELECT load_id FROM news_dlt_loads WHERE load_id IN ({placeholders})",
        list(load_ids),
    )
    return {str(row["load_id"]) for row in rows}


def _resolve_feed_snapshot_versions(
    occurrences: tuple[FeedEntryEventOccurrence, ...],
) -> dict[tuple[str, Sha256], Sha256]:
    pairs = {(value.event.entry.feed_id, value.event.feed_content_digest) for value in occurrences}
    resolved: dict[tuple[str, Sha256], Sha256] = {}
    for feed_id, digest in sorted(pairs):
        rows = catalog_query(
            "SELECT version.id FROM artifact_versions version "
            "JOIN artifacts artifact ON artifact.id = version.artifact_id "
            "WHERE artifact.id = %s AND artifact.kind = 'news_feed' "
            "AND version.content_digest = %s",
            [f"news:feed:{feed_id}", digest],
        )
        if len(rows) != 1:
            raise ValueError(
                f"Expected one snapshot for feed {feed_id} and digest {digest}, received {len(rows)}"
            )
        resolved[(feed_id, digest)] = rows[0]["id"]
    return resolved


def _existing_feed_entry_event_ids(event_ids: tuple[Sha256, ...]) -> set[Sha256]:
    existing: set[Sha256] = set()
    for offset in range(0, len(event_ids), 50):
        values = event_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in values)
        rows = catalog_query(
            f"SELECT event_id FROM news_feed_entry_events WHERE event_id IN ({placeholders})",
            list(values),
        )
        existing.update(row["event_id"] for row in rows)
    return existing


def _feed_artifact_file(
    *,
    artifact_id: str,
    artifact_kind: str,
    title: str,
    content: bytes,
    r2_key: str,
    media_type: str,
    feed_id: str | None = None,
    feed_snapshot_version_id: Sha256 | None = None,
    capture: FeedCapture | None = None,
) -> FeedArtifactFile:
    value = artifact_file(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        title=title,
        content=content,
        r2_key=r2_key,
        media_type=media_type,
    )
    return FeedArtifactFile(
        **value.model_dump(),
        feed_id=feed_id,
        feed_snapshot_version_id=feed_snapshot_version_id,
        capture=capture,
    )
