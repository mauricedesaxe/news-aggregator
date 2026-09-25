from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Jsonb

from romanian_news.catalog.report_inputs import CurrentDailyReportRecord
from romanian_news.catalog_transport import CatalogConnection, catalog_query, catalog_transaction
from romanian_news.video_digest.models import EditionId, ScheduledSlot, SlotId
from romanian_news.video_digest.selection import SlotSubjectSelection


def capture_slot_report_source(slot: ScheduledSlot) -> CurrentDailyReportRecord | None:
    """Read and freeze the report head once for this scheduled slot."""

    def capture(connection: CatalogConnection) -> CurrentDailyReportRecord | None:
        locked = connection.execute(
            "SELECT slot_id FROM video_digest_slots WHERE slot_id = %s FOR UPDATE",
            (slot.slot_id,),
        ).fetchone()
        if locked is None:
            raise ValueError("Video digest slot must be scheduled before source capture")
        stored = connection.execute(
            "SELECT source_record FROM video_digest_slot_sources WHERE slot_id = %s",
            (slot.slot_id,),
        ).fetchone()
        if stored is not None:
            return _record(slot, stored["source_record"])
        rows = connection.execute(
            """SELECT report.current_version_id AS version_id,
                      report.current_run_id AS run_id, file.content_digest, file.r2_key,
                      MAX(input_version.created_at) AS input_time
               FROM artifacts report
               JOIN artifact_files file
                 ON file.artifact_version_id = report.current_version_id
               JOIN run_inputs input
                 ON input.run_id = report.current_run_id AND input.role != 'prior_output'
               JOIN artifact_versions input_version
                 ON input_version.id = input.artifact_version_id
               WHERE report.id = %s AND report.kind = 'news_daily_report'
               GROUP BY report.current_version_id, report.current_run_id,
                        file.content_digest, file.r2_key""",
            (f"news:daily:{slot.bucharest_day.isoformat()}",),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError("Daily report source has more than one current file")
        record = (
            CurrentDailyReportRecord.model_validate(
                {"day": slot.bucharest_day, **rows[0]}, strict=False
            )
            if rows
            else None
        )
        connection.execute(
            """INSERT INTO video_digest_slot_sources
                   (slot_id, source_record, observed_at)
               VALUES (%s, %s, %s)""",
            (
                slot.slot_id,
                Jsonb(record.model_dump(mode="json")) if record is not None else None,
                datetime.now(UTC),
            ),
        )
        return record

    return catalog_transaction(capture)


def read_earlier_slot_selections(
    slot: ScheduledSlot,
) -> tuple[tuple[SlotId, SlotSubjectSelection], ...]:
    rows = catalog_query(
        """SELECT slot.slot_id, source.selection, source.selection_digest
           FROM video_digest_slots slot
           JOIN video_digest_slot_sources source ON source.slot_id = slot.slot_id
           WHERE slot.bucharest_day = %s AND slot.scheduled_at < %s
             AND slot.stage IN (
                 'claimed', 'planning', 'generating', 'assembling',
                 'subtitling', 'publishing', 'published'
             )
             AND source.selection IS NOT NULL
           ORDER BY slot.scheduled_at, slot.slot_id""",
        [slot.bucharest_day, slot.scheduled_at],
    )
    result = tuple((SlotId(str(row["slot_id"])), _selection(row["selection"])) for row in rows)
    if any(
        item.digest != row["selection_digest"] for (_, item), row in zip(result, rows, strict=True)
    ):
        raise ValueError("Earlier slot selection digest does not match stored content")
    return result


def has_unselected_earlier_edition(slot: ScheduledSlot) -> bool:
    """A legacy covered edition cannot safely be compared with a new selection."""
    rows = catalog_query(
        """SELECT 1 AS present
           FROM video_digest_slots slot
           LEFT JOIN video_digest_slot_sources source ON source.slot_id = slot.slot_id
           WHERE slot.bucharest_day = %s AND slot.scheduled_at < %s
             AND slot.stage IN (
                 'claimed', 'planning', 'generating', 'assembling',
                 'subtitling', 'publishing', 'published'
             )
             AND source.selection IS NULL
           LIMIT 1""",
        [slot.bucharest_day, slot.scheduled_at],
    )
    return bool(rows)


def record_slot_subject_selection(
    slot: ScheduledSlot, selection: SlotSubjectSelection
) -> SlotSubjectSelection:
    """Keep the first complete decision for a slot across retries."""

    def record(connection: CatalogConnection) -> SlotSubjectSelection:
        row = connection.execute(
            """SELECT source_record, selection, selection_digest
               FROM video_digest_slot_sources WHERE slot_id = %s FOR UPDATE""",
            (slot.slot_id,),
        ).fetchone()
        if row is None or row["source_record"] is None:
            raise ValueError("Selection requires a frozen report source")
        if row["selection"] is not None:
            stored = _selection(row["selection"])
            if stored.digest != str(row["selection_digest"]):
                raise ValueError("Stored selection digest does not match its content")
            return stored
        source = _record(slot, row["source_record"])
        assert source is not None
        if source.version_id != selection.report_version_id:
            raise ValueError("Selection report version differs from frozen source")
        connection.execute(
            """UPDATE video_digest_slot_sources
               SET selection = %s, selection_digest = %s, selected_at = %s
               WHERE slot_id = %s""",
            (
                Jsonb(selection.model_dump(mode="json")),
                selection.digest,
                datetime.now(UTC),
                slot.slot_id,
            ),
        )
        return selection

    return catalog_transaction(record)


def read_slot_subject_selection(slot_id: SlotId) -> SlotSubjectSelection | None:
    rows = catalog_query(
        "SELECT selection, selection_digest FROM video_digest_slot_sources WHERE slot_id = %s",
        [slot_id],
    )
    if not rows or rows[0]["selection"] is None:
        return None
    selection = _selection(rows[0]["selection"])
    if selection.digest != rows[0]["selection_digest"]:
        raise ValueError("Stored selection digest does not match its content")
    return selection


def read_edition_subject_selection(edition_id: EditionId) -> SlotSubjectSelection | None:
    rows = catalog_query(
        """SELECT source.selection, source.selection_digest,
                  edition.selection_digest AS edition_selection_digest
           FROM video_digest_editions edition
           JOIN video_digest_slots slot ON slot.edition_id = edition.edition_id
           JOIN video_digest_slot_sources source ON source.slot_id = slot.slot_id
           WHERE edition.edition_id = %s AND edition.selection_digest IS NOT NULL""",
        [edition_id],
    )
    if not rows:
        return None
    selections = tuple(_selection(row["selection"]) for row in rows)
    if any(selection != selections[0] for selection in selections):
        raise ValueError("Edition has conflicting slot selections")
    if any(
        selections[0].digest != row["selection_digest"]
        or selections[0].digest != row["edition_selection_digest"]
        for row in rows
    ):
        raise ValueError("Edition selection digest does not match stored content")
    return selections[0]


def _record(slot: ScheduledSlot, value: Any) -> CurrentDailyReportRecord | None:
    if value is None:
        return None
    record = CurrentDailyReportRecord.model_validate_json(json.dumps(value), strict=True)
    if record.day != slot.bucharest_day:
        raise ValueError("Frozen report source day differs from scheduled day")
    return record


def _selection(value: Any) -> SlotSubjectSelection:
    return SlotSubjectSelection.model_validate_json(json.dumps(value), strict=True)
