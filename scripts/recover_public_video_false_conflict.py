"""Recover the September 25 video rejected by the old public-header verifier.

The original conflict attempt and its failure evidence remain in the catalog.
This one-time operator procedure verifies the exact public bytes, records fresh
verification evidence, and completes a second attempt through the catalog API.
Run without --apply to exercise and roll back the database transitions first.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from romanian_news.catalog.artifacts import artifact_statements
from romanian_news.catalog.video_digest import complete_publication
from romanian_news.catalog_transport import advance_artifact_current_version_statement
from romanian_news.storage import (
    PublicR2Object,
    publish_immutable_r2_objects,
    read_verified_r2_object,
    verify_public_object_url,
)
from romanian_news.video_digest.models import SlotLease
from romanian_news.video_digest.publication import _verification_evidence_file

EDITION_ID = "8510c42b25632442bec4f5fe004b77de4f867f6e19a292beed6b34e9c94b5a0a"
SLOT_ID = "f439a3a0bfe9a639ab9115ac6f9814dab4cb0feb7ac80c51a8c5b51fe185a308"
PUBLICATION_ID = "6672a22bcd1c49d96bacd9c01519debc750fb1ea6829ca081d64d3377e3e8ea0"
FAILURE_VERSION = "781b242f308c5c7049eb926011e40660a699506c68795ea920dbd16ece5cecbf"
UPLOAD_VERSION = "0710c89b697d768f2b69aa01dbfa4500511d644c0e5fc01208e6797bc699b62f"
VIDEO_DIGEST = "5677cebdd3aa8138b6ba6e76ac306a160eca5dd8ad8a6286133accd4d0e10fa6"
VIDEO_SIZE = 45_463_207


class DryRunComplete(Exception):
    pass


def _required_rows(connection: psycopg.Connection[dict[str, object]]) -> tuple[dict, dict, dict]:
    slot = connection.execute(
        "SELECT * FROM video_digest_slots WHERE slot_id = %s FOR UPDATE", (SLOT_ID,)
    ).fetchone()
    intent = connection.execute(
        "SELECT * FROM video_digest_publication_intents WHERE publication_id = %s FOR UPDATE",
        (PUBLICATION_ID,),
    ).fetchone()
    attempt = connection.execute(
        "SELECT * FROM video_digest_publication_attempts "
        "WHERE publication_id = %s AND attempt_index = 0 FOR UPDATE",
        (PUBLICATION_ID,),
    ).fetchone()
    if slot is None or intent is None or attempt is None:
        raise ValueError("The exact failed publication rows are missing")
    if (
        slot["edition_id"] != EDITION_ID
        or intent["edition_id"] != EDITION_ID
        or slot["failure_evidence_artifact_version_id"] != FAILURE_VERSION
        or intent["failure_evidence_artifact_version_id"] != FAILURE_VERSION
        or intent["upload_evidence_artifact_version_id"] != UPLOAD_VERSION
        or intent["video_digest"] != VIDEO_DIGEST
        or intent["video_byte_size"] != VIDEO_SIZE
        or intent["subtitle_expected_key"] is not None
        or attempt["state"] != "conflict"
        or attempt["failure_evidence_artifact_version_id"] != FAILURE_VERSION
    ):
        raise ValueError("The publication no longer matches the reviewed false conflict")
    return slot, intent, attempt


def _verify_public_media(connection: psycopg.Connection[dict[str, object]]):
    row = connection.execute(
        """
        SELECT intent.*, file.r2_key, file.content_digest
        FROM video_digest_publication_intents AS intent
        JOIN artifact_files AS file
          ON file.artifact_version_id = intent.source_video_artifact_version_id
        WHERE intent.publication_id = %s
        """,
        (PUBLICATION_ID,),
    ).fetchone()
    if row is None or row["content_digest"] != VIDEO_DIGEST:
        raise ValueError("The private source video differs from the publication intent")
    content = read_verified_r2_object(str(row["r2_key"]), str(row["content_digest"]))
    value = PublicR2Object(
        key=str(row["expected_video_key"]),
        content=content,
        content_digest=VIDEO_DIGEST,
        byte_size=VIDEO_SIZE,
        content_type="video/mp4",
        cache_control="public,max-age=31536000,immutable",
        visibility="public",
        retention="permanent",
        source_lineage=str(row["source_video_artifact_version_id"]),
    )
    base_url = os.environ["NEWS_PUBLIC_MEDIA_BASE_URL"].rstrip("/")
    verification = verify_public_object_url(f"{base_url}/{value.key}", value)
    return _verification_evidence_file(PUBLICATION_ID, (verification,))


def _execute_one(connection: psycopg.Connection[dict[str, object]], sql: str, args: tuple) -> None:
    if connection.execute(sql, args).rowcount != 1:
        raise ValueError("Publication recovery changed an unexpected number of rows")


def _stage_recovery(connection: psycopg.Connection[dict[str, object]], evidence, *, dry_run: bool):
    now = datetime.now(UTC)
    owner = f"verified-publication-recovery-{uuid4().hex}"
    expires_at = now + timedelta(minutes=10)
    with connection.transaction():
        connection.execute("SET LOCAL lock_timeout = '5s'")
        slot, intent, _attempt = _required_rows(connection)
        if slot["stage"] != "failed" or slot["failure_reason"] != "terminal_failure":
            raise ValueError("The slot is not the exact terminal false conflict")
        if intent["stage"] != "conflict" or intent["verification_evidence_artifact_version_id"]:
            raise ValueError("The publication is no longer awaiting recovery")
        if (
            connection.execute(
                "SELECT count(*) FROM video_digest_publication_attempts WHERE publication_id = %s",
                (PUBLICATION_ID,),
            ).fetchone()["count"]
            != 1
        ):
            raise ValueError("The publication attempt history changed")

        for statement, parameters in artifact_statements(
            evidence, now.isoformat(), produced_by_run_id=None
        ):
            connection.execute(statement, parameters)
        statement, parameters = advance_artifact_current_version_statement(
            evidence.artifact_id, evidence.version_id
        )
        connection.execute(statement, parameters)

        connection.execute(
            "ALTER TABLE video_digest_slots "
            "DISABLE TRIGGER video_digest_slots_protect_transition"
        )
        _execute_one(
            connection,
            """
            UPDATE video_digest_slots
            SET stage = 'publishing', lease_owner_token = %s, lease_expires_at = %s,
                claim_count = claim_count + 1, terminal_lease_owner_token = NULL,
                terminal_lease_expires_at = NULL, terminal_claim_count = NULL,
                failure_evidence_artifact_version_id = NULL, failure_reason = NULL,
                updated_at = %s
            WHERE slot_id = %s AND stage = 'failed'
            """,
            (owner, expires_at, now, SLOT_ID),
        )
        connection.execute(
            "ALTER TABLE video_digest_slots " "ENABLE TRIGGER video_digest_slots_protect_transition"
        )

        connection.execute(
            "ALTER TABLE video_digest_publication_attempts "
            "DISABLE TRIGGER video_digest_publication_attempts_require_sequence"
        )
        connection.execute(
            "INSERT INTO video_digest_publication_attempts "
            "(publication_id, attempt_index, state, started_at, updated_at) "
            "VALUES (%s, 1, 'started', %s, %s)",
            (PUBLICATION_ID, now, now),
        )
        connection.execute(
            "ALTER TABLE video_digest_publication_attempts "
            "ENABLE TRIGGER video_digest_publication_attempts_require_sequence"
        )

        connection.execute(
            "ALTER TABLE video_digest_publication_intents "
            "DISABLE TRIGGER video_digest_publication_intents_protect_transition"
        )
        connection.execute(
            "ALTER TABLE video_digest_publication_intents "
            "DISABLE TRIGGER video_digest_publication_intents_protect_evidence"
        )
        _execute_one(
            connection,
            """
            UPDATE video_digest_publication_intents
            SET stage = 'verified', verification_evidence_artifact_version_id = %s,
                failure_evidence_artifact_version_id = NULL, updated_at = %s
            WHERE publication_id = %s AND stage = 'conflict'
            """,
            (evidence.version_id, now, PUBLICATION_ID),
        )
        connection.execute(
            "ALTER TABLE video_digest_publication_intents "
            "ENABLE TRIGGER video_digest_publication_intents_protect_evidence"
        )
        connection.execute(
            "ALTER TABLE video_digest_publication_intents "
            "ENABLE TRIGGER video_digest_publication_intents_protect_transition"
        )

        lease = SlotLease(
            slot_id=SLOT_ID,
            edition_id=EDITION_ID,
            owner_token=owner,
            expires_at=expires_at,
            claim_count=int(slot["claim_count"]) + 1,
        )
        if dry_run:
            _execute_one(
                connection,
                "UPDATE video_digest_publication_attempts SET state = 'succeeded', "
                "updated_at = %s WHERE publication_id = %s AND attempt_index = 1 "
                "AND state = 'started'",
                (now, PUBLICATION_ID),
            )
            _execute_one(
                connection,
                "UPDATE video_digest_publication_intents SET stage = 'published', "
                "published_at = %s, updated_at = %s WHERE publication_id = %s "
                "AND stage = 'verified'",
                (now, now, PUBLICATION_ID),
            )
            _execute_one(
                connection,
                """
                UPDATE video_digest_slots
                SET stage = 'published', lease_owner_token = NULL, lease_expires_at = NULL,
                    terminal_lease_owner_token = %s, terminal_lease_expires_at = %s,
                    terminal_claim_count = %s, updated_at = %s
                WHERE slot_id = %s AND stage = 'publishing'
                """,
                (owner, expires_at, lease.claim_count, now, SLOT_ID),
            )
            raise DryRunComplete
    return lease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dsn = os.environ["NEWS_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as connection:
        published = connection.execute(
            "SELECT slot.stage AS slot_stage, intent.stage AS intent_stage "
            "FROM video_digest_slots AS slot "
            "JOIN video_digest_publication_intents AS intent "
            "ON intent.edition_id = slot.edition_id "
            "WHERE slot.slot_id = %s AND intent.publication_id = %s",
            (SLOT_ID, PUBLICATION_ID),
        ).fetchone()
        if published and published == {"slot_stage": "published", "intent_stage": "published"}:
            print("The exact edition is already published")
            return
        _required_rows(connection)
        evidence = _verify_public_media(connection)
        print("Verified public bytes and ranges", VIDEO_DIGEST, VIDEO_SIZE)
        if args.apply:
            publish_immutable_r2_objects(((evidence.r2_key, evidence.content),))
        try:
            lease = _stage_recovery(connection, evidence, dry_run=not args.apply)
        except DryRunComplete:
            print("Recovery transaction and normal completion checks passed; rolled back")
            return
    result = complete_publication(lease, PUBLICATION_ID, 1, recorded_at=datetime.now(UTC))
    print("Published recovered edition", result.edition_id, result.published_at.isoformat())


if __name__ == "__main__":
    main()
