from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime

from romanian_news.articles.acquisition import (
    inspect_article_recovery_event,
    recover_quarantined_article_events,
)
from romanian_news.catalog.schema import ensure_news_catalog_schema
from romanian_news.feeds.registry import feed_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or release quarantined article work")
    subcommands = parser.add_subparsers(dest="command", required=True)
    status = subcommands.add_parser("status")
    status.add_argument("event_id")
    release = subcommands.add_parser("release")
    release.add_argument("event_id")
    release.add_argument("--expected-generation", required=True)
    release.add_argument("--requested-by", required=True)
    release.add_argument("--reason", required=True)
    release.add_argument("--requested-at")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.event_id):
        parser.error("event_id must be a lowercase SHA-256 digest")
    ensure_news_catalog_schema()
    registry = feed_registry()
    if args.command == "status":
        print(json.dumps(asdict(inspect_article_recovery_event(args.event_id, registry))))
        return 0
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_generation):
        parser.error("expected generation must be a lowercase SHA-256 digest")
    try:
        requested_at = (
            datetime.fromisoformat(args.requested_at) if args.requested_at else datetime.now(UTC)
        )
        receipt = recover_quarantined_article_events(
            (args.event_id,),
            registry,
            requested_by=args.requested_by,
            reason=args.reason,
            requested_at=requested_at,
            expected_work_generations={args.event_id: args.expected_generation},
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({**asdict(receipt), "requested_at": requested_at.astimezone(UTC).isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
