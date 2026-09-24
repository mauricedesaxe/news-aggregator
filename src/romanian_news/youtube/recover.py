from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import UUID, uuid4

from romanian_news.youtube.recovery import (
    inspect_youtube_quarantine,
    release_quarantined_youtube_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or release quarantined YouTube work")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    release = commands.add_parser("release")
    for command in (status, release):
        command.add_argument("source_id")
        command.add_argument("video_id")
    release.add_argument("--expected-generation", type=int, required=True)
    release.add_argument("--request-id", type=UUID)
    release.add_argument("--requested-at", type=datetime.fromisoformat)
    release.add_argument("--requested-by", required=True)
    release.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(asdict(inspect_youtube_quarantine(args.source_id, args.video_id))))
        return
    request_id = args.request_id or uuid4()
    requested_at = args.requested_at or datetime.now(UTC)
    print(
        json.dumps({"request_id": str(request_id), "requested_at": requested_at.isoformat()}),
        file=sys.stderr,
        flush=True,
    )
    receipt = release_quarantined_youtube_video(
        args.source_id,
        args.video_id,
        expected_generation=args.expected_generation,
        request_id=request_id,
        requested_by=args.requested_by,
        reason=args.reason,
        requested_at=requested_at,
    )
    print(json.dumps(asdict(receipt), default=str))


if __name__ == "__main__":
    main()
