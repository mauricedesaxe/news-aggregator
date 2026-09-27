"""Check a bounded archive page batch without retaining article text."""

import argparse
import json
import sys
from collections import Counter
from datetime import date

from romanian_news.archive.page_checks import check_archive_pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlet", choices=("hotnews", "digi24"), required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    checks = check_archive_pages(args.outlet, args.start, args.end, limit=args.limit)
    sys.stdout.write(
        json.dumps(
            {
                "outlet": args.outlet,
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "checked": len(checks),
                "statuses": dict(Counter(check.status for check in checks)),
                "rejections": dict(
                    Counter(check.rejection for check in checks if check.rejection is not None)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
