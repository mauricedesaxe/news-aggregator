"""Print a read-only historical replay preview without starting provider work."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date

from romanian_news.archive.replay_preflight import read_replay_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", type=date.fromisoformat)
    arguments = parser.parse_args()
    sys.stdout.write(
        json.dumps(asdict(read_replay_preflight(arguments.day)), default=str, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
