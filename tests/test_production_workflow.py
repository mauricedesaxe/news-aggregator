import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from romanian_news.worker.definitions import defs

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/romanian-news-production.yml"


def test_production_workflow_starts_every_defined_schedule() -> None:
    workflow = WORKFLOW.read_text()
    schedule_block = workflow.split("for schedule_name in (", maxsplit=1)[1].split(
        "):", maxsplit=1
    )[0]
    workflow_schedules = set(re.findall(r'"([a-z_]+)"', schedule_block))

    schedules = defs.schedules
    assert schedules is not None
    assert workflow_schedules == {schedule.name for schedule in schedules}


def test_production_workflow_rejects_noncurrent_or_inspection_activation() -> None:
    workflow = WORKFLOW.read_text()
    validation_step = workflow.split("- name: Validate operation", maxsplit=1)[1].split(
        "- name: Verify PostgreSQL catalog", maxsplit=1
    )[0]
    script = textwrap.dedent(
        validation_step.split("<<'PY'", maxsplit=1)[1].split("\n          PY", maxsplit=1)[0]
    )
    today = datetime.now(ZoneInfo("Europe/Bucharest")).date()
    common = os.environ | {"RUN_ID": ""}

    current = subprocess.run(
        (sys.executable, "-c", script),
        env=common | {"ENABLE_ARTICLE_CONTROLLER": "true", "PARTITION": today.isoformat()},
        capture_output=True,
        text=True,
    )
    old = subprocess.run(
        (sys.executable, "-c", script),
        env=common
        | {
            "ENABLE_ARTICLE_CONTROLLER": "false",
            "PARTITION": (today - timedelta(days=1)).isoformat(),
        },
        capture_output=True,
        text=True,
    )
    inspection = subprocess.run(
        (sys.executable, "-c", script),
        env=os.environ
        | {
            "ENABLE_ARTICLE_CONTROLLER": "true",
            "PARTITION": "",
            "RUN_ID": "run-1",
        },
        capture_output=True,
        text=True,
    )

    assert current.returncode == 0
    assert old.returncode != 0
    assert "current Bucharest date" in old.stderr
    assert inspection.returncode != 0
    assert "requires a current-day feed probe" in inspection.stderr
