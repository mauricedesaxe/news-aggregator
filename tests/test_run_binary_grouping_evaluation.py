from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from romanian_news.binary_grouping_evaluation import BinaryGroupingEvaluationResult
from romanian_news.tests.test_binary_grouping_evaluation import _source

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_binary_grouping_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("run_binary_grouping_evaluation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def test_dry_run_writes_complete_strict_grouping_matrix(tmp_path: Path) -> None:
    output = tmp_path / "grouping.json"

    result = runner.execute(
        runner.parse_args(
            [
                "--execution-ref",
                "git:grouping-dry-run",
                "--output",
                str(output),
                "--trials",
                "2",
                "--dry-run",
            ]
        ),
        source=_source(),
    )

    parsed = BinaryGroupingEvaluationResult.model_validate_json(output.read_bytes(), strict=True)
    assert parsed == result
    assert len(result.runs) == 6
    assert sum(len(run.cases) for run in result.runs) == 6
    assert all(case.status == "completed" for run in result.runs for case in run.cases)
