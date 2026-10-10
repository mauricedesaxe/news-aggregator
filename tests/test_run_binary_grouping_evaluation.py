from __future__ import annotations

from pathlib import Path

from romanian_news.binary_grouping_evaluation import BinaryGroupingEvaluationResult
from romanian_news.tests.test_binary_grouping_evaluation import _source
from scripts import run_binary_grouping_evaluation as runner


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
