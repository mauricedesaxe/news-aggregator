from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from romanian_news.analysis.binary_evaluation import (
    BinaryProbabilityObservation,
    BinaryQuestion,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.binary_benchmark import (
    REGISTERED_BINARY_TARGETS,
    BenchmarkId,
    BinaryBenchmarkCase,
    BinaryBenchmarkEvaluationResult,
    BinaryEvaluator,
    BinaryJudgmentCase,
    BinaryTarget,
    TargetId,
)
from romanian_news.binary_daily_theme_evaluation import (
    BinaryDailyThemeSource,
    build_daily_theme_binary_case,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.derived_binary_benchmark import (
    DERIVED_BINARY_DISPATCH,
    DerivedBinarySource,
    DerivedBinaryWorkload,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import ThemeEvaluationDaySpec
from romanian_news.tests.evaluation_factories import reference
from romanian_news.tests.test_binary_confidence_evaluation import _source as confidence_source
from romanian_news.tests.test_binary_daily_theme_evaluation import (
    _manifest_and_inputs,
    _pin,
)
from romanian_news.tests.test_binary_grouping_evaluation import _source as grouping_source
from romanian_news.tests.test_binary_ranking_evaluation import _source as ranking_source
from romanian_news.tests.test_binary_tier_evaluation import (
    _cases as tier_cases,
)
from romanian_news.tests.test_binary_tier_evaluation import (
    _source as tier_source,
)

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_derived_binary_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_derived_binary_benchmark", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def test_dispatch_covers_every_protocol_concern() -> None:
    assert set(DERIVED_BINARY_DISPATCH) == set(DERIVED_BINARY_BENCHMARKS)


@pytest.mark.parametrize("concern", tuple(DERIVED_BINARY_DISPATCH))
def test_every_registration_adapts_its_concern_source(concern: BenchmarkId) -> None:
    source = _concern_source(concern)

    workload = DERIVED_BINARY_DISPATCH[concern].prepare(source)

    assert workload.definition == DERIVED_BINARY_BENCHMARKS[concern]
    assert len(workload.cases) == workload.definition.case_count
    assert (
        sum(len(case.judgments) for case in workload.cases)
        == workload.definition.calls_per_model_trial
    )


def test_parser_defaults_to_three_trials() -> None:
    arguments = runner.parse_args(
        [
            "--concern",
            "grouping",
            "--execution-ref",
            "git:test",
            "--output",
            "result.json",
            "--dry-run",
        ]
    )

    assert arguments.trials == 3
    assert arguments.resume is False
    assert arguments.retry_errors is False


@pytest.mark.parametrize("concern", tuple(DERIVED_BINARY_DISPATCH))
def test_dry_runner_writes_complete_strict_result_for_every_concern(
    concern: BenchmarkId, tmp_path: Path
) -> None:
    output = tmp_path / f"{concern}.json"
    workload = _workload(concern)

    result = runner.execute(
        runner.parse_args(
            [
                "--concern",
                concern,
                "--execution-ref",
                f"git:{concern}",
                "--output",
                str(output),
                "--trials",
                "1",
                "--dry-run",
            ]
        ),
        workload=workload,
    )

    parsed = BinaryBenchmarkEvaluationResult.model_validate_json(output.read_bytes(), strict=True)
    assert parsed == result
    assert len(result.results) == (
        workload.definition.calls_per_model_trial * len(REGISTERED_BINARY_TARGETS)
    )
    assert all(item.status == "completed" for item in result.results)


def test_interruption_resume_reuses_terminal_judgments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "resume.json"
    arguments = runner.parse_args(
        [
            "--concern",
            "grouping",
            "--execution-ref",
            "git:resume",
            "--output",
            str(output),
            "--trials",
            "1",
            "--dry-run",
        ]
    )
    calls: Counter[tuple[str, str]] = Counter()
    evaluators = _counting_evaluators(calls)
    original_write = runner.write_binary_checkpoint
    checkpoints = 0

    def interrupting_write(*args: object, **kwargs: object) -> None:
        nonlocal checkpoints
        original_write(*args, **kwargs)
        checkpoints += 1
        if checkpoints == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "write_binary_checkpoint", interrupting_write)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(arguments, workload=_workload("grouping"), evaluators=evaluators)

    monkeypatch.setattr(runner, "write_binary_checkpoint", original_write)
    resumed = runner.execute(
        runner.parse_args(
            [
                "--concern",
                "grouping",
                "--execution-ref",
                "git:resume",
                "--output",
                str(output),
                "--trials",
                "1",
                "--dry-run",
                "--resume",
            ]
        ),
        workload=_workload("grouping"),
        evaluators=evaluators,
    )

    assert len(resumed.results) == 23 * len(REGISTERED_BINARY_TARGETS)
    assert sum(calls.values()) == len(resumed.results)


def test_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "identity.json"
    runner.execute(
        runner.parse_args(
            [
                "--concern",
                "confidence",
                "--execution-ref",
                "git:first",
                "--output",
                str(output),
                "--trials",
                "1",
                "--dry-run",
            ]
        ),
        workload=_workload("confidence"),
    )

    with pytest.raises(ValueError, match="identity"):
        runner.execute(
            runner.parse_args(
                [
                    "--concern",
                    "confidence",
                    "--execution-ref",
                    "git:other",
                    "--output",
                    str(output),
                    "--trials",
                    "1",
                    "--dry-run",
                    "--resume",
                ]
            ),
            workload=_workload("confidence"),
        )


def test_failed_judgment_is_retried_only_when_requested(tmp_path: Path) -> None:
    output = tmp_path / "retry.json"
    workload = _workload("confidence")
    calls: Counter[tuple[str, str]] = Counter()
    evaluators = _counting_evaluators(calls)
    target_id = REGISTERED_BINARY_TARGETS[0].target_id
    successful = evaluators[target_id]

    def fail_one(request: BinaryRequest, trial_ref: str) -> BinaryProbabilityObservation:
        observation = successful(request, trial_ref)
        if request.state == "state-000":
            raise RuntimeError("synthetic terminal error")
        return observation

    evaluators[target_id] = fail_one
    base_args = [
        "--concern",
        "confidence",
        "--execution-ref",
        "git:retry",
        "--output",
        str(output),
        "--trials",
        "1",
        "--dry-run",
    ]
    first = runner.execute(runner.parse_args(base_args), workload=workload, evaluators=evaluators)
    first_call_count = sum(calls.values())

    unchanged = runner.execute(
        runner.parse_args([*base_args, "--resume"]),
        workload=workload,
        evaluators=evaluators,
    )
    assert sum(calls.values()) == first_call_count
    assert sum(item.status == "failed" for item in first.results) == 1
    assert unchanged == first

    retried = runner.execute(
        runner.parse_args([*base_args, "--resume", "--retry-errors"]),
        workload=workload,
        evaluators=_counting_evaluators(calls),
    )
    assert sum(calls.values()) == first_call_count + 1
    assert all(item.status == "completed" for item in retried.results)


def test_live_execution_requires_protocol_three_trials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly three trials"):
        runner.execute(
            runner.parse_args(
                [
                    "--concern",
                    "confidence",
                    "--execution-ref",
                    "git:live",
                    "--output",
                    str(tmp_path / "live.json"),
                    "--trials",
                    "2",
                ]
            ),
            workload=_workload("confidence"),
            evaluators=_counting_evaluators(Counter()),
        )


def test_live_execution_validates_registered_call_shape_before_evaluation(
    tmp_path: Path,
) -> None:
    workload = _workload("confidence")
    malformed = DerivedBinaryWorkload(
        definition=workload.definition,
        source_artifact_id=workload.source_artifact_id,
        declared_manifest_version=workload.declared_manifest_version,
        cases=workload.cases[:-1],
    )
    calls: Counter[tuple[str, str]] = Counter()

    with pytest.raises(ValueError, match="case count"):
        runner.execute(
            runner.parse_args(
                [
                    "--concern",
                    "confidence",
                    "--execution-ref",
                    "git:live-shape",
                    "--output",
                    str(tmp_path / "live-shape.json"),
                ]
            ),
            workload=malformed,
            evaluators=_counting_evaluators(calls),
        )

    assert not calls


def _workload(concern: BenchmarkId) -> DerivedBinaryWorkload:
    definition = DERIVED_BINARY_BENCHMARKS[concern]
    judgments_per_case = (
        (3, 3, 2, 2, 9, 8)
        if concern == "daily_theme"
        else tuple(len(definition.questions) for _ in range(definition.case_count))
    )
    return DerivedBinaryWorkload(
        definition=definition,
        source_artifact_id="source-v11",
        declared_manifest_version="manifest-v11",
        cases=tuple(
            BinaryBenchmarkCase(
                case_id=f"{concern}-{index:03d}",
                identity=(f"identity-{index:03d}",),
                control=index == 0,
                judgments=tuple(
                    BinaryJudgmentCase(
                        judgment_id=(
                            question.question_id
                            if count == 1
                            else f"{question.question_id}-{judgment_index:03d}"
                        ),
                        request=_request(question, f"state-{index:03d}"),
                        expected=index % 2 == 0,
                    )
                    for judgment_index in range(count)
                    for question in (
                        definition.questions[judgment_index % len(definition.questions)],
                    )
                ),
            )
            for index, count in enumerate(judgments_per_case)
        ),
    )


def _concern_source(concern: BenchmarkId) -> DerivedBinarySource:
    if concern == "grouping":
        source = grouping_source()
        case = source.cases[0]
        return source.model_copy(
            update={
                "cases": tuple(
                    case.model_copy(update={"case_id": f"grouping-{index:03d}"})
                    for index in range(23)
                )
            }
        )
    if concern == "ranking":
        return ranking_source()
    if concern == "tier":
        return tier_source(tier_cases())
    if concern == "confidence":
        return confidence_source()
    manifest, inputs = _manifest_and_inputs()
    return BinaryDailyThemeSource(
        pin=_pin(),
        manifest_reference=reference(9_000, "evaluation-manifest").model_copy(
            update={"version_id": V11_SOURCE_ARTIFACT_ID}
        ),
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=tuple(
            build_daily_theme_binary_case(spec, inputs[spec.cluster_set.version_id])
            for spec in manifest.cases
            if isinstance(spec, ThemeEvaluationDaySpec)
        ),
    )


def _request(question: BinaryQuestion, state: str) -> BinaryRequest:
    return BinaryRequest(
        question=question,
        state=state,
        state_digest=binary_state_digest(state),
    )


def _counting_evaluators(
    calls: Counter[tuple[str, str]],
) -> dict[TargetId, BinaryEvaluator]:
    evaluators: dict[TargetId, BinaryEvaluator] = {}
    for target in REGISTERED_BINARY_TARGETS:

        def evaluate(
            request: BinaryRequest,
            trial_ref: str,
            *,
            _target: BinaryTarget = target,
            **_kwargs: object,
        ) -> BinaryProbabilityObservation:
            calls[_target.target_id, request.state_digest] += 1
            return BinaryProbabilityObservation(
                request_id=request.state_digest,
                provider_request_id=f"test:{_target.target_id}:{trial_ref}",
                model=_target.requested_model,
                probability=Decimal("0.75"),
                predicted_accepted=True,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                estimated_cost_usd=Decimal(0),
            )

        evaluators[target.target_id] = evaluate
    return evaluators
