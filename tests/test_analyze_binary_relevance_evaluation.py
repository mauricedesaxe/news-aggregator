from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryAttemptEvidence,
)
from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_25_TARGET,
    OPENROUTER_GEMINI_38_TARGET,
    TYPESAFE_JEV_TARGET,
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRelevanceCaseResult,
    BinaryRelevanceEvaluationResult,
    BinaryRelevanceRunResult,
    BinaryRelevanceTarget,
    binary_relevance_metrics,
)

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_binary_relevance_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("analyze_binary_relevance_evaluation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def test_exact_statistics_match_known_values() -> None:
    no_successes = analyzer.exact_clopper_pearson(0, 10)
    all_successes = analyzer.exact_clopper_pearson(10, 10)

    assert no_successes["lower"] == 0
    assert no_successes["upper"] == pytest.approx(0.3084971078)
    assert all_successes["lower"] == pytest.approx(0.6915028922)
    assert all_successes["upper"] == 1
    assert analyzer.exact_mcnemar_p_value(1, 9) == pytest.approx(0.021484375)
    assert analyzer.exact_mcnemar_p_value(0, 0) == 1


def test_analyzer_reports_registered_gates_pairing_stability_and_reruns(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "evaluation.json"
    artifact.write_text(_evaluation().model_dump_json(indent=2))

    report = analyzer.analyze(artifact)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    analyzer.write_reports(report, json_output, markdown_output)
    first_json = json_output.read_bytes()
    first_markdown = markdown_output.read_bytes()
    analyzer.write_reports(report, json_output, markdown_output)

    assert json_output.read_bytes() == first_json
    assert markdown_output.read_bytes() == first_markdown
    assert len(report["trials"]) == 9
    assert len(report["paired_comparisons"]) == 9
    assert all(
        trial["tp"] + trial["tn"] + trial["fp"] + trial["fn"] == 174 for trial in report["trials"]
    )
    assert all("precision_95_exact_ci" in trial for trial in report["trials"])
    assert all("p95_article_latency_ms" in trial for trial in report["trials"])
    gates = {item["target_id"]: item for item in report["gates"]["targets"]}
    assert gates["openrouter-gemini-2.5-flash"]["quality_pass"] is False
    assert gates["openrouter-gemini-3.8-flash"]["deployment_eligible"] is True
    assert gates["typesafe-jev"]["deployment_eligible"] is True
    stability = {item["target_id"]: item for item in report["stability"]}
    assert stability["openrouter-gemini-2.5-flash"]["stable_false_negative_ids"] == ["case-000"]
    assert stability["typesafe-jev"]["unstable_ids"] == ["case-121"]
    assert report["pareto"]["targets"][0]["deployment_eligible"] is False
    assert "Repeated trials" not in markdown_output.read_text()
    assert "repeated trials" in markdown_output.read_text()


def test_analyzer_strictly_rejects_coerced_case_values(tmp_path: Path) -> None:
    payload = json.loads(_evaluation().model_dump_json())
    payload["runs"][0]["cases"][0]["input_tokens"] = "2"
    artifact = tmp_path / "coerced.json"
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValidationError):
        analyzer.analyze(artifact)


def test_analyzer_rejects_incomplete_registered_shape(tmp_path: Path) -> None:
    result = _evaluation()
    shortened_runs = tuple(
        run.model_copy(
            update={
                "cases": run.cases[:-1],
                "metrics": binary_relevance_metrics(run.cases[:-1]),
            }
        )
        for run in result.runs
    )
    shortened = result.model_copy(update={"case_ids": result.case_ids[:-1], "runs": shortened_runs})
    artifact = tmp_path / "incomplete.json"
    artifact.write_text(shortened.model_dump_json())

    with pytest.raises(ValueError, match="exactly 174 cases"):
        analyzer.analyze(artifact)


def _evaluation() -> BinaryRelevanceEvaluationResult:
    targets = (
        OPENROUTER_GEMINI_25_TARGET,
        OPENROUTER_GEMINI_38_TARGET,
        TYPESAFE_JEV_TARGET,
    )
    trial_refs = ("test:trial-001", "test:trial-002", "test:trial-003")
    case_ids = tuple(f"case-{index:03d}" for index in range(174))
    runs = tuple(
        _run(target, trial_ref, trial_index, case_ids)
        for target in targets
        for trial_index, trial_ref in enumerate(trial_refs)
    )
    return BinaryRelevanceEvaluationResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        question_id=RELEVANCE_BINARY_QUESTION.question_id,
        question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        targets=targets,
        trial_refs=trial_refs,
        case_ids=case_ids,
        runs=runs,
    )


def _run(
    target: BinaryRelevanceTarget,
    trial_ref: str,
    trial_index: int,
    case_ids: tuple[str, ...],
) -> BinaryRelevanceRunResult:
    cases = tuple(
        _case(target, trial_ref, trial_index, case_id, index)
        for index, case_id in enumerate(case_ids)
    )
    return BinaryRelevanceRunResult(
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        target=target,
        trial_ref=trial_ref,
        question_id=RELEVANCE_BINARY_QUESTION.question_id,
        question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        cases=cases,
        metrics=binary_relevance_metrics(cases),
    )


def _case(
    target: BinaryRelevanceTarget,
    trial_ref: str,
    trial_index: int,
    case_id: str,
    case_index: int,
) -> BinaryRelevanceCaseResult:
    expected = case_index < 120
    verdict = expected
    if target.target_id == "openrouter-gemini-2.5-flash" and case_index == 0:
        verdict = False
    if target.target_id == "typesafe-jev" and case_index == 121 and trial_index == 1:
        verdict = True
    probability = Decimal("0.9") if verdict else Decimal("0.1")
    request_id = f"{case_index + trial_index:064x}"[-64:]
    provider_request_id = f"provider:{target.target_id}:{trial_ref}:{case_id}"
    attempt = BinaryAttemptEvidence(
        attempt_number=1,
        status="completed",
        provider_request_id=provider_request_id,
        actual_model=target.requested_model,
        http_status=200,
        error=None,
        input_tokens=2,
        output_tokens=1,
        cost_usd=Decimal("0.003"),
        latency_ms=10 + case_index,
        probability=probability,
    )
    return BinaryRelevanceCaseResult(
        status="completed",
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        case_id=case_id,
        article_version_id=f"{case_index + 1:064x}",
        target_id=target.target_id,
        provider=target.provider,
        trial_ref=trial_ref,
        requested_model=target.requested_model,
        actual_model=target.requested_model,
        question_id=RELEVANCE_BINARY_QUESTION.question_id,
        question_digest=RELEVANCE_BINARY_QUESTION.semantic_digest,
        state_digest=f"{case_index + 2:064x}",
        execution_policy_id=target.execution_policy_id,
        execution_policy_digest=target.execution_policy_digest,
        request_id=request_id,
        adapter_request_id=f"{case_index + 3:064x}",
        provider_request_id=provider_request_id,
        probability=probability,
        verdict=verdict,
        expected=expected,
        passed=verdict == expected,
        control=expected,
        input_tokens=2,
        output_tokens=1,
        wall_latency_ms=10 + case_index,
        cost_usd=Decimal("0.003"),
        cost_basis=target.cost_basis,
        errors=(),
        attempts=(attempt,),
    )
