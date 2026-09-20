from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.evaluation import EvaluationCaseResult, SubjectAssessmentEvaluationResult
from romanian_news.split_subject_assessment_experiment import (
    V11_MANIFEST_VERSION_ID,
    V11_REPORT_VERSION_IDS,
    CompletedArm,
    ExperimentAttempt,
    UnavailableArm,
)
from romanian_news.subject_assessments import SubjectAssessment, SubjectAssessmentEvidence
from scripts.analyze_split_subject_assessment_experiment import analyze, write_reports
from scripts.run_split_subject_assessment_experiment import (
    ARM_ORDER,
    SPEND_CEILING_USD,
    TRIAL_REFS,
    ArmRecord,
    InFlightArm,
    SplitExperimentResult,
    _identity,
)

REPORT_ID = V11_REPORT_VERSION_IDS[0]
SUBJECT_ID = "1" * 64


def test_completed_pairs_report_quality_whole_run_cost_latency_and_gates(
    tmp_path: Path,
) -> None:
    result = _result()
    source = _write_result(tmp_path, result)

    report = analyze(source)
    architectures = {item["architecture"]: item for item in report["architectures"]}
    candidate = architectures["candidate"]
    first_pair = report["trials"][0]["paired_delta"]

    assert report["execution_complete"] is True
    assert candidate["terminal_arms"] == candidate["completed_arms"] == 3
    assert candidate["quality"] == {
        "passes": 9,
        "total": 9,
        "tier_passes": 6,
        "tier_total": 6,
        "ranking_passes": 3,
        "ranking_total": 3,
        "control_passes": 3,
        "control_total": 3,
        "regressions_against_incumbent": 0,
        "control_regressions_against_incumbent": 0,
    }
    assert first_pair["passes"] == 0
    assert first_pair["regression_case_ids"] == []
    assert candidate["wall_latency"]["per_trial_ms"] == [
        {"trial_ref": TRIAL_REFS[0], "value": 200},
        {"trial_ref": TRIAL_REFS[1], "value": 400},
        {"trial_ref": TRIAL_REFS[2], "value": 600},
    ]
    assert candidate["wall_latency"]["p50_ms"] == 400
    assert candidate["wall_latency"]["p95_ms"] == 580
    assert candidate["wall_latency"]["total_ms"] == 1200
    assert candidate["cost"]["known_cost_usd"] == "0.033"
    assert candidate["cost"]["known_cost_by_stage_usd"] == {
        "tier": "0.003",
        "ranking": "0.030",
        "incumbent": "0",
    }
    assert candidate["structural_validity"]["valid_completed_outputs"] == 3
    assert all(
        item["valid_rationales"] and item["valid_evidence_ownership"]
        for item in candidate["structural_validity"]["per_trial"]
    )
    assert candidate["diagnostics"] == {
        "scored_completed_arms": 3,
        "per_tier_recall": {
            "main": {"passes": 3, "total": 3, "recall": 1.0},
            "worth_knowing": {"passes": 3, "total": 3, "recall": 1.0},
            "excluded": {"passes": 0, "total": 0, "recall": None},
        },
        "ranking_inversion_count": 0,
        "ranking_unavailable_count": 0,
        "merged_pair_count": 3,
        "semantic_tie_count": 3,
        "merged_tier_resolution_count": 3,
    }
    assert report["promotion"]["status"] == "pass"
    assert report["promotion"]["promotion_permitted"] is True
    assert report["uncertainty"]["status"] == "descriptive"
    assert report["uncertainty"]["resample_count"] == 27
    assert report["uncertainty"]["intervals"]["quality_pass_delta_per_trial"] == {
        "mean": 0.0,
        "lower": 0.0,
        "upper": 0.0,
    }
    assert report["uncertainty"]["intervals"]["wall_latency_delta_ms_per_trial"] == {
        "mean": 200.0,
        "lower": 121.666666666667,
        "upper": 278.333333333333,
    }

    json_output = tmp_path / "analysis.json"
    markdown_output = tmp_path / "analysis.md"
    write_reports(report, json_output, markdown_output)
    first = (json_output.read_bytes(), markdown_output.read_bytes())
    write_reports(report, json_output, markdown_output)
    assert (json_output.read_bytes(), markdown_output.read_bytes()) == first
    assert "Whole cost USD" in markdown_output.read_text()


def test_candidate_unavailable_is_worst_paired_quality_not_dropped(tmp_path: Path) -> None:
    result = _result(candidate_unavailable_trial=TRIAL_REFS[1])
    report = analyze(_write_result(tmp_path, result))
    candidate = next(
        item for item in report["architectures"] if item["architecture"] == "candidate"
    )
    paired = report["trials"][1]["paired_delta"]

    assert candidate["terminal_arms"] == 3
    assert candidate["completed_arms"] == 2
    assert candidate["unavailable_arms"] == 1
    assert candidate["quality"]["passes"] == 6
    assert candidate["quality"]["total"] == 9
    assert paired["passes"] == -3
    assert paired["tier_passes"] == -2
    assert paired["ranking_passes"] == -1
    assert paired["regression_case_ids"] == ["tier-control", "tier-case", "ranking-case"]
    assert paired["control_regression_case_ids"] == ["tier-control"]
    assert report["promotion"]["status"] == "fail"
    assert report["promotion"]["promotion_permitted"] is False


def test_incomplete_accounting_reports_explicit_lower_bound(tmp_path: Path) -> None:
    result = _result(unknown_in_flight=True)
    report = analyze(_write_result(tmp_path, result))
    candidate = next(
        item for item in report["architectures"] if item["architecture"] == "candidate"
    )
    accounting_gate = next(
        item
        for item in report["promotion"]["gates"]
        if item["gate"] == "no_omitted_provider_attempt_or_accounting_gap"
    )

    assert report["accounting"]["unknown_in_flight_count"] == 1
    assert report["accounting"]["cost_statement"] == "lower bound; accounting is incomplete"
    assert candidate["cost"]["known_cost_usd"] == "0.034"
    assert candidate["cost"]["wording"] == "lower bound; accounting is incomplete"
    assert accounting_gate["status"] == "fail"
    assert report["promotion"]["promotion_permitted"] is False


def test_request_attempt_retry_correction_and_failure_counts(tmp_path: Path) -> None:
    result = _result(with_retries=True)
    report = analyze(_write_result(tmp_path, result))
    candidate = next(
        item for item in report["architectures"] if item["architecture"] == "candidate"
    )
    requests = candidate["requests"]

    assert requests["attempt_count"] == 8
    assert requests["request_count"] == 6
    assert requests["retry_count"] == 1
    assert requests["correction_count"] == 1
    assert requests["failure_count"] == 2
    assert requests["by_stage_status"]["tier"]["retryable_error"] == 1
    assert requests["by_stage_status"]["ranking"]["rejected"] == 1
    assert requests["request_count_by_stage_status"]["tier"]["retryable_error"] == 1
    assert requests["request_count_by_stage_status"]["ranking"]["accepted"] == 2


def test_percentile_interpolates_two_values_and_incomplete_execution_blocks_promotion(
    tmp_path: Path,
) -> None:
    result = _result().model_copy(update={"records": _result().records[:-2]})
    result = _refresh_accounting(result)
    report = analyze(_write_result(tmp_path, result))
    candidate = next(
        item for item in report["architectures"] if item["architecture"] == "candidate"
    )

    assert candidate["wall_latency"]["p50_ms"] == 300
    assert candidate["wall_latency"]["p95_ms"] == 390
    assert report["execution_complete"] is False
    assert report["promotion"]["status"] == "inconclusive"
    assert report["promotion"]["promotion_permitted"] is False
    assert report["uncertainty"]["status"] == "inconclusive"
    assert report["uncertainty"]["intervals"] == {}


def test_rejects_identity_and_protocol_mismatch(tmp_path: Path) -> None:
    bad_identity = _result().model_copy(update={"identity_digest": "f" * 64})
    with pytest.raises(ValueError, match="identity or frozen protocol"):
        analyze(_write_result(tmp_path, bad_identity))

    source = _write_result(tmp_path, _result())
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": "split-subject-assessment-experiment/v2"}))
    with pytest.raises(ValueError, match="Protocol content"):
        analyze(source, protocol)


def _result(
    *,
    candidate_unavailable_trial: str | None = None,
    unknown_in_flight: bool = False,
    with_retries: bool = False,
) -> SplitExperimentResult:
    records: list[ArmRecord] = []
    for index, trial_ref in enumerate(TRIAL_REFS, start=1):
        records.append(
            ArmRecord(
                trial_ref=trial_ref,
                arm="incumbent",
                outcome=CompletedArm(
                    arm="incumbent",
                    report_version_id=REPORT_ID,
                    wall_latency_ms=index * 100,
                    attempts=(_attempt("incumbent", "incumbent", cost="0.020"),),
                    assessments=(_assessment(),),
                    accounting_complete=True,
                ),
                score=_score(),
            )
        )
        candidate_attempts = (
            _retry_attempts() if with_retries and index == 1 else _candidate_attempts(index)
        )
        if trial_ref == candidate_unavailable_trial:
            outcome = UnavailableArm(
                report_version_id=REPORT_ID,
                wall_latency_ms=index * 200,
                attempts=candidate_attempts,
                reason="provider unavailable",
                accounting_complete=all(item.accounting_complete for item in candidate_attempts),
            )
            score = None
        else:
            outcome = CompletedArm(
                arm="candidate",
                report_version_id=REPORT_ID,
                wall_latency_ms=index * 200,
                attempts=candidate_attempts,
                assessments=(_assessment(),),
                accounting_complete=all(item.accounting_complete for item in candidate_attempts),
            )
            score = _score()
        records.append(
            ArmRecord(trial_ref=trial_ref, arm="candidate", outcome=outcome, score=score)
        )
    in_flight = (
        (
            InFlightArm(
                trial_ref=TRIAL_REFS[0],
                arm="candidate",
                report_version_id=REPORT_ID,
                attempts=(_attempt("candidate-unknown", "tier", cost="0.001"),),
            ),
        )
        if unknown_in_flight
        else ()
    )
    attempts = [attempt for record in records for attempt in record.outcome.attempts]
    attempts.extend(attempt for value in in_flight for attempt in value.attempts)
    accounting_complete = not in_flight and all(item.accounting_complete for item in attempts)
    return SplitExperimentResult(
        identity_digest=_identity("analysis-test", False),
        execution_ref="analysis-test",
        mode="live",
        manifest_version_id=V11_MANIFEST_VERSION_ID,
        report_version_ids=V11_REPORT_VERSION_IDS,
        trial_refs=TRIAL_REFS,
        arm_order=ARM_ORDER,
        spend_ceiling_usd=SPEND_CEILING_USD,
        known_spend_usd=sum((item.cost_usd or Decimal(0) for item in attempts), Decimal(0)),
        accounting_complete=accounting_complete,
        records=tuple(records),
        unknown_in_flight=in_flight,
        unknown_accounting=bool(in_flight),
        crash_window="test crash window",
    )


def _refresh_accounting(result: SplitExperimentResult) -> SplitExperimentResult:
    attempts = [attempt for record in result.records for attempt in record.outcome.attempts]
    attempts.extend(attempt for value in result.unknown_in_flight for attempt in value.attempts)
    return result.model_copy(
        update={
            "known_spend_usd": sum((item.cost_usd or Decimal(0) for item in attempts), Decimal(0)),
            "accounting_complete": (
                not result.unknown_in_flight and all(item.accounting_complete for item in attempts)
            ),
        }
    )


def _score() -> SubjectAssessmentEvaluationResult:
    cases = (
        _case(
            "tier-control",
            "tier",
            control=True,
            detail="expected tier=main, assessed tier=main",
        ),
        _case(
            "tier-case",
            "tier",
            detail="expected tier=worth_knowing, assessed tier=worth_knowing",
        ),
        _case("ranking-case", "ranking"),
    )
    return SubjectAssessmentEvaluationResult(
        case_results=cases,
        merged_tier_resolutions=("merged",),
        merged_pair_count=1,
        semantic_tie_count=1,
        passed_cases=3,
        total_cases=3,
    )


def _case(
    case_id: str,
    concern: str,
    *,
    control: bool = False,
    detail: str = "passed",
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        case_id=case_id,
        concern=concern,
        passed=True,
        control=control,
        expected_positive=True,
        predicted_positive=True,
        detail=detail,
    )


def _candidate_attempts(index: int) -> tuple[ExperimentAttempt, ...]:
    return (
        _attempt(f"candidate-tier-{index}", "tier", cost="0.001"),
        _attempt(f"candidate-ranking-{index}", "ranking", cost="0.010"),
    )


def _retry_attempts() -> tuple[ExperimentAttempt, ...]:
    return (
        _attempt(
            "candidate-tier-retry",
            "tier",
            status="retryable_error",
            attempt_number=1,
            cost="0.001",
        ),
        _attempt(
            "candidate-tier-retry",
            "tier",
            attempt_number=2,
            cost="0.001",
        ),
        _attempt(
            "candidate-ranking-correction",
            "ranking",
            status="rejected",
            attempt_number=1,
            cost="0.010",
        ),
        _attempt(
            "candidate-ranking-correction",
            "ranking",
            attempt_number=2,
            cost="0.010",
        ),
    )


def _attempt(
    request: str,
    stage: str,
    *,
    status: str = "accepted",
    attempt_number: int = 1,
    cost: str,
) -> ExperimentAttempt:
    return ExperimentAttempt.model_validate(
        {
            "provider": "typesafe" if stage == "tier" else "openrouter",
            "stage": stage,
            "subject_id": SUBJECT_ID if stage == "tier" else None,
            "attempt_number": attempt_number,
            "status": status,
            "request_id": hashlib_sha256(request),
            "provider_request_id": f"provider:{request}:{attempt_number}",
            "actual_model": "test-model",
            "input_tokens": 10,
            "output_tokens": 2,
            "cost_usd": Decimal(cost),
            "latency_ms": 10,
            "error": None,
        },
        strict=True,
    )


def _assessment() -> SubjectAssessment:
    group_id = "2" * 64
    article_id = "3" * 64
    reference = ArtifactReference(
        artifact_id="test",
        version_id=article_id,
        content_digest="4" * 64,
        r2_key="test.json",
    )
    evidence = SubjectAssessmentEvidence(
        group_id=group_id,
        article=reference,
        relevance=reference,
        summary=reference,
        evidence_quote="Evidence.",
    )
    return SubjectAssessment(
        theme_id=SUBJECT_ID,
        group_ids=(group_id,),
        article_version_ids=(article_id,),
        tier="main",
        semantic_rank=1,
        rationale="Rationale.",
        evidence=(evidence,),
    )


def _write_result(tmp_path: Path, result: SplitExperimentResult) -> Path:
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


def hashlib_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
