#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from romanian_news.binary_benchmark import atomic_write
from romanian_news.evaluation import SubjectAssessmentEvaluationResult
from romanian_news.split_subject_assessment_experiment import (
    V11_MANIFEST_VERSION_ID,
    V11_REPORT_VERSION_IDS,
    CompletedArm,
    ExperimentAttempt,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.run_split_subject_assessment_experiment import (
    ARM_ORDER,
    SPEND_CEILING_USD,
    TRIAL_REFS,
    ArmName,
    ArmRecord,
    SplitExperimentResult,
    _identity,
)

PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "romanian_news"
    / "split_assessment_experiment_protocol.json"
)
PROTOCOL_SHA256 = "652f1e2325e3d7a8ea9f9c4fe21b3ab7f804008f6681a52b6ea61d0c63a6488f"
ARCHITECTURES: tuple[ArmName, ...] = ("incumbent", "candidate")
STAGES = ("tier", "ranking", "incumbent")
STATUSES = ("accepted", "rejected", "retryable_error", "terminal_error")
TIERS = ("main", "worth_knowing", "excluded")
GateStatus = Literal["pass", "fail", "inconclusive"]
_TIER_DETAIL = re.compile(
    r"^expected tier=(main|worth_knowing|excluded), assessed tier=(main|worth_knowing|excluded)$"
)


@dataclass(frozen=True)
class Arguments:
    input_path: Path
    json_output: Path
    markdown_output: Path


class TierRecall(TypedDict):
    passes: int
    total: int
    recall: float | None


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(
        description="Analyze one final frozen split subject-assessment experiment result."
    )
    _ = parser.add_argument("--input", required=True, type=Path)
    _ = parser.add_argument("--json-output", required=True, type=Path)
    _ = parser.add_argument("--markdown-output", required=True, type=Path)
    parsed = parser.parse_args(argv)
    return Arguments(
        input_path=cast(Path, parsed.input),
        json_output=cast(Path, parsed.json_output),
        markdown_output=cast(Path, parsed.markdown_output),
    )


def analyze(source_path: Path, protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    source_content = source_path.read_bytes()
    result = SplitExperimentResult.model_validate_json(source_content, strict=True)
    protocol_content = protocol_path.read_bytes()
    protocol = _require_protocol(protocol_content)
    _require_exact_result(result)

    records: dict[tuple[str, ArmName], ArmRecord] = {
        (record.trial_ref, record.arm): record for record in result.records
    }
    trial_rows = [_trial_analysis(trial_ref, records, protocol) for trial_ref in TRIAL_REFS]
    architecture_rows = [_architecture_analysis(arm, result, trial_rows) for arm in ARCHITECTURES]
    gates = _promotion_gates(result, trial_rows, architecture_rows)
    return {
        "schema_version": "split-subject-assessment-analysis-v1",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": hashlib.sha256(source_content).hexdigest(),
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": hashlib.sha256(protocol_content).hexdigest(),
            "schema_version": protocol["schema_version"],
        },
        "identity": {
            "identity_digest": result.identity_digest,
            "execution_ref": result.execution_ref,
            "mode": result.mode,
            "manifest_version_id": result.manifest_version_id,
            "report_version_ids": list(result.report_version_ids),
            "trial_refs": list(result.trial_refs),
            "arm_order": [list(order) for order in result.arm_order],
        },
        "execution_complete": len(result.records) == len(TRIAL_REFS) * len(ARCHITECTURES),
        "trials": trial_rows,
        "architectures": architecture_rows,
        "accounting": {
            "runner_known_spend_usd": str(result.known_spend_usd),
            "runner_accounting_complete": result.accounting_complete,
            "unknown_in_flight_count": len(result.unknown_in_flight),
            "unknown_in_flight_accounting": result.unknown_accounting,
            "cost_statement": _cost_statement(result.accounting_complete),
        },
        "uncertainty": _uncertainty_analysis(trial_rows),
        "promotion": gates,
    }


def write_reports(report: dict[str, Any], json_output: Path, markdown_output: Path) -> None:
    if json_output.resolve() == markdown_output.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    atomic_write(
        json_output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n",
    )
    atomic_write(markdown_output, render_markdown(report).encode())


def render_markdown(report: dict[str, Any]) -> str:
    identity = cast(dict[str, Any], report["identity"])
    promotion = cast(dict[str, Any], report["promotion"])
    lines = [
        "# Split subject-assessment experiment analysis",
        "",
        f"- Execution: `{identity['execution_ref']}` ({identity['mode']})",
        f"- Identity: `{identity['identity_digest']}`",
        f"- Execution complete: {_yes_no(report['execution_complete'])}",
        f"- Promotion status: **{promotion['status']}**",
        "",
        "## Architecture summary",
        "",
        "| Architecture | Terminal | Completed | Failed | Unavailable | Quality | Tier | Ranking | Controls | Regressions | Structural | Wall p50/p95 ms | Whole cost USD | Accounting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for value in cast(list[dict[str, Any]], report["architectures"]):
        quality = cast(dict[str, Any], value["quality"])
        latency = cast(dict[str, Any], value["wall_latency"])
        cost = cast(dict[str, Any], value["cost"])
        structural = cast(dict[str, Any], value["structural_validity"])
        lines.append(
            f"| `{value['architecture']}` | {value['terminal_arms']} | {value['completed_arms']} | "
            f"{value['failed_arms']} | {value['unavailable_arms']} | "
            f"{quality['passes']}/{quality['total']} | {quality['tier_passes']}/{quality['tier_total']} | "
            f"{quality['ranking_passes']}/{quality['ranking_total']} | "
            f"{quality['control_passes']}/{_display(quality['control_total'])} | "
            f"{quality['regressions_against_incumbent']} | "
            f"{structural['valid_completed_outputs']}/{structural['completed_typed_outputs']} | "
            f"{_display(latency['p50_ms'])}/{_display(latency['p95_ms'])} | "
            f"{cost['known_cost_usd']} | {cost['wording']} |"
        )

    lines.extend(
        [
            "",
            "## Paired trials",
            "",
            "| Trial | Incumbent status | Incumbent quality | Candidate status | Candidate quality | Pass delta | Tier delta | Ranking delta | Control regressions | Wall ms I/C |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for value in cast(list[dict[str, Any]], report["trials"]):
        paired = cast(dict[str, Any], value["paired_delta"])
        arms = cast(dict[str, dict[str, Any]], value["arms"])
        lines.append(
            f"| `{value['trial_ref']}` | {arms['incumbent']['status']} | "
            f"{_quality_cell(arms['incumbent']['quality'])} | {arms['candidate']['status']} | "
            f"{_quality_cell(arms['candidate']['quality'])} | "
            f"{_signed(paired['passes'])} | {_signed(paired['tier_passes'])} | "
            f"{_signed(paired['ranking_passes'])} | {_display(paired['control_regression_count'])} | "
            f"{_display(arms['incumbent']['wall_latency_ms'])}/{_display(arms['candidate']['wall_latency_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Quality diagnostics",
            "",
            "| Architecture | Per-tier recall (main/worth/excluded) | Ranking inversions | Ranking unavailable | Merged anchor pairs | Semantic ties | Merged tier resolutions |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for value in cast(list[dict[str, Any]], report["architectures"]):
        diagnostics = cast(dict[str, Any], value["diagnostics"])
        recall = cast(dict[str, TierRecall], diagnostics["per_tier_recall"])
        recall_text = "/".join(_recall_cell(recall[tier]) for tier in TIERS)
        lines.append(
            f"| `{value['architecture']}` | {recall_text} | "
            f"{diagnostics['ranking_inversion_count']} | {diagnostics['ranking_unavailable_count']} | "
            f"{diagnostics['merged_pair_count']} | {diagnostics['semantic_tie_count']} | "
            f"{diagnostics['merged_tier_resolution_count']} |"
        )

    lines.extend(
        [
            "",
            "## Requests and cost",
            "",
            "| Architecture | Requests | Attempts | Accepted/rejected/retryable/terminal | Retries | Corrections | Failures | Attempt latency ms | Tier/ranking/incumbent cost USD | Unknown in-flight |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for value in cast(list[dict[str, Any]], report["architectures"]):
        requests = cast(dict[str, Any], value["requests"])
        by_status = cast(dict[str, dict[str, int]], requests["by_stage_status"])
        cost = cast(dict[str, Any], value["cost"])
        by_stage_cost = cast(dict[str, str], cost["known_cost_by_stage_usd"])
        status_counts = [sum(by_status[stage][status] for stage in STAGES) for status in STATUSES]
        lines.append(
            f"| `{value['architecture']}` | {requests['request_count']} | {requests['attempt_count']} | "
            f"{'/'.join(str(count) for count in status_counts)} | {requests['retry_count']} | "
            f"{requests['correction_count']} | {requests['failure_count']} | "
            f"{requests['provider_attempt_latency_ms']} | "
            f"{by_stage_cost['tier']}/{by_stage_cost['ranking']}/{by_stage_cost['incumbent']} | "
            f"{cost['unknown_in_flight_count']} |"
        )

    lines.extend(["", "## Promotion gates", ""])
    for gate in cast(list[dict[str, Any]], promotion["gates"]):
        lines.append(f"- **{gate['status']}** `{gate['gate']}`. {gate['detail']}")
    uncertainty = cast(dict[str, Any], report["uncertainty"])
    lines.extend(["", "## Uncertainty", "", uncertainty["interpretation"], ""])
    for name, interval in cast(dict[str, dict[str, float | int]], uncertainty["intervals"]).items():
        lines.append(
            f"- `{name}`: mean {_display(interval['mean'])}; 95% interval "
            f"[{_display(interval['lower'])}, {_display(interval['upper'])}]"
        )
    lines.extend(
        [
            "",
            "## Accounting",
            "",
            f"{cast(dict[str, Any], report['accounting'])['cost_statement']}",
            "",
        ]
    )
    return "\n".join(lines)


def _trial_analysis(
    trial_ref: str,
    records: dict[tuple[str, ArmName], ArmRecord],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    incumbent = records.get((trial_ref, "incumbent"))
    candidate = records.get((trial_ref, "candidate"))
    fallback = _fallback_quality(incumbent, candidate, protocol)
    incumbent_row = _arm_trial_row(incumbent, fallback)
    candidate_row = _arm_trial_row(candidate, fallback)
    expected_subject_ids = _subject_ids(incumbent)
    incumbent_row["structural_validity"] = _structural_validity(incumbent, expected_subject_ids)
    candidate_row["structural_validity"] = _structural_validity(candidate, expected_subject_ids)
    return {
        "trial_ref": trial_ref,
        "arms": {"incumbent": incumbent_row, "candidate": candidate_row},
        "paired_delta": _paired_delta(incumbent, candidate, fallback),
    }


def _arm_trial_row(record: ArmRecord | None, fallback: dict[str, int | None]) -> dict[str, Any]:
    if record is None:
        return {
            "status": "missing",
            "wall_latency_ms": None,
            "quality": None,
            "attempts": _attempt_summary(()),
            "known_cost_usd": "0",
            "cost_wording": "not measured",
        }
    attempts = record.outcome.attempts
    complete_cost = record.outcome.accounting_complete
    return {
        "status": record.outcome.status,
        "wall_latency_ms": record.outcome.wall_latency_ms,
        "quality": _quality(record, fallback),
        "attempts": _attempt_summary(attempts),
        "known_cost_usd": str(_known_cost(attempts)),
        "cost_wording": _cost_statement(complete_cost),
    }


def _quality(record: ArmRecord, fallback: dict[str, int | None]) -> dict[str, int | None]:
    if record.score is None:
        return {
            "passes": 0,
            "total": fallback["total"],
            "tier_passes": 0,
            "tier_total": fallback["tier_total"],
            "ranking_passes": 0,
            "ranking_total": fallback["ranking_total"],
            "control_passes": 0,
            "control_total": fallback["control_total"],
        }
    return _score_quality(record.score)


def _score_quality(score: SubjectAssessmentEvaluationResult) -> dict[str, int | None]:
    tiers = tuple(item for item in score.case_results if item.concern == "tier")
    rankings = tuple(item for item in score.case_results if item.concern == "ranking")
    controls = tuple(item for item in score.case_results if item.control)
    return {
        "passes": score.passed_cases,
        "total": score.total_cases,
        "tier_passes": sum(item.passed for item in tiers),
        "tier_total": len(tiers),
        "ranking_passes": sum(item.passed for item in rankings),
        "ranking_total": len(rankings),
        "control_passes": sum(item.passed for item in controls),
        "control_total": len(controls),
    }


def _fallback_quality(
    incumbent: ArmRecord | None,
    candidate: ArmRecord | None,
    protocol: dict[str, Any],
) -> dict[str, int | None]:
    scored = next(
        (record.score for record in (incumbent, candidate) if record is not None and record.score),
        None,
    )
    if scored is not None:
        return _score_quality(scored)
    source = cast(dict[str, Any], protocol["source"])
    tier_total = cast(int, source["tier_case_count"])
    ranking_total = cast(int, source["ranking_case_count"])
    return {
        "passes": 0,
        "total": tier_total + ranking_total,
        "tier_passes": 0,
        "tier_total": tier_total,
        "ranking_passes": 0,
        "ranking_total": ranking_total,
        "control_passes": 0,
        "control_total": None,
    }


def _paired_delta(
    incumbent: ArmRecord | None,
    candidate: ArmRecord | None,
    fallback: dict[str, int | None],
) -> dict[str, Any]:
    if incumbent is None or candidate is None:
        return {
            "status": "incomplete",
            "passes": None,
            "tier_passes": None,
            "ranking_passes": None,
            "control_passes": None,
            "regression_case_ids": [],
            "improvement_case_ids": [],
            "control_regression_case_ids": [],
            "control_regression_count": None,
        }
    incumbent_quality = _quality(incumbent, fallback)
    candidate_quality = _quality(candidate, fallback)
    incumbent_cases = _case_outcomes(incumbent)
    candidate_cases = _case_outcomes(candidate)
    case_ids = tuple(dict.fromkeys((*incumbent_cases, *candidate_cases)))
    regressions = [
        case_id
        for case_id in case_ids
        if incumbent_cases.get(case_id, False) and not candidate_cases.get(case_id, False)
    ]
    improvements = [
        case_id
        for case_id in case_ids
        if candidate_cases.get(case_id, False) and not incumbent_cases.get(case_id, False)
    ]
    control_ids = {
        item.case_id
        for record in (incumbent, candidate)
        if record.score is not None
        for item in record.score.case_results
        if item.control
    }
    control_regressions = [case_id for case_id in regressions if case_id in control_ids]
    return {
        "status": "paired",
        "passes": _delta(candidate_quality, incumbent_quality, "passes"),
        "tier_passes": _delta(candidate_quality, incumbent_quality, "tier_passes"),
        "ranking_passes": _delta(candidate_quality, incumbent_quality, "ranking_passes"),
        "control_passes": _delta(candidate_quality, incumbent_quality, "control_passes"),
        "regression_case_ids": regressions,
        "improvement_case_ids": improvements,
        "control_regression_case_ids": control_regressions,
        "control_regression_count": len(control_regressions),
    }


def _architecture_analysis(
    arm: ArmName,
    result: SplitExperimentResult,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [record for record in result.records if record.arm == arm]
    unknown = [value for value in result.unknown_in_flight if value.arm == arm]
    attempts = [attempt for record in records for attempt in record.outcome.attempts]
    attempts.extend(attempt for value in unknown for attempt in value.attempts)
    qualities = [
        cast(dict[str, int | None], cast(dict[str, Any], trial["arms"])[arm]["quality"])
        for trial in trials
        if cast(dict[str, Any], trial["arms"])[arm]["quality"] is not None
    ]
    latencies = [record.outcome.wall_latency_ms for record in records]
    structural = [
        cast(dict[str, Any], cast(dict[str, Any], trial["arms"])[arm])["structural_validity"]
        for trial in trials
        if "structural_validity" in cast(dict[str, Any], trial["arms"])[arm]
    ]
    complete_accounting = (
        len(records) == len(TRIAL_REFS)
        and not unknown
        and all(record.outcome.accounting_complete for record in records)
    )
    quality = _sum_quality(qualities)
    diagnostics = _quality_diagnostics(
        record.score for record in records if record.score is not None
    )
    regression_count = sum(
        len(cast(list[str], cast(dict[str, Any], trial["paired_delta"])["regression_case_ids"]))
        for trial in trials
        if cast(dict[str, Any], trial["paired_delta"])["status"] == "paired"
    )
    control_regression_count = sum(
        cast(int, cast(dict[str, Any], trial["paired_delta"])["control_regression_count"])
        for trial in trials
        if cast(dict[str, Any], trial["paired_delta"])["control_regression_count"] is not None
    )
    quality.update(
        {
            "regressions_against_incumbent": regression_count if arm == "candidate" else 0,
            "control_regressions_against_incumbent": (
                control_regression_count if arm == "candidate" else 0
            ),
        }
    )
    return {
        "architecture": arm,
        "terminal_arms": len(records),
        "completed_arms": sum(record.outcome.status == "completed" for record in records),
        "failed_arms": sum(record.outcome.status == "failed" for record in records),
        "unavailable_arms": sum(record.outcome.status == "unavailable" for record in records),
        "missing_arms": len(TRIAL_REFS) - len(records),
        "quality": quality,
        "diagnostics": diagnostics,
        "wall_latency": {
            "per_trial_ms": [
                {
                    "trial_ref": trial_ref,
                    "value": next(
                        (
                            record.outcome.wall_latency_ms
                            for record in records
                            if record.trial_ref == trial_ref
                        ),
                        None,
                    ),
                }
                for trial_ref in TRIAL_REFS
            ],
            "observed_arms": len(latencies),
            "total_ms": sum(latencies),
            "p50_ms": _percentile(latencies, 0.50) if latencies else None,
            "p95_ms": _percentile(latencies, 0.95) if latencies else None,
            "complete": len(latencies) == len(TRIAL_REFS) and not unknown,
        },
        "requests": _attempt_summary(attempts),
        "cost": {
            "known_cost_usd": str(_known_cost(attempts)),
            "known_cost_by_stage_usd": {
                stage: str(_known_cost(item for item in attempts if item.stage == stage))
                for stage in STAGES
            },
            "accounting_complete": complete_accounting,
            "unknown_in_flight_count": len(unknown),
            "wording": _cost_statement(complete_accounting),
        },
        "structural_validity": {
            "completed_typed_outputs": sum(
                isinstance(record.outcome, CompletedArm) for record in records
            ),
            "valid_completed_outputs": sum(item["status"] == "pass" for item in structural),
            "invalid_completed_outputs": sum(item["status"] == "fail" for item in structural),
            "inconclusive_outputs": sum(item["status"] == "inconclusive" for item in structural),
            "per_trial": structural,
        },
    }


def _attempt_summary(attempts_value: Iterable[ExperimentAttempt]) -> dict[str, Any]:
    attempts = tuple(attempts_value)
    counts = Counter((item.stage, item.status) for item in attempts)
    return {
        "attempt_count": len(attempts),
        "request_count": sum(item.attempt_number == 1 for item in attempts),
        "by_stage_status": {
            stage: {status: counts[(stage, status)] for status in STATUSES} for stage in STAGES
        },
        "request_count_by_stage_status": {
            stage: {
                status: sum(
                    item.attempt_number == 1 and item.stage == stage and item.status == status
                    for item in attempts
                )
                for status in STATUSES
            }
            for stage in STAGES
        },
        "retry_count": sum(item.attempt_number > 1 and item.stage == "tier" for item in attempts),
        "correction_count": sum(
            item.attempt_number > 1 and item.stage in ("ranking", "incumbent") for item in attempts
        ),
        "failure_count": sum(
            item.status in ("rejected", "retryable_error", "terminal_error") for item in attempts
        ),
        "provider_attempt_latency_ms": sum(item.latency_ms for item in attempts),
    }


def _structural_validity(
    record: ArmRecord | None, expected_subject_ids: set[str] | None
) -> dict[str, Any]:
    if record is None:
        return {"status": "inconclusive", "reason": "arm is missing"}
    if not isinstance(record.outcome, CompletedArm):
        return {
            "status": "fail",
            "reason": f"arm ended {record.outcome.status} without a completed typed output",
        }
    assessments = record.outcome.assessments
    subject_ids = [item.theme_id for item in assessments]
    unique = bool(subject_ids) and len(subject_ids) == len(set(subject_ids))
    ranks = all(
        [item.semantic_rank for item in assessments if item.tier == tier]
        == list(range(1, sum(item.tier == tier for item in assessments) + 1))
        for tier in ("main", "worth_knowing", "excluded")
    )
    valid_rationales = all(item.rationale.strip() for item in assessments)
    valid_evidence_ownership = all(
        evidence.group_id in item.group_ids
        and evidence.article.version_id in item.article_version_ids
        and bool(evidence.evidence_quote.strip())
        for item in assessments
        for evidence in item.evidence
    )
    coverage = expected_subject_ids is not None and set(subject_ids) == expected_subject_ids
    if record.arm == "incumbent":
        coverage = unique
    valid = unique and ranks and coverage and valid_rationales and valid_evidence_ownership
    reason = (
        "completed typed output has unique exact subject coverage, contiguous tier ranks, "
        "valid rationales, and owned evidence"
        if valid
        else "completed output fails coverage, rank, rationale, or evidence ownership validation"
    )
    return {
        "status": "pass" if valid else "fail",
        "reason": reason,
        "subject_count": len(subject_ids),
        "unique_subjects": unique,
        "exact_subject_coverage": coverage,
        "contiguous_tier_ranks": ranks,
        "valid_rationales": valid_rationales,
        "valid_evidence_ownership": valid_evidence_ownership,
    }


def _promotion_gates(
    result: SplitExperimentResult,
    trials: list[dict[str, Any]],
    architectures: list[dict[str, Any]],
) -> dict[str, Any]:
    execution_complete = len(result.records) == len(TRIAL_REFS) * len(ARCHITECTURES)
    candidate = next(item for item in architectures if item["architecture"] == "candidate")
    incumbent = next(item for item in architectures if item["architecture"] == "incumbent")
    paired = [cast(dict[str, Any], trial["paired_delta"]) for trial in trials]
    candidate_terminal = candidate["terminal_arms"] == len(TRIAL_REFS)
    candidate_valid = candidate["completed_arms"] == len(TRIAL_REFS) and candidate[
        "structural_validity"
    ]["valid_completed_outputs"] == len(TRIAL_REFS)
    gates = [
        _gate(
            "zero_invalid_or_incomplete_candidate_assessments",
            "pass" if candidate_valid else "fail" if candidate_terminal else "inconclusive",
            f"candidate completed and structurally valid in {candidate['structural_validity']['valid_completed_outputs']}/{len(TRIAL_REFS)} trials",
        ),
        _gate(
            "zero_control_case_regressions",
            (
                "inconclusive"
                if any(item["status"] != "paired" for item in paired)
                else "pass"
                if sum(cast(int, item["control_regression_count"]) for item in paired) == 0
                else "fail"
            ),
            f"candidate control regressions: {sum(cast(int, item['control_regression_count']) for item in paired if item['control_regression_count'] is not None)}",
        ),
        _gate(
            "candidate_tier_and_ranking_passes_at_least_incumbent",
            (
                "inconclusive"
                if not execution_complete
                else "pass"
                if (
                    candidate["quality"]["tier_passes"] >= incumbent["quality"]["tier_passes"]
                    and candidate["quality"]["ranking_passes"]
                    >= incumbent["quality"]["ranking_passes"]
                )
                else "fail"
            ),
            "candidate versus incumbent aggregate tier/ranking passes: "
            f"{candidate['quality']['tier_passes']}/{incumbent['quality']['tier_passes']} and "
            f"{candidate['quality']['ranking_passes']}/{incumbent['quality']['ranking_passes']}",
        ),
        _gate(
            "no_omitted_provider_attempt_or_accounting_gap",
            (
                "inconclusive"
                if not execution_complete
                else "pass"
                if result.accounting_complete and not result.unknown_accounting
                else "fail"
            ),
            _cost_statement(result.accounting_complete and not result.unknown_accounting),
        ),
        _gate(
            "complete_run_latency_and_cost_include_failed_and_unavailable_arms",
            (
                "inconclusive"
                if not execution_complete
                else "pass"
                if not result.unknown_in_flight
                and all(item["wall_latency"]["complete"] for item in architectures)
                else "fail"
            ),
            "all terminal arm wall latencies and every captured attempt are included"
            if execution_complete and not result.unknown_in_flight
            else "execution or in-flight latency evidence is incomplete",
        ),
    ]
    statuses = [cast(GateStatus, item["status"]) for item in gates]
    status: GateStatus = (
        "fail" if "fail" in statuses else "inconclusive" if "inconclusive" in statuses else "pass"
    )
    return {
        "status": status,
        "promotion_permitted": status == "pass" and result.mode == "live",
        "interpretation": "Passing permits a separate guarded rollout proposal; it does not change production.",
        "gates": gates,
    }


def _require_protocol(content: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(content).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Protocol content does not match the frozen split experiment protocol")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Protocol must be a JSON object")
    protocol = cast(dict[str, Any], value)
    execution = _object(protocol.get("execution"), "protocol execution")
    source = _object(protocol.get("source"), "protocol source")
    if (
        protocol.get("schema_version") != "split-subject-assessment-experiment/v1"
        or source.get("manifest_version_id") != V11_MANIFEST_VERSION_ID
        or tuple(source.get("report_version_ids", ())) != V11_REPORT_VERSION_IDS
        or tuple(execution.get("trial_refs", ())) != TRIAL_REFS
        or tuple(tuple(item) for item in execution.get("arm_order", ())) != ARM_ORDER
        or Decimal(str(execution.get("spend_ceiling_usd"))) != SPEND_CEILING_USD
    ):
        raise ValueError("Protocol identity or execution contract does not match expectations")
    return protocol


def _require_exact_result(result: SplitExperimentResult) -> None:
    if (
        result.identity_digest != _identity(result.execution_ref, result.mode == "dry_run")
        or result.manifest_version_id != V11_MANIFEST_VERSION_ID
        or result.report_version_ids != V11_REPORT_VERSION_IDS
        or result.trial_refs != TRIAL_REFS
        or result.arm_order != ARM_ORDER
        or result.spend_ceiling_usd != SPEND_CEILING_USD
    ):
        raise ValueError("Result identity or frozen protocol fields do not match expectations")
    expected_keys = {(trial_ref, arm) for trial_ref in TRIAL_REFS for arm in ARCHITECTURES}
    keys = [(record.trial_ref, record.arm) for record in result.records]
    if len(keys) != len(set(keys)) or set(keys) - expected_keys:
        raise ValueError("Result records do not form a unique subset of frozen trial arms")
    for record in result.records:
        if (
            record.outcome.arm != record.arm
            or record.outcome.report_version_id not in V11_REPORT_VERSION_IDS
            or (record.outcome.status == "completed") != (record.score is not None)
        ):
            raise ValueError("Result arm outcome or score does not match its record")
        if record.score is not None:
            _require_score(record.score)
    completed_scores = [record.score for record in result.records if record.score is not None]
    if completed_scores:
        expected_cases = _score_case_shape(completed_scores[0])
        if any(_score_case_shape(score) != expected_cases for score in completed_scores[1:]):
            raise ValueError("Completed arm scores do not cover the same frozen evaluation cases")
    for value in result.unknown_in_flight:
        if (
            value.trial_ref,
            value.arm,
        ) not in expected_keys or value.report_version_id not in V11_REPORT_VERSION_IDS:
            raise ValueError("Unknown in-flight arm does not match the frozen workload")
    attempts = [attempt for record in result.records for attempt in record.outcome.attempts]
    attempts.extend(attempt for value in result.unknown_in_flight for attempt in value.attempts)
    known_spend = _known_cost(attempts)
    accounting_complete = not result.unknown_in_flight and all(
        item.accounting_complete for item in attempts
    )
    if (
        result.known_spend_usd != known_spend
        or result.accounting_complete != accounting_complete
        or result.unknown_accounting != bool(result.unknown_in_flight)
    ):
        raise ValueError("Result accounting summary does not match captured attempts")


def _case_outcomes(record: ArmRecord) -> dict[str, bool]:
    if record.score is None:
        return {}
    return {item.case_id: item.passed for item in record.score.case_results}


def _require_score(score: SubjectAssessmentEvaluationResult) -> None:
    case_ids = [item.case_id for item in score.case_results]
    if (
        score.total_cases != len(score.case_results)
        or score.passed_cases != sum(item.passed for item in score.case_results)
        or len(case_ids) != len(set(case_ids))
        or any(item.concern not in ("tier", "ranking") for item in score.case_results)
    ):
        raise ValueError("Arm score does not match its frozen evaluation case results")


def _score_case_shape(
    score: SubjectAssessmentEvaluationResult,
) -> tuple[tuple[str, str, bool], ...]:
    return tuple((item.case_id, item.concern, item.control) for item in score.case_results)


def _subject_ids(record: ArmRecord | None) -> set[str] | None:
    if record is None or not isinstance(record.outcome, CompletedArm):
        return None
    return {item.theme_id for item in record.outcome.assessments}


def _sum_quality(values: list[dict[str, int | None]]) -> dict[str, int | None]:
    keys = (
        "passes",
        "total",
        "tier_passes",
        "tier_total",
        "ranking_passes",
        "ranking_total",
        "control_passes",
        "control_total",
    )
    return {
        key: (
            None
            if any(value[key] is None for value in values)
            else sum(cast(int, value[key]) for value in values)
        )
        for key in keys
    }


def _quality_diagnostics(
    scores_value: Iterable[SubjectAssessmentEvaluationResult],
) -> dict[str, Any]:
    scores = tuple(scores_value)
    tier_counts: dict[str, TierRecall] = {
        tier: {
            "passes": 0,
            "total": 0,
            "recall": None,
        }
        for tier in TIERS
    }
    for score in scores:
        for case in score.case_results:
            if case.concern != "tier":
                continue
            match = _TIER_DETAIL.fullmatch(case.detail)
            if match is None:
                raise ValueError(
                    f"Tier case {case.case_id} lacks parseable expected and assessed tiers"
                )
            expected = match.group(1)
            tier_counts[expected]["total"] += 1
            tier_counts[expected]["passes"] += int(case.passed)
    for values in tier_counts.values():
        total = values["total"]
        values["recall"] = values["passes"] / total if total else None
    ranking_cases = [
        case for score in scores for case in score.case_results if case.concern == "ranking"
    ]
    return {
        "scored_completed_arms": len(scores),
        "per_tier_recall": tier_counts,
        "ranking_inversion_count": sum(case.predicted_positive is False for case in ranking_cases),
        "ranking_unavailable_count": sum(case.predicted_positive is None for case in ranking_cases),
        "merged_pair_count": sum(score.merged_pair_count for score in scores),
        "semantic_tie_count": sum(score.semantic_tie_count for score in scores),
        "merged_tier_resolution_count": sum(len(score.merged_tier_resolutions) for score in scores),
    }


def _uncertainty_analysis(trials: list[dict[str, Any]]) -> dict[str, Any]:
    measurements: dict[str, list[float]] = {
        "quality_pass_delta_per_trial": [],
        "tier_pass_delta_per_trial": [],
        "ranking_pass_delta_per_trial": [],
        "wall_latency_delta_ms_per_trial": [],
        "known_cost_delta_usd_per_trial": [],
    }
    for trial in trials:
        paired = cast(dict[str, Any], trial["paired_delta"])
        arms = cast(dict[str, dict[str, Any]], trial["arms"])
        if paired["status"] != "paired":
            return _inconclusive_uncertainty("a frozen trial pair is missing")
        for source, target in (
            ("passes", "quality_pass_delta_per_trial"),
            ("tier_passes", "tier_pass_delta_per_trial"),
            ("ranking_passes", "ranking_pass_delta_per_trial"),
        ):
            value = paired[source]
            if value is None:
                return _inconclusive_uncertainty("a paired quality delta is unavailable")
            measurements[target].append(float(value))
        incumbent_latency = arms["incumbent"]["wall_latency_ms"]
        candidate_latency = arms["candidate"]["wall_latency_ms"]
        if incumbent_latency is None or candidate_latency is None:
            return _inconclusive_uncertainty("a complete arm latency is unavailable")
        measurements["wall_latency_delta_ms_per_trial"].append(
            float(candidate_latency - incumbent_latency)
        )
        measurements["known_cost_delta_usd_per_trial"].append(
            float(
                Decimal(arms["candidate"]["known_cost_usd"])
                - Decimal(arms["incumbent"]["known_cost_usd"])
            )
        )
    intervals = {name: _exact_bootstrap_interval(values) for name, values in measurements.items()}
    return {
        "status": "descriptive",
        "method": "exact ordered nonparametric bootstrap of the three frozen paired trials",
        "confidence_level": 0.95,
        "resample_count": len(TRIAL_REFS) ** len(TRIAL_REFS),
        "intervals": intervals,
        "interpretation": (
            "Intervals describe run-to-run uncertainty across only three frozen pairs; they are "
            "not evidence of population-level statistical power. Negative latency and cost deltas "
            "favor the candidate."
        ),
    }


def _inconclusive_uncertainty(reason: str) -> dict[str, Any]:
    return {
        "status": "inconclusive",
        "method": "exact ordered nonparametric bootstrap of the three frozen paired trials",
        "confidence_level": 0.95,
        "resample_count": 0,
        "intervals": {},
        "interpretation": f"No interval is reported because {reason}.",
    }


def _exact_bootstrap_interval(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Bootstrap interval requires at least one value")
    sample_means = [
        sum(values[index] for index in sample) / len(sample)
        for sample in itertools.product(range(len(values)), repeat=len(values))
    ]
    return {
        "mean": round(sum(values) / len(values), 12),
        "lower": round(_percentile(sample_means, 0.025), 12),
        "upper": round(_percentile(sample_means, 0.975), 12),
    }


def _known_cost(attempts: Iterable[ExperimentAttempt]) -> Decimal:
    return sum((item.cost_usd or Decimal(0) for item in attempts), Decimal(0))


def _cost_statement(complete: bool) -> str:
    return "complete known provider cost" if complete else "lower bound; accounting is incomplete"


def _percentile(values: list[int] | list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _delta(
    candidate: dict[str, int | None], incumbent: dict[str, int | None], key: str
) -> int | None:
    left = candidate[key]
    right = incumbent[key]
    return None if left is None or right is None else left - right


def _gate(gate: str, status: GateStatus, detail: str) -> dict[str, str]:
    return {"gate": gate, "status": status, "detail": detail}


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _yes_no(value: object) -> str:
    return "yes" if value is True else "no"


def _display(value: object) -> str:
    return "n/a" if value is None else str(value)


def _quality_cell(value: object) -> str:
    if value is None:
        return "n/a"
    quality = cast(dict[str, int | None], value)
    return (
        f"all {quality['passes']}/{quality['total']}; "
        f"tier {quality['tier_passes']}/{quality['tier_total']}; "
        f"rank {quality['ranking_passes']}/{quality['ranking_total']}; "
        f"controls {quality['control_passes']}/{_display(quality['control_total'])}"
    )


def _signed(value: object) -> str:
    return "n/a" if value is None else f"{cast(int, value):+d}"


def _recall_cell(value: TierRecall) -> str:
    recall = value["recall"]
    rendered = "n/a" if recall is None else f"{cast(float, recall):.3f}"
    return f"{value['passes']}/{value['total']} ({rendered})"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = analyze(arguments.input_path)
    write_reports(report, arguments.json_output, arguments.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
