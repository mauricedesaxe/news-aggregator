from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import combinations
from typing import cast
from uuid import UUID

from pytest import MonkeyPatch

from romanian_news.analysis.binary_evaluation import BinaryProbabilityObservation, BinaryRequest
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET,
    build_execution_identity,
    run_registered_binary_benchmark,
)
from romanian_news.binary_daily_theme_evaluation import (
    DAILY_THEME_BINARY_BENCHMARK,
    BinaryDailyThemeSource,
    binary_daily_theme_metrics,
    build_daily_theme_binary_case,
    load_v11_binary_daily_theme_source,
    render_daily_theme_pair_state,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.evaluation import (
    EvaluationSpecProvenance,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ReportEvaluationSpec,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
)
from romanian_news.groups import NewsGroup
from romanian_news.tests.evaluation_factories import reference
from romanian_news.themes import DailyThemeInput, ThemeGroupInput

PAIR_COUNTS = (3, 3, 2, 2, 9, 8)


def test_v11_loader_adapts_exact_reviewed_day_and_pair_counts(
    monkeypatch: MonkeyPatch,
) -> None:
    manifest, inputs = _manifest_and_inputs()
    manifest_reference = reference(9_000, "evaluation-manifest").model_copy(
        update={"version_id": V11_SOURCE_ARTIFACT_ID}
    )
    reads: list[str] = []

    def references(_ids: tuple[str, ...]) -> dict[str, ArtifactReference]:
        return {V11_SOURCE_ARTIFACT_ID: manifest_reference}

    def read(key: str, _digest: str) -> bytes:
        reads.append(key)
        return manifest.model_dump_json().encode()

    def load_input(
        cluster: ArtifactReference,
        summaries: tuple[ArtifactReference, ...],
    ) -> DailyThemeInput:
        return _matching_input(inputs, cluster, summaries)

    monkeypatch.setattr(
        "romanian_news.binary_daily_theme_evaluation.read_news_evaluation_artifact_references",
        references,
    )
    monkeypatch.setattr(
        "romanian_news.binary_daily_theme_evaluation.read_verified_r2_object",
        read,
    )
    monkeypatch.setattr(
        "romanian_news.binary_daily_theme_evaluation.load_daily_theme_input",
        load_input,
    )

    source = load_v11_binary_daily_theme_source(_pin().model_dump_json())

    assert len(source.cases) == 6
    assert tuple(len(case.judgments) for case in source.cases) == PAIR_COUNTS
    assert sum(len(case.judgments) for case in source.cases) == 27
    assert reads == [manifest_reference.r2_key]


def test_rendered_pair_contains_only_frozen_neutral_event_context() -> None:
    spec, value = _day(0, pair_count=2)
    feedback_marker = UUID("00000000-0000-4000-8000-000000000001")
    first = spec.expectations[0].model_copy(
        update={
            "feedback_ids": (feedback_marker,),
            "rationale": "FORBIDDEN REVIEW RATIONALE",
            "control": True,
            "expected_same_theme": False,
        }
    )
    spec = spec.model_copy(
        update={
            "control": True,
            "expectations": (first, *spec.expectations[1:]),
            "model_output": reference(9_100, "frozen-theme-output"),
        }
    )

    state = build_daily_theme_binary_case(spec, value).judgments[0].request.state
    assert state == render_daily_theme_pair_state(
        value,
        first.left_group_id,
        first.right_group_id,
    )
    payload = cast(dict[str, object], json.loads(state))

    assert payload == {
        "day": value.day.isoformat(),
        "day_event_count": len(value.groups),
        "events": [
            {
                "group_alias": "group-001",
                "event_title": value.groups[0].value.title_ro,
                "event_summary": value.groups[0].value.summary_ro,
                "key_points": list(value.groups[0].value.key_points_ro),
            },
            {
                "group_alias": "group-002",
                "event_title": value.groups[1].value.title_ro,
                "event_summary": value.groups[1].value.summary_ro,
                "key_points": list(value.groups[1].value.key_points_ro),
            },
        ],
    }
    assert spec.model_output is not None
    forbidden = (
        "expected_same_theme",
        "must_link",
        "must_separate",
        "FORBIDDEN REVIEW RATIONALE",
        str(feedback_marker),
        spec.source_report.artifact_id,
        spec.model_output.artifact_id,
        "control",
        "observed_themes",
    )
    assert all(marker not in state for marker in forbidden)
    assert all(item.group.id not in state for item in value.groups)


def test_pair_orientation_is_label_independent_and_labels_remain_in_identity() -> None:
    spec, value = _day(0, pair_count=1)
    expectation = spec.expectations[0]
    reversed_pair = expectation.model_copy(
        update={
            "left_group_id": expectation.right_group_id,
            "right_group_id": expectation.left_group_id,
        }
    )
    reversed_case = build_daily_theme_binary_case(
        spec.model_copy(update={"expectations": (reversed_pair,)}), value
    )
    original_case = build_daily_theme_binary_case(spec, value)
    relabeled_case = build_daily_theme_binary_case(
        spec.model_copy(
            update={
                "expectations": (
                    expectation.model_copy(
                        update={"expected_same_theme": not expectation.expected_same_theme}
                    ),
                )
            }
        ),
        value,
    )

    assert reversed_case.judgments[0].request == original_case.judgments[0].request
    assert reversed_case.identity == original_case.identity
    assert relabeled_case.judgments[0].request == original_case.judgments[0].request
    assert relabeled_case.identity != original_case.identity


def test_generic_core_executes_synthetic_daily_theme_dry_run_without_providers() -> None:
    manifest, inputs = _manifest_and_inputs()
    cases = tuple(
        build_daily_theme_binary_case(spec, inputs[spec.cluster_set.version_id])
        for spec in manifest.cases
        if isinstance(spec, ThemeEvaluationDaySpec)
    )
    source = BinaryDailyThemeSource(
        pin=_pin(),
        manifest_reference=reference(9_000, "evaluation-manifest").model_copy(
            update={"version_id": V11_SOURCE_ARTIFACT_ID}
        ),
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
    )
    targets = (TYPESAFE_JEV_TARGET,)
    identity = build_execution_identity(
        DAILY_THEME_BINARY_BENCHMARK,
        source_artifact_id=source.manifest_reference.version_id,
        declared_manifest_version=source.declared_manifest_version,
        cases=source.cases,
        targets=targets,
        trial_refs=("synthetic:dry-run:trial-001",),
        execution_mode="dry_run",
        execution_ref="synthetic",
    )
    expected_by_state = {
        judgment.request.state_digest: judgment.expected
        for case in source.cases
        for judgment in case.judgments
    }

    def evaluate(
        request: BinaryRequest,
        _trial_ref: str,
        **_kwargs: object,
    ) -> BinaryProbabilityObservation:
        expected = expected_by_state[request.state_digest]
        probability = Decimal("0.75") if expected else Decimal("0.25")
        return BinaryProbabilityObservation(
            request_id=request.state_digest,
            provider_request_id=f"dry-run:{request.state_digest[:24]}",
            model=TYPESAFE_JEV_TARGET.requested_model,
            probability=probability,
            predicted_accepted=expected,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            estimated_cost_usd=Decimal(0),
        )

    result = run_registered_binary_benchmark(
        DAILY_THEME_BINARY_BENCHMARK,
        identity,
        source.cases,
        {"typesafe-jev": evaluate},
    )
    metrics = binary_daily_theme_metrics(result.results)

    assert len(result.results) == 27
    assert all(item.cost_usd == Decimal(0) for item in result.results)
    assert metrics.completed_judgments == 27
    assert metrics.must_link_total + metrics.must_separate_total == 27
    assert metrics.must_link_recall == Decimal(1)
    assert metrics.must_separate_preservation == Decimal(1)
    assert metrics.establishes_partition is False
    assert metrics.establishes_transitivity is False
    assert metrics.evaluates_report_usefulness is False


def _manifest_and_inputs() -> tuple[NewsEvaluationManifest, dict[str, DailyThemeInput]]:
    days = tuple(_day(index, pair_count=count) for index, count in enumerate(PAIR_COUNTS))
    specs = tuple(item[0] for item in days)
    inputs = {item.cluster_set.version_id: value for item, value in days}
    return (
        NewsEvaluationManifest(
            version=V11_MANIFEST_VERSION,
            reviewed_at=datetime(2026, 9, 11, tzinfo=UTC),
            issue_url="https://example.test/issues/daily-theme",
            source_feedback_ids=(),
            reports=tuple(
                ReportEvaluationSpec(
                    report=spec.source_report,
                    themes=spec.model_output,
                    cluster_set=spec.cluster_set,
                )
                for spec in specs
            ),
            cases=specs,
        ),
        inputs,
    )


def _day(index: int, *, pair_count: int) -> tuple[ThemeEvaluationDaySpec, DailyThemeInput]:
    day = date(2026, 9, 1) + timedelta(days=index)
    cluster_set = reference(1_000 + index, "cluster")
    source_report = reference(2_000 + index, "report")
    summaries = tuple(reference(3_000 + index * 10 + offset, "summary") for offset in range(6))
    groups = tuple(
        ThemeGroupInput(
            group=NewsGroup(
                id=f"{index * 10 + offset + 1:064x}",
                article_version_ids=(f"{5_000 + index * 10 + offset:064x}",),
            ),
            summary=summaries[offset],
            value=GroupSummary(
                title_ro=f"Titlul evenimentului {index}-{offset}",
                summary_ro=f"Rezumatul înghețat {index}-{offset}.",
                key_points_ro=(f"Punctul cheie {index}-{offset}.",),
                disagreements_ro=(),
                cited_article_version_ids=(f"{5_000 + index * 10 + offset:064x}",),
            ),
        )
        for offset in range(6)
    )
    pairs = tuple(combinations(range(6), 2))[:pair_count]
    if len(pairs) < pair_count:
        raise AssertionError("Synthetic day does not have enough unique pairs")
    expectations = tuple(
        ThemePairExpectation(
            case_id=f"daily-theme-{index}-pair-{pair_index}",
            feedback_ids=(),
            left_group_id=groups[left].group.id,
            right_group_id=groups[right].group.id,
            expected_same_theme=pair_index % 2 == 0,
            rationale=f"Synthetic rationale {index}-{pair_index}.",
            control=pair_index == 0,
        )
        for pair_index, (left, right) in enumerate(pairs)
    )
    return (
        ThemeEvaluationDaySpec(
            case_id=f"daily-theme-{index}",
            control=index == 0,
            provenance=EvaluationSpecProvenance(feedback_ids=()),
            day=day,
            source_report=source_report,
            cluster_set=cluster_set,
            summaries=summaries,
            expectations=expectations,
            model_output=reference(4_000 + index, "frozen-theme-output"),
        ),
        DailyThemeInput(day=day, cluster_set=cluster_set, groups=groups),
    )


def _matching_input(
    inputs: dict[str, DailyThemeInput],
    cluster_set: ArtifactReference,
    summaries: tuple[ArtifactReference, ...],
) -> DailyThemeInput:
    value = inputs[cluster_set.version_id]
    assert tuple(item.summary for item in value.groups) == summaries
    return value


def _pin() -> NewsEvaluationPin:
    return NewsEvaluationPin(
        manifest_version_id=V11_SOURCE_ARTIFACT_ID,
        baseline_version_id="f" * 64,
    )
