from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator

from romanian_news import NewsModel
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import BinaryRequest, binary_state_digest
from romanian_news.binary_benchmark import (
    BinaryBenchmarkCase,
    BinaryJudgmentCase,
    BinaryJudgmentResult,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.catalog.evaluations import read_news_evaluation_artifact_references
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import (
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
)
from romanian_news.storage import read_verified_r2_object
from romanian_news.themes import DailyThemeInput, ThemeGroupInput, load_daily_theme_input

DAILY_THEME_BINARY_BENCHMARK = DERIVED_BINARY_BENCHMARKS["daily_theme"]


class BinaryDailyThemeSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    cases: Annotated[tuple[BinaryBenchmarkCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_pinned_v11_source(self) -> BinaryDailyThemeSource:
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary daily-theme source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if self.declared_manifest_version != V11_MANIFEST_VERSION:
            raise ValueError(f"Binary daily-theme manifest version must be {V11_MANIFEST_VERSION}")
        if len(self.cases) != DAILY_THEME_BINARY_BENCHMARK.case_count:
            raise ValueError("Binary daily-theme source must contain six reviewed days")
        if (
            sum(len(case.judgments) for case in self.cases)
            != DAILY_THEME_BINARY_BENCHMARK.scored_unit_count
        ):
            raise ValueError("Binary daily-theme source must contain 27 reviewed pairs")
        return self


class BinaryDailyThemeMetrics(NewsModel):
    completed_judgments: Annotated[int, Field(ge=0)]
    failed_judgments: Annotated[int, Field(ge=0)]
    must_link_passed: Annotated[int, Field(ge=0)]
    must_link_total: Annotated[int, Field(ge=0)]
    must_link_recall: Annotated[Decimal, Field(ge=0, le=1)]
    must_separate_passed: Annotated[int, Field(ge=0)]
    must_separate_total: Annotated[int, Field(ge=0)]
    must_separate_preservation: Annotated[Decimal, Field(ge=0, le=1)]
    establishes_partition: Literal[False] = False
    establishes_transitivity: Literal[False] = False
    evaluates_report_usefulness: Literal[False] = False


def load_v11_binary_daily_theme_source(pin_content: bytes | str) -> BinaryDailyThemeSource:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    if pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID:
        raise ValueError(f"Binary daily-theme source artifact must be {V11_SOURCE_ARTIFACT_ID}")
    manifest_reference = read_news_evaluation_artifact_references((V11_SOURCE_ARTIFACT_ID,))[
        V11_SOURCE_ARTIFACT_ID
    ]
    manifest = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(
            manifest_reference.r2_key,
            manifest_reference.content_digest,
        ),
        strict=True,
    )
    if manifest.version != V11_MANIFEST_VERSION:
        raise ValueError(f"Binary daily-theme manifest version must be {V11_MANIFEST_VERSION}")
    specs = tuple(case for case in manifest.cases if isinstance(case, ThemeEvaluationDaySpec))
    if len(specs) != DAILY_THEME_BINARY_BENCHMARK.case_count:
        required_days = DAILY_THEME_BINARY_BENCHMARK.case_count
        raise ValueError(
            f"Binary daily-theme benchmark requires {required_days} day cases, received {len(specs)}"
        )
    if (
        sum(len(spec.expectations) for spec in specs)
        != DAILY_THEME_BINARY_BENCHMARK.scored_unit_count
    ):
        required_pairs = DAILY_THEME_BINARY_BENCHMARK.scored_unit_count
        raise ValueError(f"Binary daily-theme benchmark requires {required_pairs} pair judgments")
    cases = tuple(
        build_daily_theme_binary_case(
            spec,
            load_daily_theme_input(spec.cluster_set, spec.summaries),
        )
        for spec in specs
    )
    return BinaryDailyThemeSource(
        pin=pin,
        manifest_reference=manifest_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
    )


def build_daily_theme_binary_case(
    spec: ThemeEvaluationDaySpec,
    value: DailyThemeInput,
) -> BinaryBenchmarkCase:
    if value.day != spec.day or value.cluster_set != spec.cluster_set:
        raise ValueError("Binary daily-theme input does not match its frozen day")
    judgments = tuple(_build_pair_judgment(expectation, value) for expectation in spec.expectations)
    identity = (
        f"day:{spec.day.isoformat()}",
        f"cluster-set:{spec.cluster_set.version_id}",
        *(f"summary:{summary.version_id}" for summary in spec.summaries),
        f"day-control:{int(spec.control)}",
        *(_pair_identity(expectation) for expectation in spec.expectations),
    )
    return BinaryBenchmarkCase(
        case_id=spec.case_id,
        identity=identity,
        control=spec.control,
        judgments=judgments,
    )


def binary_daily_theme_metrics(
    judgments: tuple[BinaryJudgmentResult, ...],
) -> BinaryDailyThemeMetrics:
    completed = tuple(item for item in judgments if item.status == "completed")
    must_links = tuple(item for item in completed if item.expected)
    must_separates = tuple(item for item in completed if not item.expected)
    must_link_passed = sum(item.verdict is True for item in must_links)
    must_separate_passed = sum(item.verdict is False for item in must_separates)
    return BinaryDailyThemeMetrics(
        completed_judgments=len(completed),
        failed_judgments=len(judgments) - len(completed),
        must_link_passed=must_link_passed,
        must_link_total=len(must_links),
        must_link_recall=_ratio(must_link_passed, len(must_links)),
        must_separate_passed=must_separate_passed,
        must_separate_total=len(must_separates),
        must_separate_preservation=_ratio(must_separate_passed, len(must_separates)),
    )


def _build_pair_judgment(
    expectation: ThemePairExpectation,
    value: DailyThemeInput,
) -> BinaryJudgmentCase:
    pair_ids = (expectation.left_group_id, expectation.right_group_id)
    state = render_daily_theme_pair_state(value, *pair_ids)
    request = BinaryRequest(
        question=DAILY_THEME_BINARY_BENCHMARK.questions[0],
        state=state,
        state_digest=binary_state_digest(state),
    )
    return BinaryJudgmentCase(
        judgment_id=expectation.case_id,
        request=request,
        expected=expectation.expected_same_theme,
    )


def render_daily_theme_pair_state(
    value: DailyThemeInput,
    left_group_id: str,
    right_group_id: str,
) -> str:
    pair_ids = (left_group_id, right_group_id)
    if left_group_id == right_group_id:
        raise ValueError("Binary daily-theme pairs require two distinct groups")
    group_by_id = {item.group.id: item for item in value.groups}
    if any(group_id not in group_by_id for group_id in pair_ids):
        raise ValueError("Binary daily-theme pair references an unavailable group")
    aliases = {
        group_id: f"group-{index:03d}"
        for index, group_id in enumerate(sorted(group_by_id), start=1)
    }
    ordered_ids = tuple(sorted(pair_ids, key=aliases.__getitem__))
    return json.dumps(
        {
            "day": value.day.isoformat(),
            "day_event_count": len(value.groups),
            "events": [
                _render_event(aliases[group_id], group_by_id[group_id]) for group_id in ordered_ids
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_event(alias: str, item: ThemeGroupInput) -> dict[str, object]:
    return {
        "group_alias": alias,
        "event_title": item.value.title_ro,
        "event_summary": item.value.summary_ro,
        "key_points": list(item.value.key_points_ro),
    }


def _pair_identity(expectation: ThemePairExpectation) -> str:
    first, second = sorted((expectation.left_group_id, expectation.right_group_id))
    return ":".join(
        (
            "pair",
            expectation.case_id,
            first,
            second,
            f"expected={int(expectation.expected_same_theme)}",
            f"control={int(expectation.control)}",
        )
    )


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)
