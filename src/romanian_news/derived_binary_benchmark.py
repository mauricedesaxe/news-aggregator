from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from romanian_news.binary_benchmark import (
    BenchmarkId,
    BinaryBenchmarkCase,
    BinaryBenchmarkDefinition,
)
from romanian_news.binary_confidence_evaluation import (
    BinaryConfidenceSource,
    build_confidence_benchmark_cases,
    load_v11_binary_confidence_source,
)
from romanian_news.binary_daily_theme_evaluation import (
    BinaryDailyThemeSource,
    load_v11_binary_daily_theme_source,
)
from romanian_news.binary_grouping_evaluation import (
    BinaryGroupingSource,
    adapt_binary_grouping_cases,
    load_v11_binary_grouping_source,
)
from romanian_news.binary_ranking_evaluation import (
    BinaryRankingSource,
    build_ranking_benchmark_cases,
    load_v11_binary_ranking_source,
)
from romanian_news.binary_tier_evaluation import (
    BinaryTierSource,
    adapt_binary_tier_cases,
    load_v11_binary_tier_source,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS

DerivedBinarySource = (
    BinaryGroupingSource
    | BinaryRankingSource
    | BinaryTierSource
    | BinaryConfidenceSource
    | BinaryDailyThemeSource
)


@dataclass(frozen=True)
class DerivedBinaryWorkload:
    definition: BinaryBenchmarkDefinition
    source_artifact_id: str
    declared_manifest_version: str
    cases: tuple[BinaryBenchmarkCase, ...]


@dataclass(frozen=True)
class DerivedBinaryRegistration:
    definition: BinaryBenchmarkDefinition
    load_source: Callable[[bytes | str], DerivedBinarySource]
    adapt_source: Callable[[DerivedBinarySource], tuple[BinaryBenchmarkCase, ...]]

    def load(self, pin_content: bytes | str) -> DerivedBinaryWorkload:
        return self.prepare(self.load_source(pin_content))

    def prepare(self, source: DerivedBinarySource) -> DerivedBinaryWorkload:
        return DerivedBinaryWorkload(
            definition=self.definition,
            source_artifact_id=source.manifest_reference.version_id,
            declared_manifest_version=source.declared_manifest_version,
            cases=self.adapt_source(source),
        )


def _grouping_cases(source: DerivedBinarySource) -> tuple[BinaryBenchmarkCase, ...]:
    if not isinstance(source, BinaryGroupingSource):
        raise TypeError("Grouping registration requires a binary grouping source")
    return adapt_binary_grouping_cases(source.cases)


def _ranking_cases(source: DerivedBinarySource) -> tuple[BinaryBenchmarkCase, ...]:
    if not isinstance(source, BinaryRankingSource):
        raise TypeError("Ranking registration requires a binary ranking source")
    return build_ranking_benchmark_cases(source)


def _tier_cases(source: DerivedBinarySource) -> tuple[BinaryBenchmarkCase, ...]:
    if not isinstance(source, BinaryTierSource):
        raise TypeError("Tier registration requires a binary tier source")
    return adapt_binary_tier_cases(source.cases)


def _confidence_cases(source: DerivedBinarySource) -> tuple[BinaryBenchmarkCase, ...]:
    if not isinstance(source, BinaryConfidenceSource):
        raise TypeError("Confidence registration requires a binary confidence source")
    return build_confidence_benchmark_cases(source)


def _daily_theme_cases(source: DerivedBinarySource) -> tuple[BinaryBenchmarkCase, ...]:
    if not isinstance(source, BinaryDailyThemeSource):
        raise TypeError("Daily-theme registration requires a binary daily-theme source")
    return source.cases


DERIVED_BINARY_DISPATCH: Mapping[BenchmarkId, DerivedBinaryRegistration] = MappingProxyType(
    {
        "grouping": DerivedBinaryRegistration(
            DERIVED_BINARY_BENCHMARKS["grouping"],
            load_v11_binary_grouping_source,
            _grouping_cases,
        ),
        "ranking": DerivedBinaryRegistration(
            DERIVED_BINARY_BENCHMARKS["ranking"],
            load_v11_binary_ranking_source,
            _ranking_cases,
        ),
        "tier": DerivedBinaryRegistration(
            DERIVED_BINARY_BENCHMARKS["tier"],
            load_v11_binary_tier_source,
            _tier_cases,
        ),
        "confidence": DerivedBinaryRegistration(
            DERIVED_BINARY_BENCHMARKS["confidence"],
            load_v11_binary_confidence_source,
            _confidence_cases,
        ),
        "daily_theme": DerivedBinaryRegistration(
            DERIVED_BINARY_BENCHMARKS["daily_theme"],
            load_v11_binary_daily_theme_source,
            _daily_theme_cases,
        ),
    }
)


def load_v11_derived_binary_workload(
    concern: BenchmarkId, pin_content: bytes | str
) -> DerivedBinaryWorkload:
    return DERIVED_BINARY_DISPATCH[concern].load(pin_content)


def prepare_derived_binary_workload(
    concern: BenchmarkId, source: DerivedBinarySource
) -> DerivedBinaryWorkload:
    return DERIVED_BINARY_DISPATCH[concern].prepare(source)
