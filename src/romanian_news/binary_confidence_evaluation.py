from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_benchmark import (
    analyze_binary_benchmark,
    exact_binomial_interval,
)
from romanian_news.analysis.binary_evaluation import BinaryRequest, binary_state_digest
from romanian_news.articles.models import ExtractedArticle
from romanian_news.binary_benchmark import (
    BinaryBenchmarkCase,
    BinaryBenchmarkEvaluationResult,
    BinaryJudgmentCase,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.catalog.evaluations import (
    read_news_evaluation_artifact_references,
    read_news_evaluation_report_lineage,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import (
    ConfidenceEvaluationSpec,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ReportEvaluationSpec,
)
from romanian_news.groups import DailyClusterSet
from romanian_news.storage import read_verified_r2_object

CONFIDENCE_ARTICLE_BODY_CHARACTERS = 12_000
CONFIDENCE_BENCHMARK = DERIVED_BINARY_BENCHMARKS["confidence"]


class BinaryConfidenceCase(NewsModel):
    case_id: str
    control: bool
    report: ArtifactReference
    group_id: Sha256
    articles: Annotated[tuple[tuple[ArtifactReference, ExtractedArticle], ...], Field(min_length=1)]
    expected_sufficient: bool


class BinaryConfidenceSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    cases: Annotated[tuple[BinaryConfidenceCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_pinned_v11_source(self) -> BinaryConfidenceSource:
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary confidence source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if self.declared_manifest_version != V11_MANIFEST_VERSION:
            raise ValueError(f"Binary confidence manifest version must be {V11_MANIFEST_VERSION}")
        if len(self.cases) != CONFIDENCE_BENCHMARK.case_count:
            raise ValueError(
                f"Binary confidence benchmark requires {CONFIDENCE_BENCHMARK.case_count} cases, received {len(self.cases)}"
            )
        return self


def load_v11_binary_confidence_source(pin_content: bytes | str) -> BinaryConfidenceSource:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    if pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID:
        raise ValueError(f"Binary confidence source artifact must be {V11_SOURCE_ARTIFACT_ID}")
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
        raise ValueError(f"Binary confidence manifest version must be {V11_MANIFEST_VERSION}")
    specs = tuple(case for case in manifest.cases if isinstance(case, ConfidenceEvaluationSpec))
    if len(specs) != CONFIDENCE_BENCHMARK.case_count:
        raise ValueError(
            f"Binary confidence benchmark requires {CONFIDENCE_BENCHMARK.case_count} cases, received {len(specs)}"
        )
    reports = {report.report.version_id: report for report in manifest.reports}
    return BinaryConfidenceSource(
        pin=pin,
        manifest_reference=manifest_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=tuple(_load_confidence_case(spec, reports) for spec in specs),
    )


def render_confidence_state(case: BinaryConfidenceCase) -> str:
    articles = tuple(sorted(case.articles, key=lambda item: item[0].version_id))
    return "\n\n".join(
        _render_article(index, reference, article)
        for index, (reference, article) in enumerate(articles, start=1)
    )


def build_confidence_benchmark_cases(
    source: BinaryConfidenceSource,
) -> tuple[BinaryBenchmarkCase, ...]:
    question = CONFIDENCE_BENCHMARK.questions[0]
    return tuple(
        BinaryBenchmarkCase(
            case_id=case.case_id,
            identity=(
                case.report.version_id,
                case.group_id,
                *(reference.version_id for reference, _article in case.articles),
            ),
            control=case.control,
            judgments=(
                BinaryJudgmentCase(
                    judgment_id=question.question_id,
                    request=_confidence_request(case),
                    expected=case.expected_sufficient,
                ),
            ),
        )
        for case in source.cases
    )


def analyze_binary_confidence_benchmark(
    result: BinaryBenchmarkEvaluationResult,
    cases: tuple[BinaryBenchmarkCase, ...],
) -> dict[str, object]:
    if result.identity.benchmark != "confidence":
        raise ValueError("Binary confidence analysis requires a confidence benchmark result")
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != CONFIDENCE_BENCHMARK.case_count or set(case_by_id) != {
        case.case_id for case in result.identity.cases
    }:
        raise ValueError("Binary confidence analysis requires the exact four registered cases")

    report = cast(dict[str, object], analyze_binary_benchmark(CONFIDENCE_BENCHMARK, result))
    trial_metrics: list[dict[str, object]] = []
    for value in cast(list[object], report["trials"]):
        trial = cast(Mapping[str, object], value)
        target_id = str(trial["target_id"])
        trial_ref = str(trial["trial_ref"])
        rows = tuple(
            row
            for row in result.results
            if row.target_id == target_id and row.trial_ref == trial_ref
        )
        controls = tuple(row for row in rows if case_by_id[row.case_id].control)
        completed_controls = tuple(row for row in controls if row.status == "completed")
        control_correct = sum(row.passed for row in completed_controls)
        trial_metrics.append(
            {
                **trial,
                "error_ids": [
                    row.case_id for row in rows if row.status == "completed" and not row.passed
                ],
                "unavailable_ids": [row.case_id for row in rows if row.status == "failed"],
                "control_preservation": (
                    control_correct / len(completed_controls) if completed_controls else None
                ),
                "control_preservation_95_exact_interval": (
                    exact_binomial_interval(control_correct, len(completed_controls))
                    if completed_controls
                    else None
                ),
            }
        )
    report["trials"] = trial_metrics
    report["analysis_mode"] = "descriptive_only"
    report["winner"] = None
    return report


def _load_confidence_case(
    spec: ConfidenceEvaluationSpec,
    reports: Mapping[Sha256, ReportEvaluationSpec],
) -> BinaryConfidenceCase:
    report_version_id = spec.provenance.report_version_id
    group_id = spec.provenance.group_id
    if report_version_id is None or group_id is None:
        raise ValueError("Binary confidence cases require frozen report and group provenance")
    report_spec = reports.get(report_version_id)
    if report_spec is None:
        raise ValueError("Binary confidence case references an unavailable frozen report")
    report_reference = report_spec.report
    cluster_reference = report_spec.cluster_set
    lineage = read_news_evaluation_report_lineage(report_version_id)
    if lineage.report != report_reference or lineage.inputs.cluster_set != (cluster_reference,):
        raise ValueError("Binary confidence report lineage does not match the pinned manifest")
    cluster_set = DailyClusterSet.model_validate_json(
        read_verified_r2_object(cluster_reference.r2_key, cluster_reference.content_digest),
        strict=True,
    )
    groups = {group.id: group for group in cluster_set.groups}
    group = groups.get(group_id)
    if group is None:
        raise ValueError("Binary confidence group is absent from the frozen cluster set")
    references = read_news_evaluation_artifact_references(group.article_version_ids)
    articles = tuple(
        (
            references[article_id],
            ExtractedArticle.model_validate_json(
                read_verified_r2_object(
                    references[article_id].r2_key,
                    references[article_id].content_digest,
                ),
                strict=True,
            ),
        )
        for article_id in group.article_version_ids
    )
    return BinaryConfidenceCase(
        case_id=spec.case_id,
        control=spec.control,
        report=report_reference,
        group_id=group_id,
        articles=articles,
        expected_sufficient=spec.expected_sufficient,
    )


def _confidence_request(case: BinaryConfidenceCase) -> BinaryRequest:
    state = render_confidence_state(case)
    return BinaryRequest(
        question=CONFIDENCE_BENCHMARK.questions[0],
        state=state,
        state_digest=binary_state_digest(state),
    )


def _render_article(
    index: int,
    reference: ArtifactReference,
    article: ExtractedArticle,
) -> str:
    return "\n".join(
        (
            f"Source article {index}",
            f"Article version: {reference.version_id}",
            f"Published: {article.published_at.isoformat()}",
            f"Outlet: {article.outlet_id}",
            f"URL: {article.canonical_url}",
            f"Title: {article.title}",
            "Body:",
            article.body[:CONFIDENCE_ARTICLE_BODY_CHARACTERS],
        )
    )
