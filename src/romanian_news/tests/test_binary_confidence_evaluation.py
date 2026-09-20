from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import cast

import pytest
from pydantic import HttpUrl

from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.articles.models import ExtractedArticle
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET,
    build_binary_evaluators,
    build_execution_identity,
    run_registered_binary_benchmark,
)
from romanian_news.binary_confidence_evaluation import (
    CONFIDENCE_BENCHMARK,
    BinaryConfidenceCase,
    BinaryConfidenceSource,
    analyze_binary_confidence_benchmark,
    build_confidence_benchmark_cases,
    load_v11_binary_confidence_source,
    render_confidence_state,
)
from romanian_news.binary_relevance_evaluation import (
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
)
from romanian_news.catalog.evaluations import NewsEvaluationReportLineage
from romanian_news.evaluation import (
    ConfidenceEvaluationSpec,
    EvaluationSpecProvenance,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ReportEvaluationSpec,
    ReportInputReferences,
)
from romanian_news.groups import DailyClusterSet, NewsGroup

_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_REPORT = "e" * 64
_CLUSTER = "f" * 64


def test_state_contains_only_source_evidence() -> None:
    case = _case(
        expected_sufficient=False,
        control=True,
        body="Human-reviewed evidence body.",
    )

    state = render_confidence_state(case)

    assert "Human-reviewed evidence body." in state
    assert "Article version:" in state
    forbidden = (
        "expected_sufficient",
        "observed_sufficient",
        "uncertainty_ro",
        "GroupSummary",
        "control",
        "feedback",
        "complete-report-output-marker",
    )
    assert all(value not in state for value in forbidden)


def test_loader_reconstructs_only_upstream_group_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_reference = _reference(V11_SOURCE_ARTIFACT_ID, "manifest")
    report_reference = _reference(_REPORT, "report")
    cluster_reference = _reference(_CLUSTER, "cluster")
    article_ids = (_A, _B, _C, _D)
    article_references = {
        value: _reference(value, f"article-{index}") for index, value in enumerate(article_ids)
    }
    specs = tuple(
        ConfidenceEvaluationSpec(
            case_id=f"confidence-{index}",
            control=index == 0,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(),
                report_version_id=_REPORT,
                group_id=article_id,
            ),
            expected_sufficient=index % 2 == 0,
        )
        for index, article_id in enumerate(article_ids)
    )
    manifest = NewsEvaluationManifest.model_construct(
        version=V11_MANIFEST_VERSION,
        reviewed_at=datetime(2026, 9, 11, tzinfo=UTC),
        issue_url="https://example.test/review",
        prior_manifest=None,
        source_feedback_ids=(),
        reports=(
            ReportEvaluationSpec(
                report=report_reference,
                cluster_set=cluster_reference,
            ),
        ),
        cases=specs,
        archived_daily_theme_judgments=(),
        excluded_feedback=(),
        feedback_reviews=(),
        unreviewed_themes=(),
        score_curation=(),
    )
    cluster_set = DailyClusterSet(
        day=date(2026, 9, 10),
        algorithm="test",
        threshold=0.5,
        embedding_model="test",
        article_version_ids=article_ids,
        relevance_version_ids=article_ids,
        embedding_version_ids=article_ids,
        merges=(),
        groups=tuple(NewsGroup(id=value, article_version_ids=(value,)) for value in article_ids),
    )
    payloads = {
        "manifest": manifest.model_dump_json().encode(),
        "cluster": cluster_set.model_dump_json().encode(),
        **{
            f"article-{index}": _article(article_id, f"evidence-{index}").model_dump_json().encode()
            for index, article_id in enumerate(article_ids)
        },
    }
    read_keys: list[str] = []

    def read_object(key: str, _digest: str) -> bytes:
        read_keys.append(key)
        return payloads[key]

    def references(ids: tuple[str, ...]) -> dict[str, ArtifactReference]:
        if ids == (V11_SOURCE_ARTIFACT_ID,):
            return {V11_SOURCE_ARTIFACT_ID: manifest_reference}
        return {value: article_references[value] for value in ids}

    def report_lineage(_version_id: str) -> NewsEvaluationReportLineage:
        return NewsEvaluationReportLineage(
            report=report_reference,
            inputs=ReportInputReferences(
                cluster_set=(cluster_reference,),
                summary=(_reference("1" * 64, "forbidden-summary"),),
                sentiment=(_reference("2" * 64, "forbidden-sentiment"),),
            ),
        )

    monkeypatch.setattr(
        "romanian_news.binary_confidence_evaluation.read_news_evaluation_artifact_references",
        references,
    )
    monkeypatch.setattr(
        "romanian_news.binary_confidence_evaluation.read_news_evaluation_report_lineage",
        report_lineage,
    )
    monkeypatch.setattr(
        "romanian_news.binary_confidence_evaluation.read_verified_r2_object", read_object
    )
    pin = NewsEvaluationPin(
        manifest_version_id=V11_SOURCE_ARTIFACT_ID,
        baseline_version_id="3" * 64,
    )

    source = load_v11_binary_confidence_source(pin.model_dump_json())

    assert len(source.cases) == 4
    assert tuple(case.expected_sufficient for case in source.cases) == (
        True,
        False,
        True,
        False,
    )
    assert "report" not in read_keys
    assert "forbidden-summary" not in read_keys
    assert "forbidden-sentiment" not in read_keys


def test_generic_adapter_executes_four_dry_cases_without_provider_calls() -> None:
    source = _source()
    cases = build_confidence_benchmark_cases(source)
    identity = build_execution_identity(
        CONFIDENCE_BENCHMARK,
        source_artifact_id=V11_SOURCE_ARTIFACT_ID,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("confidence:dry-run:trial-001",),
        execution_mode="dry_run",
        execution_ref="test:confidence",
    )

    result = run_registered_binary_benchmark(
        CONFIDENCE_BENCHMARK,
        identity,
        cases,
        build_binary_evaluators((TYPESAFE_JEV_TARGET,), dry_run=True),
    )
    analysis = analyze_binary_confidence_benchmark(result, cases)

    assert len(cases) == 4
    assert len(result.results) == 4
    assert all(row.status == "completed" for row in result.results)
    assert analysis["analysis_mode"] == "descriptive_only"
    assert analysis["winner"] is None
    trial = cast(Mapping[str, object], cast(list[object], analysis["trials"])[0])
    assert trial["completed"] == 4
    interval = cast(Mapping[str, object], trial["accuracy_95_exact_interval"])
    assert interval["total"] == 4
    assert set(cast(list[str], trial["error_ids"])).issubset({case.case_id for case in cases})


def test_source_requires_exactly_four_cases() -> None:
    source = _source()

    with pytest.raises(ValueError, match="requires 4 cases"):
        _ = BinaryConfidenceSource(
            pin=source.pin,
            manifest_reference=source.manifest_reference,
            declared_manifest_version=V11_MANIFEST_VERSION,
            cases=source.cases[:3],
        )


def _source() -> BinaryConfidenceSource:
    return BinaryConfidenceSource(
        pin=NewsEvaluationPin(
            manifest_version_id=V11_SOURCE_ARTIFACT_ID,
            baseline_version_id="3" * 64,
        ),
        manifest_reference=_reference(V11_SOURCE_ARTIFACT_ID, "manifest"),
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=tuple(
            _case(
                case_id=f"confidence-{index}",
                group_id=value,
                expected_sufficient=index % 2 == 0,
                control=index == 0,
                body=f"Evidence {index}",
            )
            for index, value in enumerate((_A, _B, _C, _D))
        ),
    )


def _case(
    *,
    case_id: str = "confidence-0",
    group_id: str = _A,
    expected_sufficient: bool,
    control: bool,
    body: str,
) -> BinaryConfidenceCase:
    reference = _reference(group_id, f"article-{group_id[0]}")
    return BinaryConfidenceCase(
        case_id=case_id,
        control=control,
        report=_reference(_REPORT, "report"),
        group_id=group_id,
        articles=((reference, _article(group_id, body)),),
        expected_sufficient=expected_sufficient,
    )


def _article(article_id: str, body: str) -> ExtractedArticle:
    return ExtractedArticle(
        article_id=article_id,
        outlet_id="digi24",
        canonical_url=HttpUrl(f"https://example.test/{article_id}"),
        title=f"Source title {article_id[0]}",
        body=body,
        author=None,
        published_at=datetime(2026, 9, 10, 12, tzinfo=UTC),
        source_updated_at=None,
        bucharest_day=date(2026, 9, 10),
        material_digest=article_id,
        extraction_digest=article_id,
    )


def _reference(version_id: str, key: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"artifact:{key}",
        version_id=version_id,
        content_digest="9" * 64,
        r2_key=key,
    )
