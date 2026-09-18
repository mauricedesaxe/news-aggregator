import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from pydantic import ValidationError

from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.relevance import (
    RELEVANCE_POLICY_V1,
    RelevanceDecision,
    relevance_is_accepted,
)
from romanian_news.analysis.relevance_v3 import ImpactDecision
from romanian_news.catalog import evaluations
from romanian_news.evaluation import (
    ConfidenceEvaluationCase,
    ConfidenceEvaluationSpec,
    EvaluationSpecProvenance,
    ExcludedFeedbackScore,
    FeedbackReview,
    GroupingEvaluationCase,
    GroupingEvaluationSpec,
    NewsEvaluationBaseline,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ProjectedFeedbackScore,
    RankingEvaluationCase,
    RankingEvaluationSpec,
    ReaderPresentationEvaluationCase,
    ReaderPresentationEvaluationSpec,
    RelevanceConcernDisposition,
    RelevanceEvaluationCase,
    RelevanceEvaluationSpec,
    RelevanceV3EvaluationDecision,
    ReportEvaluationSpec,
    ReportGroupObservation,
    SummaryFormatEvaluationCase,
    SummaryFormatEvaluationSpec,
    SupersededConcernDisposition,
    TierEvaluationCase,
    TierEvaluationSpec,
    UnreviewedTheme,
)
from romanian_news.feedback import ThemeFeedbackTarget
from romanian_news.groups import DailyClusterSet
from romanian_news.tests.evaluation_factories import (
    DAY,
    embedded_article,
    relevance_decision,
    synthetic_dataset,
    synthetic_manifest,
)


def _manifest_reference_ids(value: object) -> set[str]:
    if isinstance(value, list):
        return {version_id for item in value for version_id in _manifest_reference_ids(item)}
    if not isinstance(value, dict):
        return set()
    if set(value) == {"artifact_id", "version_id", "content_digest", "r2_key"}:
        return {str(value["version_id"])}
    return {version_id for item in value.values() for version_id in _manifest_reference_ids(item)}


def _patch_catalog_references(monkeypatch, manifest: NewsEvaluationManifest) -> None:
    references = {
        reference.version_id: reference for reference in evaluations._manifest_inputs(manifest)
    }
    monkeypatch.setattr(
        evaluations,
        "catalog_query",
        lambda _sql, parameters: [
            references[version_id].model_dump(mode="python")
            for version_id in parameters
            if version_id in references
        ],
    )


def _reference(artifact_id: str, digest: str, key: str) -> ArtifactReference:
    version_id = hashlib.sha256(f"{artifact_id}\0{digest}".encode()).hexdigest()
    return ArtifactReference(
        artifact_id=artifact_id,
        version_id=version_id,
        content_digest=digest,
        r2_key=key,
    )


def _relevance_manifest() -> tuple[NewsEvaluationManifest, dict[str, bytes]]:
    feedback_id = UUID("00000000-0000-4000-8000-000000000001")
    article = _reference("news:article:test", "1" * 64, "article.json")
    relevance = _reference("news:relevance:test", "2" * 64, "relevance.json")
    manifest = NewsEvaluationManifest(
        version="synthetic-v1",
        reviewed_at=datetime(2026, 9, 4, tzinfo=UTC),
        issue_url="https://example.com/issues/1",
        source_feedback_ids=(feedback_id,),
        reports=(),
        cases=(
            RelevanceEvaluationSpec(
                case_id="accepts-national-policy",
                provenance=EvaluationSpecProvenance(feedback_ids=(feedback_id,)),
                expected_accepted=True,
                article=article,
                model_output=relevance,
            ),
        ),
    )
    payloads = {
        article.r2_key: json.dumps(
            {
                "article_id": "3" * 64,
                "outlet_id": "example",
                "canonical_url": "https://example.com/article",
                "title": "National policy changes",
                "body": "The policy applies across Romania.",
                "author": None,
                "published_at": "2026-09-04T08:00:00Z",
                "source_updated_at": None,
                "bucharest_day": "2026-09-04",
                "material_digest": "4" * 64,
                "extraction_digest": "5" * 64,
            }
        ).encode(),
        relevance.r2_key: json.dumps(
            {
                "article_version_id": article.version_id,
                "decision": {
                    "national_reach": "nationwide",
                    "consequence_magnitude": "routine",
                    "political_relevance": "strong",
                    "economic_relevance": "none",
                    "romania_relevance": "strong",
                    "confidence": 0.9,
                    "evidence_quote": "applies across Romania",
                    "reason_ro": "Politică națională.",
                },
            }
        ).encode(),
    }
    return manifest, payloads


def _v7_membership_manifest() -> NewsEvaluationManifest:
    old_id = UUID("00000000-0000-4000-8000-000000000091")
    replacement_id = UUID("00000000-0000-4000-8000-000000000092")
    report = _reference("news:daily:v7", "a" * 64, "v7-report.json")
    cluster = _reference("news:clusters:v7", "b" * 64, "v7-cluster.json")
    target = ThemeFeedbackTarget(report_version_id=report.version_id, theme_id="c" * 64)
    return NewsEvaluationManifest(
        version="synthetic-v7",
        reviewed_at=datetime(2026, 9, 10, tzinfo=UTC),
        issue_url="https://example.test/issues/370",
        source_feedback_ids=(old_id, replacement_id),
        reports=(ReportEvaluationSpec(report=report, cluster_set=cluster),),
        cases=(
            RelevanceEvaluationSpec(
                case_id="v7-relevance",
                provenance=EvaluationSpecProvenance(
                    feedback_ids=(replacement_id,),
                    report_version_id=report.version_id,
                    group_id="d" * 64,
                ),
                expected_accepted=True,
                article=_reference("news:article:v7", "d" * 64, "v7-article.json"),
                model_output=_reference("news:relevance:v7", "e" * 64, "v7-relevance.json"),
            ),
        ),
        feedback_reviews=(
            FeedbackReview(
                feedback_id=old_id,
                target=target,
                concerns=(SupersededConcernDisposition(superseded_by_feedback_id=replacement_id),),
            ),
            FeedbackReview(
                feedback_id=replacement_id,
                target=target,
                concerns=(
                    RelevanceConcernDisposition(
                        judgment="relevant",
                        case_ids=("v7-relevance",),
                        rationale="The replacement carries the executable judgment.",
                    ),
                ),
            ),
        ),
        unreviewed_themes=(
            UnreviewedTheme(
                subject_position=2,
                target=ThemeFeedbackTarget(
                    report_version_id=report.version_id,
                    theme_id="f" * 64,
                ),
            ),
        ),
    )


def test_hydrate_manifest_preserves_v7_review_metadata(monkeypatch) -> None:
    manifest = _v7_membership_manifest()
    target = manifest.feedback_reviews[1].target
    assert isinstance(target, ThemeFeedbackTarget)
    source = synthetic_dataset()
    snapshot = source.reports[0].model_copy(
        update={
            "report": manifest.reports[0].report,
            "cluster_set": manifest.reports[0].cluster_set,
            "report_group_ids": ("d" * 64, source.reports[0].report_group_ids[1]),
            "report_theme_ids": (
                target.theme_id,
                manifest.unreviewed_themes[0].target.theme_id,
            ),
        }
    )
    case = source.cases[0].model_copy(
        update={
            "case_id": manifest.cases[0].case_id,
            "provenance": source.cases[0].provenance.model_copy(
                update={
                    "feedback_ids": (manifest.source_feedback_ids[1],),
                    "report": snapshot.report,
                    "group_id": "d" * 64,
                }
            ),
        }
    )
    monkeypatch.setattr(evaluations, "_hydrate_report", lambda _spec: (snapshot, object()))
    monkeypatch.setattr(
        evaluations,
        "_hydrate_case",
        lambda _spec, _reports, _cluster_sets: case,
    )

    dataset = evaluations.hydrate_news_evaluation_manifest(manifest)

    assert dataset.feedback_reviews == manifest.feedback_reviews
    assert dataset.unreviewed_themes == manifest.unreviewed_themes

    with pytest.raises(ValidationError, match="target does not match its report position"):
        NewsEvaluationDataset(
            version=dataset.version,
            reviewed_at=dataset.reviewed_at,
            issue_url=dataset.issue_url,
            source_feedback_ids=dataset.source_feedback_ids,
            reports=(snapshot.model_copy(update={"report_theme_ids": ("0" * 64,) * 2}),),
            cases=dataset.cases,
            feedback_reviews=dataset.feedback_reviews,
            unreviewed_themes=dataset.unreviewed_themes,
        )


def test_hydrated_theme_feedback_rejects_a_case_from_another_theme(monkeypatch) -> None:
    manifest = _v7_membership_manifest()
    source = synthetic_dataset()
    target = manifest.feedback_reviews[1].target
    assert isinstance(target, ThemeFeedbackTarget)
    target_theme_id = target.theme_id
    other_theme_id = "f" * 64
    snapshot = source.reports[0].model_copy(
        update={
            "report": manifest.reports[0].report,
            "cluster_set": manifest.reports[0].cluster_set,
            "report_group_ids": ("d" * 64, source.reports[0].report_group_ids[1]),
            "report_theme_ids": (other_theme_id, target_theme_id),
        }
    )
    case = source.cases[0].model_copy(
        update={
            "case_id": manifest.cases[0].case_id,
            "provenance": source.cases[0].provenance.model_copy(
                update={
                    "feedback_ids": (manifest.source_feedback_ids[1],),
                    "report": snapshot.report,
                    "group_id": "d" * 64,
                }
            ),
        }
    )
    monkeypatch.setattr(evaluations, "_hydrate_report", lambda _spec: (snapshot, object()))
    monkeypatch.setattr(
        evaluations,
        "_hydrate_case",
        lambda _spec, _reports, _cluster_sets: case,
    )

    with pytest.raises(ValidationError, match="linked case must belong to its target theme"):
        evaluations.hydrate_news_evaluation_manifest(
            manifest.model_copy(update={"unreviewed_themes": ()})
        )


def test_publish_v7_manifest_records_all_membership_and_supersession(monkeypatch) -> None:
    manifest = _v7_membership_manifest()
    batches = []
    _patch_catalog_references(monkeypatch, manifest)
    monkeypatch.setattr(evaluations, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(evaluations, "run_status", lambda _run_id: None)
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda _values: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)
    monkeypatch.setattr(
        evaluations,
        "read_news_evaluation_feedback",
        lambda _ids: tuple(
            SimpleNamespace(
                feedback_id=review.feedback_id,
                target=review.target,
                created_at=datetime(2026, 9, 10, tzinfo=UTC),
            )
            for review in manifest.feedback_reviews
        ),
    )

    reference = evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    feedback = [
        parameters
        for sql, parameters in batches[0]
        if "INTO news_evaluation_feedback_sources" in sql
    ]
    assert feedback == [
        [reference.version_id, 0, str(manifest.source_feedback_ids[0]), "excluded"],
        [reference.version_id, 1, str(manifest.source_feedback_ids[1]), "represented"],
    ]


def test_publish_v7_manifest_rejects_changed_feedback_target_before_writes(monkeypatch) -> None:
    manifest = _v7_membership_manifest()
    batches = []
    changed = manifest.feedback_reviews[1]
    monkeypatch.setattr(
        evaluations,
        "read_news_evaluation_feedback",
        lambda _ids: (
            SimpleNamespace(
                feedback_id=changed.feedback_id,
                target=changed.target.model_copy(update={"theme_id": "0" * 64}),
            ),
        ),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    with pytest.raises(ValueError, match="do not match reviewed IDs and targets"):
        evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    assert batches == []


def test_publish_v7_manifest_rejects_older_superseding_feedback_before_writes(
    monkeypatch,
) -> None:
    manifest = _v7_membership_manifest()
    old, replacement = manifest.feedback_reviews
    created_at = datetime(2026, 9, 10, tzinfo=UTC)
    batches = []
    monkeypatch.setattr(
        evaluations,
        "read_news_evaluation_feedback",
        lambda _ids: (
            SimpleNamespace(
                feedback_id=old.feedback_id,
                target=old.target,
                created_at=created_at,
            ),
            SimpleNamespace(
                feedback_id=replacement.feedback_id,
                target=replacement.target,
                created_at=created_at - timedelta(seconds=1),
            ),
        ),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    with pytest.raises(ValueError, match="Superseding feedback must be newer"):
        evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    assert batches == []


def test_hydrate_manifest_loads_exact_artifacts(monkeypatch) -> None:
    manifest, payloads = _relevance_manifest()
    manifest = manifest.model_copy(
        update={"prior_manifest": _reference("prior", "9" * 64, "prior.json")}
    )
    reads = []

    def read(key, digest):
        reads.append((key, digest))
        return payloads[key]

    monkeypatch.setattr(evaluations, "read_verified_r2_object", read)

    dataset = evaluations.hydrate_news_evaluation_manifest(manifest)

    case = dataset.cases[0]
    assert isinstance(case, RelevanceEvaluationCase)
    assert isinstance(case.model_response, RelevanceDecision)
    assert relevance_is_accepted(case.model_response, RELEVANCE_POLICY_V1) is True
    assert dataset.feedback_reviews == ()
    assert dataset.unreviewed_themes == ()
    assert reads == [("article.json", "1" * 64), ("relevance.json", "2" * 64)]


def test_hydrate_relevance_preserves_v3_context_and_impact_decisions(monkeypatch) -> None:
    manifest, payloads = _relevance_manifest()
    spec = manifest.cases[0]
    assert isinstance(spec, RelevanceEvaluationSpec)
    relevance = spec.model_output
    payloads[relevance.r2_key] = json.dumps(
        {
            "article_version_id": spec.article.version_id,
            "context": {
                "decision": {
                    "subject_role": "principal",
                    "news_cycle": "current_cycle",
                    "romanian_consequence": "direct",
                    "certainty": "clear",
                    "evidence_quote": "Romania is the main subject.",
                    "reason_ro": "România este subiectul principal.",
                }
            },
            "impact": {
                "decision": {
                    "consequence_status": "realized",
                    "effect_basis": "actual_consequence",
                    "effect_scope": "national_market",
                    "magnitude": "major",
                    "political_relevance": "none",
                    "economic_relevance": "strong",
                    "quantified": True,
                    "certainty": "clear",
                    "evidence_quote": "National market evidence.",
                    "reason_ro": "Impact economic național.",
                }
            },
        }
    ).encode()
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda key, _digest: payloads[key],
    )

    dataset = evaluations.hydrate_news_evaluation_manifest(manifest)

    case = dataset.cases[0]
    assert isinstance(case, RelevanceEvaluationCase)
    assert isinstance(case.model_response, RelevanceV3EvaluationDecision)
    assert case.model_response.context.subject_role == "principal"
    assert case.model_response.impact is not None
    assert case.model_response.impact.magnitude == "major"


def test_hydrate_manifest_propagates_digest_failures(monkeypatch) -> None:
    manifest, _payloads = _relevance_manifest()
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda _key, _digest: (_ for _ in ()).throw(ValueError("digest mismatch")),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        evaluations.hydrate_news_evaluation_manifest(manifest)


def test_hydrate_manifest_rejects_mismatched_payload_identity(monkeypatch) -> None:
    manifest, payloads = _relevance_manifest()
    payload = json.loads(payloads["relevance.json"])
    payload["article_version_id"] = "0" * 64
    payloads["relevance.json"] = json.dumps(payload).encode()
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda key, _digest: payloads[key],
    )

    with pytest.raises(ValueError, match="another article version"):
        evaluations.hydrate_news_evaluation_manifest(manifest)


def _hydration_fixture():
    manifest = synthetic_manifest()
    dataset = synthetic_dataset()
    first = embedded_article(1)
    second = embedded_article(2)
    report = dataset.reports[0].model_copy(
        update={
            "report": manifest.reports[0].report,
            "cluster_set": manifest.reports[0].cluster_set,
        }
    )
    ranking_case = dataset.cases[3]
    assert isinstance(ranking_case, RankingEvaluationCase)
    cluster_set = DailyClusterSet(
        day=DAY,
        algorithm="synthetic",
        threshold=0.72,
        embedding_model="synthetic",
        article_version_ids=(first.article.version_id, second.article.version_id),
        relevance_version_ids=(first.relevance.version_id, second.relevance.version_id),
        embedding_version_ids=(first.embedding.version_id, second.embedding.version_id),
        merges=(),
        groups=ranking_case.groups,
    )
    summary_source = manifest.cases[1]
    assert isinstance(summary_source, SummaryFormatEvaluationSpec)
    summary_spec = summary_source.model_copy(
        update={
            "provenance": summary_source.provenance.model_copy(
                update={"group_id": report.report_group_ids[0]}
            )
        }
    )
    manifest = manifest.model_copy(
        update={"cases": (manifest.cases[0], summary_spec, *manifest.cases[2:])}
    )
    payloads = {
        first.article.r2_key: first.value.model_dump_json().encode(),
        second.article.r2_key: second.value.model_dump_json().encode(),
        first.relevance.r2_key: json.dumps(
            {
                "article_version_id": first.article.version_id,
                "decision": relevance_decision(accepted=True, major=True).model_dump(mode="json"),
            }
        ).encode(),
        second.relevance.r2_key: json.dumps(
            {
                "article_version_id": second.article.version_id,
                "decision": relevance_decision(accepted=False).model_dump(mode="json"),
            }
        ).encode(),
        first.embedding.r2_key: json.dumps(
            {
                "article_version_id": first.article.version_id,
                "relevance_version_id": first.relevance.version_id,
                "vector": first.vector,
            }
        ).encode(),
        second.embedding.r2_key: json.dumps(
            {
                "article_version_id": second.article.version_id,
                "relevance_version_id": second.relevance.version_id,
                "vector": second.vector,
            }
        ).encode(),
        manifest.reports[0].cluster_set.r2_key: cluster_set.model_dump_json().encode(),
        summary_spec.model_output.r2_key: json.dumps(
            {
                "group_id": report.report_group_ids[0],
                "summary": {
                    "title_ro": "Titlu sintetic complet",
                    "summary_ro": "Rezumat sintetic complet.",
                    "key_points_ro": ["Primul punct."],
                },
            }
        ).encode(),
    }
    reports = {report.report.version_id: report}
    cluster_sets = {report.report.version_id: cluster_set}
    return manifest, reports, cluster_sets, payloads


def _record_artifact_reads(monkeypatch, payloads):
    reads = []

    def read(key, digest):
        reads.append((key, digest))
        return payloads[key]

    monkeypatch.setattr(evaluations, "read_verified_r2_object", read)
    return reads


def _assert_case_metadata(case, spec, reports) -> None:
    assert case.case_id == spec.case_id
    assert case.control == spec.control
    assert case.provenance.feedback_ids == spec.provenance.feedback_ids
    assert case.provenance.report == reports[spec.provenance.report_version_id].report
    assert case.provenance.group_id == spec.provenance.group_id


def test_hydrate_relevance_preserves_provenance_fields_and_read_order(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[0]
    assert isinstance(spec, RelevanceEvaluationSpec)
    reads = _record_artifact_reads(monkeypatch, payloads)

    case = evaluations._hydrate_case(spec, reports, cluster_sets)

    assert isinstance(case, RelevanceEvaluationCase)
    _assert_case_metadata(case, spec, reports)
    assert case.provenance.articles == (spec.article,)
    assert case.provenance.model_outputs == (spec.model_output,)
    assert case.expected_accepted == spec.expected_accepted
    assert case.model_response == relevance_decision(accepted=True, major=True)
    assert reads == [
        (spec.article.r2_key, spec.article.content_digest),
        (spec.model_output.r2_key, spec.model_output.content_digest),
    ]


def test_hydrate_summary_preserves_provenance_artifact_and_read_order(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[1]
    assert isinstance(spec, SummaryFormatEvaluationSpec)
    reads = _record_artifact_reads(monkeypatch, payloads)

    case = evaluations._hydrate_case(spec, reports, cluster_sets)

    assert isinstance(case, SummaryFormatEvaluationCase)
    _assert_case_metadata(case, spec, reports)
    assert case.provenance.articles == spec.articles
    assert case.provenance.model_outputs == (spec.model_output,)
    assert case.artifact.model_dump() == {
        "title": "Titlu sintetic complet",
        "summary": "Rezumat sintetic complet.",
        "key_points": ("Primul punct.",),
    }
    assert reads == [
        *((item.r2_key, item.content_digest) for item in spec.articles),
        (spec.model_output.r2_key, spec.model_output.content_digest),
    ]


def test_hydrate_grouping_preserves_provenance_fields_and_read_order(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[2]
    assert isinstance(spec, GroupingEvaluationSpec)
    reads = _record_artifact_reads(monkeypatch, payloads)

    case = evaluations._hydrate_case(spec, reports, cluster_sets)

    assert isinstance(case, GroupingEvaluationCase)
    _assert_case_metadata(case, spec, reports)
    assert case.provenance.articles == tuple(item.article for item in spec.articles)
    assert case.provenance.model_outputs == tuple(
        reference for item in spec.articles for reference in (item.relevance, item.embedding)
    )
    assert case.source_cluster_set == spec.source_cluster_set
    assert case.day == spec.day
    assert tuple(item.article for item in case.articles) == tuple(
        item.article for item in spec.articles
    )
    assert case.left_article_version_id == spec.left_article_version_id
    assert case.right_article_version_id == spec.right_article_version_id
    assert case.expected_same_group == spec.expected_same_group
    assert reads == [
        (spec.source_cluster_set.r2_key, spec.source_cluster_set.content_digest),
        *(
            (reference.r2_key, reference.content_digest)
            for item in spec.articles
            for reference in (item.article, item.relevance, item.embedding)
        ),
    ]


def test_ranking_relevance_reads_current_impact_decisions(monkeypatch) -> None:
    article_version_id = "1" * 64
    reference = _reference("news:relevance:v3", "2" * 64, "relevance-v3.json")
    payload = {
        "article_version_id": article_version_id,
        "impact": {
            "decision": {
                "consequence_status": "realized",
                "effect_basis": "actual_consequence",
                "effect_scope": "national_market",
                "magnitude": "major",
                "political_relevance": "none",
                "economic_relevance": "strong",
                "quantified": True,
                "certainty": "clear",
                "evidence_quote": "National market evidence.",
                "reason_ro": "Impact economic național.",
            }
        },
    }
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda _key, _digest: json.dumps(payload).encode(),
    )

    decision = evaluations.read_evaluation_ranking_decision(reference, article_version_id)

    assert isinstance(decision, ImpactDecision)
    assert decision.effect_scope == "national_market"
    assert decision.magnitude == "major"


def test_hydrate_ranking_preserves_provenance_fields_and_read_order(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[3]
    assert isinstance(spec, RankingEvaluationSpec)
    reads = _record_artifact_reads(monkeypatch, payloads)

    case = evaluations._hydrate_case(spec, reports, cluster_sets)

    assert isinstance(case, RankingEvaluationCase)
    _assert_case_metadata(case, spec, reports)
    assert case.provenance.articles == tuple(item.article for item in spec.relevance)
    assert case.provenance.model_outputs == tuple(item.model_output for item in spec.relevance)
    assert case.groups == cluster_sets[manifest.reports[0].report.version_id].groups
    assert tuple(item.article_version_id for item in case.relevance) == tuple(
        item.article.version_id for item in spec.relevance
    )
    assert case.higher_group_id == spec.higher_group_id
    assert case.lower_group_id == spec.lower_group_id
    assert reads == [
        *((item.model_output.r2_key, item.model_output.content_digest) for item in spec.relevance),
        *((item.article.r2_key, item.article.content_digest) for item in spec.relevance),
    ]


def test_hydrate_reader_preserves_empty_provenance_and_expectation(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[4]
    assert isinstance(spec, ReaderPresentationEvaluationSpec)
    reads = _record_artifact_reads(monkeypatch, payloads)

    case = evaluations._hydrate_case(spec, reports, cluster_sets)

    assert isinstance(case, ReaderPresentationEvaluationCase)
    _assert_case_metadata(case, spec, reports)
    assert case.provenance.articles == ()
    assert case.provenance.model_outputs == ()
    assert case.expectation == spec.expectation
    assert reads == []


def test_hydrate_tier_and_confidence_from_report_group_observation() -> None:
    manifest, reports, cluster_sets, _payloads = _hydration_fixture()
    report_id = manifest.reports[0].report.version_id
    report = reports[report_id]
    group_id = report.report_group_ids[0]
    reports[report_id] = report.model_copy(
        update={
            "group_observations": tuple(
                ReportGroupObservation(
                    group_id=current_group_id,
                    uncertainty_disclosed=current_group_id == group_id,
                )
                for current_group_id in report.report_group_ids
            )
        }
    )
    provenance = EvaluationSpecProvenance(
        feedback_ids=(),
        report_version_id=report_id,
        group_id=group_id,
    )

    tier = evaluations._hydrate_case(
        TierEvaluationSpec(
            case_id="tier",
            provenance=provenance,
            expected_tier="worth_knowing",
        ),
        reports,
        cluster_sets,
    )
    confidence = evaluations._hydrate_case(
        ConfidenceEvaluationSpec(
            case_id="confidence",
            provenance=provenance,
            expected_sufficient=False,
        ),
        reports,
        cluster_sets,
    )

    assert isinstance(tier, TierEvaluationCase)
    assert tier.observed_tier == "main"
    assert isinstance(confidence, ConfidenceEvaluationCase)
    assert confidence.observed_sufficient is False


def test_hydrate_summary_rejects_another_group_after_article_verification(monkeypatch) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[1]
    assert isinstance(spec, SummaryFormatEvaluationSpec)
    payload = json.loads(payloads[spec.model_output.r2_key])
    payload["group_id"] = "f" * 64
    payloads[spec.model_output.r2_key] = json.dumps(payload).encode()
    reads = []
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda key, digest: reads.append((key, digest)) or payloads[key],
    )

    with pytest.raises(ValueError, match="Summary payload references another group"):
        evaluations._hydrate_case(spec, reports, cluster_sets)

    assert reads == [
        (spec.articles[0].r2_key, spec.articles[0].content_digest),
        (spec.model_output.r2_key, spec.model_output.content_digest),
    ]


def test_hydrate_grouping_rejects_mismatched_cluster_inputs_after_verification(
    monkeypatch,
) -> None:
    manifest, reports, cluster_sets, payloads = _hydration_fixture()
    spec = manifest.cases[2]
    assert isinstance(spec, GroupingEvaluationSpec)
    source = json.loads(payloads[spec.source_cluster_set.r2_key])
    source["relevance_version_ids"][0] = "f" * 64
    payloads[spec.source_cluster_set.r2_key] = json.dumps(source).encode()
    reads = []
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda key, digest: reads.append((key, digest)) or payloads[key],
    )

    with pytest.raises(ValueError, match="Grouping spec does not match its source cluster set"):
        evaluations._hydrate_case(spec, reports, cluster_sets)

    assert reads == [
        (spec.source_cluster_set.r2_key, spec.source_cluster_set.content_digest),
        *(
            (reference.r2_key, reference.content_digest)
            for item in spec.articles
            for reference in (item.article, item.relevance, item.embedding)
        ),
    ]


@pytest.mark.parametrize("mismatch", ("missing", "duplicate"))
def test_hydrate_ranking_rejects_coverage_mismatches_before_artifact_reads(
    monkeypatch,
    mismatch,
) -> None:
    manifest, reports, cluster_sets, _payloads = _hydration_fixture()
    spec = manifest.cases[3]
    assert isinstance(spec, RankingEvaluationSpec)
    relevance = spec.relevance
    mismatched_relevance = relevance[:1] if mismatch == "missing" else (*relevance, relevance[0])
    spec = spec.model_copy(update={"relevance": mismatched_relevance})
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda _key, _digest: (_ for _ in ()).throw(AssertionError("unexpected read")),
    )

    with pytest.raises(ValueError, match="Ranking specs must exactly cover ranked group articles"):
        evaluations._hydrate_case(spec, reports, cluster_sets)


def test_synthetic_manifest_is_strict_and_compact() -> None:
    manifest = synthetic_manifest()
    prior = _reference("news:evaluation-manifest:prior", "9" * 64, "prior.json")
    manifest = manifest.model_copy(update={"prior_manifest": prior})
    pin = NewsEvaluationPin(manifest_version_id="1" * 64, baseline_version_id="2" * 64)
    payload = manifest.model_dump(mode="json")

    assert len(manifest.source_feedback_ids) == 5
    assert {case.concern for case in manifest.cases} == {
        "relevance",
        "summary_format",
        "grouping",
        "ranking",
        "reader_presentation",
    }
    serialized = json.dumps(payload)
    assert '"body"' not in serialized
    assert '"model_response"' not in serialized
    assert all(set(report) == {"report", "themes", "cluster_set"} for report in payload["reports"])
    assert set(pin.model_dump()) == {"manifest_version_id", "baseline_version_id"}
    assert "body" not in pin.model_dump_json()
    assert manifest.prior_manifest == prior

    incomplete = manifest.model_dump(mode="json")
    del incomplete["prior_manifest"]["content_digest"]
    with pytest.raises(ValidationError, match="content_digest"):
        NewsEvaluationManifest.model_validate_json(json.dumps(incomplete), strict=True)

    payload["cases"][0]["unexpected"] = []

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)


def test_publish_manifest_records_atomic_concern_score_curation(monkeypatch) -> None:
    manifest = synthetic_manifest()
    report_version_id = manifest.reports[0].report.version_id
    feedback_ids = manifest.source_feedback_ids
    prose_output = _reference("news:theme-prose:score", "d" * 64, "theme-prose.json")
    assignment_output = _reference("news:theme-assignment:score", "e" * 64, "theme-assignment.json")
    decisions = (
        ProjectedFeedbackScore(
            feedback_id=feedback_ids[0],
            concern="language",
            report_version_id=report_version_id,
            model_output=prose_output,
            model_attempt_id="7" * 64,
            polarity="negative",
            rationale="The generated prose uses the wrong language.",
        ),
        ProjectedFeedbackScore(
            feedback_id=feedback_ids[0],
            concern="grouping",
            report_version_id=report_version_id,
            model_output=assignment_output,
            model_attempt_id="8" * 64,
            polarity="positive",
            rationale="The assigned groups belong together.",
        ),
        *(
            ExcludedFeedbackScore(
                feedback_id=feedback_id,
                reason="presentation",
                report_version_id=report_version_id,
                rationale="The decision does not assess model output.",
            )
            for feedback_id in feedback_ids[1:]
        ),
    )
    manifest = manifest.model_copy(update={"score_curation": decisions})
    batches = []
    _patch_catalog_references(monkeypatch, manifest)
    monkeypatch.setattr(evaluations, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(evaluations, "run_status", lambda _run_id: None)
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda _values: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    published = evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    curation = [
        parameters for sql, parameters in batches[0] if "INTO news_feedback_score_curation" in sql
    ]
    assert len(curation) == len(decisions)
    expected_language_score_id = str(
        uuid5(
            NAMESPACE_URL,
            "chartly:news-feedback-score:"
            f"{published.version_id}:{feedback_ids[0]}:language:{'7' * 64}",
        )
    )
    assert curation[0] == [
        published.version_id,
        manifest.version,
        0,
        str(feedback_ids[0]),
        "language",
        None,
        "projected",
        report_version_id,
        prose_output.version_id,
        "7" * 64,
        "negative",
        "The generated prose uses the wrong language.",
        expected_language_score_id,
    ]
    assert curation[0][-1] != curation[1][-1]
    other_manifest = evaluations._score_curation_statements(manifest, "f" * 64)
    assert curation[0][-1] != other_manifest[0][1][-1]
    assert curation[2][4:10] == [None, "presentation", "excluded", report_version_id, None, None]
    input_ids = {parameters[2] for sql, parameters in batches[0] if "INTO run_inputs" in sql}
    assert prose_output.version_id in input_ids
    assert assignment_output.version_id in input_ids


def test_publish_manifest_records_exact_inputs_and_feedback_membership(monkeypatch) -> None:
    manifest = synthetic_manifest()
    prior = _reference("news:evaluation-manifest:prior", "9" * 64, "prior.json")
    manifest = manifest.model_copy(update={"prior_manifest": prior})

    batches = []
    objects = []
    _patch_catalog_references(monkeypatch, manifest)
    monkeypatch.setattr(evaluations, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(evaluations, "run_status", lambda _run_id: None)
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda values: objects.extend(values)
        or SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    reference = evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    assert reference.artifact_id == f"news:evaluation-manifest:{manifest.version}"

    assert objects == [(reference.r2_key, objects[0][1])]
    inputs = [parameters for sql, parameters in batches[0] if "INTO run_inputs" in sql]
    expected = _manifest_reference_ids(manifest.model_dump(mode="json"))
    assert {parameters[2] for parameters in inputs} == expected
    assert prior.version_id in {parameters[2] for parameters in inputs}

    feedback = [
        parameters
        for sql, parameters in batches[0]
        if "INTO news_evaluation_feedback_sources" in sql
    ]
    excluded = {item.feedback_id for item in manifest.excluded_feedback}
    assert feedback == [
        [
            reference.version_id,
            position,
            str(feedback_id),
            "excluded" if feedback_id in excluded else "represented",
        ]
        for position, feedback_id in enumerate(manifest.source_feedback_ids)
    ]


def test_publish_manifest_reuses_completed_publication(monkeypatch) -> None:
    manifest, _payloads = _relevance_manifest()
    _patch_catalog_references(monkeypatch, manifest)
    monkeypatch.setattr(evaluations, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(evaluations, "run_status", lambda _run_id: "completed")
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda _values: (_ for _ in ()).throw(AssertionError("unexpected upload")),
    )
    monkeypatch.setattr(
        evaluations,
        "catalog_batch",
        lambda _statements: (_ for _ in ()).throw(AssertionError("unexpected catalog write")),
    )

    first = evaluations.publish_news_evaluation_manifest(manifest, "git:test")
    second = evaluations.publish_news_evaluation_manifest(manifest, "git:test")

    assert first == second


def test_publish_manifest_rejects_changed_content_for_same_version(monkeypatch) -> None:
    manifest, _payloads = _relevance_manifest()
    _patch_catalog_references(monkeypatch, manifest)
    monkeypatch.setattr(
        evaluations,
        "current_artifact_file",
        lambda _artifact_id: SimpleNamespace(content_digest="f" * 64),
    )

    with pytest.raises(ValueError, match="already has different content"):
        evaluations.publish_news_evaluation_manifest(manifest, "git:test")


@pytest.mark.parametrize("field", ("content_digest", "r2_key"))
def test_publish_manifest_rejects_forged_existing_reference_before_writes(
    monkeypatch, field
) -> None:
    manifest = synthetic_manifest()
    references = evaluations._manifest_inputs(manifest)
    catalog = {reference.version_id: reference for reference in references}
    report = manifest.reports[0].report
    forged = report.model_copy(
        update={field: "f" * 64 if field == "content_digest" else "forged.json"}
    )
    payload = manifest.model_dump(mode="json")
    payload["reports"][0]["report"] = forged.model_dump(mode="json")
    manifest = NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)
    monkeypatch.setattr(
        evaluations,
        "catalog_query",
        lambda _sql, parameters: [
            catalog[version_id].model_dump(mode="python") for version_id in parameters
        ],
    )
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda _values: (_ for _ in ()).throw(AssertionError("unexpected R2 write")),
    )
    monkeypatch.setattr(
        evaluations,
        "catalog_batch",
        lambda _statements: (_ for _ in ()).throw(AssertionError("unexpected catalog write")),
    )

    with pytest.raises(ValueError, match="does not match the catalog"):
        evaluations.publish_news_evaluation_manifest(manifest, "git:test")


def test_publish_baseline_binds_manifest_as_exact_input(monkeypatch) -> None:
    manifest, _payloads = _relevance_manifest()
    manifest_reference = _reference(
        "news:evaluation-manifest:synthetic-v1", "3" * 64, "manifest.json"
    )
    baseline = NewsEvaluationBaseline(
        manifest=manifest_reference,
        dataset_version=manifest.version,
        cases=(),
        deterministic_checks=(),
        aggregates=(),
    )
    batches = []
    monkeypatch.setattr(evaluations, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(evaluations, "run_status", lambda _run_id: None)
    monkeypatch.setattr(evaluations, "_catalog_reference", lambda _version_id: manifest_reference)
    monkeypatch.setattr(
        evaluations,
        "publish_immutable_r2_objects",
        lambda _values: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    reference = evaluations.publish_news_evaluation_baseline(baseline, "git:evaluator")

    inputs = [parameters for sql, parameters in batches[0] if "INTO run_inputs" in sql]
    assert reference.artifact_id == "news:evaluation-baseline:synthetic-v1"
    assert [parameters[2] for parameters in inputs] == [manifest_reference.version_id]


def test_load_release_resolves_pin_verifies_bytes_and_hydrates(monkeypatch) -> None:
    manifest, payloads = _relevance_manifest()
    manifest_content = evaluations.canonical_json(manifest.model_dump(mode="json"))
    manifest_reference = _reference(
        "news:evaluation-manifest:synthetic-v1",
        hashlib.sha256(manifest_content).hexdigest(),
        "manifest.json",
    )
    baseline = NewsEvaluationBaseline(
        manifest=manifest_reference,
        dataset_version=manifest.version,
        cases=(),
        deterministic_checks=(),
        aggregates=(),
    )
    baseline_content = evaluations.canonical_json(baseline.model_dump(mode="json"))
    baseline_reference = _reference(
        "news:evaluation-baseline:synthetic-v1",
        hashlib.sha256(baseline_content).hexdigest(),
        "baseline.json",
    )
    references = {
        manifest_reference.version_id: manifest_reference,
        baseline_reference.version_id: baseline_reference,
    }
    contents = {
        manifest_reference.r2_key: manifest_content,
        baseline_reference.r2_key: baseline_content,
        **payloads,
    }
    reads = []

    def query(_sql, parameters):
        reference = references[parameters[0]]
        return [reference.model_dump(mode="python")]

    def read(key, digest):
        reads.append((key, digest))
        return contents[key]

    monkeypatch.setattr(evaluations, "catalog_query", query)
    monkeypatch.setattr(evaluations, "read_verified_r2_object", read)
    pin = NewsEvaluationPin(
        manifest_version_id=manifest_reference.version_id,
        baseline_version_id=baseline_reference.version_id,
    )

    release = evaluations.load_news_evaluation_release(pin.model_dump_json())

    case = release.dataset.cases[0]
    assert isinstance(case, RelevanceEvaluationCase)
    assert isinstance(case.model_response, RelevanceDecision)
    assert relevance_is_accepted(case.model_response, RELEVANCE_POLICY_V1) is True
    assert reads[:2] == [
        (manifest_reference.r2_key, manifest_reference.content_digest),
        (baseline_reference.r2_key, baseline_reference.content_digest),
    ]
    assert set(pin.model_dump()) == {"manifest_version_id", "baseline_version_id"}
    assert "body" not in pin.model_dump_json()


def test_load_release_rejects_baseline_for_another_manifest(monkeypatch) -> None:
    manifest, _payloads = _relevance_manifest()
    manifest_content = evaluations.canonical_json(manifest.model_dump(mode="json"))
    manifest_reference = _reference(
        "news:evaluation-manifest:synthetic-v1",
        hashlib.sha256(manifest_content).hexdigest(),
        "manifest.json",
    )
    other_manifest = manifest_reference.model_copy(update={"artifact_id": "news:evaluation:other"})
    baseline = NewsEvaluationBaseline(
        manifest=other_manifest,
        dataset_version=manifest.version,
        cases=(),
        deterministic_checks=(),
        aggregates=(),
    )
    baseline_content = evaluations.canonical_json(baseline.model_dump(mode="json"))
    baseline_reference = _reference(
        "news:evaluation-baseline:synthetic-v1",
        hashlib.sha256(baseline_content).hexdigest(),
        "baseline.json",
    )
    references = {
        manifest_reference.version_id: manifest_reference,
        baseline_reference.version_id: baseline_reference,
    }
    contents = {
        manifest_reference.r2_key: manifest_content,
        baseline_reference.r2_key: baseline_content,
    }
    monkeypatch.setattr(
        evaluations,
        "catalog_query",
        lambda _sql, parameters: [references[parameters[0]].model_dump(mode="python")],
    )
    monkeypatch.setattr(
        evaluations,
        "read_verified_r2_object",
        lambda key, _digest: contents[key],
    )
    pin = NewsEvaluationPin(
        manifest_version_id=manifest_reference.version_id,
        baseline_version_id=baseline_reference.version_id,
    )

    with pytest.raises(ValueError, match="does not bind"):
        evaluations.load_news_evaluation_release(pin.model_dump_json())


def test_projection_receipt_boundary_preserves_identity_and_write_order(monkeypatch) -> None:
    manifest_version_id = "1" * 64
    row = {
        "provider": "langfuse",
        "manifest_artifact_version_id": manifest_version_id,
        "projection_kind": "experiment",
        "dataset_id": "dataset-id",
        "dataset_name": "dataset-name",
        "experiment_id": "experiment-id",
        "experiment_name": "experiment-name",
        "experiment_url": "https://example.com/experiment",
        "implementation_ref": "git:abc123",
    }
    queries = []
    batches = []
    monkeypatch.setattr(
        evaluations,
        "catalog_query",
        lambda sql, parameters: queries.append((sql, parameters)) or [row],
    )
    monkeypatch.setattr(evaluations, "catalog_batch", batches.append)

    receipt = evaluations.read_news_evaluation_projection_receipt(
        "langfuse", manifest_version_id, "experiment", "git:abc123"
    )
    assert receipt is not None
    assert receipt.model_dump(mode="json") == row
    evaluations.record_news_evaluation_projection_receipt(receipt)

    assert queries[0][1] == ["langfuse", manifest_version_id, "experiment", "git:abc123"]
    assert batches[0][0][1][:-1] == [
        "langfuse",
        manifest_version_id,
        "experiment",
        "dataset-id",
        "dataset-name",
        "experiment-id",
        "experiment-name",
        "https://example.com/experiment",
        "git:abc123",
    ]
    datetime.fromisoformat(batches[0][0][1][-1])


def test_projection_receipt_models_reject_cross_kind_states() -> None:
    with pytest.raises(ValidationError):
        evaluations.NewsEvaluationDatasetReceipt.model_validate(
            {
                "provider": "langfuse",
                "manifest_artifact_version_id": "1" * 64,
                "dataset_id": "dataset-id",
                "dataset_name": "dataset-name",
                "experiment_id": "experiment-id",
            }
        )

    with pytest.raises(ValidationError):
        evaluations.NewsEvaluationExperimentReceipt(
            provider="langfuse",
            manifest_artifact_version_id="1" * 64,
            dataset_id="dataset-id",
            dataset_name="dataset-name",
            experiment_id="experiment-id",
            experiment_name="experiment-name",
            experiment_url=None,
            implementation_ref=" ",
        )


def test_evaluation_feedback_boundary_parses_rows_and_preserves_query_order(monkeypatch) -> None:
    feedback_ids = (
        UUID("00000000-0000-4000-8000-000000000001"),
        UUID("00000000-0000-4000-8000-000000000002"),
    )
    calls = []
    monkeypatch.setattr(
        evaluations,
        "catalog_query",
        lambda sql, parameters: calls.append((sql, parameters))
        or [
            {
                "feedback_id": str(feedback_ids[0]),
                "report_version_id": "1" * 64,
                "target_kind": "group",
                "theme_id": None,
                "group_id": "2" * 64,
                "article_version_id": None,
                "rating": "positive",
                "note": None,
                "actor": "owner",
                "created_at": "2026-09-03T20:37:00+00:00",
            }
        ],
    )

    rows = evaluations.read_news_evaluation_feedback(feedback_ids)

    assert rows[0].feedback_id == feedback_ids[0]
    assert rows[0].target.kind == "group"
    assert rows[0].created_at == datetime(2026, 9, 3, 20, 37, tzinfo=UTC)
    assert "ORDER BY report_version_id, created_at, feedback_id" in calls[0][0]
    assert calls[0][1] == [str(feedback_id) for feedback_id in feedback_ids]


def test_report_lineage_boundary_returns_exact_ordered_inputs(monkeypatch) -> None:
    report = _reference("news:daily:2026-09-03", "1" * 64, "report.json")
    cluster = _reference("news:cluster:2026-09-03", "2" * 64, "cluster.json")
    summary = _reference("news:summary:test", "3" * 64, "summary.json")
    calls = []

    def query(sql, parameters):
        calls.append((sql, parameters))
        if "artifact.kind = 'news_daily_report'" in sql:
            return [{**report.model_dump(mode="python"), "produced_by_run_id": "run-id"}]
        return [
            {"position": 2, "role": "cluster_set", **cluster.model_dump(mode="python")},
            {"position": 3, "role": "summary", **summary.model_dump(mode="python")},
        ]

    monkeypatch.setattr(evaluations, "catalog_query", query)

    lineage = evaluations.read_news_evaluation_report_lineage(report.version_id)

    assert lineage.report == report
    assert lineage.inputs.cluster_set == (cluster,)
    assert lineage.inputs.summary == (summary,)
    assert "input.role != 'prior_output'" in calls[1][0]
    assert "ORDER BY input.position" in calls[1][0]
    assert calls[1][1] == ["run-id"]


def test_evaluation_artifact_reference_boundary_batches_fifty_versions(monkeypatch) -> None:
    references = tuple(
        _reference(f"news:article:{position}", f"{position:064x}", f"{position}.json")
        for position in range(1, 52)
    )
    by_version = {reference.version_id: reference for reference in references}
    batches = []

    def query(_sql, parameters):
        batches.append(tuple(parameters))
        return [by_version[version_id].model_dump(mode="python") for version_id in parameters]

    monkeypatch.setattr(evaluations, "catalog_query", query)

    result = evaluations.read_news_evaluation_artifact_references(
        tuple(reference.version_id for reference in references)
    )

    assert result == by_version
    assert tuple(len(batch) for batch in batches) == (50, 1)
