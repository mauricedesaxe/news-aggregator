import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

import romanian_news.national_consequence_evaluation as consequence
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.evaluation import (
    EvaluationSpecProvenance,
    ExcludedConcernDisposition,
    ExecutableConcernDisposition,
    FeedbackReview,
    NewsEvaluationManifest,
    RankingEvaluationSpec,
    ReportEvaluationSpec,
    TierConcernDisposition,
    TierEvaluationSpec,
)
from romanian_news.feedback import GroupFeedbackTarget
from romanian_news.national_consequence_evaluation import (
    BET_CHANGES_FEEDBACK_ID,
    BET_CHANGES_TIER_CASE_ID,
    DATASET_VERSION,
    DEFICIT_MORATORIUM_FEEDBACK_ID,
    DEFICIT_MORATORIUM_REVIEW,
    NEW_FEEDBACK_REVIEWS,
    QUESTIONED_PNL_CRITICISM_FEEDBACK_ID,
    QUESTIONED_PNL_CRITICISM_REVIEW,
    RANKING_SPECS,
    REPORT_2026_09_02,
    REPORT_2026_09_03,
    REVIEWED_AT,
    SEPTEMBER_2_GROUPS,
    SEPTEMBER_3_GROUPS,
    V8_MANIFEST_VERSION_ID,
    V10_DATASET_VERSION,
    V10_MANIFEST_VERSION_ID,
)
from romanian_news.september_10_evaluation import REPORT_VERSION_ID as REPORT_2026_09_10


def _report_reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:daily:{version_id[:10]}",
        version_id=version_id,
        content_digest=version_id,
        r2_key=f"news/recorded/{version_id[:10]}.json",
    )


METROREX_GROUP_ID = "6ff7f531f9adf2485ce61c77c71af37b32b2a3148900da7c8eff136099a09f9e"
STB_GROUP_ID = "2c876749d928c2fd12a4c70128187694844999545bbf505b8303177c43f3045a"


def _expected_manifest() -> NewsEvaluationManifest:
    ranking_cases = tuple(
        RankingEvaluationSpec(
            case_id=spec.case_id,
            control=spec.control,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(UUID(spec.feedback_id),),
                report_version_id=spec.report_version_id,
            ),
            relevance=(),
            higher_group_id=spec.higher_group_id,
            lower_group_id=spec.lower_group_id,
        )
        for spec in RANKING_SPECS
    )
    return NewsEvaluationManifest(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=consequence.ISSUE_URL,
        prior_manifest=_report_reference(V8_MANIFEST_VERSION_ID),
        source_feedback_ids=(
            QUESTIONED_PNL_CRITICISM_FEEDBACK_ID,
            DEFICIT_MORATORIUM_FEEDBACK_ID,
        ),
        reports=(
            ReportEvaluationSpec(
                report=_report_reference(REPORT_2026_09_02),
                cluster_set=_report_reference(REPORT_2026_09_02),
            ),
            ReportEvaluationSpec(
                report=_report_reference(REPORT_2026_09_03),
                cluster_set=_report_reference(REPORT_2026_09_03),
            ),
        ),
        cases=ranking_cases,
        feedback_reviews=NEW_FEEDBACK_REVIEWS,
    )


def _pinned_v10_manifest(*, with_tier_case: bool = True) -> NewsEvaluationManifest:
    concerns = []
    if with_tier_case:
        concerns.append(
            TierConcernDisposition(
                judgment="worth_knowing",
                case_ids=(BET_CHANGES_TIER_CASE_ID,),
                rationale="The subject is useful financial context.",
            )
        )
    else:
        concerns.append(
            ExcludedConcernDisposition(
                concern="tier",
                reason="conditional",
                rationale="Placeholder disposition for the without-judgment variant.",
            )
        )
    cases: list[RankingEvaluationSpec | TierEvaluationSpec] = [
        RankingEvaluationSpec(
            case_id="inherited-case",
            provenance=EvaluationSpecProvenance(
                feedback_ids=(),
                report_version_id=REPORT_2026_09_02,
            ),
            relevance=(),
            higher_group_id="a" * 64,
            lower_group_id="b" * 64,
        )
    ]
    if with_tier_case:
        cases.append(
            TierEvaluationSpec(
                case_id=BET_CHANGES_TIER_CASE_ID,
                control=False,
                provenance=EvaluationSpecProvenance(
                    feedback_ids=(BET_CHANGES_FEEDBACK_ID,),
                    report_version_id=REPORT_2026_09_10,
                    group_id=STB_GROUP_ID,
                ),
                expected_tier="worth_knowing",
            )
        )
    return NewsEvaluationManifest(
        version=V10_DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=consequence.ISSUE_URL,
        prior_manifest=_report_reference(V10_MANIFEST_VERSION_ID),
        source_feedback_ids=(BET_CHANGES_FEEDBACK_ID,),
        reports=(
            ReportEvaluationSpec(
                report=_report_reference(REPORT_2026_09_02),
                cluster_set=_report_reference(REPORT_2026_09_02),
            ),
            ReportEvaluationSpec(
                report=_report_reference(REPORT_2026_09_03),
                cluster_set=_report_reference(REPORT_2026_09_03),
            ),
            ReportEvaluationSpec(
                report=_report_reference(REPORT_2026_09_10),
                cluster_set=_report_reference(REPORT_2026_09_10),
            ),
        ),
        cases=tuple(cases),
        feedback_reviews=(
            FeedbackReview(
                feedback_id=BET_CHANGES_FEEDBACK_ID,
                target=GroupFeedbackTarget(
                    report_version_id=REPORT_2026_09_10,
                    group_id=STB_GROUP_ID,
                ),
                concerns=tuple(concerns),
            ),
        ),
    )


def test_v9_declaration_covers_exact_reviewed_feedback_and_anchors() -> None:
    assert str(QUESTIONED_PNL_CRITICISM_FEEDBACK_ID) == "2e080f78-746b-42ab-9fe9-e67879c7df02"
    assert str(DEFICIT_MORATORIUM_FEEDBACK_ID) == "2cef857b-3ca8-4978-807d-070e858fbbb7"
    assert QUESTIONED_PNL_CRITICISM_REVIEW.target.model_dump(mode="json") == {
        "kind": "group",
        "report_version_id": REPORT_2026_09_02,
        "group_id": SEPTEMBER_2_GROUPS["questioned_pnl_criticism"],
    }
    assert DEFICIT_MORATORIUM_REVIEW.target.model_dump(mode="json") == {
        "kind": "group",
        "report_version_id": REPORT_2026_09_03,
        "group_id": SEPTEMBER_3_GROUPS["deficit_moratorium"],
    }
    assert {
        spec.case_id: (spec.higher_group_id, spec.lower_group_id, spec.control)
        for spec in RANKING_SPECS
    } == {
        "sep02-government-collapse-ranks-above-questioned-pnl-criticism": (
            SEPTEMBER_2_GROUPS["government_collapse"],
            SEPTEMBER_2_GROUPS["questioned_pnl_criticism"],
            False,
        ),
        "sep03-deficit-moratorium-ranks-above-opinion-led-commentary": (
            SEPTEMBER_3_GROUPS["deficit_moratorium"],
            SEPTEMBER_3_GROUPS["opinion_led_commentary"],
            False,
        ),
        "sep03-deficit-moratorium-stays-above-irrelevant-climate-control": (
            SEPTEMBER_3_GROUPS["deficit_moratorium"],
            SEPTEMBER_3_GROUPS["irrelevant_climate_control"],
            True,
        ),
    }


def test_v9_review_dispositions_link_every_ranking_case() -> None:
    linked = {
        case_id
        for review in NEW_FEEDBACK_REVIEWS
        for disposition in review.concerns
        if isinstance(disposition, ExecutableConcernDisposition)
        for case_id in disposition.case_ids
    }
    assert linked == {spec.case_id for spec in RANKING_SPECS}


def test_v9_manifest_serializes_and_parses_strictly() -> None:
    manifest = _expected_manifest()
    content = manifest.model_dump_json()

    parsed = NewsEvaluationManifest.model_validate_json(content, strict=True)

    assert parsed == manifest


def test_v9_builder_wires_ranking_cases_under_mocked_boundaries(monkeypatch) -> None:
    expected = _expected_manifest()
    expected_by_id = {case.case_id: case for case in expected.cases}
    prior_snapshots = tuple(
        SimpleNamespace(
            report=_report_reference(version_id),
            themes=None,
            cluster_set=_report_reference(version_id),
        )
        for version_id in (REPORT_2026_09_02, REPORT_2026_09_03)
    )
    prior = NewsEvaluationManifest(
        version="news-evaluation-2026-09-10-v8",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/prior",
        source_feedback_ids=(),
        reports=(),
        cases=(),
    )
    snapshots = {
        REPORT_2026_09_02: SimpleNamespace(
            report="sept2-report", report_group_ids=tuple(SEPTEMBER_2_GROUPS.values())
        ),
        REPORT_2026_09_03: SimpleNamespace(
            report="sept3-report", report_group_ids=tuple(SEPTEMBER_3_GROUPS.values())
        ),
    }

    monkeypatch.setattr(
        consequence,
        "read_evaluation_artifact",
        lambda _reference: prior.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "read_news_evaluation_artifact_references",
        lambda ids: {V8_MANIFEST_VERSION_ID: _report_reference(V8_MANIFEST_VERSION_ID)},
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "hydrate_news_evaluation_manifest",
        lambda _manifest: SimpleNamespace(
            version="news-evaluation-2026-09-10-v8",
            source_feedback_ids=(),
            reports=prior_snapshots,
            cases=(),
            feedback_reviews=(),
        ),
    )
    monkeypatch.setattr(
        consequence,
        "read_evaluation_report_bundle",
        lambda report_version_id: SimpleNamespace(snapshot=snapshots[report_version_id]),
    )
    monkeypatch.setattr(
        consequence,
        "build_ranking_case",
        lambda spec, _bundle: expected_by_id[spec.case_id],
    )
    monkeypatch.setattr(consequence, "evaluation_case_spec", lambda case: case)
    monkeypatch.setattr(
        consequence,
        "NewsEvaluationDataset",
        lambda **values: SimpleNamespace(**values),
    )

    manifest = consequence.build_national_consequence_evaluation_manifest()

    assert manifest == expected


def test_v9_builder_rejects_unfrozen_reviewed_reports(monkeypatch) -> None:
    prior = NewsEvaluationManifest(
        version="news-evaluation-2026-09-10-v8",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/prior",
        source_feedback_ids=(),
        reports=(),
        cases=(),
    )

    monkeypatch.setattr(
        consequence,
        "read_evaluation_artifact",
        lambda _reference: prior.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "read_news_evaluation_artifact_references",
        lambda ids: {V8_MANIFEST_VERSION_ID: _report_reference(V8_MANIFEST_VERSION_ID)},
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "hydrate_news_evaluation_manifest",
        lambda _manifest: prior,
    )

    with pytest.raises(ValueError, match="must freeze the reviewed September 2 and 3 reports"):
        consequence.build_national_consequence_evaluation_manifest()


def test_v9_builder_rejects_missing_reviewed_anchors(monkeypatch) -> None:
    prior = NewsEvaluationManifest(
        version="news-evaluation-2026-09-10-v8",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/prior",
        source_feedback_ids=(),
        reports=(),
        cases=(),
    )
    prior_snapshots = tuple(
        SimpleNamespace(
            report=_report_reference(version_id),
            themes=None,
            cluster_set=_report_reference(version_id),
        )
        for version_id in (REPORT_2026_09_02, REPORT_2026_09_03)
    )
    snapshots = {
        REPORT_2026_09_02: SimpleNamespace(report="sept2-report", report_group_ids=()),
        REPORT_2026_09_03: SimpleNamespace(report="sept3-report", report_group_ids=()),
    }

    monkeypatch.setattr(
        consequence,
        "read_evaluation_artifact",
        lambda _reference: prior.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "read_news_evaluation_artifact_references",
        lambda ids: {V8_MANIFEST_VERSION_ID: _report_reference(V8_MANIFEST_VERSION_ID)},
    )
    monkeypatch.setattr(
        consequence.evaluation_catalog,
        "hydrate_news_evaluation_manifest",
        lambda _manifest: SimpleNamespace(
            version="news-evaluation-2026-09-10-v8",
            source_feedback_ids=(),
            reports=prior_snapshots,
            cases=(),
            feedback_reviews=(),
        ),
    )
    monkeypatch.setattr(
        consequence,
        "read_evaluation_report_bundle",
        lambda report_version_id: SimpleNamespace(snapshot=snapshots[report_version_id]),
    )

    with pytest.raises(ValueError, match="absent from their frozen reports"):
        consequence.build_national_consequence_evaluation_manifest()


def test_duplicate_reviewed_anchor_pairs_are_rejected_by_the_manifest() -> None:
    duplicate = _expected_manifest().model_dump(mode="json")
    duplicate["cases"].append(duplicate["cases"][0])

    with pytest.raises(ValidationError, match="case IDs must be unique"):
        NewsEvaluationManifest.model_validate_json(json.dumps(duplicate), strict=True)


def test_v11_resolution_retires_the_financial_signals_tier_case(monkeypatch) -> None:
    pinned = _pinned_v10_manifest()
    monkeypatch.setattr(
        consequence,
        "_pinned_prior",
        lambda version_id, expected: (_report_reference(V10_MANIFEST_VERSION_ID), pinned),
    )

    resolved = consequence.build_resolved_tier_evaluation_manifest()

    assert resolved.version == consequence.RESOLUTION_DATASET_VERSION
    assert resolved.prior_manifest == _report_reference(V10_MANIFEST_VERSION_ID)
    assert all(case.case_id != BET_CHANGES_TIER_CASE_ID for case in resolved.cases)
    assert len(resolved.cases) == len(pinned.cases) - 1
    review = resolved.feedback_reviews[0]
    tiers = [item for item in review.concerns if item.concern == "tier"]
    assert len(tiers) == 1
    disposition = tiers[0]
    assert isinstance(disposition, ExcludedConcernDisposition)
    assert disposition.reason == "conditional"
    assert disposition.rationale == (
        "The standalone subject is useful financial context; once grouped with the BET "
        "movement and BVB decline into the reviewed financial-signals subject, the subject "
        "keeps their reviewed main tier."
    )


def test_v11_resolution_requires_the_pinned_tier_case(monkeypatch) -> None:
    pinned = _pinned_v10_manifest(with_tier_case=False)
    monkeypatch.setattr(
        consequence,
        "_pinned_prior",
        lambda version_id, expected: (_report_reference(V10_MANIFEST_VERSION_ID), pinned),
    )

    with pytest.raises(ValueError, match="does not carry the reviewed tier cases"):
        consequence.build_resolved_tier_evaluation_manifest()


def test_tier_conflict_resolution_expects_the_worth_knowing_judgment() -> None:
    review = FeedbackReview(
        feedback_id=BET_CHANGES_FEEDBACK_ID,
        target=GroupFeedbackTarget(
            report_version_id=REPORT_2026_09_10,
            group_id=STB_GROUP_ID,
        ),
        concerns=(
            TierConcernDisposition(
                judgment="main",
                case_ids=(BET_CHANGES_TIER_CASE_ID,),
                rationale="Unexpected reviewed judgment.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects the retired standalone tier judgment"):
        consequence._resolve_tier_conflicts(review)
