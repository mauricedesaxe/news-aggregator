import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from romanian_news import september_10_evaluation as september
from romanian_news.evaluation import (
    ConfidenceEvaluationSpec,
    EvaluationSpecProvenance,
    ExcludedFeedbackScore,
    ExecutableConcernDisposition,
    NewsEvaluationManifest,
    RankingEvaluationSpec,
    RelevanceEvaluationSpec,
    ReportEvaluationSpec,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
    TierConcernDisposition,
    TierEvaluationSpec,
)
from romanian_news.feedback import NewsFeedbackEvent
from romanian_news.september_10_evaluation import (
    CLUSTER_REFERENCE,
    CONFIDENCE_SPECS,
    CURRENT_FEEDBACK_IDS,
    DATASET_VERSION,
    FEEDBACK_REVIEWS,
    GROUP_IDS,
    ISSUE_URL,
    PRIOR_MANIFEST_REFERENCE,
    RANKING_SPECS,
    RELEVANCE_SPECS,
    REPORT_REFERENCE,
    REPORT_VERSION_ID,
    REVIEWED_AT,
    SOURCE_FEEDBACK_IDS,
    THEME_IDS,
    THEME_JUDGMENTS,
    THEMES_REFERENCE,
    TIER_SPECS,
    UNREVIEWED_THEMES,
    require_exact_september_10_feedback,
)

EXPECTED_FEEDBACK_IDS = (
    "4f08b504-ea65-4baa-93f0-9aec04d69753",
    "4f87f587-61f0-4259-881e-ab836a459977",
    "351ede8c-9398-4051-afd3-8ac9a3ef3f55",
    "37399f0e-f030-45f5-b5cb-fb32c0f5af3c",
    "5fcfb650-9c77-4604-9f04-656a6e31a6f9",
    "12712b04-127f-4260-83bf-dd923ffc2213",
    "b2f92241-5a4a-4425-be07-5f646b42ddbc",
    "736f39d8-4b30-4da9-820e-bda2c29f5917",
    "a44faaff-9135-43fb-b749-c7e71914b804",
    "2b82f2e5-1bf6-48bf-bf64-caaf953fd6fb",
    "380ab371-7eb0-4c71-ad1d-4651a117e086",
    "e2f3019c-423f-4ea8-a66b-ce22d6678164",
    "0c071aa1-9cb0-4ac4-bf78-fa36f3114349",
    "44a396e9-8613-4001-9b5f-384b613cafb5",
    "ebdc6e3f-7fa7-4e63-a296-1159bcb45f90",
    "b19a8101-5c35-4e31-b88e-3e5a7c9261c4",
    "650de0b3-3307-4f60-a24d-3c452f46ac87",
    "f844f25c-02db-4726-9913-13e0045a608c",
    "099eabde-d07c-4740-ba53-42e5ee0afdc5",
    "9ee26633-222a-4445-b934-90532b6c0551",
    "aba588b6-5411-45aa-8457-7e0bde6f7408",
    "b485f4d4-0405-4828-9302-0e55d43758aa",
    "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
    "fe4c9127-8c7b-4528-942a-07ed1a9d58be",
    "cf08cc52-5f32-47a2-8de4-e3d687383960",
    "a395ade1-9f1d-4a3c-87dd-17fe3fcf3328",
    "7adad8d4-623a-4892-8cf2-b3cdb79c3479",
    "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
    "6c92cf32-87cb-43c0-aee3-0428699df6f2",
    "7dab7cb3-b7ce-4e9a-b7d0-1cfddabeb122",
    "1e5226d4-f060-428f-a5b6-7713821cb565",
    "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
)
EXPECTED_TARGET_POSITIONS = (
    1,
    None,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    19,
    20,
    21,
    21,
    22,
    23,
    24,
    25,
    25,
    26,
    27,
    28,
    29,
)


def _manifest() -> NewsEvaluationManifest:
    relevance_cases = tuple(
        RelevanceEvaluationSpec(
            case_id=case_id,
            control=spec.control,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(UUID(spec.feedback_id),),
                report_version_id=REPORT_VERSION_ID,
                group_id=spec.group_id,
            ),
            expected_accepted=spec.expected_accepted,
            article=REPORT_REFERENCE.model_copy(
                update={
                    "artifact_id": f"news:article:{case_id}",
                    "version_id": f"{position:064x}",
                }
            ),
            model_output=THEMES_REFERENCE.model_copy(
                update={
                    "artifact_id": f"news:relevance:{case_id}",
                    "version_id": f"{position + 100:064x}",
                }
            ),
        )
        for position, (spec, case_id) in enumerate(
            ((spec, case_id) for spec in RELEVANCE_SPECS for case_id in spec.case_ids),
            start=1,
        )
    )
    ranking_cases = tuple(
        RankingEvaluationSpec(
            case_id=spec.case_id,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(UUID(spec.feedback_id),),
                report_version_id=REPORT_VERSION_ID,
            ),
            relevance=(),
            higher_group_id=spec.higher_group_id,
            lower_group_id=spec.lower_group_id,
        )
        for spec in RANKING_SPECS
    )
    expectations = tuple(
        ThemePairExpectation(
            case_id=judgment.judgment_id,
            feedback_ids=judgment.feedback_ids,
            left_group_id=judgment.left_group_id,
            right_group_id=judgment.right_group_id,
            expected_same_theme=judgment.expected_same_theme,
            rationale=judgment.rationale,
        )
        for judgment in THEME_JUDGMENTS
    )
    theme_case = ThemeEvaluationDaySpec(
        case_id="2026-09-10-c1369f9a2420-daily-themes",
        provenance=EvaluationSpecProvenance(
            feedback_ids=tuple(
                dict.fromkeys(
                    feedback_id
                    for expectation in expectations
                    for feedback_id in expectation.feedback_ids
                )
            ),
            report_version_id=REPORT_VERSION_ID,
        ),
        day=date(2026, 9, 10),
        source_report=REPORT_REFERENCE,
        cluster_set=CLUSTER_REFERENCE,
        summaries=(),
        expectations=expectations,
        model_output=THEMES_REFERENCE,
    )
    tier_cases = tuple(
        TierEvaluationSpec(
            case_id=spec.case_id,
            control=spec.control,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(UUID(spec.feedback_id),),
                report_version_id=REPORT_VERSION_ID,
                group_id=spec.group_id,
            ),
            expected_tier=spec.expected_tier,
        )
        for spec in TIER_SPECS
    )
    confidence_cases = tuple(
        ConfidenceEvaluationSpec(
            case_id=spec.case_id,
            control=spec.control,
            provenance=EvaluationSpecProvenance(
                feedback_ids=(UUID(spec.feedback_id),),
                report_version_id=REPORT_VERSION_ID,
                group_id=spec.group_id,
            ),
            expected_sufficient=spec.expected_sufficient,
        )
        for spec in CONFIDENCE_SPECS
    )
    return NewsEvaluationManifest(
        version=DATASET_VERSION,
        reviewed_at=REVIEWED_AT,
        issue_url=ISSUE_URL,
        prior_manifest=PRIOR_MANIFEST_REFERENCE,
        source_feedback_ids=SOURCE_FEEDBACK_IDS,
        reports=(
            ReportEvaluationSpec(
                report=REPORT_REFERENCE,
                themes=THEMES_REFERENCE,
                cluster_set=CLUSTER_REFERENCE,
            ),
        ),
        cases=(
            *relevance_cases,
            *ranking_cases,
            *tier_cases,
            *confidence_cases,
            theme_case,
        ),
        feedback_reviews=FEEDBACK_REVIEWS,
        unreviewed_themes=UNREVIEWED_THEMES,
    )


def test_v7_declaration_covers_exact_events_targets_and_concerns() -> None:
    assert tuple(str(value) for value in SOURCE_FEEDBACK_IDS) == EXPECTED_FEEDBACK_IDS
    assert len(set(SOURCE_FEEDBACK_IDS)) == 32
    assert len(FEEDBACK_REVIEWS) == 32
    for review, expected_position in zip(FEEDBACK_REVIEWS, EXPECTED_TARGET_POSITIONS, strict=True):
        if expected_position is None:
            assert review.target.kind == "report"
        else:
            assert review.target.kind == "theme"
            assert review.target.theme_id == THEME_IDS[expected_position]
    assert all(
        len({item.concern for item in review.concerns}) == len(review.concerns)
        for review in FEEDBACK_REVIEWS
    )

    current = tuple(
        review for review in FEEDBACK_REVIEWS if review.concerns[0].kind != "superseded"
    )
    expected_targets = {review.target.model_dump_json() for review in current}
    assert len(current) == 29
    assert len(expected_targets) == 29
    assert {review.target.report_version_id for review in current} == {REPORT_VERSION_ID}
    assert {review.target.theme_id for review in current if review.target.kind == "theme"} == {
        theme_id for position, theme_id in THEME_IDS.items() if position != 17
    }


def test_v7_supersession_and_unreviewed_subject_are_exact() -> None:
    superseded = {
        str(review.feedback_id): str(review.concerns[0].superseded_by_feedback_id)
        for review in FEEDBACK_REVIEWS
        if review.concerns[0].kind == "superseded"
    }

    assert superseded == {
        "380ab371-7eb0-4c71-ad1d-4651a117e086": "e2f3019c-423f-4ea8-a66b-ce22d6678164",
        "b485f4d4-0405-4828-9302-0e55d43758aa": "33c1dccc-e4d6-4ec5-9775-7c5a53fd952b",
        "7adad8d4-623a-4892-8cf2-b3cdb79c3479": "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
    }
    assert len(UNREVIEWED_THEMES) == 1
    assert UNREVIEWED_THEMES[0].subject_position == 17
    assert UNREVIEWED_THEMES[0].target.theme_id == THEME_IDS[17]
    assert all(review.target != UNREVIEWED_THEMES[0].target for review in FEEDBACK_REVIEWS)


def test_firm_relevance_tiers_and_confidence_link_to_executable_cases() -> None:
    reviews = {str(review.feedback_id): review for review in FEEDBACK_REVIEWS}
    assert not any(
        item.kind == "excluded" and item.reason == "unsupported"
        for review in FEEDBACK_REVIEWS
        for item in review.concerns
    )

    castle = reviews["cf08cc52-5f32-47a2-8de4-e3d687383960"]
    castle_relevance = next(item for item in castle.concerns if item.concern == "relevance")
    assert castle_relevance.kind == "represented"
    assert castle_relevance.judgment == "irrelevant"
    assert castle_relevance.case_ids == ("sep10-subject-23-article-1-relevance",)

    expected_tiers = {
        "f844f25c-02db-4726-9913-13e0045a608c": "excluded",
        "fe4c9127-8c7b-4528-942a-07ed1a9d58be": "excluded",
        "cf08cc52-5f32-47a2-8de4-e3d687383960": "excluded",
        "9ee26633-222a-4445-b934-90532b6c0551": "worth_knowing",
    }
    assert {
        feedback_id: next(
            item.judgment
            for item in reviews[feedback_id].concerns
            if isinstance(item, TierConcernDisposition)
        )
        for feedback_id in expected_tiers
    } == expected_tiers
    subject_21_tier = next(
        item
        for item in reviews["33c1dccc-e4d6-4ec5-9775-7c5a53fd952b"].concerns
        if item.concern == "tier"
    )
    assert subject_21_tier.kind == "excluded"
    assert subject_21_tier.reason == "conditional"
    assert sum(len(spec.case_ids) for spec in RELEVANCE_SPECS) == 27
    assert len(TIER_SPECS) == 27
    assert len(CONFIDENCE_SPECS) == 4
    assert len(RANKING_SPECS) == 13
    assert len(THEME_JUDGMENTS) == 1

    manifest = _manifest()
    assert Counter(case.concern for case in manifest.cases) == {
        "relevance": 27,
        "ranking": 13,
        "tier": 27,
        "confidence": 4,
        "daily_theme": 1,
    }


def test_only_explicit_public_transport_relationship_is_a_theme_case() -> None:
    assert tuple(judgment.judgment_id for judgment in THEME_JUDGMENTS) == (
        "sep10-metrorex-and-stb-share-public-transport-theme",
    )
    reviews = {str(review.feedback_id): review for review in FEEDBACK_REVIEWS}
    for feedback_id in (
        "31ee14fc-e827-4c82-a3ab-f23afb8739b6",
        "6c92cf32-87cb-43c0-aee3-0428699df6f2",
    ):
        assert all(item.concern != "grouping" for item in reviews[feedback_id].concerns)
    for feedback_id in (
        "1e5226d4-f060-428f-a5b6-7713821cb565",
        "c2619b7c-fb8b-4dfa-8ebe-3012b5e8f5b9",
    ):
        grouping = next(
            item for item in reviews[feedback_id].concerns if item.concern == "grouping"
        )
        assert grouping.kind == "excluded"
        assert grouping.reason == "ambiguous"


def test_pnrr_ranking_compares_against_every_earlier_subject() -> None:
    expected = {
        "sep10-pnrr-ranks-above-research-funding",
        "sep10-pnrr-ranks-above-aur-afd",
        "sep10-pnrr-ranks-above-cyber-readiness",
        "sep10-pnrr-ranks-above-metrorex",
        "sep10-pnrr-ranks-above-hydropower-law",
    }
    assert {
        spec.case_id for spec in RANKING_SPECS if spec.feedback_id == CURRENT_FEEDBACK_IDS[6]
    } == expected
    review = next(
        review for review in FEEDBACK_REVIEWS if str(review.feedback_id) == CURRENT_FEEDBACK_IDS[6]
    )
    disposition = next(item for item in review.concerns if item.concern == "ranking")
    assert isinstance(disposition, ExecutableConcernDisposition)
    assert set(disposition.case_ids) == expected


@pytest.mark.parametrize(
    ("concern", "judgment"),
    (("relevance", "irrelevant"), ("tier", "main"), ("confidence", "supported")),
)
def test_v7_represented_judgment_must_match_linked_case(
    concern: str,
    judgment: str,
) -> None:
    payload = _manifest().model_dump(mode="json")
    disposition = next(
        item
        for review in payload["feedback_reviews"]
        for item in review["concerns"]
        if item["concern"] == concern and "case_ids" in item
    )
    disposition["judgment"] = judgment

    with pytest.raises(ValidationError, match="judgment must match its linked case"):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)


def test_v7_case_links_validate_both_ways() -> None:
    manifest = _manifest()
    executable_ids = {
        review.feedback_id
        for review in manifest.feedback_reviews
        if any(hasattr(item, "case_ids") for item in review.concerns)
    }
    provenance_ids = {
        feedback_id for case in manifest.cases for feedback_id in case.provenance.feedback_ids
    }

    assert provenance_ids == executable_ids

    payload = manifest.model_dump(mode="json")
    review = next(
        item
        for item in payload["feedback_reviews"]
        if any("case_ids" in concern for concern in item["concerns"])
    )
    concern = next(item for item in review["concerns"] if "case_ids" in item)
    concern["case_ids"] = ["missing-case"]
    with pytest.raises(ValidationError, match="match evaluation provenance both ways"):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)


def test_v7_review_validation_rejects_duplicate_provenance_and_report_mismatch() -> None:
    manifest = _manifest()
    duplicate = manifest.model_dump(mode="json")
    duplicate["cases"][0]["provenance"]["feedback_ids"].append(
        duplicate["cases"][0]["provenance"]["feedback_ids"][0]
    )
    with pytest.raises(ValidationError, match="provenance feedback IDs must be unique"):
        NewsEvaluationManifest.model_validate_json(json.dumps(duplicate), strict=True)

    mismatch = manifest.model_dump(mode="json")
    other_report = REPORT_REFERENCE.model_copy(
        update={"artifact_id": "news:daily:other", "version_id": "f" * 64}
    )
    mismatch["reports"].append(
        ReportEvaluationSpec(
            report=other_report,
            cluster_set=CLUSTER_REFERENCE,
        ).model_dump(mode="json")
    )
    mismatch["cases"][0]["provenance"]["report_version_id"] = other_report.version_id
    with pytest.raises(ValidationError, match="feedback target reports must match"):
        NewsEvaluationManifest.model_validate_json(json.dumps(mismatch), strict=True)


def test_v7_review_validation_rejects_score_curation() -> None:
    manifest = _manifest()
    changed = manifest.model_copy(
        update={
            "score_curation": (
                ExcludedFeedbackScore(
                    feedback_id=manifest.source_feedback_ids[0],
                    reason="presentation",
                    report_version_id=REPORT_VERSION_ID,
                    rationale="Reviewed manifests keep score curation separate.",
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match="cannot use legacy curation fields"):
        NewsEvaluationManifest.model_validate_json(changed.model_dump_json(), strict=True)


def test_only_non_deterministic_concerns_stay_manifest_only() -> None:
    manifest = _manifest()
    assert manifest.score_curation == ()
    assert {case.concern for case in manifest.cases} == {
        "relevance",
        "ranking",
        "tier",
        "confidence",
        "daily_theme",
    }
    manifest_only = {
        item.concern
        for review in manifest.feedback_reviews
        for item in review.concerns
        if not hasattr(item, "case_ids")
    }
    assert manifest_only == {
        "submission",
        "grouping",
        "relevance",
        "tier",
        "context",
        "research",
        "overall_report",
    }


def test_v7_manifest_serializes_and_parses_strictly() -> None:
    manifest = _manifest()
    content = manifest.model_dump_json()

    parsed = NewsEvaluationManifest.model_validate_json(content, strict=True)

    assert parsed == manifest
    assert '"feedback_reviews"' in content
    assert '"unreviewed_themes"' in content
    assert '"note"' not in content


def test_production_feedback_check_accepts_only_exact_targets_and_newer_expansions() -> None:
    started = datetime(2026, 9, 10, 9, tzinfo=UTC)
    events = tuple(
        NewsFeedbackEvent(
            feedback_id=review.feedback_id,
            target=review.target,
            rating=None,
            note="Reviewed evidence",
            actor="owner",
            created_at=started + timedelta(minutes=position),
        )
        for position, review in enumerate(FEEDBACK_REVIEWS)
    )

    require_exact_september_10_feedback(events)

    changed = events[0].model_copy(
        update={"target": events[0].target.model_copy(update={"theme_id": GROUP_IDS[1]})}
    )
    with pytest.raises(ValueError, match="target changed"):
        require_exact_september_10_feedback((changed, *events[1:]))


def test_production_builder_wires_every_executable_concern_under_mocked_boundaries(
    monkeypatch,
) -> None:
    expected = _manifest()
    expected_by_id = {case.case_id: case for case in expected.cases}
    prior = NewsEvaluationManifest(
        version="news-evaluation-2026-09-07-v6.3",
        reviewed_at=REVIEWED_AT,
        issue_url="https://example.test/prior",
        source_feedback_ids=(),
        reports=(),
        cases=(),
    )
    events = tuple(
        NewsFeedbackEvent(
            feedback_id=review.feedback_id,
            target=review.target,
            rating=None,
            note="Reviewed evidence",
            actor="owner",
            created_at=datetime(2026, 9, 10, 9, tzinfo=UTC) + timedelta(minutes=position),
        )
        for position, review in enumerate(FEEDBACK_REVIEWS)
    )
    snapshot = SimpleNamespace(
        report=REPORT_REFERENCE,
        themes=THEMES_REFERENCE,
        cluster_set=CLUSTER_REFERENCE,
        report_group_ids=tuple(GROUP_IDS.values()),
    )
    bundle = SimpleNamespace(snapshot=snapshot)
    calls = Counter()

    monkeypatch.setattr(
        september,
        "read_evaluation_artifact",
        lambda _reference: prior.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        september.evaluation_catalog,
        "hydrate_news_evaluation_manifest",
        lambda _manifest: SimpleNamespace(reports=(), cases=()),
    )
    monkeypatch.setattr(september, "read_evaluation_report_bundle", lambda _id: bundle)
    monkeypatch.setattr(
        september.evaluation_catalog,
        "read_news_evaluation_feedback",
        lambda _ids: events,
    )
    monkeypatch.setattr(
        september,
        "build_relevance_cases",
        lambda spec, _bundle: calls.update(relevance=1)
        or tuple(expected_by_id[case_id] for case_id in spec.case_ids),
    )
    monkeypatch.setattr(
        september,
        "build_ranking_case",
        lambda spec, _bundle: calls.update(ranking=1) or expected_by_id[spec.case_id],
    )
    monkeypatch.setattr(
        september,
        "build_tier_case",
        lambda spec, _bundle: calls.update(tier=1) or expected_by_id[spec.case_id],
    )
    monkeypatch.setattr(
        september,
        "build_confidence_case",
        lambda spec, _bundle: calls.update(confidence=1) or expected_by_id[spec.case_id],
    )
    theme_case = next(case for case in expected.cases if case.concern == "daily_theme")
    monkeypatch.setattr(
        september,
        "build_theme_day_cases",
        lambda *_args: calls.update(grouping=1) or (theme_case,),
    )
    monkeypatch.setattr(september, "evaluation_case_spec", lambda case: case)
    monkeypatch.setattr(
        september,
        "NewsEvaluationDataset",
        lambda **values: SimpleNamespace(**values),
    )

    manifest = september.build_september_10_evaluation_manifest()

    assert manifest == expected
    assert calls == {
        "relevance": 23,
        "ranking": 13,
        "tier": 27,
        "confidence": 4,
        "grouping": 1,
    }
