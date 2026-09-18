import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from romanian_news import september_12_evaluation as september
from romanian_news.evaluation import (
    EvaluationSpecProvenance,
    ExecutableConcernDisposition,
    NewsEvaluationManifest,
    ReportEvaluationSpec,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
    TierConcernDisposition,
    TierEvaluationSpec,
)
from romanian_news.feedback import NewsFeedbackEvent
from romanian_news.september_12_evaluation import (
    CLUSTER_REFERENCE,
    DATASET_VERSION,
    FEEDBACK_REVIEWS,
    GROUP_IDS,
    GROUP_ORDER,
    ISSUE_URL,
    PRIOR_MANIFEST_REFERENCE,
    REPORT_REFERENCE,
    REPORT_VERSION_ID,
    REVIEWED_AT,
    SOURCE_FEEDBACK_IDS,
    THEME_JUDGMENTS,
    THEMES_REFERENCE,
    TIER_SPECS,
    require_exact_september_12_feedback,
)

EXPECTED_FEEDBACK_IDS = (
    "d2019784-7d3f-4803-952c-1f40411a40f9",
    "134a7f64-c053-485a-8417-3b99591d11b0",
    "fa0d49b7-11e4-46d9-b989-ae2cf2a358d4",
    "139398fd-cef6-43d4-a3ec-ce6119967e4c",
    "f9164813-7673-4129-b2aa-546edaa9762f",
    "99d6391c-19ff-49a9-9707-78c54598bf77",
    "5134001d-38e4-4847-acb9-f0ebbca95783",
    "5600bd39-6988-40f3-a294-f37a232bcbce",
    "c8bef42e-1869-4637-9e5a-bbf8840750bc",
    "c7ae76eb-088a-42c9-acca-a38b72f8f5e1",
    "cd53196c-8e09-4514-a8e2-c532e520aa23",
    "f6f2dc24-7c24-4224-b9d5-5a7cdb452823",
    "3fabeb8b-fc8a-429f-bbbb-ae82275ff7f8",
    "42736350-e762-4fac-925c-20ea096f26dc",
    "71bfad89-75ce-4807-bd80-f03f8988e041",
    "76360535-c261-4790-b7a2-d5f01c1c3b4f",
    "7577642e-1161-4449-b384-df5d7f21ba7f",
    "a25348bf-71e0-4c71-a13c-d3747eeaedc8",
    "dd235eef-a59c-456e-a3d3-3ee87a358062",
    "5dd6bd0f-fc6f-40ea-8f68-c994a7e0c06c",
    "04fb2df4-cc3a-4820-8670-e16c8124deb5",
    "a98866cc-09d8-473d-8019-1f4709a25985",
    "a70b81a1-16dc-46e1-bd50-9b143b09a403",
    "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82",
)

SUPERSEDED = {
    "cd53196c-8e09-4514-a8e2-c532e520aa23": "42736350-e762-4fac-925c-20ea096f26dc",
    "f6f2dc24-7c24-4224-b9d5-5a7cdb452823": "3fabeb8b-fc8a-429f-bbbb-ae82275ff7f8",
    "76360535-c261-4790-b7a2-d5f01c1c3b4f": "7577642e-1161-4449-b384-df5d7f21ba7f",
    "a25348bf-71e0-4c71-a13c-d3747eeaedc8": "5dd6bd0f-fc6f-40ea-8f68-c994a7e0c06c",
    "dd235eef-a59c-456e-a3d3-3ee87a358062": "04fb2df4-cc3a-4820-8670-e16c8124deb5",
    "a98866cc-09d8-473d-8019-1f4709a25985": "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82",
    "a70b81a1-16dc-46e1-bd50-9b143b09a403": "a5dcaba6-9a0a-474a-9bcb-b1ec39394c82",
}


def _manifest() -> NewsEvaluationManifest:
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
        case_id="2026-09-12-5929257f0bf6-daily-themes",
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
        day=date(2026, 9, 12),
        source_report=REPORT_REFERENCE,
        cluster_set=CLUSTER_REFERENCE,
        summaries=(),
        expectations=expectations,
        model_output=THEMES_REFERENCE,
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
            *tier_cases,
            theme_case,
        ),
        feedback_reviews=FEEDBACK_REVIEWS,
    )


def test_v12_declaration_covers_exact_events_targets_and_concerns() -> None:
    assert tuple(str(value) for value in SOURCE_FEEDBACK_IDS) == EXPECTED_FEEDBACK_IDS
    assert len(set(SOURCE_FEEDBACK_IDS)) == 24
    assert len(FEEDBACK_REVIEWS) == 24
    assert {review.target.report_version_id for review in FEEDBACK_REVIEWS} == {REPORT_VERSION_ID}
    assert all(
        len({item.concern for item in review.concerns}) == len(review.concerns)
        for review in FEEDBACK_REVIEWS
    )


def test_v12_supersession_is_exact() -> None:
    superseded = {
        str(review.feedback_id): str(review.concerns[0].superseded_by_feedback_id)
        for review in FEEDBACK_REVIEWS
        if review.concerns[0].kind == "superseded"
    }
    assert superseded == SUPERSEDED
    replacement_ids = {str(value) for value in SUPERSEDED.values()}
    assert not replacement_ids & set(SUPERSEDED)


def test_v12_grouping_cases_are_exact() -> None:
    assert len(TIER_SPECS) == 15
    assert {
        spec.expected_tier: sum(
            1 for item in TIER_SPECS if item.expected_tier == spec.expected_tier
        )
        for spec in TIER_SPECS
    } == {
        "main": 12,
        "worth_knowing": 3,
    }
    assert sum(1 for spec in TIER_SPECS if spec.control) == 11
    assert len(THEME_JUDGMENTS) == 1
    judgment = THEME_JUDGMENTS[0]
    assert judgment.expected_same_theme is True
    assert judgment.left_group_id == GROUP_IDS[3]
    assert judgment.right_group_id == GROUP_IDS[2]

    reviews = {str(review.feedback_id): review for review in FEEDBACK_REVIEWS}
    merge = next(
        item
        for item in reviews["fa0d49b7-11e4-46d9-b989-ae2cf2a358d4"].concerns
        if item.concern == "grouping"
    )
    assert isinstance(merge, ExecutableConcernDisposition)
    assert merge.case_ids == (judgment.judgment_id,)

    manifest = _manifest()
    assert Counter(case.concern for case in manifest.cases) == {
        "tier": 15,
        "daily_theme": 1,
    }


def test_v12_represented_judgment_must_match_linked_case() -> None:
    payload = _manifest().model_dump(mode="json")
    disposition = next(
        item
        for review in payload["feedback_reviews"]
        for item in review["concerns"]
        if item["concern"] == "tier"
    )
    disposition["judgment"] = "worth_knowing" if disposition["judgment"] == "main" else "main"
    with pytest.raises(ValidationError, match="judgment must match its linked case"):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)


def test_v12_manifest_serializes_and_parses_strictly() -> None:
    manifest = _manifest()
    content = manifest.model_dump_json()

    parsed = NewsEvaluationManifest.model_validate_json(content, strict=True)

    assert parsed == manifest
    assert '"feedback_reviews"' in content
    assert '"note"' not in content


def test_production_feedback_check_accepts_only_exact_targets_and_newer_expansions() -> None:
    started = datetime(2026, 9, 12, 13, tzinfo=UTC)
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

    require_exact_september_12_feedback(events)

    changed = events[0].model_copy(
        update={"target": events[0].target.model_copy(update={"theme_id": GROUP_IDS[1]})}
    )
    with pytest.raises(ValueError, match="target changed"):
        require_exact_september_12_feedback((changed, *events[1:]))


def test_production_builder_wires_every_executable_concern_under_mocked_boundaries(
    monkeypatch,
) -> None:
    expected = _manifest()
    expected_by_id = {case.case_id: case for case in expected.cases}
    prior = NewsEvaluationManifest(
        version="news-evaluation-2026-09-11-v11",
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
            created_at=datetime(2026, 9, 12, 13, tzinfo=UTC) + timedelta(minutes=position),
        )
        for position, review in enumerate(FEEDBACK_REVIEWS)
    )
    snapshot = SimpleNamespace(
        report=REPORT_REFERENCE,
        themes=THEMES_REFERENCE,
        cluster_set=CLUSTER_REFERENCE,
        report_group_ids=GROUP_ORDER,
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
        lambda _manifest: SimpleNamespace(
            reports=(),
            cases=(),
            feedback_reviews=(),
            source_feedback_ids=(),
        ),
    )
    monkeypatch.setattr(september, "read_evaluation_report_bundle", lambda _id: bundle)
    monkeypatch.setattr(
        september.evaluation_catalog,
        "read_news_evaluation_feedback",
        lambda _ids: events,
    )
    monkeypatch.setattr(
        september,
        "build_tier_case",
        lambda spec, _bundle: calls.update(tier=1) or expected_by_id[spec.case_id],
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

    manifest = september.build_september_12_evaluation_manifest()

    assert manifest == expected
    assert calls == {
        "tier": 15,
        "grouping": 1,
    }


def test_v12_tier_controls_pin_reviewed_placements() -> None:
    reviews = {str(review.feedback_id): review for review in FEEDBACK_REVIEWS}
    for feedback_id in (
        "5dd6bd0f-fc6f-40ea-8f68-c994a7e0c06c",
        "04fb2df4-cc3a-4820-8670-e16c8124deb5",
    ):
        tier = next(
            item
            for item in reviews[feedback_id].concerns
            if isinstance(item, TierConcernDisposition)
        )
        assert tier.judgment == "worth_knowing"
    for feedback_id in (
        "c7ae76eb-088a-42c9-acca-a38b72f8f5e1",
        "42736350-e762-4fac-925c-20ea096f26dc",
        "3fabeb8b-fc8a-429f-bbbb-ae82275ff7f8",
        "71bfad89-75ce-4807-bd80-f03f8988e041",
    ):
        tier = next(
            item
            for item in reviews[feedback_id].concerns
            if isinstance(item, TierConcernDisposition)
        )
        assert tier.judgment == "main"
        assert next(spec for spec in TIER_SPECS if spec.feedback_id == feedback_id).control is False
