import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from romanian_news import evaluation
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.analysis.relevance_v3 import ContextDecision, ImpactDecision
from romanian_news.evaluation import (
    ConfidenceEvaluationCase,
    EvaluationCaseResult,
    EvaluationProvenance,
    ExcludedFeedback,
    ExcludedFeedbackScore,
    ExecutableConcernDisposition,
    FeedbackReview,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    NonExecutableFeedback,
    ProjectedFeedbackScore,
    RelevanceConcernDisposition,
    RelevanceV3EvaluationDecision,
    SupersededConcernDisposition,
    ThemeEvaluationDayCase,
    ThemePairExpectation,
    ThemeReportExpectation,
    TierConcernDisposition,
    TierEvaluationCase,
    build_news_evaluation_baseline,
    compare_news_evaluation_to_baseline,
    evaluate_news_dataset,
    evaluate_theme_output,
)
from romanian_news.feedback import ThemeFeedbackTarget
from romanian_news.groups import NewsGroup
from romanian_news.tests.evaluation_factories import (
    DAY,
    embedded_article,
    reference,
    synthetic_dataset,
    synthetic_manifest,
)
from romanian_news.themes import (
    LEGACY_SPARSE_THEME_POLICY,
    DailyTheme,
    DailyThemeInput,
    DailyThemeSet,
    EmptyThemeConstruction,
    SparseDailyThemeSet,
    ThemeGroupInput,
    sparse_theme_policy_digest,
)

MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:synthetic-v1",
    version_id="1" * 64,
    content_digest="2" * 64,
    r2_key="news/evaluations/manifests/synthetic-v1.json",
)


def test_evaluation_result_rejects_a_passing_unavailable_prediction() -> None:
    with pytest.raises(ValidationError, match="unavailable prediction cannot pass"):
        EvaluationCaseResult(
            case_id="invalid",
            concern="daily_theme",
            passed=True,
            control=False,
            expected_positive=False,
            predicted_positive=None,
            detail="No valid partition.",
        )


def test_dataset_rejects_duplicate_expanded_case_ids() -> None:
    dataset = synthetic_dataset()
    duplicated = dataset.model_copy(update={"cases": (*dataset.cases, dataset.cases[0])})

    with pytest.raises(ValidationError, match="Evaluation case IDs must be unique"):
        NewsEvaluationDataset.model_validate_json(duplicated.model_dump_json(), strict=True)


def test_manifest_rejects_duplicate_expanded_case_ids() -> None:
    manifest = synthetic_manifest()
    duplicated = manifest.model_copy(update={"cases": (*manifest.cases, manifest.cases[0])})

    with pytest.raises(ValidationError, match="Evaluation case IDs must be unique"):
        NewsEvaluationManifest.model_validate_json(duplicated.model_dump_json(), strict=True)


def test_historical_manifest_without_score_curation_parses() -> None:
    payload = synthetic_manifest().model_dump(mode="json")

    assert "score_curation" not in payload
    assert (
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True).score_curation
        == ()
    )


def test_v6_manifest_round_trips_without_v7_fields() -> None:
    content = synthetic_manifest().model_dump_json()

    assert '"feedback_reviews"' not in content
    assert '"unreviewed_themes"' not in content
    parsed = NewsEvaluationManifest.model_validate_json(content, strict=True)
    assert parsed.feedback_reviews == ()
    assert parsed.unreviewed_themes == ()
    assert parsed.model_dump_json() == content


def test_feedback_review_variants_reject_invalid_field_combinations() -> None:
    target = ThemeFeedbackTarget(report_version_id="1" * 64, theme_id="2" * 64)
    feedback_id = UUID("00000000-0000-4000-8000-000000000001")

    with pytest.raises(ValidationError, match="Field required"):
        TierConcernDisposition.model_validate(
            {
                "kind": "represented",
                "concern": "tier",
                "judgment": "main",
                "rationale": "Tier must link to an executable case.",
            }
        )
    relevance = RelevanceConcernDisposition(
        judgment="irrelevant",
        case_ids=("relevance-case",),
        rationale="The reviewed subject is not relevant.",
    )
    assert relevance.model_dump() == {
        "kind": "represented",
        "concern": "relevance",
        "judgment": "irrelevant",
        "case_ids": ("relevance-case",),
        "rationale": "The reviewed subject is not relevant.",
    }

    with pytest.raises(ValidationError, match="Superseded feedback cannot retain"):
        FeedbackReview(
            feedback_id=feedback_id,
            target=target,
            concerns=(
                SupersededConcernDisposition(
                    superseded_by_feedback_id=UUID("00000000-0000-4000-8000-000000000002")
                ),
                ExecutableConcernDisposition(
                    concern="ranking",
                    case_ids=("ranking-case",),
                    rationale="A current judgment cannot survive supersession.",
                ),
            ),
        )


def test_score_curation_allows_several_concerns_for_one_feedback() -> None:
    manifest = synthetic_manifest()
    feedback_ids = manifest.source_feedback_ids
    report_version_id = manifest.reports[0].report.version_id
    decisions = (
        ProjectedFeedbackScore(
            feedback_id=feedback_ids[0],
            concern="language",
            report_version_id=report_version_id,
            model_output=reference(701, "theme-prose"),
            model_attempt_id="7" * 64,
            polarity="negative",
            rationale="The prose uses the wrong language.",
        ),
        ProjectedFeedbackScore(
            feedback_id=feedback_ids[0],
            concern="grouping",
            report_version_id=report_version_id,
            model_output=reference(702, "theme-assignment"),
            model_attempt_id="8" * 64,
            polarity="positive",
            rationale="The assignment is correct.",
        ),
        *(
            ExcludedFeedbackScore(
                feedback_id=feedback_id,
                reason="presentation",
                report_version_id=report_version_id,
                rationale="This decision does not assess a model concern.",
            )
            for feedback_id in feedback_ids[1:]
        ),
    )

    curated = manifest.model_copy(update={"score_curation": decisions})
    parsed = NewsEvaluationManifest.model_validate_json(curated.model_dump_json(), strict=True)

    assert len(parsed.score_curation) == len(feedback_ids) + 1


def test_score_exclusions_distinguish_missing_and_combined_observations() -> None:
    feedback_id = UUID("00000000-0000-4000-8000-000000000099")
    report_version_id = "9" * 64

    missing = ExcludedFeedbackScore(
        feedback_id=feedback_id,
        reason="no_model_observation",
        concern="grouping",
        report_version_id=report_version_id,
        rationale="The frozen report has no theme model output.",
    )
    combined = ExcludedFeedbackScore(
        feedback_id=feedback_id,
        reason="no_concern_specific_observation",
        concern="language",
        report_version_id=report_version_id,
        polarity="negative",
        rationale="One observation produced assignment and prose.",
    )

    assert missing.polarity is None
    assert combined.polarity == "negative"
    with pytest.raises(ValidationError, match="require a concern and polarity"):
        ExcludedFeedbackScore(
            feedback_id=feedback_id,
            reason="no_concern_specific_observation",
            concern="language",
            report_version_id=report_version_id,
            rationale="Missing polarity is invalid.",
        )


def test_score_curation_requires_exact_source_feedback_coverage() -> None:
    manifest = synthetic_manifest()
    decision = ExcludedFeedbackScore(
        feedback_id=manifest.source_feedback_ids[0],
        reason="ambiguous",
        report_version_id=manifest.reports[0].report.version_id,
        rationale="The concern cannot be determined.",
    )
    incomplete = manifest.model_copy(update={"score_curation": (decision,)})

    with pytest.raises(ValidationError, match="exactly cover every declared feedback ID"):
        NewsEvaluationManifest.model_validate_json(incomplete.model_dump_json(), strict=True)


def test_evaluator_runs_each_concern() -> None:
    dataset = synthetic_dataset()

    report = evaluate_news_dataset(dataset)

    assert {result.concern for result in report.case_results} == {
        "relevance",
        "summary_format",
        "grouping",
        "ranking",
        "reader_presentation",
    }
    assert all(result.passed for result in report.case_results)
    assert {aggregate.concern for aggregate in report.aggregates} == {
        "relevance",
        "grouping",
        "ranking",
    }
    assert report.limitations == (evaluation.RELEVANCE_LIMITATION,)


def test_deterministic_evaluator_excludes_daily_themes() -> None:
    first = embedded_article(1)
    second = embedded_article(2)
    left_group_id = reference(10, "group").version_id
    right_group_id = reference(11, "group").version_id
    theme_case = ThemeEvaluationDayCase(
        case_id="daily-theme-day",
        provenance=EvaluationProvenance(feedback_ids=()),
        source_report=reference(600),
        input=DailyThemeInput(
            day=DAY,
            cluster_set=reference(601),
            groups=(
                ThemeGroupInput(
                    group=NewsGroup(
                        id=left_group_id,
                        article_version_ids=(first.article.version_id,),
                    ),
                    summary=reference(602),
                    value=GroupSummary(
                        title_ro="Primul grup sintetic",
                        summary_ro="Primul grup sintetic adună știri naționale.",
                        key_points_ro=("Primul punct.",),
                        disagreements_ro=(),
                        cited_article_version_ids=(first.article.version_id,),
                    ),
                ),
                ThemeGroupInput(
                    group=NewsGroup(
                        id=right_group_id,
                        article_version_ids=(second.article.version_id,),
                    ),
                    summary=reference(603),
                    value=GroupSummary(
                        title_ro="Al doilea grup sintetic",
                        summary_ro="Al doilea grup sintetic adună știri locale.",
                        key_points_ro=("Al doilea punct.",),
                        disagreements_ro=(),
                        cited_article_version_ids=(second.article.version_id,),
                    ),
                ),
            ),
        ),
        expectations=(
            ThemePairExpectation(
                case_id="daily-theme-pair",
                feedback_ids=(),
                left_group_id=left_group_id,
                right_group_id=right_group_id,
                expected_same_theme=True,
                rationale="Reviewed theme pair.",
            ),
        ),
    )
    dataset = synthetic_dataset()
    with_theme_case = dataset.model_copy(update={"cases": (*dataset.cases, theme_case)})

    report = evaluate_news_dataset(with_theme_case)

    assert report.case_results == evaluate_news_dataset(dataset).case_results
    assert all(result.concern != "daily_theme" for result in report.case_results)
    assert all(aggregate.concern != "daily_theme" for aggregate in report.aggregates)


def test_theme_output_scores_relationships_and_complete_report() -> None:
    groups = tuple(
        ThemeGroupInput(
            group=NewsGroup(id=str(index) * 64, article_version_ids=(str(index + 3) * 64,)),
            summary=reference(610 + index),
            value=GroupSummary(
                title_ro=f"Grupul {index}",
                summary_ro=f"Rezumatul {index}.",
                key_points_ro=(f"Punctul {index}.",),
                disagreements_ro=(),
                cited_article_version_ids=(str(index + 3) * 64,),
            ),
        )
        for index in range(1, 4)
    )
    value = DailyThemeInput(day=DAY, cluster_set=reference(609), groups=groups)
    case = ThemeEvaluationDayCase(
        case_id="reader-subject-day",
        provenance=EvaluationProvenance(feedback_ids=()),
        source_report=reference(608),
        input=value,
        expectations=(
            ThemePairExpectation(
                case_id="related",
                feedback_ids=(),
                left_group_id=groups[0].group.id,
                right_group_id=groups[1].group.id,
                expected_same_theme=True,
                rationale="The groups share one reviewed reader subject.",
            ),
            ThemePairExpectation(
                case_id="separate",
                feedback_ids=(),
                left_group_id=groups[0].group.id,
                right_group_id=groups[2].group.id,
                expected_same_theme=False,
                rationale="The groups cover unrelated subjects.",
            ),
        ),
        report_expectation=ThemeReportExpectation(),
    )
    policy_digest = sparse_theme_policy_digest(LEGACY_SPARSE_THEME_POLICY)
    theme_set = SparseDailyThemeSet.model_construct(
        day=DAY,
        request_id="a" * 64,
        policy=LEGACY_SPARSE_THEME_POLICY,
        policy_digest=policy_digest,
        cluster_set=value.cluster_set,
        groups=tuple(item.group for item in groups),
        summary_inputs=tuple(item.summary for item in groups),
        construction=EmptyThemeConstruction(),
        themes=(
            DailyTheme(
                id="b" * 64,
                title="Subiect comun",
                summary="Rezumat comun.",
                group_ids=(groups[0].group.id, groups[1].group.id),
                article_version_ids=(
                    *groups[0].group.article_version_ids,
                    *groups[1].group.article_version_ids,
                ),
            ),
            DailyTheme(
                id="c" * 64,
                title="Subiect separat",
                summary="Rezumat separat.",
                group_ids=(groups[2].group.id,),
                article_version_ids=groups[2].group.article_version_ids,
            ),
        ),
    )

    metrics = evaluate_theme_output(case, theme_set)

    assert metrics.must_link_recall == 1.0
    assert metrics.must_separate_preservation == 1.0
    assert metrics.singleton_rate == 1 / 3
    assert metrics.useful


def test_evaluator_expands_observed_theme_tier_and_confidence_concerns() -> None:
    dataset = synthetic_dataset()
    grouping = next(item for item in dataset.cases if item.concern == "grouping")
    left_id = grouping.left_article_version_id
    right_id = grouping.right_article_version_id
    left_group = NewsGroup(id="7" * 64, article_version_ids=(left_id,))
    right_group = NewsGroup(id="8" * 64, article_version_ids=(right_id,))
    theme_output = DailyThemeSet.model_construct(
        themes=(
            DailyTheme(
                id="9" * 64,
                title="Tema sintetică",
                summary="Rezumat sintetic.",
                group_ids=(left_group.id, right_group.id),
                article_version_ids=(left_id, right_id),
            ),
        )
    )
    theme_case = ThemeEvaluationDayCase.model_construct(
        case_id="theme-parent",
        provenance=EvaluationProvenance(feedback_ids=()),
        source_report=reference(800, "report"),
        input=DailyThemeInput(
            day=DAY,
            cluster_set=reference(801, "cluster"),
            groups=(),
        ),
        expectations=(
            ThemePairExpectation(
                case_id="theme-pair",
                feedback_ids=(),
                left_group_id=left_group.id,
                right_group_id=right_group.id,
                expected_same_theme=True,
                rationale="The groups belong together.",
            ),
        ),
        model_output=reference(802, "themes"),
        observed_themes=theme_output,
    )
    tier_case = TierEvaluationCase(
        case_id="tier-case",
        provenance=EvaluationProvenance(feedback_ids=()),
        expected_tier="worth_knowing",
        observed_tier="main",
    )
    confidence_case = ConfidenceEvaluationCase(
        case_id="confidence-case",
        provenance=EvaluationProvenance(feedback_ids=()),
        expected_sufficient=False,
        observed_sufficient=False,
    )

    report = evaluate_news_dataset(
        dataset.model_copy(
            update={
                "cases": (*dataset.cases, theme_case, tier_case, confidence_case),
            }
        )
    )

    by_id = {item.case_id: item for item in report.case_results}
    assert by_id["theme-pair"].passed is True
    assert by_id["theme-pair"].concern == "grouping"
    assert by_id["tier-case"].passed is False
    assert by_id["confidence-case"].passed is True
    assert {item.concern for item in report.aggregates} == {
        "relevance",
        "grouping",
        "ranking",
        "tier",
        "confidence",
    }


def test_v3_relevance_uses_context_and_impact_together() -> None:
    dataset = synthetic_dataset()
    case = next(item for item in dataset.cases if item.concern == "relevance")
    changed = case.model_copy(
        update={
            "expected_accepted": False,
            "model_response": RelevanceV3EvaluationDecision(
                context=ContextDecision(
                    subject_role="absent",
                    news_cycle="current_cycle",
                    romanian_consequence="absent",
                    certainty="clear",
                    evidence_quote="No Romanian subject.",
                    reason_ro="Articolul nu are un subiect românesc.",
                ),
                impact=ImpactDecision(
                    consequence_status="realized",
                    effect_basis="actual_consequence",
                    effect_scope="national_market",
                    magnitude="major",
                    political_relevance="none",
                    economic_relevance="strong",
                    quantified=True,
                    certainty="clear",
                    evidence_quote="A major national market effect.",
                    reason_ro="Impact economic național major.",
                ),
            ),
        }
    )

    report = evaluate_news_dataset(dataset.model_copy(update={"cases": (changed,), "reports": ()}))

    assert report.case_results[0].predicted_positive is False
    assert report.case_results[0].passed is True


def test_v3_relevance_rejects_a_context_early_exit_without_impact() -> None:
    dataset = synthetic_dataset()
    case = next(item for item in dataset.cases if item.concern == "relevance")
    changed = case.model_copy(
        update={
            "expected_accepted": False,
            "model_response": RelevanceV3EvaluationDecision(
                context=ContextDecision(
                    subject_role="absent",
                    news_cycle="current_cycle",
                    romanian_consequence="absent",
                    certainty="clear",
                    evidence_quote="No Romanian subject.",
                    reason_ro="Articolul nu are un subiect românesc.",
                ),
                impact=None,
            ),
        }
    )

    report = evaluate_news_dataset(dataset.model_copy(update={"cases": (changed,), "reports": ()}))

    assert report.case_results[0].predicted_positive is False
    assert report.case_results[0].passed is True


def test_same_group_case_requires_every_labeled_article_in_one_group(monkeypatch) -> None:
    dataset = synthetic_dataset()
    case = next(item for item in dataset.cases if item.concern == "grouping")
    third = embedded_article(3, axis=1)
    changed = case.model_copy(update={"articles": (*case.articles, third)})
    first_ids = tuple(item.article.version_id for item in changed.articles[:2])
    monkeypatch.setattr(
        evaluation,
        "cluster_articles",
        lambda *_args, **_kwargs: SimpleNamespace(
            cluster_set=SimpleNamespace(
                groups=(
                    NewsGroup(id="3" * 64, article_version_ids=first_ids),
                    NewsGroup(id="4" * 64, article_version_ids=(third.article.version_id,)),
                )
            )
        ),
    )

    report = evaluate_news_dataset(dataset.model_copy(update={"cases": (changed,), "reports": ()}))

    assert report.case_results[0].passed is False


def test_report_completeness_requires_exact_inputs() -> None:
    dataset = synthetic_dataset()
    snapshot = dataset.reports[0]
    inputs = snapshot.report_inputs
    changed = snapshot.model_copy(
        update={
            "report_inputs": inputs.model_copy(
                update={
                    "cluster_set": (MANIFEST_REFERENCE,),
                    "relevance": (inputs.relevance[0], inputs.relevance[0]),
                }
            )
        }
    )

    report = evaluate_news_dataset(dataset.model_copy(update={"reports": (changed,)}))

    assert report.deterministic_checks[0].passed is False
    assert report.deterministic_checks[0].detail == (
        "report cluster-set input is not exact; report relevance inputs are not exact"
    )


def test_report_completeness_requires_the_exact_themes_input() -> None:
    dataset = synthetic_dataset()
    snapshot = dataset.reports[0]
    themes = MANIFEST_REFERENCE.model_copy(update={"artifact_id": "news:themes:test"})
    changed = snapshot.model_copy(update={"themes": themes})

    report = evaluate_news_dataset(dataset.model_copy(update={"reports": (changed,)}))

    assert report.deterministic_checks[0].passed is False
    assert report.deterministic_checks[0].detail == "report themes input is not exact"


def test_baseline_accepts_equal_results_and_reports_regressions() -> None:
    report = evaluate_news_dataset(synthetic_dataset())
    baseline = build_news_evaluation_baseline(report, MANIFEST_REFERENCE)

    assert compare_news_evaluation_to_baseline(report, baseline, MANIFEST_REFERENCE) == ()

    first = report.case_results[0]
    regressed = report.model_copy(
        update={
            "case_results": (
                first.model_copy(update={"passed": False}),
                *report.case_results[1:],
            )
        }
    )
    assert compare_news_evaluation_to_baseline(regressed, baseline, MANIFEST_REFERENCE) == (
        f"case regressed: {first.case_id}",
    )

    changed_identity = MANIFEST_REFERENCE.model_copy(
        update={"artifact_id": "news:evaluation-manifest:other"}
    )
    assert compare_news_evaluation_to_baseline(report, baseline, changed_identity) == (
        "manifest reference changed",
    )


def test_old_daily_theme_baseline_requires_reviewed_replacement() -> None:
    report = evaluate_news_dataset(synthetic_dataset())
    baseline = build_news_evaluation_baseline(report, MANIFEST_REFERENCE)
    old_theme_case = baseline.cases[0].model_copy(
        update={"case_id": "old-theme-case", "concern": "daily_theme"}
    )
    old_theme_aggregate = baseline.aggregates[0].model_copy(update={"concern": "daily_theme"})
    old_baseline = baseline.model_copy(
        update={
            "cases": (*baseline.cases, old_theme_case),
            "aggregates": (*baseline.aggregates, old_theme_aggregate),
        }
    )

    regressions = compare_news_evaluation_to_baseline(report, old_baseline, MANIFEST_REFERENCE)

    assert regressions == (
        "case IDs changed: added=[], removed=['old-theme-case']",
        "aggregate IDs changed: added=[], removed=['daily_theme']",
    )


def test_manifest_reports_the_first_feedback_assignment_error() -> None:
    manifest = synthetic_manifest()

    duplicate_source = manifest.model_dump(mode="json")
    duplicate_source["source_feedback_ids"].append(duplicate_source["source_feedback_ids"][0])
    duplicate_source["reports"].append(duplicate_source["reports"][0])
    with pytest.raises(ValidationError, match="Evaluation feedback IDs must be unique"):
        NewsEvaluationManifest.model_validate_json(json.dumps(duplicate_source), strict=True)

    represented_exclusion = manifest.model_dump(mode="json")
    represented_exclusion["excluded_feedback"] = [
        {
            "feedback_id": represented_exclusion["source_feedback_ids"][0],
            "report_version_id": represented_exclusion["reports"][0]["report"]["version_id"],
            "reason": "non_executable",
        }
    ]
    represented_exclusion["reports"].append(represented_exclusion["reports"][0])
    with pytest.raises(ValidationError, match="Excluded feedback cannot also be represented"):
        NewsEvaluationManifest.model_validate_json(json.dumps(represented_exclusion), strict=True)


def test_manifest_checks_coverage_before_duplicate_exclusions_and_reports() -> None:
    manifest = synthetic_manifest()
    payload = manifest.model_dump(mode="json")
    first_feedback_id, second_feedback_id = payload["source_feedback_ids"][:2]
    payload["cases"][0]["provenance"]["feedback_ids"] = []
    payload["cases"][1]["provenance"]["feedback_ids"] = []
    exclusion = {
        "feedback_id": first_feedback_id,
        "report_version_id": payload["reports"][0]["report"]["version_id"],
        "reason": "non_executable",
    }
    payload["excluded_feedback"] = [exclusion, exclusion]
    payload["reports"].append(payload["reports"][0])

    with pytest.raises(
        ValidationError,
        match="Evaluation specs, theme judgments, and exclusions must exactly cover every declared feedback ID",
    ):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)

    payload["cases"][1]["provenance"]["feedback_ids"] = [second_feedback_id]
    with pytest.raises(ValidationError, match="Excluded feedback IDs must be unique"):
        NewsEvaluationManifest.model_validate_json(json.dumps(payload), strict=True)


def test_dataset_rejects_unrepresented_feedback() -> None:
    dataset = synthetic_dataset()
    payload = dataset.model_dump(mode="json")
    payload["source_feedback_ids"] = ["00000000-0000-4000-8000-000000000001"]

    with pytest.raises(ValidationError, match="exactly cover every declared feedback ID"):
        NewsEvaluationDataset.model_validate_json(json.dumps(payload), strict=True)


def test_dataset_accepts_non_executable_feedback_as_explicitly_dispositioned() -> None:
    dataset = synthetic_dataset()
    feedback_id = UUID("00000000-0000-4000-8000-000000000099")
    changed = dataset.model_copy(
        update={
            "source_feedback_ids": (*dataset.source_feedback_ids, feedback_id),
            "excluded_feedback": (
                NonExecutableFeedback(
                    feedback_id=feedback_id,
                    report_version_id=dataset.reports[0].report.version_id,
                    reason="non_executable",
                ),
            ),
        }
    )

    parsed = NewsEvaluationDataset.model_validate_json(changed.model_dump_json(), strict=True)

    assert parsed.excluded_feedback == changed.excluded_feedback


def test_non_executable_feedback_rejects_supersession_fields() -> None:
    dataset = synthetic_dataset()
    feedback_id = "00000000-0000-4000-8000-000000000099"
    payload = dataset.model_dump(mode="json")
    payload["source_feedback_ids"].append(feedback_id)
    payload["excluded_feedback"] = [
        {
            "feedback_id": feedback_id,
            "report_version_id": dataset.reports[0].report.version_id,
            "reason": "non_executable",
            "superseded_by_feedback_id": "00000000-0000-4000-8000-000000000098",
        }
    ]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NewsEvaluationDataset.model_validate_json(json.dumps(payload), strict=True)


def test_superseded_feedback_can_point_to_a_non_executable_replacement() -> None:
    dataset = synthetic_dataset()
    old_id = UUID("00000000-0000-4000-8000-000000000098")
    replacement_id = UUID("00000000-0000-4000-8000-000000000099")
    report_version_id = dataset.reports[0].report.version_id
    changed = dataset.model_copy(
        update={
            "source_feedback_ids": (*dataset.source_feedback_ids, old_id, replacement_id),
            "excluded_feedback": (
                ExcludedFeedback(
                    feedback_id=old_id,
                    report_version_id=report_version_id,
                    reason="superseded",
                    superseded_by_feedback_id=replacement_id,
                ),
                NonExecutableFeedback(
                    feedback_id=replacement_id,
                    report_version_id=report_version_id,
                    reason="non_executable",
                ),
            ),
        }
    )

    parsed = NewsEvaluationDataset.model_validate_json(changed.model_dump_json(), strict=True)

    assert parsed.excluded_feedback == changed.excluded_feedback


def test_case_variants_reject_fields_from_another_concern() -> None:
    payload = synthetic_dataset().model_dump(mode="json")
    payload["cases"][0]["artifact"] = {"title": "Unexpected summary"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NewsEvaluationDataset.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    ("summary", "passed"),
    (
        ("Dl. Popescu a prezentat planul. Guvernul îl aplică.", True),
        ("Prima. A doua. A treia.", False),
        ("„Prima.” „A doua.” A treia.", False),
    ),
)
def test_summary_sentence_limits(summary: str, passed: bool) -> None:
    dataset = synthetic_dataset()
    original = next(item for item in dataset.cases if item.concern == "summary_format")
    changed = original.model_copy(
        update={"artifact": original.artifact.model_copy(update={"summary": summary})}
    )

    report = evaluate_news_dataset(dataset.model_copy(update={"cases": (changed,), "reports": ()}))

    assert report.case_results[0].passed is passed


def test_pin_contains_the_published_national_consequence_release() -> None:
    pin = NewsEvaluationPin.model_validate_json(evaluation.PIN_PATH.read_bytes(), strict=True)

    assert pin.manifest_version_id == (
        "083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"
    )
    assert pin.baseline_version_id == (
        "78f4738f918280d467b7cba6f5e64f09f76cf5f710817640a06f7ae879cea77d"
    )


def test_cli_loads_the_pinned_catalog_release_without_credentials(monkeypatch, capsys) -> None:
    from romanian_news.catalog import evaluations

    dataset = synthetic_dataset()
    report = evaluate_news_dataset(dataset)
    baseline = build_news_evaluation_baseline(report, MANIFEST_REFERENCE)
    loaded = []

    def load_release(content):
        loaded.append(content)
        return SimpleNamespace(
            dataset=dataset,
            baseline=baseline,
            manifest_reference=MANIFEST_REFERENCE,
        )

    monkeypatch.setattr(evaluations, "load_news_evaluation_release", load_release)

    assert evaluations.main(()) == 0
    assert loaded == [evaluation.PIN_PATH.read_bytes()]
    assert json.loads(capsys.readouterr().out)["dataset_version"] == "synthetic-v1"


def test_cli_returns_failure_for_a_catalog_baseline_regression(monkeypatch, capsys) -> None:
    from romanian_news.catalog import evaluations

    dataset = synthetic_dataset()
    report = evaluate_news_dataset(dataset)
    baseline = build_news_evaluation_baseline(report, MANIFEST_REFERENCE)
    failed_case = baseline.cases[0].model_copy(update={"case_id": "missing-case"})
    release = SimpleNamespace(
        dataset=dataset,
        baseline=baseline.model_copy(update={"cases": (failed_case, *baseline.cases[1:])}),
        manifest_reference=MANIFEST_REFERENCE,
    )
    monkeypatch.setattr(evaluations, "load_news_evaluation_release", lambda _content: release)

    assert evaluations.main(()) == 1
    assert "case IDs changed" in capsys.readouterr().err
