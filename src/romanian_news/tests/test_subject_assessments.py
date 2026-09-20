import json
from datetime import date

import pytest
from pydantic import ValidationError

from romanian_news import subject_assessments as assessment_module
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.evaluation import (
    EvaluationProvenance,
    RankingEvaluationCase,
    TierEvaluationCase,
    evaluate_subject_assessment,
    subject_tier_label_conflicts,
)
from romanian_news.groups import NewsGroup
from romanian_news.subject_assessments import (
    PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
    DailySubjectAssessmentInput,
    DailySubjectAssessmentSet,
    EmptySubjectAssessmentConstruction,
    ModelSubjectAssessmentConstruction,
    SubjectAssessment,
    SubjectAssessmentEvidence,
    SubjectEvidenceInput,
    SubjectSummaryInput,
    compare_subject_anchors,
    construct_daily_subject_assessments,
    parse_subject_assessment_response,
    subject_assessment_policy_digest,
    subject_assessment_request_id,
)
from romanian_news.themes import (
    PRODUCTION_THEME_POLICY,
    DailyTheme,
    EmptyThemeConstruction,
    ReaderSubjectDailyThemeSet,
    daily_theme_id,
    sparse_theme_policy_digest,
)

DAY = date(2026, 9, 10)


def test_response_requires_exact_subject_coverage_and_subject_evidence() -> None:
    value = _input()
    first, second = value.theme_set.themes
    article = value.evidence[0].article.version_id
    valid = {
        "main": [
            {
                "subject": "subject_01",
                "rationale": "A national policy changes household costs.",
                "evidence_articles": ["article_01"],
            }
        ],
        "worth_knowing": [
            {
                "subject": "subject_02",
                "rationale": "The local change remains useful context.",
                "evidence_articles": ["article_02"],
            }
        ],
        "excluded": [],
    }

    context = assessment_module._assessment_context(value)
    assert "subject_01" in context and "article_01" in context and "event_01" in context
    assert first.id not in context and article not in context
    assert context.count('"uncertainty":"Statusul oficial este neclar."') == 1
    assert context.count('"uncertainty":null') == 1

    parsed = parse_subject_assessment_response(json.dumps(valid), value)
    assert tuple(item.subject for item in parsed.main) == ("subject_01",)

    missing = {**valid, "worth_knowing": []}
    with pytest.raises(ValueError, match="every supplied subject exactly once"):
        parse_subject_assessment_response(json.dumps(missing), value)

    wrong_evidence = json.loads(json.dumps(valid))
    wrong_evidence["main"][0]["evidence_articles"] = ["article_02"]
    with pytest.raises(ValueError, match="belong to the assessed subject"):
        parse_subject_assessment_response(json.dumps(wrong_evidence), value)


def test_empty_day_skips_inference_and_retains_exact_identity(monkeypatch) -> None:
    value = _input(empty=True)
    monkeypatch.setattr(
        "romanian_news.subject_assessments.openrouter_client",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected model call")),
    )

    output = construct_daily_subject_assessments(value)

    assert output.assessment_set.assessments == ()
    assert output.assessment_set.subject_ids == ()
    assert isinstance(output.assessment_set.construction, EmptySubjectAssessmentConstruction)
    assert output.assessment_set.request_id == subject_assessment_request_id(value)
    assert DailySubjectAssessmentSet.model_validate_json(output.content, strict=True) == (
        output.assessment_set
    )


def test_assessment_set_rejects_missing_subject_and_nonsemantic_tie() -> None:
    value = _input()
    first, second = value.theme_set.themes
    evidence = tuple(
        SubjectAssessmentEvidence(
            group_id=item.group_id,
            article=item.article,
            relevance=item.relevance,
            summary=item.summary,
            evidence_quote=item.evidence_quote,
        )
        for item in value.evidence
    )
    request_id = subject_assessment_request_id(value)
    policy = PRODUCTION_SUBJECT_ASSESSMENT_POLICY
    common = {
        "day": DAY,
        "request_id": request_id,
        "policy": policy,
        "policy_digest": subject_assessment_policy_digest(policy),
        "themes": value.themes,
        "summary_inputs": tuple(item.reference for item in value.summaries),
        "relevance_inputs": value.relevance,
        "construction": ModelSubjectAssessmentConstruction.model_construct(),
        "subject_ids": (first.id, second.id),
    }
    assessments = (
        SubjectAssessment(
            theme_id=first.id,
            group_ids=first.group_ids,
            article_version_ids=first.article_version_ids,
            tier="main",
            semantic_rank=1,
            rationale="National consequence.",
            evidence=(evidence[0],),
        ),
        SubjectAssessment(
            theme_id=second.id,
            group_ids=second.group_ids,
            article_version_ids=second.article_version_ids,
            tier="main",
            semantic_rank=1,
            rationale="Another consequence.",
            evidence=(evidence[1],),
        ),
    )
    payload = {**common, "assessments": assessments}
    with pytest.raises(ValidationError, match="ordered tier positions"):
        DailySubjectAssessmentSet.model_validate(payload)


def test_stable_group_anchors_resolve_per_trial_and_reject_merged_subjects() -> None:
    value = _input()
    assessment_set = _assessment_set(value)
    first_group, second_group = value.theme_set.groups

    assert compare_subject_anchors(
        assessment_set,
        value.theme_set,
        first_group.id,
        second_group.id,
    )

    merged = _reader_theme_set(value).model_copy(
        update={
            "themes": (
                DailyTheme(
                    id="e" * 64,
                    title="Merged",
                    summary="Merged subject.",
                    group_ids=(first_group.id, second_group.id),
                    article_version_ids=(
                        *first_group.article_version_ids,
                        *second_group.article_version_ids,
                    ),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="resolve to one final subject"):
        compare_subject_anchors(
            assessment_set,
            merged,
            first_group.id,
            second_group.id,
        )


def test_assessment_evaluation_uses_stable_anchors_and_fails_a_merged_pair() -> None:
    value = _input()
    first_group, second_group = value.theme_set.groups
    case = RankingEvaluationCase.model_construct(
        case_id="stable-ranking",
        control=False,
        provenance=EvaluationProvenance(feedback_ids=()),
        higher_group_id=first_group.id,
        lower_group_id=second_group.id,
    )

    result = evaluate_subject_assessment((case,), _reader_theme_set(value), _assessment_set(value))

    assert result.passed_cases == result.total_cases == 1
    assert result.merged_pair_count == result.semantic_tie_count == 0

    merged = _reader_theme_set(value).model_copy(
        update={
            "themes": (
                DailyTheme(
                    id="e" * 64,
                    title="Merged",
                    summary="Merged subject.",
                    group_ids=(first_group.id, second_group.id),
                    article_version_ids=(
                        *first_group.article_version_ids,
                        *second_group.article_version_ids,
                    ),
                ),
            )
        }
    )
    merged_result = evaluate_subject_assessment((case,), merged, _assessment_set(value))

    assert merged_result.merged_pair_count == 1
    assert merged_result.case_results[0].passed is False
    assert "merged into one subject" in merged_result.case_results[0].detail


def test_assessment_evaluation_fails_a_semantic_tie_before_any_id_tiebreak() -> None:
    value = _input()
    assessment_set = _assessment_set(value)
    first, second = assessment_set.assessments
    tied = assessment_set.model_copy(
        update={
            "assessments": (
                first,
                second.model_copy(update={"tier": "main", "semantic_rank": 1}),
            )
        }
    )
    first_group, second_group = value.theme_set.groups
    case = RankingEvaluationCase.model_construct(
        case_id="semantic-tie",
        control=False,
        provenance=EvaluationProvenance(feedback_ids=()),
        higher_group_id=first_group.id,
        lower_group_id=second_group.id,
    )

    result = evaluate_subject_assessment((case,), _reader_theme_set(value), tied)

    assert result.semantic_tie_count == 1
    assert result.case_results[0].passed is False
    assert "tie before the technical ID" in result.case_results[0].detail


def test_conflicting_frozen_tier_labels_report_exact_merged_subject() -> None:
    value = _input()
    first_group, second_group = value.theme_set.groups
    merged = _reader_theme_set(value).model_copy(
        update={
            "themes": (
                DailyTheme(
                    id="e" * 64,
                    title="Merged",
                    summary="Merged subject.",
                    group_ids=(first_group.id, second_group.id),
                    article_version_ids=(
                        *first_group.article_version_ids,
                        *second_group.article_version_ids,
                    ),
                ),
            )
        }
    )
    cases = (
        TierEvaluationCase(
            case_id="main-label",
            provenance=EvaluationProvenance(feedback_ids=(), group_id=first_group.id),
            expected_tier="main",
            observed_tier="main",
        ),
        TierEvaluationCase(
            case_id="excluded-label",
            provenance=EvaluationProvenance(feedback_ids=(), group_id=second_group.id),
            expected_tier="excluded",
            observed_tier="main",
        ),
    )

    conflicts = subject_tier_label_conflicts(cases, merged)

    assert len(conflicts) == 1
    assert conflicts[0].subject_id == "e" * 64
    assert conflicts[0].labels == ("excluded", "main")
    assert conflicts[0].governing_case_id is None
    assert conflicts[0].superseded_case_ids == ()
    with pytest.raises(ValueError, match="without a reviewed merge rule"):
        evaluate_subject_assessment(cases, merged, _assessment_set(value))


def test_merged_worth_knowing_label_keeps_the_governing_main_case() -> None:
    value = _input()
    first_group, second_group = value.theme_set.groups
    merged = _reader_theme_set(value).model_copy(
        update={
            "themes": (
                DailyTheme(
                    id="e" * 64,
                    title="Merged",
                    summary="Merged subject.",
                    group_ids=(first_group.id, second_group.id),
                    article_version_ids=(
                        *first_group.article_version_ids,
                        *second_group.article_version_ids,
                    ),
                ),
            )
        }
    )
    cases = (
        TierEvaluationCase(
            case_id="main-label",
            provenance=EvaluationProvenance(feedback_ids=(), group_id=first_group.id),
            expected_tier="main",
            observed_tier="main",
        ),
        TierEvaluationCase(
            case_id="worth-knowing-label",
            provenance=EvaluationProvenance(feedback_ids=(), group_id=second_group.id),
            expected_tier="worth_knowing",
            observed_tier="main",
        ),
    )

    conflicts = subject_tier_label_conflicts(cases, merged)

    assert len(conflicts) == 1
    assert conflicts[0].governing_case_id == "main-label"
    assert conflicts[0].superseded_case_ids == ("worth-knowing-label",)
    assessment = _assessment_set(value).assessments[0]
    prior = _assessment_set(value)
    merged_assessment_set = DailySubjectAssessmentSet.model_construct(
        day=prior.day,
        request_id=prior.request_id,
        policy=prior.policy,
        policy_digest=prior.policy_digest,
        themes=prior.themes,
        summary_inputs=prior.summary_inputs,
        relevance_inputs=prior.relevance_inputs,
        construction=prior.construction,
        subject_ids=("e" * 64,),
        assessments=(
            SubjectAssessment(
                theme_id="e" * 64,
                group_ids=(first_group.id, second_group.id),
                article_version_ids=merged.themes[0].article_version_ids,
                tier="main",
                semantic_rank=1,
                rationale=assessment.rationale,
                evidence=assessment.evidence,
            ),
        ),
    )

    result = evaluate_subject_assessment(cases, merged, merged_assessment_set)

    assert [item.case_id for item in result.case_results] == ["main-label"]
    assert all(item.passed for item in result.case_results)
    assert len(result.merged_tier_resolutions) == 1
    assert "keeps the governing main case" in result.merged_tier_resolutions[0]


def _assessment_set(value: DailySubjectAssessmentInput) -> DailySubjectAssessmentSet:
    first, second = value.theme_set.themes
    policy = PRODUCTION_SUBJECT_ASSESSMENT_POLICY
    evidence = tuple(
        SubjectAssessmentEvidence(
            group_id=item.group_id,
            article=item.article,
            relevance=item.relevance,
            summary=item.summary,
            evidence_quote=item.evidence_quote,
        )
        for item in value.evidence
    )
    return DailySubjectAssessmentSet.model_construct(
        day=DAY,
        request_id=subject_assessment_request_id(value),
        policy=policy,
        policy_digest=subject_assessment_policy_digest(policy),
        themes=value.themes,
        summary_inputs=tuple(item.reference for item in value.summaries),
        relevance_inputs=value.relevance,
        construction=EmptySubjectAssessmentConstruction(),
        subject_ids=(first.id, second.id),
        assessments=(
            SubjectAssessment(
                theme_id=first.id,
                group_ids=first.group_ids,
                article_version_ids=first.article_version_ids,
                tier="main",
                semantic_rank=1,
                rationale="National consequence.",
                evidence=(evidence[0],),
            ),
            SubjectAssessment(
                theme_id=second.id,
                group_ids=second.group_ids,
                article_version_ids=second.article_version_ids,
                tier="worth_knowing",
                semantic_rank=1,
                rationale="Useful context.",
                evidence=(evidence[1],),
            ),
        ),
    )


def _reader_theme_set(value: DailySubjectAssessmentInput) -> ReaderSubjectDailyThemeSet:
    theme_set = value.theme_set
    assert isinstance(theme_set, ReaderSubjectDailyThemeSet)
    return theme_set


def _input(*, empty: bool = False) -> DailySubjectAssessmentInput:
    cluster = _reference(1, "clusters")
    themes_reference = _reference(2, "themes")
    groups = (
        ()
        if empty
        else (
            NewsGroup(id="a" * 64, article_version_ids=("1" * 64,)),
            NewsGroup(id="b" * 64, article_version_ids=("2" * 64,)),
        )
    )
    summaries = tuple(_reference(10 + index, "summary") for index in range(len(groups)))
    policy_digest = sparse_theme_policy_digest(PRODUCTION_THEME_POLICY)
    themes = tuple(
        DailyTheme(
            id=daily_theme_id(DAY, (group.id,), policy_digest),
            title=f"Subject {index}",
            summary=f"Summary {index}.",
            group_ids=(group.id,),
            article_version_ids=group.article_version_ids,
        )
        for index, group in enumerate(groups, start=1)
    )
    theme_set = ReaderSubjectDailyThemeSet.model_construct(
        day=DAY,
        request_id="c" * 64,
        policy=PRODUCTION_THEME_POLICY,
        policy_digest=policy_digest,
        cluster_set=cluster,
        groups=groups,
        summary_inputs=summaries,
        construction=EmptyThemeConstruction(),
        themes=themes,
    )
    summary_inputs = tuple(
        SubjectSummaryInput(
            group_id=group.id,
            reference=reference,
            summary=GroupSummary(
                title_ro=f"Titlu {index}",
                summary_ro=f"Rezumat {index}.",
                key_points_ro=(f"Punct {index}.",),
                disagreements_ro=(),
                uncertainty_ro="Statusul oficial este neclar." if index == 1 else None,
                cited_article_version_ids=group.article_version_ids,
            ),
        )
        for index, (group, reference) in enumerate(zip(groups, summaries, strict=True), start=1)
    )
    relevance = tuple(_reference(20 + index, "relevance") for index in range(len(groups)))
    evidence = tuple(
        SubjectEvidenceInput(
            theme_id=theme.id,
            group_id=group.id,
            article=_reference(
                int(group.article_version_ids[0][0], 16), "article", group.article_version_ids[0]
            ),
            relevance=relevance_reference,
            summary=summary_reference,
            evidence_quote=f"Evidence {index}.",
        )
        for index, (theme, group, relevance_reference, summary_reference) in enumerate(
            zip(themes, groups, relevance, summaries, strict=True), start=1
        )
    )
    value = DailySubjectAssessmentInput.model_construct(
        day=DAY,
        themes=themes_reference,
        theme_set=theme_set,
        summaries=summary_inputs,
        relevance=relevance,
        evidence=evidence,
    )
    assert isinstance(value.theme_set, ReaderSubjectDailyThemeSet)
    return value


def _reference(index: int, kind: str, version_id: str | None = None) -> ArtifactReference:
    value = version_id or f"{index:064x}"
    return ArtifactReference(
        artifact_id=f"news:{kind}:{index}",
        version_id=value,
        content_digest=f"{index + 100:064x}",
        r2_key=f"news/{kind}/{index}.json",
    )


def test_stored_summary_payload_parses_json_lists(monkeypatch) -> None:
    reference = _reference(9, "summary")
    summary = GroupSummary(
        title_ro="Titlu",
        summary_ro="Rezumat.",
        key_points_ro=("Punct unu.", "Punct doi."),
        disagreements_ro=(),
        cited_article_version_ids=(_reference(1, "article").version_id,),
    )
    payload = {
        "group_id": _input().summaries[0].group_id,
        "summary": json.loads(summary.model_dump_json()),
    }
    monkeypatch.setattr(
        assessment_module,
        "read_verified_r2_object",
        lambda _r2_key, _digest: json.dumps(payload).encode(),
    )

    parsed = assessment_module._read_summary(reference, payload["group_id"])

    assert parsed == summary
