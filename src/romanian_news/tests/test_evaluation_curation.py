import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from romanian_news import Sha256
from romanian_news import evaluation_curation as curation
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.catalog import evaluations
from romanian_news.evaluation import (
    DailyThemeJudgment,
    EvaluationProvenance,
    ExcludedFeedback,
    ExcludedFeedbackScore,
    FeedbackScoreCuration,
    NewsEvaluationDataset,
    NonExecutableFeedback,
    ReportEvaluationSnapshot,
    ReportGroupInputs,
    ReportInputReferences,
    ThemeEvaluationDayCase,
    ThemePairExpectation,
)
from romanian_news.evaluation_curation import (
    DATASET_VERSION,
    DEFAULT_OUTPUT_PATH,
    EXCLUSIONS,
    FEEDBACK_TARGETS,
    GROUPING_SPECS,
    PRIOR_MANIFEST_REFERENCE,
    RANKING_SPECS,
    RELEVANCE_SPECS,
    SOURCE_FEEDBACK_IDS,
    SOURCE_MANIFEST_REFERENCE,
    THEME_JUDGMENTS,
    _FeedbackTargetExpectation,
    _load_prior_dataset,
    _report_group_inputs,
    _report_themes_reference,
    _require_feedback_target,
    _require_report_matches_cluster,
    _require_supersession,
    _without_feedback,
)
from romanian_news.feedback import GroupFeedbackTarget, NewsFeedbackEvent, ReportFeedbackTarget
from romanian_news.groups import DailyClusterSet, EmbeddedArticleReference, NewsGroup
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportSectionV2,
    DailyReportV2,
    ReportArticle,
    ReportEvent,
)
from romanian_news.tests.evaluation_factories import (
    embedded_article,
    synthetic_dataset,
    synthetic_manifest,
)
from romanian_news.themes import (
    DailyTheme,
    DailyThemeInput,
    DailyThemeSet,
    ModelThemeConstruction,
    SparseDailyThemeSet,
    SparseThemeConstruction,
    ThemeGroupInput,
    ThemeModelAttemptEvidence,
    ThemeStageEvidence,
)


def _reference(label: str, digit: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{label}",
        version_id=digit * 64,
        content_digest=digit * 64,
        r2_key=f"news/{label}.json",
    )


def test_curation_loads_the_exact_published_v5_manifest_through_catalog(monkeypatch) -> None:
    manifest = synthetic_manifest()
    dataset = synthetic_dataset()
    calls = []
    monkeypatch.setattr(
        "romanian_news.evaluation_curation.read_evaluation_artifact",
        lambda reference: calls.append(("read", reference)) or manifest.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        "romanian_news.catalog.evaluations.hydrate_news_evaluation_manifest",
        lambda value: calls.append(("hydrate", value)) or dataset,
    )

    assert _load_prior_dataset() == dataset
    assert calls == [("read", SOURCE_MANIFEST_REFERENCE), ("hydrate", manifest)]
    assert SOURCE_MANIFEST_REFERENCE.model_dump() == {
        "artifact_id": "news:evaluation-manifest:news-evaluation-2026-09-05-v5",
        "version_id": "b973c0f3f05e65f93d097fbdbfb1461533351a28f1c9bcd0b52e0fe35e5056d6",
        "content_digest": "c2fad2de91cf4e7d25503caff07d6557713f111feecbcc1110d1f22e55bb4440",
        "r2_key": (
            "news/evaluations/manifests/news-evaluation-2026-09-05-v5/"
            "c2fad2de91cf4e7d25503caff07d6557713f111feecbcc1110d1f22e55bb4440.json"
        ),
    }


def test_model_output_route_uses_verified_response_and_exact_accepted_attempt(
    monkeypatch,
) -> None:
    output = _reference("relevance:output", "a")
    queries = []
    monkeypatch.setattr(
        curation,
        "read_evaluation_artifact",
        lambda reference: json.dumps(
            {
                "request_id": "b" * 64,
                "provider_responses": [{"id": "rejected"}, {"id": "accepted"}],
            }
        ).encode(),
    )
    monkeypatch.setattr(
        curation.evaluation_catalog,
        "read_news_model_attempt_id",
        lambda request_id, response_id: queries.append((request_id, response_id)) or "c" * 64,
    )

    route = curation._model_output_observation(output)

    assert route == curation._ModelObservationRoute(output=output, attempt_id="c" * 64)
    assert queries == [("b" * 64, "accepted")]


def test_theme_stage_requires_payload_and_catalog_attempt_identity(monkeypatch) -> None:
    output = _reference("themes:output", "d")
    attempt = ThemeModelAttemptEvidence.model_construct(
        attempt_id="e" * 64,
        response_id="accepted",
        status="accepted",
    )
    stage = ThemeStageEvidence.model_construct(
        request_id="f" * 64,
        call=ModelCall.model_construct(response_id="accepted"),
        attempts=(attempt,),
    )
    monkeypatch.setattr(curation, "_resolve_accepted_attempt", lambda *_args: "e" * 64)

    assert curation._theme_stage_observation(output, stage) == curation._ModelObservationRoute(
        output=output,
        attempt_id="e" * 64,
    )

    monkeypatch.setattr(curation, "_resolve_accepted_attempt", lambda *_args: "1" * 64)
    with pytest.raises(ValueError, match="differs from the catalog"):
        curation._theme_stage_observation(output, stage)


@pytest.mark.parametrize(
    ("combined_observation", "expected_decision", "expected_reason"),
    (
        (False, "projected", None),
        (True, "excluded", "no_concern_specific_observation"),
    ),
)
def test_complete_score_curation_projects_only_model_owned_concerns(
    monkeypatch,
    combined_observation: bool,
    expected_decision: str,
    expected_reason: str | None,
) -> None:
    feedback_ids = tuple(UUID(f"00000000-0000-4000-8000-{index:012d}") for index in range(101, 106))
    dataset = synthetic_dataset()
    report = dataset.reports[0]
    relevance_output = _reference("relevance:score", "a")
    cases = []
    for concern, feedback_id in zip(
        ("relevance", "grouping", "ranking"), feedback_ids, strict=False
    ):
        case = next(item for item in dataset.cases if item.concern == concern)
        provenance = EvaluationProvenance(
            feedback_ids=(feedback_id,),
            report=report.report,
            model_outputs=(relevance_output,) if concern == "relevance" else (),
        )
        cases.append(case.model_copy(update={"provenance": provenance}))
    dataset = dataset.model_copy(
        update={"source_feedback_ids": feedback_ids, "cases": tuple(cases)}
    )
    manifest = synthetic_manifest().model_copy(
        update={
            "source_feedback_ids": feedback_ids,
            "excluded_feedback": (
                NonExecutableFeedback(
                    feedback_id=feedback_ids[3],
                    report_version_id=report.report.version_id,
                    reason="non_executable",
                    concern="language",
                    rationale="The reviewed text uses the wrong language.",
                ),
                NonExecutableFeedback(
                    feedback_id=feedback_ids[4],
                    report_version_id=report.report.version_id,
                    reason="non_executable",
                    concern="presentation",
                    rationale="The report presentation is not model-owned.",
                ),
            ),
        }
    )
    route = curation._ModelObservationRoute(output=relevance_output, attempt_id="b" * 64)
    theme_output = _reference("themes:score", "c")
    report = report.model_copy(update={"themes": theme_output})
    dataset = dataset.model_copy(update={"reports": (report,)})
    theme_route = curation._ModelObservationRoute(output=theme_output, attempt_id="d" * 64)
    monkeypatch.setattr(curation, "_model_output_observation", lambda _output: route)
    theme_set = _score_theme_set(combined_observation)
    monkeypatch.setattr(curation, "_read_theme_set", lambda _snapshot: theme_set)
    monkeypatch.setattr(
        curation,
        "_theme_observation",
        lambda _snapshot, _stage: _score_theme_observation(
            combined_observation, theme_set, theme_route
        ),
    )
    feedback = (
        NewsFeedbackEvent(
            feedback_id=feedback_ids[3],
            target=ReportFeedbackTarget(report_version_id=report.report.version_id),
            rating="negative",
            actor="owner",
            created_at=datetime(2026, 9, 3, 20, 37, tzinfo=UTC),
        ),
    )

    decisions = curation._build_feedback_score_curation(manifest, dataset, feedback)

    by_feedback = {
        feedback_id: tuple(
            decision for decision in decisions if decision.feedback_id == feedback_id
        )
        for feedback_id in feedback_ids
    }
    _assert_relevance_score(by_feedback[feedback_ids[0]][0])
    _assert_grouping_exclusion(by_feedback[feedback_ids[1]][0])
    _assert_ranking_exclusion(by_feedback[feedback_ids[2]][0])
    _assert_language_score(
        by_feedback[feedback_ids[3]][0],
        expected_decision,
        expected_reason,
    )
    _assert_presentation_exclusion(by_feedback[feedback_ids[4]][0])


def _assert_relevance_score(decision: FeedbackScoreCuration) -> None:
    assert decision.decision == "projected"
    assert decision.concern == "relevance"


def _assert_grouping_exclusion(decision: FeedbackScoreCuration) -> None:
    assert decision.decision == "excluded"
    assert decision.reason == "no_model_observation"
    assert decision.concern == "grouping"
    assert decision.polarity == "positive"


def _assert_ranking_exclusion(decision: FeedbackScoreCuration) -> None:
    assert decision.decision == "excluded"
    assert decision.reason == "no_model_observation"
    assert decision.concern == "ranking"


def _assert_language_score(
    decision: FeedbackScoreCuration,
    expected_decision: str,
    expected_reason: str | None,
) -> None:
    assert decision.concern == "language"
    assert decision.polarity == "negative"
    assert decision.decision == expected_decision
    assert getattr(decision, "reason", None) == expected_reason


def _assert_presentation_exclusion(decision: FeedbackScoreCuration) -> None:
    assert decision.decision == "excluded"
    assert decision.reason == "presentation"


def _score_theme_set(
    combined_observation: bool,
) -> DailyThemeSet | SparseDailyThemeSet:
    if combined_observation:
        return DailyThemeSet.model_construct(construction=ModelThemeConstruction.model_construct())
    return SparseDailyThemeSet.model_construct(
        construction=SparseThemeConstruction.model_construct()
    )


def _score_theme_observation(
    combined_observation: bool,
    theme_set: DailyThemeSet | SparseDailyThemeSet,
    route: curation._ModelObservationRoute,
) -> tuple[DailyThemeSet | SparseDailyThemeSet, curation._ModelObservationRoute]:
    if combined_observation:
        raise AssertionError("combined output has no prose stage")
    return theme_set, route


@pytest.mark.parametrize("combined_observation", (False, True))
def test_theme_grouping_requires_a_concern_specific_assignment_route(
    monkeypatch,
    combined_observation: bool,
) -> None:
    feedback_id = UUID("00000000-0000-4000-8000-000000000201")
    dataset = synthetic_dataset()
    themes_reference = _reference("themes", "a")
    report = dataset.reports[0].model_copy(update={"themes": themes_reference})
    dataset = dataset.model_copy(update={"reports": (report,)})
    left_id = "1" * 64
    right_id = "2" * 64
    expectation = ThemePairExpectation(
        case_id="reviewed-theme-pair",
        feedback_ids=(feedback_id,),
        left_group_id=left_id,
        right_group_id=right_id,
        expected_same_theme=True,
        rationale="The groups should share one theme.",
    )
    case = ThemeEvaluationDayCase.model_construct(
        case_id="theme-day",
        provenance=EvaluationProvenance(feedback_ids=(feedback_id,)),
        source_report=report.report,
        expectations=(expectation,),
    )
    themes = (
        DailyTheme.model_construct(id="3" * 64, group_ids=(left_id,)),
        DailyTheme.model_construct(id="4" * 64, group_ids=(right_id,)),
    )
    theme_set = (
        DailyThemeSet.model_construct(
            construction=ModelThemeConstruction.model_construct(), themes=themes
        )
        if combined_observation
        else SparseDailyThemeSet.model_construct(
            construction=SparseThemeConstruction.model_construct(), themes=themes
        )
    )
    assignment_route = curation._ModelObservationRoute(
        output=themes_reference,
        attempt_id="5" * 64,
    )
    monkeypatch.setattr(curation, "_read_theme_set", lambda _snapshot: theme_set)
    monkeypatch.setattr(
        curation,
        "_theme_observation",
        lambda _snapshot, stage: (theme_set, assignment_route)
        if not combined_observation and stage == "assignment"
        else (_ for _ in ()).throw(AssertionError("wrong stage")),
    )
    decisions = {feedback_id: []}

    curation._append_theme_grouping_scores(
        case,
        (feedback_id,),
        dataset,
        decisions,
    )

    decision = decisions[feedback_id][0]
    assert decision.concern == "grouping"
    assert decision.polarity == "negative"
    if combined_observation:
        assert decision.decision == "excluded"
        assert decision.reason == "no_concern_specific_observation"
    else:
        assert decision.decision == "projected"
        assert decision.model_attempt_id == "5" * 64


def test_score_builder_creates_a_new_manifest_over_immutable_v62(monkeypatch) -> None:
    prior = synthetic_manifest().model_copy(update={"version": DATASET_VERSION})
    prior_reference = _reference("evaluation:v6.2", "f")
    decisions = tuple(
        ExcludedFeedbackScore(
            feedback_id=feedback_id,
            reason="presentation",
            report_version_id=prior.reports[0].report.version_id,
            rationale="This test decision is not model-owned.",
        )
        for feedback_id in prior.source_feedback_ids
    )
    monkeypatch.setattr(curation, "SOURCE_FEEDBACK_IDS", prior.source_feedback_ids)
    monkeypatch.setattr(
        curation.evaluation_catalog,
        "read_news_evaluation_artifact_references",
        lambda version_ids: {version_ids[0]: prior_reference},
    )
    monkeypatch.setattr(
        curation, "read_evaluation_artifact", lambda _reference: prior.model_dump_json().encode()
    )
    monkeypatch.setattr(
        curation.evaluation_catalog,
        "hydrate_news_evaluation_manifest",
        lambda _manifest: synthetic_dataset(),
    )
    monkeypatch.setattr(
        curation.evaluation_catalog,
        "read_news_evaluation_feedback",
        lambda _feedback_ids: (),
    )
    monkeypatch.setattr(curation, "read_evaluation_report_bundle", lambda _version_id: object())
    monkeypatch.setattr(curation, "_require_exact_feedback_source", lambda *_args: None)
    monkeypatch.setattr(
        curation,
        "_build_feedback_score_curation",
        lambda *_args: decisions,
    )

    result = curation.build_news_feedback_score_manifest()

    assert prior.score_curation == ()
    assert result.version == curation.SCORE_DATASET_VERSION
    assert result.issue_url == curation.SCORE_ISSUE_URL
    assert result.prior_manifest == prior_reference
    assert result.score_curation == decisions


def test_score_curation_command_selects_the_v63_builder(monkeypatch, tmp_path) -> None:
    output = tmp_path / "score-manifest.json"
    calls = []
    monkeypatch.setattr(
        curation,
        "write_news_feedback_score_manifest",
        lambda path: calls.append(path),
    )

    assert curation.main(("--score-curation", "--output", str(output))) == 0
    assert calls == [output]


def test_curation_constants_cover_the_exact_reviewed_feedback_set() -> None:
    assigned = (
        {UUID(item.feedback_id) for item in RELEVANCE_SPECS}
        | {UUID(item.feedback_id) for item in GROUPING_SPECS}
        | {UUID(item.feedback_id) for item in RANKING_SPECS}
        | {feedback_id for judgment in THEME_JUDGMENTS for feedback_id in judgment.feedback_ids}
        | {item.feedback_id for item in EXCLUSIONS}
    )

    assert len(SOURCE_FEEDBACK_IDS) == 30
    assert len(set(SOURCE_FEEDBACK_IDS)) == 30
    assert assigned == set(SOURCE_FEEDBACK_IDS)
    assert set(FEEDBACK_TARGETS) == set(SOURCE_FEEDBACK_IDS)
    assert len({target.model_dump_json() for target in FEEDBACK_TARGETS.values()}) == 24
    assert sum(item.reason == "superseded" for item in EXCLUSIONS) == 6


def test_v62_constants_record_v61_lineage() -> None:
    assert DATASET_VERSION == "news-evaluation-2026-09-07-v6.2"
    assert (
        Path("data/news/evaluations/news-evaluation-2026-09-07-v6.2-manifest.json")
        == DEFAULT_OUTPUT_PATH
    )
    assert PRIOR_MANIFEST_REFERENCE.model_dump() == {
        "artifact_id": "news:evaluation-manifest:news-evaluation-2026-09-07-v6.1",
        "version_id": "17b39ccf990097441b87537304df5110f61c05aa730f34de31d6c2873fdb2364",
        "content_digest": "638c1f32b50db5a95a3a510783aa65db2dfc158ae6508c89feea2b921885ed7e",
        "r2_key": (
            "news/evaluations/manifests/news-evaluation-2026-09-07-v6.1/"
            "638c1f32b50db5a95a3a510783aa65db2dfc158ae6508c89feea2b921885ed7e.json"
        ),
    }

    dataset = synthetic_dataset()
    feedback_id = UUID("00000000-0000-4000-8000-000000000099")
    inherited = dataset.cases[0].model_copy(
        update={
            "provenance": dataset.cases[0].provenance.model_copy(
                update={"feedback_ids": (feedback_id,)}
            )
        }
    )
    assert _without_feedback((inherited,))[0].provenance.feedback_ids == ()


def test_report_group_inputs_map_payload_group_ids_instead_of_input_order(monkeypatch) -> None:
    first_group_id = "1" * 64
    second_group_id = "2" * 64
    first_summary = _reference("summary:first", "3")
    second_summary = _reference("summary:second", "4")
    first_sentiment = _reference("sentiment:first", "5")
    second_sentiment = _reference("sentiment:second", "6")
    group_ids = {
        first_summary.version_id: first_group_id,
        second_summary.version_id: second_group_id,
        first_sentiment.version_id: first_group_id,
        second_sentiment.version_id: second_group_id,
    }
    monkeypatch.setattr(
        "romanian_news.evaluation_curation.read_evaluation_artifact",
        lambda reference: json.dumps({"group_id": group_ids[reference.version_id]}).encode(),
    )

    result = _report_group_inputs(
        (
            NewsGroup(id=second_group_id, article_version_ids=("7" * 64,)),
            NewsGroup(id=first_group_id, article_version_ids=("8" * 64,)),
        ),
        (second_summary, first_summary),
        (first_sentiment, second_sentiment),
    )

    assert tuple(item.group_id for item in result) == (first_group_id, second_group_id)
    assert tuple(item.summary for item in result) == (first_summary, second_summary)
    assert tuple(item.sentiment for item in result) == (first_sentiment, second_sentiment)


def test_report_group_inputs_reject_duplicate_payload_groups(monkeypatch) -> None:
    group_id = "1" * 64
    first_summary = _reference("summary:first", "3")
    second_summary = _reference("summary:second", "4")
    sentiment = _reference("sentiment:first", "5")
    monkeypatch.setattr(
        "romanian_news.evaluation_curation.read_evaluation_artifact",
        lambda _reference: json.dumps({"group_id": group_id}).encode(),
    )

    with pytest.raises(ValueError, match="duplicate summary inputs"):
        _report_group_inputs(
            (NewsGroup(id=group_id, article_version_ids=("7" * 64,)),),
            (first_summary, second_summary),
            (sentiment,),
        )


def test_feedback_target_expectation_accepts_only_its_exact_raw_target() -> None:
    feedback_id = UUID("00000000-0000-4000-8000-000000000001")
    target = GroupFeedbackTarget(
        report_version_id="1" * 64,
        group_id="2" * 64,
    )
    expectation = _FeedbackTargetExpectation(feedback_id=feedback_id, target=target)
    row = _feedback(feedback_id, target.report_version_id, target.group_id)

    _require_feedback_target({feedback_id: row}, expectation)

    row = row.model_copy(
        update={
            "target": GroupFeedbackTarget(
                report_version_id=target.report_version_id,
                group_id="3" * 64,
            )
        }
    )
    with pytest.raises(ValueError, match=str(feedback_id)):
        _require_feedback_target({feedback_id: row}, expectation)


def _feedback(
    feedback_id: UUID,
    report_version_id: Sha256,
    group_id: Sha256,
    created_at: datetime = datetime(2026, 9, 3, 20, 37, tzinfo=UTC),
) -> NewsFeedbackEvent:
    return NewsFeedbackEvent(
        feedback_id=feedback_id,
        target=GroupFeedbackTarget(
            report_version_id=report_version_id,
            group_id=group_id,
        ),
        rating="positive",
        note=None,
        actor="owner",
        created_at=created_at,
    )


def _supersession_rows() -> tuple[dict[UUID, NewsFeedbackEvent], ExcludedFeedback]:
    old_id = UUID("00000000-0000-4000-8000-000000000001")
    replacement_id = UUID("00000000-0000-4000-8000-000000000002")
    report_version_id = "1" * 64
    group_id = "2" * 64
    rows = {
        old_id: _feedback(old_id, report_version_id, group_id),
        replacement_id: _feedback(
            replacement_id,
            report_version_id,
            group_id,
            datetime(2026, 9, 3, 21, 11, tzinfo=UTC),
        ),
    }
    exclusion = ExcludedFeedback(
        feedback_id=old_id,
        report_version_id=report_version_id,
        reason="superseded",
        superseded_by_feedback_id=replacement_id,
    )
    return rows, exclusion


def test_supersession_requires_the_same_exact_feedback_target() -> None:
    rows, exclusion = _supersession_rows()
    rows[exclusion.superseded_by_feedback_id] = rows[
        exclusion.superseded_by_feedback_id
    ].model_copy(
        update={
            "target": GroupFeedbackTarget(
                report_version_id=exclusion.report_version_id,
                group_id="3" * 64,
            )
        }
    )

    with pytest.raises(ValueError, match="same exact report target"):
        _require_supersession(rows, exclusion)


def test_supersession_requires_the_replacement_to_be_newer() -> None:
    rows, exclusion = _supersession_rows()
    rows[exclusion.superseded_by_feedback_id] = rows[
        exclusion.superseded_by_feedback_id
    ].model_copy(update={"created_at": datetime(2026, 9, 3, 20, 0, tzinfo=UTC)})

    with pytest.raises(ValueError, match="newer by reader ordering"):
        _require_supersession(rows, exclusion)


def test_v6_specs_include_direct_grouping_theme_and_ranking_cases() -> None:
    duplicate_cases = tuple(item for item in GROUPING_SPECS if not item.control)
    assert len(duplicate_cases) == 2
    assert all(item.right_group_id is not None for item in duplicate_cases)
    assert all(item.expected_same_group for item in duplicate_cases)

    negative_theme = next(
        item for item in THEME_JUDGMENTS if item.judgment_id.endswith("events-differ")
    )
    assert negative_theme.expected_same_theme is False
    assert negative_theme.feedback_ids == (UUID("2b89d953-caaf-481b-93d4-82ede5fe128c"),)

    assert [item.control for item in RANKING_SPECS] == [True, False, True]
    assert all(item.control for item in RELEVANCE_SPECS if item.expected_accepted)


def test_cross_group_case_includes_every_article_from_both_groups(monkeypatch) -> None:
    articles = tuple(embedded_article(index) for index in range(1, 5))
    left_group_id = _reference("group:left", "1").version_id
    right_group_id = _reference("group:right", "2").version_id
    sections = {
        left_group_id: SimpleNamespace(
            articles=tuple(
                SimpleNamespace(article_version_id=item.article.version_id) for item in articles[:2]
            )
        ),
        right_group_id: SimpleNamespace(
            articles=tuple(
                SimpleNamespace(article_version_id=item.article.version_id) for item in articles[2:]
            )
        ),
    }
    embedded_by_id = {item.article.version_id: item for item in articles}
    references: dict[Sha256, EmbeddedArticleReference] = {
        item.article.version_id: EmbeddedArticleReference(
            article=item.article,
            relevance=item.relevance,
            embedding=item.embedding,
        )
        for item in articles
    }
    day = articles[0].value.bucharest_day
    bundle = curation.ReportBundle(
        snapshot=ReportEvaluationSnapshot(
            report=_reference("report", "3"),
            cluster_set=_reference("cluster", "4"),
            report_article_version_ids=(),
            report_group_ids=(),
            cluster_articles=(),
            group_inputs=(),
            report_inputs=ReportInputReferences(
                cluster_set=(_reference("cluster", "4"),), summary=(), sentiment=()
            ),
        ),
        report=DailyReport(
            day=day,
            accepted_article_count=0,
            theme_count=0,
            group_count=0,
            sections=(),
        ),
        cluster_set=DailyClusterSet(
            day=day,
            algorithm="test",
            threshold=0.72,
            embedding_model="test",
            article_version_ids=(),
            relevance_version_ids=(),
            embedding_version_ids=(),
            merges=(),
            groups=(),
        ),
        articles=references,
        relevance={},
    )
    spec = curation.GroupingSpec(
        case_id="cross-group",
        feedback_id="00000000-0000-4000-8000-000000000001",
        report_version_id=_reference("report", "3").version_id,
        left_group_id=left_group_id,
        right_group_id=right_group_id,
        control=False,
    )
    monkeypatch.setattr(curation, "_section", lambda _report, group_id: sections[group_id])
    monkeypatch.setattr(
        curation,
        "_load_embedded",
        lambda reference: embedded_by_id[reference.article.version_id],
    )

    case = curation._grouping_case(spec, bundle)

    assert case.articles == articles
    assert case.left_article_version_id == articles[0].article.version_id
    assert case.right_article_version_id == articles[2].article.version_id


def test_v6_non_executable_feedback_records_a_concern_and_rationale() -> None:
    values = tuple(item for item in EXCLUSIONS if item.reason == "non_executable")

    assert len(values) == 8
    assert {item.concern for item in values} == {
        "language",
        "presentation",
        "operations",
        "ambiguous",
        "contradictory",
    }
    assert all(item.rationale for item in values)


def test_report_themes_reference_requires_exact_fresh_report_lineage(monkeypatch) -> None:
    group_id = "1" * 64
    theme_id = "2" * 64
    cluster = _reference("cluster", "3")
    themes = _reference("themes", "4")
    summary = _reference("summary", "5")
    report = DailyReportV2(
        day=date(2026, 9, 6),
        accepted_article_count=0,
        theme_count=1,
        group_count=1,
        sections=(
            DailyReportSectionV2(
                theme_id=theme_id,
                title="Theme",
                summary="Theme summary",
                events=(
                    ReportEvent(
                        group_id=group_id,
                        title_ro="Event",
                        summary_ro="Event summary",
                        key_points_ro=(),
                        disagreements_ro=(),
                        sentiment_label="neutral",
                        sentiment_score=0,
                        sentiment_rationale_ro="No difference.",
                        articles=(),
                    ),
                ),
            ),
        ),
    )
    theme_set = SimpleNamespace(
        day=report.day,
        cluster_set=cluster,
        summary_inputs=(summary,),
        themes=(
            SimpleNamespace(
                id=theme_id,
                title="Theme",
                summary="Theme summary",
                group_ids=(group_id,),
            ),
        ),
    )
    monkeypatch.setattr(curation, "read_evaluation_artifact", lambda _reference: b"themes")
    monkeypatch.setattr(curation, "parse_daily_theme_set", lambda _content: theme_set)
    inputs = {
        "themes": [themes],
        "cluster_set": [cluster],
        "relevance": [],
        "summary": [summary],
        "sentiment": [],
    }

    assert _report_themes_reference(report, inputs, cluster) == themes

    theme_set.themes[0].group_ids = ("6" * 64,)
    with pytest.raises(ValueError, match="differs from theme input"):
        _report_themes_reference(report, inputs, cluster)


def test_report_snapshot_rejects_group_article_membership_drift() -> None:
    group_id = "1" * 64
    first_article_id = "2" * 64
    second_article_id = "3" * 64
    report = ArchivedDailyReport(
        day=date(2026, 9, 3),
        accepted_article_count=1,
        group_count=1,
        sections=(
            ArchivedDailyReportSection(
                group_id=group_id,
                title_ro="Titlu",
                summary_ro="Rezumat",
                key_points_ro=(),
                disagreements_ro=(),
                sentiment_label="neutral",
                sentiment_score=0.0,
                sentiment_rationale_ro="Fără diferențe.",
                articles=(
                    ReportArticle(
                        article_version_id=first_article_id,
                        outlet_id="example",
                        title="Primul articol",
                        canonical_url="https://example.com/first",
                        sentiment_label="neutral",
                        sentiment_score=0.0,
                    ),
                ),
            ),
        ),
    )
    cluster_set = DailyClusterSet(
        day=date(2026, 9, 3),
        algorithm="test",
        threshold=0.72,
        embedding_model="test",
        article_version_ids=(first_article_id, second_article_id),
        relevance_version_ids=("4" * 64, "5" * 64),
        embedding_version_ids=("6" * 64, "7" * 64),
        merges=(),
        groups=(
            NewsGroup(
                id=group_id,
                article_version_ids=(first_article_id, second_article_id),
            ),
        ),
    )

    with pytest.raises(ValueError, match="articles do not match cluster group"):
        _require_report_matches_cluster(report, cluster_set)


_PRIOR_THEME_JUDGMENT_IDS = {
    date(2026, 9, 2): (
        "sep02-government-crisis-and-psd-aur",
        "sep02-psd-aur-and-aur-participation",
        "sep02-psd-aur-and-udmr-position",
        "sep02-tise-and-government-formation",
        "sep02-climate-and-government-differ",
    ),
    date(2026, 9, 3): (
        "sep03-political-system-and-security-services",
        "sep03-political-system-and-pahontu",
        "sep03-security-services-and-pahontu",
        "sep03-integrity-law-and-ani-risk",
        "sep03-ai-and-political-system-differ",
        "sep03-global-and-romanian-demography-differ",
    ),
}
_GROUP_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@pytest.fixture
def four_day_theme_dataset(monkeypatch) -> NewsEvaluationDataset:
    prior_snapshots = tuple(
        _theme_snapshot(
            day,
            _numbered_reference(f"report:{day.isoformat()}", day.day * 100),
            tuple(f"{day.day * 100 + offset:064x}" for offset in range(5)),
        )
        for day in _PRIOR_THEME_JUDGMENT_IDS
    )
    prior_judgments = tuple(
        _theme_judgment(
            judgment_id=judgment_id,
            snapshot=snapshot,
            feedback_index=day.day * 100 + index,
            pair=_GROUP_PAIRS[index],
        )
        for day, snapshot in zip(_PRIOR_THEME_JUDGMENT_IDS, prior_snapshots, strict=True)
        for index, judgment_id in enumerate(_PRIOR_THEME_JUDGMENT_IDS[day])
    )
    prior = NewsEvaluationDataset(
        version="prior-v4",
        reviewed_at=datetime(2026, 9, 4, tzinfo=UTC),
        issue_url="https://example.test/issues/204",
        source_feedback_ids=tuple(
            feedback_id for judgment in prior_judgments for feedback_id in judgment.feedback_ids
        ),
        reports=prior_snapshots,
        cases=(),
        archived_daily_theme_judgments=prior_judgments,
    )
    current_snapshots = tuple(
        _current_theme_snapshot(day, report_version_id)
        for day, report_version_id in zip(
            (date(2026, 9, 4), date(2026, 9, 5)),
            curation.REPORT_VERSION_IDS,
            strict=True,
        )
    )
    inputs = {
        snapshot.cluster_set.version_id: _theme_input(snapshot, day)
        for snapshot, day in zip(
            (*prior_snapshots, *current_snapshots),
            (*_PRIOR_THEME_JUDGMENT_IDS, date(2026, 9, 4), date(2026, 9, 5)),
            strict=True,
        )
    }
    bundles = {
        snapshot.report.version_id: SimpleNamespace(snapshot=snapshot)
        for snapshot in current_snapshots
    }
    cluster_sets = {value.cluster_set.version_id: _cluster_set(value) for value in inputs.values()}
    current_feedback_ids = tuple(
        feedback_id
        for judgment in curation.THEME_JUDGMENTS
        for feedback_id in judgment.feedback_ids
    )
    monkeypatch.setattr(curation, "RELEVANCE_SPECS", ())
    monkeypatch.setattr(curation, "GROUPING_SPECS", ())
    monkeypatch.setattr(curation, "RANKING_SPECS", ())
    monkeypatch.setattr(curation, "EXCLUSIONS", ())
    monkeypatch.setattr(curation, "_load_prior_dataset", lambda: prior)
    monkeypatch.setattr(curation, "read_evaluation_report_bundle", bundles.__getitem__)
    monkeypatch.setattr(
        evaluations,
        "read_news_evaluation_feedback",
        lambda _feedback_ids: tuple(
            SimpleNamespace(feedback_id=value) for value in current_feedback_ids
        ),
    )
    monkeypatch.setattr(curation, "_require_exact_feedback_source", lambda *_args: None)
    monkeypatch.setattr(
        curation,
        "read_evaluation_artifact",
        lambda reference: cluster_sets[reference.version_id].model_dump_json().encode(),
    )

    def load_theme_input(cluster_set, summaries):
        value = inputs[cluster_set.version_id]
        summary_by_group = {item.group.id: item.summary for item in value.groups}
        assert summaries == tuple(summary_by_group[item.group.id] for item in value.groups)
        return value

    monkeypatch.setattr(curation, "load_daily_theme_input", load_theme_input)
    return curation._build_hydrated_dataset()


def test_hydrated_dataset_executes_all_prior_theme_judgments(
    four_day_theme_dataset,
) -> None:
    cases = _theme_cases(four_day_theme_dataset)
    expectations = _theme_expectations(cases)
    prior_ids = {case_id for values in _PRIOR_THEME_JUDGMENT_IDS.values() for case_id in values}

    assert {case.input.day for case in cases} == {
        date(2026, 9, 2),
        date(2026, 9, 3),
        date(2026, 9, 4),
        date(2026, 9, 5),
    }
    assert prior_ids <= set(expectations)
    assert all(expectations[case_id].feedback_ids == () for case_id in prior_ids)
    assert all(
        expectations[case_id].control and not expectations[case_id].expected_same_theme
        for case_id in prior_ids
        if "differ" in case_id
    )
    assert four_day_theme_dataset.archived_daily_theme_judgments == ()


def test_hydrated_dataset_preserves_current_theme_expectations(
    four_day_theme_dataset,
) -> None:
    expectations = _theme_expectations(_theme_cases(four_day_theme_dataset))

    assert all(
        expectations[judgment.judgment_id].feedback_ids == judgment.feedback_ids
        for judgment in curation.THEME_JUDGMENTS
    )


def _theme_cases(dataset: NewsEvaluationDataset):
    return tuple(case for case in dataset.cases if case.concern == "daily_theme")


def _theme_expectations(cases):
    return {expectation.case_id: expectation for case in cases for expectation in case.expectations}


def _current_theme_snapshot(day: date, report_version_id: str) -> ReportEvaluationSnapshot:
    judgments = tuple(
        item for item in curation.THEME_JUDGMENTS if item.report_version_id == report_version_id
    )
    group_ids = tuple(
        dict.fromkeys(
            group_id
            for judgment in judgments
            for group_id in (judgment.left_group_id, judgment.right_group_id)
        )
    )
    return _theme_snapshot(
        day,
        _numbered_reference(f"report:{day.isoformat()}", day.day * 100).model_copy(
            update={"version_id": report_version_id}
        ),
        (*group_ids, f"{day.day * 1000:064x}"),
    )


def _theme_snapshot(
    day: date,
    report: ArtifactReference,
    group_ids: tuple[str, ...],
) -> ReportEvaluationSnapshot:
    base = synthetic_dataset().reports[0]
    cluster_set = _numbered_reference(f"cluster:{day.isoformat()}", day.day * 100 + 1)
    return base.model_copy(
        update={
            "report": report,
            "cluster_set": cluster_set,
            "report_group_ids": group_ids,
            "group_inputs": tuple(
                ReportGroupInputs(
                    group_id=group_id,
                    summary=_numbered_reference(
                        f"summary:{day.isoformat()}:{offset}", day.day * 100 + offset + 2
                    ),
                    sentiment=_numbered_reference(
                        f"sentiment:{day.isoformat()}:{offset}", day.day * 100 + offset + 20
                    ),
                )
                for offset, group_id in enumerate(group_ids)
            ),
        }
    )


def _theme_input(snapshot: ReportEvaluationSnapshot, day: date) -> DailyThemeInput:
    return DailyThemeInput(
        day=day,
        cluster_set=snapshot.cluster_set,
        groups=tuple(
            ThemeGroupInput(
                group=NewsGroup(
                    id=item.group_id,
                    article_version_ids=(f"{day.day * 1000 + index:064x}",),
                ),
                summary=item.summary,
                value=GroupSummary(
                    title_ro=f"Tema {index}",
                    summary_ro="Rezumat verificat.",
                    key_points_ro=("Fapt verificat.",),
                    disagreements_ro=(),
                    uncertainty_ro=None,
                    cited_article_version_ids=(f"{day.day * 1000 + index:064x}",),
                ),
            )
            for index, item in enumerate(reversed(snapshot.group_inputs))
        ),
    )


def _cluster_set(value: DailyThemeInput) -> DailyClusterSet:
    article_ids = tuple(
        article_id for item in value.groups for article_id in item.group.article_version_ids
    )
    return DailyClusterSet(
        day=value.day,
        algorithm="test",
        threshold=0.72,
        embedding_model="test",
        article_version_ids=article_ids,
        relevance_version_ids=tuple(f"{index + 10000:064x}" for index in range(len(article_ids))),
        embedding_version_ids=tuple(f"{index + 20000:064x}" for index in range(len(article_ids))),
        merges=(),
        groups=tuple(item.group for item in value.groups),
    )


def _theme_judgment(
    *,
    judgment_id: str,
    snapshot: ReportEvaluationSnapshot,
    feedback_index: int,
    pair: tuple[int, int],
) -> DailyThemeJudgment:
    return DailyThemeJudgment(
        judgment_id=judgment_id,
        feedback_ids=(UUID(f"00000000-0000-4000-8000-{feedback_index:012d}"),),
        report_version_id=snapshot.report.version_id,
        left_group_id=snapshot.report_group_ids[pair[0]],
        right_group_id=snapshot.report_group_ids[pair[1]],
        expected_same_theme="differ" not in judgment_id,
        rationale="Reviewed daily theme relationship.",
    )


def _numbered_reference(label: str, index: int) -> ArtifactReference:
    digest = f"{index:064x}"
    return ArtifactReference(
        artifact_id=f"news:{label}",
        version_id=digest,
        content_digest=digest,
        r2_key=f"news/{label}.json",
    )
