import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from romanian_news import reports as reports_module
from romanian_news import themes as themes_module
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.groups.models import (
    ArticleSentiment,
    GroupSentiment,
    GroupSummary,
    SentimentAssessment,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import DailyClusterSet, NewsGroup
from romanian_news.reports import (
    ArchivedDailyReport,
    ArchivedDailyReportSection,
    DailyReport,
    DailyReportConstruction,
    DailyReportInput,
    DailyReportOutput,
    ReportArticleSource,
    ReportEvent,
    RetrospectiveCoverage,
    RetrospectiveDailyReport,
    WeeklyReport,
    build_daily_report_from_construction,
    build_retrospective_daily_report,
    build_weekly_report,
    daily_report_request_id,
    parse_daily_report,
)
from romanian_news.subject_assessments import (
    PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
    DailySubjectAssessmentInput,
    DailySubjectAssessmentSet,
    ModelSubjectAssessmentConstruction,
    SubjectAssessment,
    SubjectAssessmentEvidence,
    subject_assessment_policy_digest,
    subject_assessment_request_id,
)
from romanian_news.themes import (
    LEGACY_THEME_POLICY,
    DailyTheme,
    DailyThemeInput,
    DailyThemeSet,
    ModelThemeConstruction,
    ThemeGroupInput,
    ThemeModelAttemptEvidence,
    ThemeModelMessage,
    daily_theme_id,
    daily_theme_request_id,
    theme_policy_digest,
)


@pytest.fixture(autouse=True)
def _isolate_input_cache(monkeypatch):
    """Keep the module-level input cache out of staleness unit tests."""
    monkeypatch.setattr(
        "romanian_news.catalog.artifacts.current_artifact_references",
        lambda _artifact_ids: (),
    )
    reports_module.clear_daily_report_input_cache()
    yield
    reports_module.clear_daily_report_input_cache()


def test_weekly_report_uses_seven_exact_daily_versions(monkeypatch) -> None:
    week_start = date(2026, 8, 24)
    daily = {
        week_start + timedelta(days=offset): (
            _reference(str(offset) * 64),
            DailyReport(
                day=week_start + timedelta(days=offset),
                accepted_article_count=offset + 1,
                theme_count=0,
                group_count=1,
                sections=(),
            ),
        )
        for offset in range(7)
    }
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)

    output = build_weekly_report(week_start)
    replay = build_weekly_report(week_start)

    assert output.content == replay.content
    assert output.request_id == replay.request_id
    assert output.report.week_end == date(2026, 8, 30)
    assert output.report.accepted_article_count == 28
    assert output.report.group_count == 7
    assert tuple(day.daily_report_version_id for day in output.report.days) == tuple(
        reference.version_id for reference, _report in daily.values()
    )


def test_provisional_daily_report_changes_identity_and_survives_weekly_round_trip(
    monkeypatch,
) -> None:
    specs = (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),)
    output = _build_ranked_report(
        monkeypatch,
        specs,
        coverage_status="provisional",
    )
    complete = _build_ranked_report(monkeypatch, specs, coverage_status="complete")
    assert output.report.coverage_status == "provisional"
    assert parse_daily_report(output.content).coverage_status == "provisional"
    assert output.request_id != complete.request_id
    assert output.content_digest != complete.content_digest
    assert output.request_id != daily_report_request_id(
        DailyReportInput(
            day=output.report.day,
            themes=output.themes,
            assessments=output.assessments,
            cluster_set=output.cluster_set,
            summaries=output.summaries,
            sentiments=output.sentiments,
        ),
        "complete",
    )

    week_start = output.report.day - timedelta(days=output.report.day.weekday())
    daily = {
        week_start + timedelta(days=offset): (
            _reference(str(offset) * 64),
            output.report
            if week_start + timedelta(days=offset) == output.report.day
            else DailyReport(
                day=week_start + timedelta(days=offset),
                accepted_article_count=0,
                theme_count=0,
                group_count=0,
                coverage_status="complete",
                sections=(),
            ),
        )
        for offset in range(7)
    }
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)
    weekly = build_weekly_report(week_start)
    assert weekly.report.coverage_status == "provisional"
    assert WeeklyReport.model_validate_json(weekly.content, strict=True).coverage_status == (
        "provisional"
    )


def test_old_daily_report_parses_with_unknown_coverage() -> None:
    legacy = DailyReport(
        day=date(2026, 8, 24),
        accepted_article_count=0,
        theme_count=0,
        group_count=0,
        sections=(),
    ).model_dump(mode="json", exclude={"coverage_status"})
    assert parse_daily_report(json.dumps(legacy).encode()).coverage_status == "unknown"


def test_retrospective_report_requires_capture_evidence_and_survives_weekly_round_trip(
    monkeypatch,
) -> None:
    week_start = date(2025, 9, 15)
    coverage = RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 10, 5, tzinfo=UTC),
        included_outlets=("hotnews", "digi24"),
        discovered_url_count=200,
        verified_page_count=80,
        captured_article_count=40,
        coverage_note="Only two publisher archives were included.",
    )
    historical = RetrospectiveDailyReport(
        day=week_start,
        accepted_article_count=25,
        theme_count=0,
        group_count=0,
        sections=(),
        retrospective=coverage,
    )
    assert parse_daily_report(historical.model_dump_json().encode()) == historical
    with pytest.raises(ValueError):
        RetrospectiveDailyReport.model_validate(
            {key: value for key, value in historical.model_dump().items() if key != "retrospective"}
        )

    with pytest.raises(ValueError, match="schema version 4"):
        RetrospectiveDailyReport.model_validate({**historical.model_dump(), "schema_version": 3})
    with pytest.raises(ValueError, match="schema version 3"):
        DailyReport(
            schema_version=4,
            day=week_start,
            accepted_article_count=0,
            theme_count=0,
            group_count=0,
            sections=(),
        )
    daily = {
        week_start + timedelta(days=offset): (
            _reference(str(offset) * 64),
            historical
            if offset == 0
            else DailyReport(
                day=week_start + timedelta(days=offset),
                accepted_article_count=1,
                theme_count=0,
                group_count=0,
                sections=(),
            ),
        )
        for offset in range(7)
    }
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)

    weekly = build_weekly_report(week_start)
    round_trip = WeeklyReport.model_validate_json(weekly.content, strict=True)
    assert isinstance(round_trip.days[0].report, RetrospectiveDailyReport)
    assert round_trip.days[0].report.retrospective == coverage


def test_retrospective_output_changes_with_coverage_and_keeps_exact_inputs() -> None:
    day = date(2025, 9, 19)
    live = DailyReport(
        day=day,
        accepted_article_count=2,
        theme_count=0,
        group_count=0,
        sections=(),
    )
    base = DailyReportOutput(
        request_id="a" * 64,
        report=live,
        themes=_reference("1" * 64),
        assessments=_reference("2" * 64),
        cluster_set=_reference("3" * 64),
        summaries=(),
        sentiments=(),
        content_digest="b" * 64,
        content=live.model_dump_json().encode(),
    )
    coverage = RetrospectiveCoverage(
        capture_started_at=datetime(2026, 9, 27, 10, tzinfo=UTC),
        capture_ended_at=datetime(2026, 9, 27, 11, tzinfo=UTC),
        included_outlets=("hotnews", "digi24"),
        discovered_url_count=20,
        verified_page_count=5,
        captured_article_count=3,
        coverage_note="Two outlets have verified captures.",
    )

    first = build_retrospective_daily_report(base, coverage)
    replay = build_retrospective_daily_report(base, coverage)
    revised = build_retrospective_daily_report(
        base, coverage.model_copy(update={"captured_article_count": 4})
    )

    assert first == replay
    assert first.request_id != base.request_id
    assert first.request_id != revised.request_id
    assert first.content_digest != revised.content_digest
    assert parse_daily_report(first.content) == first.report
    assert first.themes == base.themes
    assert first.assessments == base.assessments
    assert first.cluster_set == base.cluster_set
    with pytest.raises(ValueError, match="Accepted articles exceed retrospective captures"):
        build_retrospective_daily_report(
            base, coverage.model_copy(update={"captured_article_count": 1})
        )


def test_weekly_report_reconstructs_a_mixed_current_and_archived_week(monkeypatch) -> None:
    week_start = date(2026, 8, 24)
    daily = {}
    for offset in range(7):
        day = week_start + timedelta(days=offset)
        report = (
            ArchivedDailyReport(
                day=day,
                accepted_article_count=2,
                group_count=1,
                sections=(),
            )
            if offset == 0
            else DailyReport(
                day=day,
                accepted_article_count=1,
                theme_count=0,
                group_count=1,
                sections=(),
            )
        )
        daily[day] = (_reference(f"{offset + 1:064x}"), report)
    monkeypatch.setattr("romanian_news.reports._read_daily_report", daily.__getitem__)

    output = build_weekly_report(week_start)

    assert isinstance(output.report.days[0].report, ArchivedDailyReport)
    assert isinstance(output.report.days[1].report, DailyReport)
    assert output.report.accepted_article_count == 8


def test_daily_report_section_accepts_no_material_uncertainty() -> None:
    section = ReportEvent.model_validate(
        {
            "group_id": "a" * 64,
            "title_ro": "Title",
            "summary_ro": "Summary",
            "key_points_ro": (),
            "disagreements_ro": (),
            "uncertainty_ro": None,
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_rationale_ro": "Neutral account",
            "articles": (),
        }
    )

    assert section.uncertainty_ro is None


def test_weekly_report_requires_a_monday_and_all_seven_days(monkeypatch) -> None:
    with pytest.raises(ValueError, match="Monday"):
        build_weekly_report(date(2026, 8, 25))

    monkeypatch.setattr(
        "romanian_news.reports._read_daily_report",
        lambda day: (
            _reference("a" * 64),
            DailyReport(
                day=day, accepted_article_count=0, theme_count=0, group_count=0, sections=()
            ),
        )
        if day < date(2026, 8, 27)
        else (_ for _ in ()).throw(ValueError("missing daily report")),
    )

    with pytest.raises(ValueError, match="missing daily report"):
        build_weekly_report(date(2026, 8, 24))


def _reference(version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:daily:{version_id[0]}",
        version_id=version_id,
        content_digest="f" * 64,
        r2_key=f"news/reports/{version_id}.json",
    )


def test_daily_report_uses_assessment_order_independent_of_theme_order(
    monkeypatch,
) -> None:
    specs = (
        ("a", "Local traffic restriction", 8, "narrow", "strong", "strong"),
        ("b", "Routine national policy", 1, "routine", "strong", "none"),
        ("f", "Romania loses PNRR funds", 1, "major", "strong", "strong"),
    )

    first = _build_ranked_report(monkeypatch, specs)
    second = _build_ranked_report(monkeypatch, tuple(reversed(specs)))

    expected = [
        "Romania loses PNRR funds",
        "Routine national policy",
        "Local traffic restriction",
    ]
    assert [section.title for section in _fresh_report(first).sections] == expected
    assert [section.title for section in _fresh_report(second).sections] == expected


def test_daily_report_uses_each_assessment_rank_exactly(
    monkeypatch,
) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (
            ("c", "One strong dimension with many articles", 12, "routine", "strong", "none"),
            ("b", "Both strong dimensions with one article", 1, "routine", "strong", "strong"),
            (
                "f",
                "One major member in a small group",
                2,
                ("routine", "major"),
                ("strong", "strong"),
                ("none", "none"),
            ),
            ("a", "One strong dimension with one article", 1, "routine", "strong", "none"),
        ),
    )

    assert [section.title for section in _fresh_report(output).sections] == [
        "One major member in a small group",
        "Both strong dimensions with one article",
        "One strong dimension with many articles",
        "One strong dimension with one article",
    ]


def test_daily_report_keeps_two_event_boundaries_inside_one_theme(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (
            ("a", "Coalition talks", 1, "major", "strong", "strong"),
            ("b", "Prime minister search", 1, "major", "strong", "strong"),
        ),
        combine_themes=True,
    )

    report = _fresh_report(output)
    assert report.theme_count == 1
    assert report.group_count == 2
    assert [event.group_id for event in report.sections[0].events] == [
        "a" * 64,
        "b" * 64,
    ]
    assert [
        tuple(article.article_version_id for article in event.articles)
        for event in report.sections[0].events
    ] == [(f"{1:064x}",), (f"{2:064x}",)]


def test_daily_report_preserves_prefixed_source_titles(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),),
    )

    assert _fresh_report(output).sections[0].events[0].articles[0].title == (
        "BREAKING: LIVE: Source title " + "0" * 63 + "1"
    )


def test_daily_report_records_assessment_in_output_and_request_identity(monkeypatch) -> None:
    output = _build_ranked_report(
        monkeypatch,
        (("a", "Romania loses PNRR funds", 1, "major", "strong", "strong"),),
    )
    without_assessment = reports_module._sha256(
        reports_module._canonical_json(
            {
                "theme_version_id": output.themes.version_id,
                "cluster_set_version_id": output.cluster_set.version_id,
                "operation": "news.publish_daily",
                "report_schema": DailyReport.model_json_schema(),
                "sentiment_version_ids": [value.version_id for value in output.sentiments],
                "summary_version_ids": [value.version_id for value in output.summaries],
            }
        )
    )

    assert output.assessments.artifact_id == "news:subject-assessments"
    assert output.request_id != without_assessment


def test_archived_daily_report_section_keeps_permissive_generated_text() -> None:
    section = ArchivedDailyReportSection.model_validate(
        {
            "group_id": "a" * 64,
            "title_ro": "x" * 101,
            "summary_ro": "x" * 501,
            "key_points_ro": (),
            "disagreements_ro": ("x" * 300,),
            "uncertainty_ro": None,
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "sentiment_rationale_ro": "Archived rationale",
            "articles": (),
        }
    )

    assert len(section.title_ro) == 101
    assert len(section.summary_ro) == 501
    assert section.key_points_ro == ()


def test_recorded_daily_report_input_reads_immutable_run_lineage(monkeypatch) -> None:
    day = date(2026, 9, 3)
    roles = ("themes", "assessments", "cluster_set", "summary", "sentiment")
    rows = [
        {
            "parameters_json": {"day": day.isoformat()},
            "position": position,
            "role": role,
            "artifact_id": f"artifact:{role}",
            "version_id": str(position + 1) * 64,
            "content_digest": "a" * 64,
            "r2_key": f"objects/{role}.json",
        }
        for position, role in enumerate(roles)
    ]
    monkeypatch.setattr("romanian_news.catalog.report_inputs.catalog_query", lambda *_args: rows)

    result = reports_module.read_recorded_daily_report_input(day)

    assert result.themes.artifact_id == "artifact:themes"
    assert result.assessments.artifact_id == "artifact:assessments"
    assert result.cluster_set.artifact_id == "artifact:cluster_set"
    assert tuple(value.artifact_id for value in result.summaries) == ("artifact:summary",)
    assert tuple(value.artifact_id for value in result.sentiments) == ("artifact:sentiment",)


def _daily_input(day: date) -> DailyReportInput:
    return DailyReportInput(
        day=day,
        themes=_reference("9" * 64),
        assessments=_reference("8" * 64),
        cluster_set=_reference(f"{day.day:064x}"),
        summaries=(),
        sentiments=(),
    )


def _fresh_report(output: DailyReportOutput) -> DailyReport:
    report = output.report
    if not isinstance(report, DailyReport):
        raise AssertionError("Expected a freshly constructed daily report")
    return report


def _ranked_report_members(specs):
    groups = []
    article_ids = []
    article_metadata = {}
    group_ranks = {}
    summary_references = []
    summary_values = {}
    sentiment_references = []
    sentiment_values = {}
    themes = []
    next_article = 1
    for index, (
        group_character,
        title,
        article_count,
        magnitude,
        _political,
        _economic,
    ) in enumerate(specs):
        member_ids = tuple(
            f"{position:064x}" for position in range(next_article, next_article + article_count)
        )
        next_article += article_count
        group = NewsGroup(id=group_character * 64, article_version_ids=member_ids)
        groups.append(group)
        article_ids.extend(member_ids)
        magnitude_values = magnitude if isinstance(magnitude, tuple) else (magnitude,)
        group_ranks[group.id] = -max(
            {"narrow": 0, "routine": 1, "major": 2}[item] for item in magnitude_values
        )
        for _member_index, article_id in enumerate(member_ids):
            article_metadata[article_id] = ReportArticleSource(
                outlet_id="test",
                title=f"BREAKING: LIVE: Source title {article_id}",
                canonical_url=f"https://example.test/{article_id}",
            )
        summary_reference = _analysis_reference(f"summary-{index}", f"{index + 10:064x}")
        summary_references.append(summary_reference)
        summary_values[summary_reference.artifact_id] = GroupSummary(
            title_ro=title,
            summary_ro=f"Summary for {title}.",
            key_points_ro=(f"Key point for {title}.",),
            disagreements_ro=(),
            uncertainty_ro=None,
            cited_article_version_ids=member_ids,
        )
        sentiment_reference = _analysis_reference(f"sentiment-{index}", f"{index + 20:064x}")
        sentiment_references.append(sentiment_reference)
        sentiment_values[sentiment_reference.artifact_id] = GroupSentiment(
            overall=SentimentAssessment(
                label="neutral", score=0.0, confidence=0.9, rationale_ro="Neutral"
            ),
            articles=tuple(
                ArticleSentiment(
                    article_version_id=article_id,
                    label="neutral",
                    score=0.0,
                    confidence=0.9,
                    rationale_ro="Neutral",
                    evidence_quote="Evidence",
                )
                for article_id in member_ids
            ),
        )
        themes.append(
            DailyTheme(
                id=f"{index + 40:064x}",
                title=title,
                summary=f"Theme summary for {title}.",
                group_ids=(group.id,),
                article_version_ids=member_ids,
            )
        )
    return (
        groups,
        article_ids,
        article_metadata,
        group_ranks,
        summary_references,
        summary_values,
        sentiment_references,
        sentiment_values,
        themes,
    )


def _ranked_assessment_set(
    day,
    themes,
    groups,
    summary_references,
    relevance_by_article,
    article_group,
    group_ranks,
    theme_reference,
    theme_set,
):
    summary_by_group = dict(zip((group.id for group in groups), summary_references, strict=True))
    ordered_themes = sorted(
        themes,
        key=lambda theme: (
            min(group_ranks[group_id] for group_id in theme.group_ids),
            theme.title,
        ),
    )
    assessments = tuple(
        SubjectAssessment(
            theme_id=theme.id,
            group_ids=theme.group_ids,
            article_version_ids=theme.article_version_ids,
            tier="main",
            semantic_rank=rank,
            rationale=f"Consequence rationale for {theme.title}.",
            evidence=(
                SubjectAssessmentEvidence(
                    group_id=article_group[theme.article_version_ids[0]],
                    article=_analysis_reference(
                        f"article-{theme.article_version_ids[0]}",
                        theme.article_version_ids[0],
                    ),
                    relevance=relevance_by_article[theme.article_version_ids[0]],
                    summary=summary_by_group[article_group[theme.article_version_ids[0]]],
                    evidence_quote="Evidence",
                ),
            ),
        )
        for rank, theme in enumerate(ordered_themes, start=1)
    )
    assessment_reference = _analysis_reference("subject-assessments", "8" * 64)
    assessment_set = DailySubjectAssessmentSet.model_construct(
        day=day,
        request_id=subject_assessment_request_id(
            DailySubjectAssessmentInput.model_construct(
                day=day,
                themes=theme_reference,
                theme_set=theme_set,
                summaries=tuple(
                    SimpleNamespace(reference=reference) for reference in summary_references
                ),
                relevance=tuple(relevance_by_article.values()),
            )
        ),
        policy=PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
        policy_digest=subject_assessment_policy_digest(PRODUCTION_SUBJECT_ASSESSMENT_POLICY),
        themes=theme_reference,
        summary_inputs=tuple(summary_references),
        relevance_inputs=tuple(relevance_by_article.values()),
        construction=ModelSubjectAssessmentConstruction.model_construct(),
        subject_ids=tuple(theme.id for theme in themes),
        assessments=assessments,
    )
    return assessment_reference, assessment_set


def _build_ranked_report(
    monkeypatch, specs, *, combine_themes: bool = False, coverage_status: str = "unknown"
):
    day = date(2026, 8, 31)
    (
        groups,
        article_ids,
        article_metadata,
        group_ranks,
        summary_references,
        summary_values,
        sentiment_references,
        sentiment_values,
        themes,
    ) = _ranked_report_members(specs)
    if combine_themes:
        themes = [
            DailyTheme(
                id="4" * 64,
                title="Shared theme",
                summary="The events belong to one theme.",
                group_ids=tuple(group.id for group in groups),
                article_version_ids=tuple(article_ids),
            )
        ]
    cluster_reference = _analysis_reference("clusters", "c" * 64)
    theme_reference = _analysis_reference("themes", "9" * 64)
    cluster_set = DailyClusterSet(
        day=day,
        algorithm="complete-link-cosine-v1",
        threshold=0.78,
        embedding_model="test/model",
        article_version_ids=tuple(article_ids),
        relevance_version_ids=tuple(f"{index + 100:064x}" for index in range(len(article_ids))),
        embedding_version_ids=tuple(f"{index + 200:064x}" for index in range(len(article_ids))),
        merges=(),
        groups=tuple(groups),
    )
    policy_digest = theme_policy_digest(LEGACY_THEME_POLICY)
    themes = [
        theme.model_copy(update={"id": daily_theme_id(day, theme.group_ids, policy_digest)})
        for theme in themes
    ]
    theme_input = DailyThemeInput(
        day=day,
        cluster_set=cluster_reference,
        groups=tuple(
            ThemeGroupInput(
                group=group,
                summary=summary_reference,
                value=summary_values[summary_reference.artifact_id],
            )
            for group, summary_reference in zip(groups, summary_references, strict=True)
        ),
    )
    theme_set = DailyThemeSet(
        day=day,
        request_id=daily_theme_request_id(theme_input, LEGACY_THEME_POLICY),
        policy=LEGACY_THEME_POLICY,
        policy_digest=policy_digest,
        cluster_set=cluster_reference,
        groups=tuple(groups),
        summary_inputs=tuple(summary_references),
        construction=_theme_construction(tuple(themes)),
        themes=tuple(themes),
    )
    relevance_references = tuple(
        _analysis_reference(f"relevance-{index}", version_id)
        for index, version_id in enumerate(cluster_set.relevance_version_ids)
    )
    relevance_by_article = dict(zip(article_ids, relevance_references, strict=True))
    article_group = {
        article_id: group.id for group in groups for article_id in group.article_version_ids
    }
    assessment_reference, assessment_set = _ranked_assessment_set(
        day,
        themes,
        groups,
        summary_references,
        relevance_by_article,
        article_group,
        group_ranks,
        theme_reference,
        theme_set,
    )
    inputs = DailyReportInput(
        day=day,
        themes=theme_reference,
        assessments=assessment_reference,
        cluster_set=cluster_reference,
        summaries=tuple(summary_references),
        sentiments=tuple(sentiment_references),
    )
    return build_daily_report_from_construction(
        DailyReportConstruction(
            inputs=inputs,
            theme_set=theme_set,
            assessment_set=assessment_set,
            cluster_set=cluster_set,
            articles=article_metadata,
            summaries=summary_values,
            sentiments=sentiment_values,
        ),
        coverage_status=coverage_status,
    )


def _theme_construction(themes: tuple[DailyTheme, ...]) -> ModelThemeConstruction:
    messages = (
        ThemeModelMessage(role="system", content="System"),
        ThemeModelMessage(role="user", content="Input"),
    )
    response_content = json.dumps(
        {
            "themes": [
                {
                    "title": theme.title,
                    "summary": theme.summary,
                    "group_ids": theme.group_ids,
                }
                for theme in themes
            ]
        },
        ensure_ascii=False,
    )
    return ModelThemeConstruction(
        messages=messages,
        input_digest=themes_module._sha256(
            themes_module._canonical_json([message.model_dump(mode="json") for message in messages])
        ),
        response_schema_digest=themes_module._response_schema_digest(),
        call=ModelCall(
            response_id="response",
            model="test/model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        attempts=(
            ThemeModelAttemptEvidence(
                attempt_id="6" * 64,
                response_id="response",
                status="accepted",
                error=None,
                response_content=response_content,
                response_content_digest=themes_module._sha256(response_content.encode()),
                provider_response={
                    "id": "response",
                    "choices": [{"message": {"content": response_content}}],
                },
            ),
        ),
    )


def _analysis_reference(name: str, version_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=version_id,
        content_digest="f" * 64,
        r2_key=f"news/{name}.json",
    )


def _member_value(value: object, index: int) -> object:
    return value[index] if isinstance(value, tuple) else value
