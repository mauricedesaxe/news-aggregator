from datetime import date
from types import SimpleNamespace

from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.relevance import RelevanceOutput
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import analysis, clusters, reports
from romanian_news.catalog_transport import advance_artifact_current_version_from_run_statement
from romanian_news.groups import DailyClusterOutput, DailyClusterSet
from romanian_news.reports import (
    DailyReport,
    DailyReportOutput,
    WeeklyReport,
    WeeklyReportOutput,
    report_run_id,
)

DAY = date(2026, 9, 5)
IMPLEMENTATION_REF = "git:test"


def test_existing_cluster_run_rechecks_guarded_head_advance(monkeypatch) -> None:
    output = DailyClusterOutput.model_construct(
        request_id="1" * 64,
        content_digest="2" * 64,
        cluster_set=DailyClusterSet.model_construct(day=DAY),
        articles=(),
        content=b"clusters",
    )
    batches = []
    monkeypatch.setattr(clusters, "run_exists", lambda _run_id: True)
    monkeypatch.setattr(clusters, "catalog_batch", batches.append)

    publication = clusters.publish_daily_clusters(output, IMPLEMENTATION_REF)

    assert batches == [
        [
            advance_artifact_current_version_from_run_statement(
                f"news:clusters:{DAY.isoformat()}", publication.version_id, publication.run_id
            )
        ]
    ]


def test_new_cluster_run_inserts_all_inputs_in_one_statement(monkeypatch) -> None:
    articles = tuple(
        SimpleNamespace(
            article=_reference(f"article-{index}", str(index + 1)),
            relevance=_reference(f"relevance-{index}", str(index + 3)),
            embedding=_reference(f"embedding-{index}", str(index + 5)),
        )
        for index in range(2)
    )
    output = DailyClusterOutput.model_construct(
        request_id="1" * 64,
        content_digest="2" * 64,
        cluster_set=DailyClusterSet.model_construct(
            day=DAY, algorithm="average-link-cosine-v1", embedding_model="model", threshold=0.72
        ),
        articles=articles,
        content=b"clusters",
    )
    batches = []
    monkeypatch.setattr(clusters, "run_exists", lambda _run_id: False)
    monkeypatch.setattr(
        clusters,
        "publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(clusters, "catalog_batch", batches.append)

    clusters.publish_daily_clusters(output, IMPLEMENTATION_REF)

    input_statements = [
        statement
        for statement, _parameters in batches[0]
        if statement.startswith("INSERT INTO run_inputs")
    ]
    assert len(input_statements) == 1


def test_existing_daily_report_run_rechecks_guarded_head_advance(monkeypatch) -> None:
    output = DailyReportOutput.model_construct(
        request_id="3" * 64,
        report=DailyReport.model_construct(day=DAY),
        content_digest="4" * 64,
        content=b"daily",
    )
    batches = []
    monkeypatch.setattr(reports, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(reports, "run_status", lambda _run_id: "completed")
    monkeypatch.setattr(reports, "catalog_batch", batches.append)

    publication = reports.publish_daily_report(output, IMPLEMENTATION_REF)

    assert publication.run_id == report_run_id(output.request_id, IMPLEMENTATION_REF)
    assert batches == [
        [
            advance_artifact_current_version_from_run_statement(
                f"news:daily:{DAY.isoformat()}", publication.version_id, publication.run_id
            )
        ]
    ]


def test_existing_weekly_report_run_rechecks_guarded_head_advance(monkeypatch) -> None:
    output = WeeklyReportOutput.model_construct(
        request_id="5" * 64,
        report=WeeklyReport.model_construct(week_start=DAY),
        content_digest="6" * 64,
        content=b"weekly",
    )
    batches = []
    monkeypatch.setattr(reports, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(reports, "run_status", lambda _run_id: "completed")
    monkeypatch.setattr(reports, "catalog_batch", batches.append)

    publication = reports.publish_weekly_report(output, IMPLEMENTATION_REF)

    assert publication.run_id == report_run_id(output.request_id, IMPLEMENTATION_REF)
    assert batches == [
        [
            advance_artifact_current_version_from_run_statement(
                f"news:weekly:{DAY.isoformat()}", publication.version_id, publication.run_id
            )
        ]
    ]


def test_existing_analysis_run_rechecks_guarded_head_advance(monkeypatch) -> None:
    output = RelevanceOutput.model_construct(
        request_id="7" * 64,
        article=_reference("article", "8"),
        call=ModelCall(
            response_id="response-1",
            model="model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        content=b"analysis",
    )
    run_id = analysis._analysis_run_id(output)
    batches = []
    monkeypatch.setattr(
        analysis,
        "publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=0, reused_objects=1),
    )
    monkeypatch.setattr(analysis, "existing_run_ids", lambda _run_ids: frozenset({run_id}))
    monkeypatch.setattr(analysis, "catalog_batch", batches.append)

    publication = analysis.publish_relevance_outputs((output,), IMPLEMENTATION_REF)
    file = analysis._analysis_file(output, run_id)

    assert publication.run_ids == (run_id,)
    assert batches == [
        [
            advance_artifact_current_version_from_run_statement(
                file.artifact_id, file.version_id, run_id
            )
        ]
    ]


def _reference(name: str, digest_character: str) -> ArtifactReference:
    digest = digest_character * 64
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=digest,
        content_digest=digest,
        r2_key=f"news/{name}.json",
    )
