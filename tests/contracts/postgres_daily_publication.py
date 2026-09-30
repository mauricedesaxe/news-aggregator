from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ClientError

from romanian_news import storage
from romanian_news import themes as construction_module
from romanian_news.analysis import corrected_structured
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import clusters, themes
from romanian_news.catalog.artifacts import current_artifact_file
from romanian_news.catalog.reports import publish_daily_report
from romanian_news.groups import DailyClusterOutput, DailyClusterSet, NewsGroup
from romanian_news.identity import canonical_json, sha256
from romanian_news.reports import (
    DailyReport,
    DailyReportInput,
    DailyReportOutput,
    DailyReportSection,
    ReportArticle,
    ReportEvent,
    ReportSubjectCitation,
    daily_report_request_id,
    report_run_id,
)
from romanian_news.themes import (
    DailyThemeInput,
    DailyThemeOutput,
    SparseThemeConstruction,
    ThemeGroupInput,
    construct_daily_themes,
    daily_theme_run_id,
)
from tests.postgres_catalog import TEST_POSTGRES_DSN, PostgresCatalog, postgres_catalog_fixture

pytestmark = pytest.mark.skipif(
    TEST_POSTGRES_DSN is None,
    reason="NEWS_TEST_POSTGRES_DSN is required",
)

postgres_catalog = postgres_catalog_fixture("daily_publication")

FIXTURE = (
    Path(__file__).parents[2] / "src" / "romanian_news" / "tests" / "fixtures"
) / "synthetic_v1_theme_set.json"
CAPTURED_AT = "2026-09-01T06:00:00+00:00"


class FakeR2Client:
    """In-memory stand-in for the R2 S3 client at the storage network boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]) -> None:
        self.objects[Key] = Body


@pytest.fixture
def fake_r2(monkeypatch: pytest.MonkeyPatch) -> FakeR2Client:
    client = FakeR2Client()
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    return client


def test_daily_theme_publication_persists_model_evidence_and_ordered_run_inputs(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = DailyThemeOutput.model_validate_json(FIXTURE.read_text())
    for reference in (output.theme_set.cluster_set, *output.theme_set.summary_inputs):
        _seed_input(postgres_catalog, reference)

    publication = themes.publish_daily_themes(output, "git:contract")

    theme_set = output.theme_set
    assert publication.status == "published"
    assert publication.uploaded_objects == 1
    assert publication.run_id == daily_theme_run_id(theme_set.request_id, "recorded-response")
    assert _run_input_rows(postgres_catalog, publication.run_id) == [
        ("cluster_set", theme_set.cluster_set.version_id),
        *[("summary", summary.version_id) for summary in theme_set.summary_inputs],
    ]
    run = _run_row(postgres_catalog, publication.run_id)
    assert run["operation_key"] == "news.construct_daily_themes"
    assert run["status"] == "completed"
    assert run["parameters_json"]["day"] == theme_set.day.isoformat()
    assert _model_call_row(postgres_catalog, publication.reference.version_id) == pytest.approx(
        {
            "model": "recorded/model",
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": 0.0,
            "response_count": 1,
        }
    )

    assert _head_row(postgres_catalog, f"news:themes:{theme_set.day.isoformat()}") == (
        publication.reference.version_id,
        publication.run_id,
    )
    assert _output_rows(postgres_catalog, publication.run_id) == [
        (publication.reference.version_id, "output")
    ]
    assert themes.read_daily_theme_reference(theme_set.day) == publication.reference
    assert (
        sha256(fake_r2.objects[publication.reference.r2_key])
        == publication.reference.content_digest
    )

    two_stage = _sparse_output(monkeypatch, merged=True, day=date(2026, 9, 2))
    for reference in (two_stage.theme_set.cluster_set, *two_stage.theme_set.summary_inputs):
        _seed_input(postgres_catalog, reference)
    construction = two_stage.theme_set.construction
    assert isinstance(construction, SparseThemeConstruction)
    assert construction.merged_prose is not None

    two_stage_publication = themes.publish_daily_themes(two_stage, "git:contract")

    assert two_stage_publication.status == "published"
    assert two_stage_publication.run_id == daily_theme_run_id(
        two_stage.theme_set.request_id,
        (construction.assignment.call.response_id, construction.merged_prose.call.response_id),
    )
    assert _run_input_rows(postgres_catalog, two_stage_publication.run_id) == [
        ("cluster_set", two_stage.theme_set.cluster_set.version_id),
        *[("summary", summary.version_id) for summary in two_stage.theme_set.summary_inputs],
    ]
    assert _model_call_row(
        postgres_catalog, two_stage_publication.reference.version_id
    ) == pytest.approx(
        {
            "model": "google/gemini-3.8-flash",
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.02,
            "response_count": 2,
        }
    )
    assert (
        themes.read_daily_theme_reference(two_stage.theme_set.day)
        == two_stage_publication.reference
    )

    one_stage = _sparse_output(monkeypatch, merged=False, day=date(2026, 9, 3))
    for reference in (one_stage.theme_set.cluster_set, *one_stage.theme_set.summary_inputs):
        _seed_input(postgres_catalog, reference)
    one_stage_construction = one_stage.theme_set.construction
    assert isinstance(one_stage_construction, SparseThemeConstruction)

    one_stage_publication = themes.publish_daily_themes(one_stage, "git:contract")

    assert one_stage_publication.run_id == daily_theme_run_id(
        one_stage.theme_set.request_id,
        (one_stage_construction.assignment.call.response_id,),
    )
    assert _model_call_row(
        postgres_catalog, one_stage_publication.reference.version_id
    ) == pytest.approx(
        {
            "model": "google/gemini-3.8-flash",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.01,
            "response_count": 1,
        }
    )


def test_cluster_publication_persists_ordered_inputs_in_one_batch(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> None:
    articles = tuple(
        SimpleNamespace(
            article=_reference(f"article-{index}", str(index + 1)),
            relevance=_reference(f"relevance-{index}", str(index + 3)),
            embedding=_reference(f"embedding-{index}", str(index + 5)),
        )
        for index in range(2)
    )
    for article in articles:
        for reference in (article.article, article.relevance, article.embedding):
            _seed_input(postgres_catalog, reference)
    content = b'{"clusters":[]}'
    output = DailyClusterOutput.model_construct(
        request_id="a" * 64,
        content_digest=sha256(content),
        cluster_set=DailyClusterSet.model_construct(
            day=date(2026, 9, 6),
            algorithm="average-link-cosine-v1",
            embedding_model="model",
            threshold=0.72,
        ),
        articles=articles,
        content=content,
    )

    publication = clusters.publish_daily_clusters(output, "git:contract")

    assert publication.uploaded_objects == 1
    assert output.content_digest == sha256(fake_r2.objects[next(iter(fake_r2.objects))])
    assert _run_input_rows(postgres_catalog, output.request_id) == [
        (role, reference.version_id)
        for article in articles
        for role, reference in (
            ("article", article.article),
            ("relevance", article.relevance),
            ("embedding", article.embedding),
        )
    ]


def test_daily_report_publication_records_assessment_lineage_rows(
    postgres_catalog: PostgresCatalog,
    fake_r2: FakeR2Client,
) -> None:
    day = date(2026, 9, 4)
    themes_reference = _reference("themes:2026-09-04", "9")
    assessments_reference = _reference("subject-assessments:2026-09-04", "8")
    cluster_reference = _reference("clusters:2026-09-04", "7")
    summary_reference = _reference("summary:group-a", "6")
    sentiment_reference = _reference("sentiment:group-a", "5")
    inputs = DailyReportInput(
        day=day,
        themes=themes_reference,
        assessments=assessments_reference,
        cluster_set=cluster_reference,
        summaries=(summary_reference,),
        sentiments=(sentiment_reference,),
    )
    article_version = "1" * 64
    report = DailyReport(
        day=day,
        accepted_article_count=1,
        theme_count=1,
        group_count=1,
        sections=(
            DailyReportSection(
                theme_id="4" * 64,
                title="Budget policy",
                summary="The draft budget dominates the day.",
                tier="main",
                semantic_rank=1,
                consequence_rationale="Budget decisions with direct national effects.",
                citations=(
                    ReportSubjectCitation(
                        article_version_id=article_version,
                        evidence_quote="Guvernul a publicat proiectul de buget.",
                    ),
                ),
                events=(
                    ReportEvent(
                        group_id="c" * 64,
                        title_ro="Dezbaterea bugetară",
                        summary_ro="Guvernul a publicat proiectul de buget pentru anul viitor.",
                        key_points_ro=("Deficitul estimat rămâne problema centrală.",),
                        disagreements_ro=(),
                        sentiment_label="mixed",
                        sentiment_score=-0.1,
                        sentiment_rationale_ro="Tonul combină prudența cu așteptări moderate.",
                        articles=(
                            ReportArticle(
                                article_version_id=article_version,
                                outlet_id="presa-exemplu",
                                title="Analiza proiectului de buget",
                                canonical_url="https://example.com/analiza",
                                sentiment_label="neutral",
                                sentiment_score=0.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    content = canonical_json(report.model_dump(mode="json"))
    output = DailyReportOutput(
        request_id=daily_report_request_id(inputs),
        report=report,
        themes=themes_reference,
        assessments=assessments_reference,
        cluster_set=cluster_reference,
        summaries=(summary_reference,),
        sentiments=(sentiment_reference,),
        content_digest=sha256(content),
        content=content,
    )
    for reference in (
        themes_reference,
        assessments_reference,
        cluster_reference,
        summary_reference,
        sentiment_reference,
    ):
        _seed_input(postgres_catalog, reference)

    publication = publish_daily_report(output, "git:contract")

    assert publication.status == "published"
    assert publication.uploaded_objects == 1
    assert publication.run_id == report_run_id(output.request_id, "git:contract")
    assert _run_input_rows(postgres_catalog, publication.run_id) == [
        ("themes", themes_reference.version_id),
        ("assessments", assessments_reference.version_id),
        ("cluster_set", cluster_reference.version_id),
        ("summary", summary_reference.version_id),
        ("sentiment", sentiment_reference.version_id),
    ]
    run = _run_row(postgres_catalog, publication.run_id)
    assert run["operation_key"] == "news.publish_daily"
    assert run["status"] == "completed"
    assert run["parameters_json"] == {"day": day.isoformat()}
    assert _head_row(postgres_catalog, f"news:daily:{day.isoformat()}") == (
        publication.version_id,
        publication.run_id,
    )
    assert _output_rows(postgres_catalog, publication.run_id) == [
        (publication.version_id, "output")
    ]
    model_call_count = postgres_catalog.execute(
        "SELECT count(*) AS count FROM news_model_calls"
    ).fetchone()
    assert model_call_count is not None
    assert not model_call_count["count"]
    current = current_artifact_file(f"news:daily:{day.isoformat()}")
    assert current is not None
    assert current.version_id == publication.version_id
    assert current.content_digest == output.content_digest
    assert sha256(fake_r2.objects[current.r2_key]) == output.content_digest


def _seed_input(catalog: PostgresCatalog, reference: ArtifactReference) -> None:
    catalog.execute(
        "INSERT INTO artifacts (id, kind, title, authority_class, lifecycle_state, "
        "visibility, created_at) VALUES (%s, 'news_input', 'Seeded input', 'derived', "
        "'current', 'private', %s) ON CONFLICT DO NOTHING",
        (reference.artifact_id, CAPTURED_AT),
    )
    catalog.execute(
        "INSERT INTO artifact_versions (id, artifact_id, schema_version, content_digest, "
        "created_at) VALUES (%s, %s, 1, %s, %s) ON CONFLICT DO NOTHING",
        (reference.version_id, reference.artifact_id, reference.content_digest, CAPTURED_AT),
    )
    catalog.execute(
        "UPDATE artifacts SET current_version_id = %s WHERE id = %s",
        (reference.version_id, reference.artifact_id),
    )


def _run_input_rows(catalog: PostgresCatalog, run_id: str) -> list[tuple[str, str]]:
    rows = catalog.execute(
        "SELECT role, artifact_version_id FROM run_inputs WHERE run_id = %s ORDER BY position",
        (run_id,),
    ).fetchall()
    return [(str(row["role"]), str(row["artifact_version_id"])) for row in rows]


def _run_row(catalog: PostgresCatalog, run_id: str) -> dict[str, Any]:
    row = catalog.execute(
        "SELECT operation_key, status, parameters_json FROM runs WHERE id = %s", (run_id,)
    ).fetchone()
    assert row is not None
    return row


def _model_call_row(catalog: PostgresCatalog, version_id: str) -> dict[str, object]:
    row = catalog.execute(
        "SELECT model, input_tokens, output_tokens, cost_usd, response_count "
        "FROM news_model_calls WHERE artifact_version_id = %s",
        (version_id,),
    ).fetchone()
    assert row is not None
    return row


def _head_row(catalog: PostgresCatalog, artifact_id: str) -> tuple[str, str]:
    row = catalog.execute(
        "SELECT current_version_id, current_run_id FROM artifacts WHERE id = %s", (artifact_id,)
    ).fetchone()
    assert row is not None
    return (str(row["current_version_id"]), str(row["current_run_id"]))


def _output_rows(catalog: PostgresCatalog, run_id: str) -> list[tuple[str, str]]:
    rows = catalog.execute(
        "SELECT artifact_version_id, role FROM run_outputs WHERE run_id = %s", (run_id,)
    ).fetchall()
    return [(str(row["artifact_version_id"]), str(row["role"])) for row in rows]


def _sparse_output(monkeypatch: pytest.MonkeyPatch, *, merged: bool, day: date) -> DailyThemeOutput:
    groups = (
        NewsGroup(id="a" * 64, article_version_ids=("1" * 64,)),
        NewsGroup(id="b" * 64, article_version_ids=("2" * 64,)),
    )
    value = DailyThemeInput(
        day=day,
        cluster_set=_reference("clusters", "3"),
        groups=tuple(
            ThemeGroupInput(
                group=group,
                summary=_reference(f"summary-{index}", str(index + 4)),
                value=GroupSummary(
                    title_ro=f"Event {index}",
                    summary_ro=f"Prima {index}. A doua. A treia.",
                    key_points_ro=("Punct",),
                    disagreements_ro=(),
                    cited_article_version_ids=group.article_version_ids,
                ),
            )
            for index, group in enumerate(groups)
        ),
    )
    labels = (1, 1) if merged else (1, 2)
    responses = [
        _response(
            {"assignments": {f"g{index}": label for index, label in enumerate(labels, start=1)}},
            "assignment",
        )
    ]
    if merged:
        responses.append(
            _response(
                {"themes": {"theme_01": {"title": "Tema", "summary": "Rezumat."}}},
                "prose",
            )
        )
    response_iterator = iter(responses)
    monkeypatch.setattr(construction_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(corrected_structured.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: next(response_iterator))
            )
        ),
    )
    monkeypatch.setattr(
        corrected_structured,
        "record_model_attempt",
        lambda response, **_kwargs: SimpleNamespace(
            attempt_id=hashlib.sha256(response.id.encode()).hexdigest(),
            response_id=response.id,
        ),
    )
    return construct_daily_themes(value)


def _reference(name: str, character: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=character * 64,
        content_digest="0" * 64,
        r2_key=f"news/{name}.json",
    )


def _response(payload: dict[str, object], response_id: str):
    content = json.dumps(payload)
    provider_payload = {
        "id": response_id,
        "model": "google/gemini-3.8-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
        "choices": [{"message": {"content": content}}],
    }
    return SimpleNamespace(
        id=response_id,
        model="google/gemini-3.8-flash",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=(SimpleNamespace(message=SimpleNamespace(content=content)),),
        model_dump=lambda *, mode: provider_payload,
    )
