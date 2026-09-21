import hashlib
from datetime import date
from types import SimpleNamespace

from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import subject_assessments as catalog
from romanian_news.subject_assessments import (
    PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
    DailySubjectAssessmentOutput,
    DailySubjectAssessmentSet,
    EmptySubjectAssessmentConstruction,
    subject_assessment_policy_digest,
)

DAY = date(2026, 9, 10)


def test_assessment_publication_records_exact_theme_summary_and_relevance_lineage(
    monkeypatch,
) -> None:
    themes = _reference("themes", "1")
    summaries = (_reference("summary", "2"), _reference("summary", "3"))
    relevance = (_reference("relevance", "4"), _reference("relevance", "5"))
    assessment_set = DailySubjectAssessmentSet.model_construct(
        day=DAY,
        request_id="6" * 64,
        policy=PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
        policy_digest=subject_assessment_policy_digest(PRODUCTION_SUBJECT_ASSESSMENT_POLICY),
        themes=themes,
        summary_inputs=summaries,
        relevance_inputs=relevance,
        construction=EmptySubjectAssessmentConstruction(),
        subject_ids=(),
        assessments=(),
    )
    output = DailySubjectAssessmentOutput.model_construct(
        assessment_set=assessment_set,
        content_digest=hashlib.sha256(b"assessment").hexdigest(),
        content=b"assessment",
    )
    batches = []
    monkeypatch.setattr(catalog, "read_completed_subject_assessment", lambda _run_id: None)
    monkeypatch.setattr(catalog, "run_status", lambda _run_id: None)
    monkeypatch.setattr(catalog, "current_artifact_file", lambda _artifact_id: None)
    monkeypatch.setattr(
        catalog,
        "publish_immutable_r2_objects",
        lambda _objects: SimpleNamespace(uploaded_objects=1, reused_objects=0),
    )
    monkeypatch.setattr(catalog, "catalog_batch", batches.append)

    publication = catalog.publish_daily_subject_assessments(output, "git:test")

    inputs = [parameters for sql, parameters in batches[0] if "INTO run_inputs" in sql]
    assert [(parameters[2], parameters[3]) for parameters in inputs] == [
        (themes.version_id, "themes"),
        (summaries[0].version_id, "summary"),
        (summaries[1].version_id, "summary"),
        (relevance[0].version_id, "relevance"),
        (relevance[1].version_id, "relevance"),
    ]
    assert publication.reference.content_digest == output.content_digest


def _reference(kind: str, character: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"news:{kind}:{character}",
        version_id=character * 64,
        content_digest=character * 64,
        r2_key=f"news/{kind}/{character}.json",
    )
