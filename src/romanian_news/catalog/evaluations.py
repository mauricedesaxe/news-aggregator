from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal, overload
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, StringConstraints, TypeAdapter

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.analysis.relevance_v3 import ContextDecision, ImpactDecision
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog.artifacts import (
    ArtifactFile,
    artifact_file,
    artifact_statements,
    canonical_json,
    current_artifact_file,
    run_output_statement,
    run_status,
    sha256,
)
from romanian_news.catalog_transport import (
    advance_artifact_current_version_from_run_statement,
    catalog_batch,
    catalog_query,
)
from romanian_news.evaluation import (
    PIN_PATH,
    ArticleRelevanceInput,
    ConfidenceEvaluationCase,
    ConfidenceEvaluationSpec,
    EvaluationProvenance,
    FrozenGeneratedSummary,
    GroupingEvaluationCase,
    GroupingEvaluationSpec,
    NewsEvaluationBaseline,
    NewsEvaluationCase,
    NewsEvaluationDataset,
    NewsEvaluationManifest,
    NewsEvaluationPin,
    NewsEvaluationSpec,
    RankingEvaluationCase,
    RankingEvaluationSpec,
    ReaderPresentationEvaluationCase,
    ReaderPresentationEvaluationSpec,
    RelevanceEvaluationCase,
    RelevanceEvaluationSpec,
    RelevanceV3EvaluationDecision,
    ReportEvaluationSnapshot,
    ReportEvaluationSpec,
    ReportGroupInputs,
    ReportGroupObservation,
    ReportInputReferences,
    SummaryFormatEvaluationCase,
    SummaryFormatEvaluationSpec,
    ThemeEvaluationDayCase,
    ThemeEvaluationDaySpec,
    Tier,
    TierEvaluationCase,
    TierEvaluationSpec,
    compare_news_evaluation_to_baseline,
    evaluate_news_dataset,
)
from romanian_news.feedback import NewsFeedbackEvent, news_feedback_event_from_row
from romanian_news.groups import (
    DailyClusterSet,
    EmbeddedArticle,
    EmbeddedArticleReference,
    NewsGroup,
)
from romanian_news.reports import (
    ArchivedDailyReport,
    DailyReport,
    DailyReportDocument,
    DailyReportSection,
    parse_daily_report,
    report_section_events,
)
from romanian_news.storage import (
    publish_immutable_r2_objects,
    read_verified_r2_object,
)
from romanian_news.subject_assessments import parse_daily_subject_assessment_set
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    DailyThemeSet,
    ReaderSubjectDailyThemeSet,
    SparseDailyThemeSet,
    load_daily_theme_input,
    parse_daily_theme_set,
)


class LoadedNewsEvaluationRelease(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    baseline_reference: ArtifactReference
    manifest: NewsEvaluationManifest
    baseline: NewsEvaluationBaseline
    dataset: NewsEvaluationDataset


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class NewsEvaluationDatasetReceipt(NewsModel):
    provider: Literal["langfuse"]
    manifest_artifact_version_id: Sha256
    projection_kind: Literal["dataset"] = "dataset"
    dataset_id: str
    dataset_name: str
    experiment_id: None = None
    experiment_name: None = None
    experiment_url: None = None
    implementation_ref: Literal[""] = ""


class NewsEvaluationExperimentReceipt(NewsModel):
    provider: Literal["langfuse"]
    manifest_artifact_version_id: Sha256
    projection_kind: Literal["experiment"] = "experiment"
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None
    implementation_ref: NonEmptyString


NewsEvaluationProjectionReceipt = Annotated[
    NewsEvaluationDatasetReceipt | NewsEvaluationExperimentReceipt,
    Field(discriminator="projection_kind"),
]
_PROJECTION_RECEIPT_ADAPTER: TypeAdapter[NewsEvaluationProjectionReceipt] = TypeAdapter(
    NewsEvaluationProjectionReceipt
)
_SHA256_ADAPTER = TypeAdapter(Sha256)


class NewsRelevanceExperimentReceipt(NewsModel):
    provider: Literal["langfuse"]
    manifest_artifact_version_id: Sha256
    policy_digest: Sha256
    implementation_ref: NonEmptyString
    policy_id: NonEmptyString
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None


class NewsThemeExperimentReceipt(NewsModel):
    provider: Literal["langfuse"]
    manifest_artifact_version_id: Sha256
    policy_digest: Sha256
    implementation_ref: NonEmptyString
    policy_id: NonEmptyString
    dataset_id: str
    dataset_name: str
    experiment_id: str
    experiment_name: str
    experiment_url: str | None


class NewsEvaluationReportLineage(NewsModel):
    report: ArtifactReference
    inputs: ReportInputReferences


def hydrate_news_evaluation_manifest(manifest: NewsEvaluationManifest) -> NewsEvaluationDataset:
    """Load exact manifest artifacts into the pure evaluator's runtime dataset."""
    hydrated_reports = tuple(_hydrate_report(spec) for spec in manifest.reports)
    snapshots = tuple(value[0] for value in hydrated_reports)
    reports = {snapshot.report.version_id: snapshot for snapshot in snapshots}
    cluster_sets = {
        snapshot.report.version_id: cluster_set for snapshot, cluster_set in hydrated_reports
    }
    cases = tuple(_hydrate_case(spec, reports, cluster_sets) for spec in manifest.cases)
    return NewsEvaluationDataset(
        version=manifest.version,
        reviewed_at=manifest.reviewed_at,
        issue_url=manifest.issue_url,
        source_feedback_ids=manifest.source_feedback_ids,
        reports=snapshots,
        cases=cases,
        archived_daily_theme_judgments=manifest.archived_daily_theme_judgments,
        excluded_feedback=manifest.excluded_feedback,
        feedback_reviews=manifest.feedback_reviews,
        unreviewed_themes=manifest.unreviewed_themes,
    )


def publish_news_evaluation_manifest(
    manifest: NewsEvaluationManifest,
    implementation_ref: str,
) -> ArtifactReference:
    """Publish one reference-only manifest and its exact feedback membership."""
    if manifest.feedback_reviews:
        _require_reviewed_feedback_matches(
            manifest,
            read_news_evaluation_feedback(manifest.source_feedback_ids),
        )
    inputs = _manifest_inputs(manifest)
    _require_catalog_references(inputs)
    content = canonical_json(manifest.model_dump(mode="json"))
    digest = sha256(content)
    file = artifact_file(
        artifact_id=f"news:evaluation-manifest:{manifest.version}",
        artifact_kind="news_evaluation_manifest",
        title=f"Romanian news evaluation manifest {manifest.version}",
        content=content,
        r2_key=f"news/evaluations/manifests/{manifest.version}/{digest}.json",
        media_type="application/json",
    )
    excluded_ids = _excluded_feedback_ids(manifest)
    feedback_statements = [
        (
            "INSERT INTO news_evaluation_feedback_sources VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                file.version_id,
                position,
                str(feedback_id),
                "excluded" if feedback_id in excluded_ids else "represented",
            ],
        )
        for position, feedback_id in enumerate(manifest.source_feedback_ids)
    ]
    score_curation_statements = _score_curation_statements(manifest, file.version_id)
    return _publish_evaluation_artifact(
        file=file,
        operation_key="news.publish_evaluation_manifest",
        implementation_ref=implementation_ref,
        parameters={"manifest_version": manifest.version},
        inputs=inputs,
        extra_statements=[*feedback_statements, *score_curation_statements],
    )


def _require_reviewed_feedback_matches(
    manifest: NewsEvaluationManifest,
    feedback: tuple[NewsFeedbackEvent, ...],
) -> None:
    actual = {item.feedback_id: item for item in feedback}
    expected = {item.feedback_id: item.target for item in manifest.feedback_reviews}
    if (
        len(actual) != len(feedback)
        or {feedback_id: item.target for feedback_id, item in actual.items()} != expected
    ):
        raise ValueError("Immutable feedback events do not match reviewed IDs and targets")
    for review in manifest.feedback_reviews:
        superseded = next(
            (item for item in review.concerns if item.kind == "superseded"),
            None,
        )
        if superseded is None:
            continue
        replacement = actual[superseded.superseded_by_feedback_id]
        source = actual[review.feedback_id]
        if (replacement.created_at, str(replacement.feedback_id)) <= (
            source.created_at,
            str(source.feedback_id),
        ):
            raise ValueError("Superseding feedback must be newer than the feedback it replaces")


def _excluded_feedback_ids(manifest: NewsEvaluationManifest) -> set[UUID]:
    legacy_ids = {item.feedback_id for item in manifest.excluded_feedback}
    superseded_ids = {
        review.feedback_id
        for review in manifest.feedback_reviews
        if review.concerns[0].kind == "superseded"
    }
    return legacy_ids | superseded_ids


def _score_curation_statements(
    manifest: NewsEvaluationManifest,
    manifest_artifact_version_id: Sha256,
) -> list[tuple[str, list[object]]]:
    statements = []
    for position, decision in enumerate(manifest.score_curation):
        if decision.decision == "projected":
            score_id = str(
                uuid5(
                    NAMESPACE_URL,
                    "chartly:news-feedback-score:"
                    f"{manifest_artifact_version_id}:{decision.feedback_id}:"
                    f"{decision.concern}:{decision.model_attempt_id}",
                )
            )
            model_output_version_id = decision.model_output.version_id
            model_attempt_id = decision.model_attempt_id
            polarity = decision.polarity
            concern = decision.concern
            exclusion_reason = None
        else:
            score_id = None
            model_output_version_id = None
            model_attempt_id = None
            polarity = decision.polarity
            concern = decision.concern
            exclusion_reason = decision.reason
        statements.append(
            (
                "INSERT INTO news_feedback_score_curation "
                "(manifest_artifact_version_id, manifest_version, position, feedback_id, "
                "concern, exclusion_reason, decision, report_version_id, "
                "model_output_version_id, model_attempt_id, polarity, rationale, score_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    manifest_artifact_version_id,
                    manifest.version,
                    position,
                    str(decision.feedback_id),
                    concern,
                    exclusion_reason,
                    decision.decision,
                    decision.report_version_id,
                    model_output_version_id,
                    model_attempt_id,
                    polarity,
                    decision.rationale,
                    score_id,
                ],
            )
        )
    return statements


def publish_news_evaluation_baseline(
    baseline: NewsEvaluationBaseline,
    implementation_ref: str,
) -> ArtifactReference:
    """Publish one baseline bound to its exact manifest and evaluator."""
    if _catalog_reference(baseline.manifest.version_id) != baseline.manifest:
        raise ValueError("Evaluation baseline manifest reference does not match the catalog")
    content = canonical_json(baseline.model_dump(mode="json"))
    digest = sha256(content)

    file = artifact_file(
        artifact_id=f"news:evaluation-baseline:{baseline.dataset_version}",
        artifact_kind="news_evaluation_baseline",
        title=f"Romanian news evaluation baseline {baseline.dataset_version}",
        content=content,
        r2_key=f"news/evaluations/baselines/{baseline.dataset_version}/{digest}.json",
        media_type="application/json",
    )
    return _publish_evaluation_artifact(
        file=file,
        operation_key="news.publish_evaluation_baseline",
        implementation_ref=implementation_ref,
        parameters={"dataset_version": baseline.dataset_version},
        inputs=(baseline.manifest,),
    )


def load_news_evaluation_release(pin_content: bytes | str) -> LoadedNewsEvaluationRelease:
    """Resolve a compact pin and verify its exact manifest and baseline."""
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    manifest_reference = _catalog_reference(pin.manifest_version_id)
    baseline_reference = _catalog_reference(pin.baseline_version_id)
    manifest = NewsEvaluationManifest.model_validate_json(
        read_verified_r2_object(
            manifest_reference.r2_key,
            manifest_reference.content_digest,
        ),
        strict=True,
    )
    baseline = NewsEvaluationBaseline.model_validate_json(
        read_verified_r2_object(
            baseline_reference.r2_key,
            baseline_reference.content_digest,
        ),
        strict=True,
    )
    if baseline.manifest != manifest_reference:
        raise ValueError("Evaluation baseline does not bind the pinned manifest")
    if baseline.dataset_version != manifest.version:
        raise ValueError("Evaluation baseline and manifest versions differ")
    return LoadedNewsEvaluationRelease(
        pin=pin,
        manifest_reference=manifest_reference,
        baseline_reference=baseline_reference,
        manifest=manifest,
        baseline=baseline,
        dataset=hydrate_news_evaluation_manifest(manifest),
    )


@overload
def read_news_evaluation_projection_receipt(
    provider: Literal["langfuse"],
    manifest_version_id: Sha256,
    projection_kind: Literal["dataset"],
    implementation_ref: Literal[""],
) -> NewsEvaluationDatasetReceipt | None: ...


@overload
def read_news_evaluation_projection_receipt(
    provider: Literal["langfuse"],
    manifest_version_id: Sha256,
    projection_kind: Literal["experiment"],
    implementation_ref: str,
) -> NewsEvaluationExperimentReceipt | None: ...


def read_news_evaluation_projection_receipt(
    provider: Literal["langfuse"],
    manifest_version_id: Sha256,
    projection_kind: Literal["dataset", "experiment"],
    implementation_ref: str,
) -> NewsEvaluationProjectionReceipt | None:
    rows = catalog_query(
        """SELECT provider, manifest_artifact_version_id, projection_kind,
                  dataset_id, dataset_name, experiment_id, experiment_name,
                  experiment_url, implementation_ref
           FROM news_evaluation_projections
           WHERE provider = %s
             AND manifest_artifact_version_id = %s
             AND projection_kind = %s
             AND implementation_ref = %s""",
        [provider, manifest_version_id, projection_kind, implementation_ref],
    )
    if len(rows) > 1:
        raise ValueError("Evaluation projection receipt identity is not unique")
    return _PROJECTION_RECEIPT_ADAPTER.validate_python(rows[0], strict=False) if rows else None


def record_news_evaluation_projection_receipt(
    receipt: NewsEvaluationProjectionReceipt,
) -> None:
    catalog_batch(
        [
            (
                "INSERT INTO news_evaluation_projections "
                "(provider, manifest_artifact_version_id, projection_kind, dataset_id, "
                "dataset_name, experiment_id, experiment_name, experiment_url, "
                "implementation_ref, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    receipt.provider,
                    receipt.manifest_artifact_version_id,
                    receipt.projection_kind,
                    receipt.dataset_id,
                    receipt.dataset_name,
                    receipt.experiment_id,
                    receipt.experiment_name,
                    receipt.experiment_url,
                    receipt.implementation_ref,
                    datetime.now(UTC).isoformat(),
                ],
            )
        ]
    )


def read_news_relevance_experiment_receipt(
    provider: Literal["langfuse"],
    manifest_version_id: Sha256,
    policy_digest: Sha256,
    implementation_ref: str,
) -> NewsRelevanceExperimentReceipt | None:
    rows = catalog_query(
        """SELECT provider, manifest_artifact_version_id, policy_digest,
                  implementation_ref, policy_id, dataset_id, dataset_name,
                  experiment_id, experiment_name, experiment_url
           FROM news_relevance_experiments
           WHERE provider = %s
             AND manifest_artifact_version_id = %s
             AND policy_digest = %s
             AND implementation_ref = %s""",
        [provider, manifest_version_id, policy_digest, implementation_ref],
    )
    if len(rows) > 1:
        raise ValueError("Relevance experiment receipt identity is not unique")
    return NewsRelevanceExperimentReceipt.model_validate(rows[0], strict=False) if rows else None


def record_news_relevance_experiment_receipt(
    receipt: NewsRelevanceExperimentReceipt,
) -> None:
    catalog_batch(
        [
            (
                "INSERT INTO news_relevance_experiments "
                "(provider, manifest_artifact_version_id, policy_digest, implementation_ref, "
                "policy_id, dataset_id, dataset_name, experiment_id, experiment_name, "
                "experiment_url, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    receipt.provider,
                    receipt.manifest_artifact_version_id,
                    receipt.policy_digest,
                    receipt.implementation_ref,
                    receipt.policy_id,
                    receipt.dataset_id,
                    receipt.dataset_name,
                    receipt.experiment_id,
                    receipt.experiment_name,
                    receipt.experiment_url,
                    datetime.now(UTC).isoformat(),
                ],
            )
        ]
    )


def read_news_theme_experiment_receipt(
    provider: Literal["langfuse"],
    manifest_version_id: Sha256,
    policy_digest: Sha256,
    implementation_ref: str,
) -> NewsThemeExperimentReceipt | None:
    rows = catalog_query(
        """SELECT provider, manifest_artifact_version_id, policy_digest,
                  implementation_ref, policy_id, dataset_id, dataset_name,
                  experiment_id, experiment_name, experiment_url
           FROM news_theme_experiments
           WHERE provider = %s
             AND manifest_artifact_version_id = %s
             AND policy_digest = %s
             AND implementation_ref = %s""",
        [provider, manifest_version_id, policy_digest, implementation_ref],
    )
    if len(rows) > 1:
        raise ValueError("Theme experiment receipt identity is not unique")
    return NewsThemeExperimentReceipt.model_validate(rows[0], strict=False) if rows else None


def record_news_theme_experiment_receipt(receipt: NewsThemeExperimentReceipt) -> None:
    catalog_batch(
        [
            (
                "INSERT INTO news_theme_experiments "
                "(provider, manifest_artifact_version_id, policy_digest, implementation_ref, "
                "policy_id, dataset_id, dataset_name, experiment_id, experiment_name, "
                "experiment_url, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [
                    receipt.provider,
                    receipt.manifest_artifact_version_id,
                    receipt.policy_digest,
                    receipt.implementation_ref,
                    receipt.policy_id,
                    receipt.dataset_id,
                    receipt.dataset_name,
                    receipt.experiment_id,
                    receipt.experiment_name,
                    receipt.experiment_url,
                    datetime.now(UTC).isoformat(),
                ],
            )
        ]
    )


def read_news_evaluation_feedback(
    feedback_ids: tuple[UUID, ...],
) -> tuple[NewsFeedbackEvent, ...]:
    if not feedback_ids:
        return ()
    placeholders = ", ".join("%s" for _ in feedback_ids)
    rows = catalog_query(
        f"""SELECT feedback_id, report_version_id, target_kind, theme_id, group_id,
                   article_version_id, rating, note, actor, created_at
            FROM news_feedback
            WHERE feedback_id IN ({placeholders})
            ORDER BY report_version_id, created_at, feedback_id""",
        [str(feedback_id) for feedback_id in feedback_ids],
    )
    return tuple(news_feedback_event_from_row(row) for row in rows)


def read_news_model_attempt_id(request_id: Sha256, response_id: str) -> Sha256:
    """Resolve one accepted model attempt from its provider response identity."""
    rows = catalog_query(
        """SELECT attempt_id
           FROM news_model_attempts
           WHERE request_id = %s AND response_id = %s AND status = 'accepted'
           ORDER BY observed_at, attempt_id""",
        [request_id, response_id],
    )
    if len(rows) != 1:
        raise ValueError(
            "Exact accepted model attempt is unresolved: "
            f"request_id={request_id}, response_id={response_id}, matches={len(rows)}"
        )
    return _SHA256_ADAPTER.validate_python(rows[0]["attempt_id"], strict=True)


def read_news_evaluation_report_lineage(
    report_version_id: Sha256,
) -> NewsEvaluationReportLineage:
    rows = catalog_query(
        """SELECT version.artifact_id, version.id AS version_id,
                  version.produced_by_run_id, file.content_digest, file.r2_key
           FROM artifact_versions version
           JOIN artifacts artifact ON artifact.id = version.artifact_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE version.id = %s AND artifact.kind = 'news_daily_report'""",
        [report_version_id],
    )
    if len(rows) != 1 or rows[0]["produced_by_run_id"] is None:
        raise ValueError(f"Exact report version is unavailable: {report_version_id}")
    inputs = _report_inputs_for_run(str(rows[0]["produced_by_run_id"]))
    return NewsEvaluationReportLineage(
        report=_reference(rows[0]),
        inputs=ReportInputReferences(**{role: tuple(values) for role, values in inputs.items()}),
    )


def read_news_evaluation_artifact_references(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, ArtifactReference]:
    return _version_references(version_ids)


def _publish_evaluation_artifact(
    *,
    file: ArtifactFile,
    operation_key: str,
    implementation_ref: str,
    parameters: dict[str, str],
    inputs: tuple[ArtifactReference, ...],
    extra_statements: Sequence[tuple[str, Sequence[object]]] | None = None,
) -> ArtifactReference:
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    current = current_artifact_file(file.artifact_id)
    if current is not None and current.content_digest != file.content_digest:
        raise ValueError(f"{file.artifact_kind} already has different content")
    run_id = sha256(f"{operation_key}\0{file.version_id}\0{implementation_ref}".encode())
    status = run_status(run_id)
    if status == "completed":
        return _file_reference(file)
    if status is not None:
        raise RuntimeError(f"Evaluation publication run is incomplete: {status}")

    publish_immutable_r2_objects(((file.r2_key, file.content),))
    timestamp = datetime.now(UTC).isoformat()
    statements: list[tuple[str, Sequence[object]]] = [
        (
            "INSERT INTO runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                operation_key,
                "python",
                implementation_ref,
                canonical_json(parameters).decode(),
                "chartly",
                "publishing",
                f"{operation_key}:{run_id}",
                None,
                timestamp,
                None,
            ],
        )
    ]
    statements.extend(
        (
            "INSERT INTO run_inputs VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                run_id,
                position,
                reference.version_id,
                "evaluation_source",
                None,
                reference.content_digest,
                "whole_file",
                None,
            ],
        )
        for position, reference in enumerate(inputs)
    )
    statements.extend(artifact_statements(file, timestamp, produced_by_run_id=run_id))
    statements.extend(extra_statements or ())
    statements.append(run_output_statement(run_id, file))
    statements.append(
        advance_artifact_current_version_from_run_statement(
            file.artifact_id,
            file.version_id,
            run_id,
        )
    )
    statements.append(
        (
            "UPDATE runs SET status = 'completed', completed_at = %s "
            "WHERE id = %s AND status = 'publishing'",
            [timestamp, run_id],
        )
    )
    catalog_batch(statements)
    return _file_reference(file)


def _catalog_reference(version_id: Sha256) -> ArtifactReference:
    rows = catalog_query(
        """SELECT version.artifact_id, version.id AS version_id,
                  file.content_digest, file.r2_key
           FROM artifact_versions version
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE version.id = %s""",
        [version_id],
    )
    if len(rows) != 1:
        raise ValueError(f"Evaluation artifact version is unavailable: {version_id}")
    return _reference(rows[0])


def _require_catalog_references(references: tuple[ArtifactReference, ...]) -> None:
    expected = {reference.version_id: reference for reference in references}
    catalog = {}
    version_ids = tuple(expected)
    for offset in range(0, len(version_ids), 50):
        batch = version_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in batch)
        rows = catalog_query(
            f"""SELECT version.artifact_id, version.id AS version_id,
                       file.content_digest, file.r2_key
                FROM artifact_versions version
                JOIN artifact_files file ON file.artifact_version_id = version.id
                WHERE version.id IN ({placeholders})""",
            list(batch),
        )
        for row in rows:
            reference = _reference(row)
            existing = catalog.get(reference.version_id)
            if existing is not None and existing != reference:
                raise ValueError("Evaluation manifest input has conflicting catalog identities")
            catalog[reference.version_id] = reference
    if catalog != expected:
        raise ValueError("Evaluation manifest input does not match the catalog")


def _file_reference(file: ArtifactFile) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=file.artifact_id,
        version_id=file.version_id,
        content_digest=file.content_digest,
        r2_key=file.r2_key,
    )


def _hydrate_report(
    spec: ReportEvaluationSpec,
) -> tuple[ReportEvaluationSnapshot, DailyClusterSet]:
    report = parse_daily_report(json.dumps(_read_payload(spec.report), ensure_ascii=False).encode())
    cluster_set = DailyClusterSet.model_validate_json(
        json.dumps(_read_payload(spec.cluster_set), ensure_ascii=False), strict=True
    )
    _require_report_matches_cluster(report, cluster_set)
    inputs = _report_inputs(spec.report)
    if inputs["cluster_set"] != [spec.cluster_set]:
        raise ValueError("Manifest cluster set does not match the exact report input")
    expected_themes = [spec.themes] if spec.themes is not None else []
    if inputs["themes"] != expected_themes:
        raise ValueError("Manifest themes do not match the exact report input")
    if spec.themes is not None:
        theme_set = parse_daily_theme_set(
            json.dumps(_read_payload(spec.themes), ensure_ascii=False).encode()
        )
        _require_report_matches_themes(
            report,
            theme_set,
            spec.cluster_set,
            tuple(inputs["summary"]),
        )
    _require_report_matches_assessment(report, inputs, spec.themes)
    references = _version_references(
        (
            *cluster_set.article_version_ids,
            *cluster_set.relevance_version_ids,
            *cluster_set.embedding_version_ids,
        )
    )
    articles = tuple(
        EmbeddedArticleReference(
            article=references[article_id],
            relevance=references[relevance_id],
            embedding=references[embedding_id],
        )
        for article_id, relevance_id, embedding_id in zip(
            cluster_set.article_version_ids,
            cluster_set.relevance_version_ids,
            cluster_set.embedding_version_ids,
            strict=True,
        )
    )
    group_inputs = _report_group_inputs(
        cluster_set.groups,
        tuple(inputs["summary"]),
        tuple(inputs["sentiment"]),
    )
    events = report_section_events(report)
    snapshot = ReportEvaluationSnapshot(
        report=spec.report,
        themes=spec.themes,
        assessments=(inputs["assessments"][0] if inputs["assessments"] else None),
        cluster_set=spec.cluster_set,
        report_article_version_ids=tuple(
            article.article_version_id for event in events for article in event.articles
        ),
        report_group_ids=tuple(event.group_id for event in events),
        cluster_articles=articles,
        group_inputs=group_inputs,
        report_inputs=ReportInputReferences(
            themes=tuple(inputs["themes"]),
            assessments=tuple(inputs["assessments"]),
            cluster_set=tuple(inputs["cluster_set"]),
            relevance=tuple(inputs["relevance"]),
            summary=tuple(inputs["summary"]),
            sentiment=tuple(inputs["sentiment"]),
        ),
        group_observations=tuple(
            ReportGroupObservation(
                group_id=event.group_id,
                uncertainty_disclosed=event.uncertainty_ro is not None,
                tier=_report_group_tiers(report)[event.group_id],
            )
            for event in events
        ),
        report_theme_ids=(
            ()
            if isinstance(report, ArchivedDailyReport)
            else tuple(section.theme_id for section in report.sections for _event in section.events)
        ),
    )
    return snapshot, cluster_set


def _hydrate_case(
    spec: NewsEvaluationSpec,
    reports: dict[Sha256, ReportEvaluationSnapshot],
    cluster_sets: dict[Sha256, DailyClusterSet],
) -> NewsEvaluationCase:
    report = _case_report(spec, reports)
    match spec:
        case RelevanceEvaluationSpec():
            return _hydrate_relevance_case(spec, report)
        case SummaryFormatEvaluationSpec():
            return _hydrate_summary_case(spec, report)
        case GroupingEvaluationSpec():
            return _hydrate_grouping_case(spec, report)
        case RankingEvaluationSpec():
            return _hydrate_ranking_case(spec, report, cluster_sets)
        case TierEvaluationSpec():
            return _hydrate_tier_case(spec, report)
        case ConfidenceEvaluationSpec():
            return _hydrate_confidence_case(spec, report)
        case ReaderPresentationEvaluationSpec():
            return _hydrate_reader_presentation_case(spec, report)
        case ThemeEvaluationDaySpec():
            return _hydrate_theme_day_case(spec, report)
    raise TypeError(f"Unsupported evaluation spec: {type(spec).__name__}")


def _hydrate_theme_day_case(
    spec: ThemeEvaluationDaySpec,
    report: ReportEvaluationSnapshot | None,
) -> ThemeEvaluationDayCase:
    if report is None or report.report != spec.source_report:
        raise ValueError("Theme evaluation requires its exact frozen source report")
    value = load_daily_theme_input(spec.cluster_set, spec.summaries)
    if value.day != spec.day:
        raise ValueError("Theme evaluation inputs belong to another day")
    if spec.model_output is not None and spec.model_output != report.themes:
        raise ValueError("Theme evaluation requires the exact report theme output")
    observed_themes = (
        parse_daily_theme_set(
            json.dumps(_read_payload(spec.model_output), ensure_ascii=False).encode()
        )
        if spec.model_output is not None
        else None
    )
    return ThemeEvaluationDayCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=spec.source_report,
        ),
        source_report=spec.source_report,
        input=value,
        expectations=spec.expectations,
        report_expectation=spec.report_expectation,
        model_output=spec.model_output,
        observed_themes=observed_themes,
    )


def _hydrate_tier_case(
    spec: TierEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> TierEvaluationCase:
    group_id = spec.provenance.group_id
    if report is None or group_id is None:
        raise ValueError("Tier evaluation requires a frozen report group")
    observations = {item.group_id: item for item in report.group_observations}
    if group_id not in observations:
        raise ValueError("Tier evaluation group is absent from the frozen report")
    return TierEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report,
            group_id=group_id,
        ),
        expected_tier=spec.expected_tier,
        observed_tier=observations[group_id].tier,
    )


def _hydrate_confidence_case(
    spec: ConfidenceEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> ConfidenceEvaluationCase:
    group_id = spec.provenance.group_id
    if report is None or group_id is None:
        raise ValueError("Confidence evaluation requires a frozen report group")
    observations = {item.group_id: item for item in report.group_observations}
    observation = observations.get(group_id)
    if observation is None:
        raise ValueError("Confidence evaluation group is absent from the frozen report")
    return ConfidenceEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report,
            group_id=group_id,
        ),
        expected_sufficient=spec.expected_sufficient,
        observed_sufficient=not observation.uncertainty_disclosed,
    )


def _hydrate_relevance_case(
    spec: RelevanceEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> RelevanceEvaluationCase:
    _read_article(spec.article)
    decision = read_evaluation_relevance_observation(spec.model_output, spec.article.version_id)
    return RelevanceEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report if report is not None else None,
            group_id=spec.provenance.group_id,
            articles=(spec.article,),
            model_outputs=(spec.model_output,),
        ),
        expected_accepted=spec.expected_accepted,
        model_response=decision,
    )


def _hydrate_summary_case(
    spec: SummaryFormatEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> SummaryFormatEvaluationCase:
    for article in spec.articles:
        _read_article(article)
    payload = _read_payload(spec.model_output)
    _require_group_identity(payload, spec.provenance.group_id, "Summary")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Summary payload must contain a summary object")
    return SummaryFormatEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report if report is not None else None,
            group_id=spec.provenance.group_id,
            articles=spec.articles,
            model_outputs=(spec.model_output,),
        ),
        artifact=FrozenGeneratedSummary.model_validate_json(
            json.dumps(
                {
                    "title": summary.get("title_ro"),
                    "summary": summary.get("summary_ro"),
                    "key_points": summary.get("key_points_ro"),
                },
                ensure_ascii=False,
            ),
            strict=True,
        ),
    )


def _hydrate_grouping_case(
    spec: GroupingEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> GroupingEvaluationCase:
    source_cluster_set = DailyClusterSet.model_validate_json(
        json.dumps(_read_payload(spec.source_cluster_set), ensure_ascii=False), strict=True
    )
    if source_cluster_set.day != spec.day:
        raise ValueError("Grouping cluster set belongs to another day")
    articles = tuple(_read_embedded(article) for article in spec.articles)
    _require_grouping_inputs(source_cluster_set, spec.articles)
    return GroupingEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report if report is not None else None,
            group_id=spec.provenance.group_id,
            articles=tuple(item.article for item in spec.articles),
            model_outputs=tuple(
                reference
                for item in spec.articles
                for reference in (item.relevance, item.embedding)
            ),
        ),
        source_cluster_set=spec.source_cluster_set,
        day=spec.day,
        articles=articles,
        left_article_version_id=spec.left_article_version_id,
        right_article_version_id=spec.right_article_version_id,
        expected_same_group=spec.expected_same_group,
    )


def _hydrate_ranking_case(
    spec: RankingEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
    cluster_sets: dict[Sha256, DailyClusterSet],
) -> RankingEvaluationCase:
    report_version_id = spec.provenance.report_version_id
    if report is None or report_version_id is None:
        raise ValueError("Ranking specs require a frozen report")
    cluster_set = cluster_sets[report_version_id]
    groups = _ranking_groups(cluster_set.groups, spec.higher_group_id, spec.lower_group_id)
    required_ids = {article_id for group in groups for article_id in group.article_version_ids}
    supplied_ids = {item.article.version_id for item in spec.relevance}
    if supplied_ids != required_ids or len(supplied_ids) != len(spec.relevance):
        raise ValueError("Ranking specs must exactly cover ranked group articles")
    relevance = tuple(
        ArticleRelevanceInput(
            article_version_id=item.article.version_id,
            decision=read_evaluation_ranking_decision(item.model_output, item.article.version_id),
        )
        for item in spec.relevance
    )
    for item in spec.relevance:
        _read_article(item.article)
    return RankingEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report,
            group_id=spec.provenance.group_id,
            articles=tuple(item.article for item in spec.relevance),
            model_outputs=tuple(item.model_output for item in spec.relevance),
        ),
        groups=groups,
        relevance=relevance,
        higher_group_id=spec.higher_group_id,
        lower_group_id=spec.lower_group_id,
        observed_group_order=report.report_group_ids,
    )


def _hydrate_reader_presentation_case(
    spec: ReaderPresentationEvaluationSpec,
    report: ReportEvaluationSnapshot | None,
) -> ReaderPresentationEvaluationCase:
    return ReaderPresentationEvaluationCase(
        case_id=spec.case_id,
        control=spec.control,
        provenance=EvaluationProvenance(
            feedback_ids=spec.provenance.feedback_ids,
            report=report.report if report is not None else None,
            group_id=spec.provenance.group_id,
        ),
        expectation=spec.expectation,
    )


def _case_report(
    spec: NewsEvaluationSpec, reports: dict[Sha256, ReportEvaluationSnapshot]
) -> ReportEvaluationSnapshot | None:
    report_version_id = spec.provenance.report_version_id
    return reports[report_version_id] if report_version_id is not None else None


def _report_inputs(report: ArtifactReference) -> dict[str, list[ArtifactReference]]:
    rows = catalog_query(
        """SELECT input.position, input.role, version.artifact_id,
                  version.id AS version_id, file.content_digest, file.r2_key
           FROM artifact_versions report_version
           JOIN run_inputs input ON input.run_id = report_version.produced_by_run_id
           JOIN artifact_versions version ON version.id = input.artifact_version_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE report_version.id = %s AND input.role != 'prior_output'
           ORDER BY input.position""",
        [report.version_id],
    )
    return _report_input_references(rows)


def _report_inputs_for_run(run_id: str) -> dict[str, list[ArtifactReference]]:
    rows = catalog_query(
        """SELECT input.position, input.role, version.artifact_id,
                  version.id AS version_id, file.content_digest, file.r2_key
           FROM run_inputs input
           JOIN artifact_versions version ON version.id = input.artifact_version_id
           JOIN artifact_files file ON file.artifact_version_id = version.id
           WHERE input.run_id = %s AND input.role != 'prior_output'
           ORDER BY input.position""",
        [run_id],
    )
    return _report_input_references(rows)


def _report_input_references(
    rows: list[dict[str, object]],
) -> dict[str, list[ArtifactReference]]:
    inputs: dict[str, list[ArtifactReference]] = {
        "themes": [],
        "assessments": [],
        "cluster_set": [],
        "relevance": [],
        "summary": [],
        "sentiment": [],
    }
    for row in rows:
        role = str(row["role"])
        if role not in inputs:
            raise ValueError(f"Unexpected report input role: {role}")
        inputs[role].append(_reference(row))
    if len(inputs["cluster_set"]) != 1:
        raise ValueError("Exact report run must have one cluster-set input")
    return inputs


def _version_references(version_ids: tuple[Sha256, ...]) -> dict[Sha256, ArtifactReference]:
    references = {}
    for offset in range(0, len(version_ids), 50):
        batch = version_ids[offset : offset + 50]
        placeholders = ", ".join("%s" for _ in batch)
        rows = catalog_query(
            f"""SELECT version.artifact_id, version.id AS version_id,
                       file.content_digest, file.r2_key
                FROM artifact_versions version
                JOIN artifact_files file ON file.artifact_version_id = version.id
                WHERE version.id IN ({placeholders})""",
            list(batch),
        )
        references.update({str(row["version_id"]): _reference(row) for row in rows})
    if set(references) != set(version_ids):
        raise ValueError("Frozen report references unavailable inputs")
    return references


def _report_group_inputs(
    groups: tuple[NewsGroup, ...],
    summaries: tuple[ArtifactReference, ...],
    sentiments: tuple[ArtifactReference, ...],
) -> tuple[ReportGroupInputs, ...]:
    summary_by_group = _references_by_group("summary", summaries)
    sentiment_by_group = _references_by_group("sentiment", sentiments)
    expected_group_ids = {group.id for group in groups}
    if set(summary_by_group) != expected_group_ids or set(sentiment_by_group) != expected_group_ids:
        raise ValueError("Exact report group inputs do not cover the cluster groups")
    return tuple(
        ReportGroupInputs(
            group_id=group_id,
            summary=summary_by_group[group_id],
            sentiment=sentiment_by_group[group_id],
        )
        for group_id in sorted(expected_group_ids)
    )


def _references_by_group(
    role: str, references: tuple[ArtifactReference, ...]
) -> dict[Sha256, ArtifactReference]:
    values = {}
    for reference in references:
        payload = _read_payload(reference)
        group_id = payload.get("group_id")
        if not isinstance(group_id, str) or len(group_id) != 64:
            raise ValueError(f"Exact report has invalid {role} group identity")
        if group_id in values:
            raise ValueError(f"Exact report has duplicate {role} inputs for group {group_id}")
        values[group_id] = reference
    return values


def _read_embedded(reference: EmbeddedArticleReference) -> EmbeddedArticle:
    article = _read_article(reference.article)
    _read_relevance(reference.relevance, reference.article.version_id)
    payload = _read_payload(reference.embedding)
    if payload.get("article_version_id") != reference.article.version_id:
        raise ValueError("Embedding payload references another article version")
    if payload.get("relevance_version_id") != reference.relevance.version_id:
        raise ValueError("Embedding payload references another relevance version")
    vector = payload.get("vector")
    if not isinstance(vector, list):
        raise ValueError(f"Embedding payload has no vector: {reference.embedding.version_id}")
    return EmbeddedArticle(
        article=reference.article,
        relevance=reference.relevance,
        embedding=reference.embedding,
        value=article,
        vector=tuple(float(coordinate) for coordinate in vector),
    )


def _read_article(reference: ArtifactReference) -> ExtractedArticle:
    return ExtractedArticle.model_validate_json(
        json.dumps(_read_payload(reference), ensure_ascii=False), strict=True
    )


def _read_relevance(reference: ArtifactReference, article_version_id: Sha256) -> RelevanceDecision:
    payload = _read_payload(reference)
    if payload.get("article_version_id") != article_version_id:
        raise ValueError("Relevance payload references another article version")
    return RelevanceDecision.model_validate(payload.get("decision"), strict=True)


def read_evaluation_relevance_observation(
    reference: ArtifactReference, article_version_id: Sha256
) -> RelevanceDecision | RelevanceV3EvaluationDecision:
    payload = _read_payload(reference)
    if payload.get("article_version_id") != article_version_id:
        raise ValueError("Relevance payload references another article version")
    if "decision" in payload:
        return RelevanceDecision.model_validate(payload["decision"], strict=True)
    context = payload.get("context")
    if not isinstance(context, dict):
        raise ValueError("V3 relevance payload has no context decision")
    impact = payload.get("impact")
    if impact is not None and not isinstance(impact, dict):
        raise ValueError("V3 relevance payload has an invalid impact decision")
    return RelevanceV3EvaluationDecision(
        context=ContextDecision.model_validate(context.get("decision"), strict=True),
        impact=(
            ImpactDecision.model_validate(impact.get("decision"), strict=True)
            if impact is not None
            else None
        ),
    )


def read_evaluation_ranking_decision(
    reference: ArtifactReference, article_version_id: Sha256
) -> RelevanceDecision | ImpactDecision:
    payload = _read_payload(reference)
    if payload.get("article_version_id") != article_version_id:
        raise ValueError("Relevance payload references another article version")
    if "decision" in payload:
        return RelevanceDecision.model_validate(payload["decision"], strict=True)
    impact = payload.get("impact")
    if not isinstance(impact, dict):
        raise ValueError("Ranking relevance payload has no impact decision")
    return ImpactDecision.model_validate(impact.get("decision"), strict=True)


def _read_payload(reference: ArtifactReference) -> dict[str, object]:
    content = read_verified_r2_object(reference.r2_key, reference.content_digest)
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Evaluation artifact payload must be a JSON object")
    return payload


def _require_group_identity(
    payload: dict[str, object], group_id: Sha256 | None, label: str
) -> None:
    if group_id is None or payload.get("group_id") != group_id:
        raise ValueError(f"{label} payload references another group")


def _require_grouping_inputs(
    cluster_set: DailyClusterSet, articles: tuple[EmbeddedArticleReference, ...]
) -> None:
    expected = {
        article_id: (relevance_id, embedding_id)
        for article_id, relevance_id, embedding_id in zip(
            cluster_set.article_version_ids,
            cluster_set.relevance_version_ids,
            cluster_set.embedding_version_ids,
            strict=True,
        )
    }
    for article in articles:
        if expected.get(article.article.version_id) != (
            article.relevance.version_id,
            article.embedding.version_id,
        ):
            raise ValueError("Grouping spec does not match its source cluster set")


def _ranking_groups(
    groups: tuple[NewsGroup, ...], higher_group_id: Sha256, lower_group_id: Sha256
) -> tuple[NewsGroup, ...]:
    groups_by_id = {group.id: group for group in groups}
    if higher_group_id not in groups_by_id or lower_group_id not in groups_by_id:
        raise ValueError("Ranking spec references a group outside its report")
    return groups_by_id[higher_group_id], groups_by_id[lower_group_id]


def _require_report_matches_themes(
    report: DailyReportDocument,
    theme_set: (
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
    ),
    cluster_set: ArtifactReference,
    summaries: tuple[ArtifactReference, ...],
) -> None:
    if isinstance(report, ArchivedDailyReport):
        raise ValueError("Archived reports cannot reference themes")
    if (
        theme_set.day != report.day
        or theme_set.cluster_set != cluster_set
        or theme_set.summary_inputs != summaries
    ):
        raise ValueError("Manifest themes do not match the report inputs")
    report_themes = {section.theme_id: section for section in report.sections}
    source_themes = {theme.id: theme for theme in theme_set.themes}
    if (
        len(report_themes) != len(report.sections)
        or len(source_themes) != len(theme_set.themes)
        or set(report_themes) != set(source_themes)
    ):
        raise ValueError("Manifest themes do not match the report sections")
    for theme_id, section in report_themes.items():
        theme = source_themes[theme_id]
        if (
            section.title != theme.title
            or section.summary != theme.summary
            or set(event.group_id for event in section.events) != set(theme.group_ids)
        ):
            raise ValueError(f"Manifest report theme differs from its input: {theme_id}")


def _require_report_matches_assessment(
    report: DailyReportDocument,
    inputs: dict[str, list[ArtifactReference]],
    themes: ArtifactReference | None,
) -> None:
    if not isinstance(report, DailyReport):
        return
    if len(inputs["assessments"]) != 1 or themes is None:
        raise ValueError("Schema-v3 report must have one assessment input")
    assessment_set = parse_daily_subject_assessment_set(
        json.dumps(_read_payload(inputs["assessments"][0]), ensure_ascii=False).encode()
    )
    if assessment_set.themes != themes:
        raise ValueError("Report assessment input references another theme set")
    assessments = {item.theme_id: item for item in assessment_set.assessments}
    for section in report.sections:
        assessment = assessments.get(section.theme_id)
        if assessment is None or (
            section.tier,
            section.semantic_rank,
            section.consequence_rationale,
            tuple(
                (citation.article_version_id, citation.evidence_quote)
                for citation in section.citations
            ),
        ) != (
            assessment.tier,
            assessment.semantic_rank,
            assessment.rationale,
            tuple(
                (evidence.article.version_id, evidence.evidence_quote)
                for evidence in assessment.evidence
            ),
        ):
            raise ValueError("Schema-v3 report differs from its assessment input")


def _report_group_tiers(report: DailyReportDocument) -> dict[Sha256, Tier]:
    if isinstance(report, ArchivedDailyReport):
        return {section.group_id: "main" for section in report.sections}
    return {
        event.group_id: section.tier if isinstance(section, DailyReportSection) else "main"
        for section in report.sections
        for event in section.events
    }


def _require_report_matches_cluster(
    report: DailyReportDocument,
    cluster_set: DailyClusterSet,
) -> None:
    if isinstance(report, ArchivedDailyReport):
        events = report.sections
    else:
        events = tuple(event for section in report.sections for event in section.events)
    report_groups = {event.group_id: event for event in events}
    cluster_groups = {group.id: group for group in cluster_set.groups}
    if len(report_groups) != len(events) or len(cluster_groups) != len(cluster_set.groups):
        raise ValueError("Exact report or cluster set has duplicate groups")
    if set(report_groups) != set(cluster_groups):
        raise ValueError("Exact report groups do not match the cluster set")
    for group_id, event in report_groups.items():
        report_articles = tuple(article.article_version_id for article in event.articles)
        cluster_articles = cluster_groups[group_id].article_version_ids
        if (
            len(report_articles) != len(set(report_articles))
            or len(cluster_articles) != len(set(cluster_articles))
            or set(report_articles) != set(cluster_articles)
        ):
            raise ValueError(f"Exact report group articles do not match cluster group {group_id}")


def _reference(row: dict[str, object]) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=str(row["artifact_id"]),
        version_id=str(row["version_id"]),
        content_digest=str(row["content_digest"]),
        r2_key=str(row["r2_key"]),
    )


def _manifest_inputs(manifest: NewsEvaluationManifest) -> tuple[ArtifactReference, ...]:
    references: dict[Sha256, ArtifactReference] = {}

    def include(reference: ArtifactReference) -> None:
        existing = references.get(reference.version_id)
        if existing is not None and existing != reference:
            raise ValueError("Manifest repeats an artifact version with conflicting identity")
        references[reference.version_id] = reference

    if manifest.prior_manifest is not None:
        include(manifest.prior_manifest)
    for report in manifest.reports:
        include(report.report)
        if report.themes is not None:
            include(report.themes)
        include(report.cluster_set)
    for decision in manifest.score_curation:
        if decision.decision == "projected":
            include(decision.model_output)
    for case in manifest.cases:
        if case.concern == "relevance":
            include(case.article)
            include(case.model_output)
        elif case.concern == "summary_format":
            for reference in case.articles:
                include(reference)
            include(case.model_output)
        elif case.concern == "grouping":
            include(case.source_cluster_set)
            for article in case.articles:
                include(article.article)
                include(article.relevance)
                include(article.embedding)
        elif case.concern == "daily_theme":
            include(case.source_report)
            include(case.cluster_set)
            for reference in case.summaries:
                include(reference)
            if case.model_output is not None:
                include(case.model_output)
        elif case.concern == "ranking":
            for item in case.relevance:
                include(item.article)
                include(item.model_output)
    return tuple(references[version_id] for version_id in sorted(references))


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    release = load_news_evaluation_release(PIN_PATH.read_bytes())
    report = evaluate_news_dataset(release.dataset)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    regressions = compare_news_evaluation_to_baseline(
        report,
        release.baseline,
        release.manifest_reference,
    )

    for regression in regressions:
        print(regression, file=sys.stderr)
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
