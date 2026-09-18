from __future__ import annotations

from datetime import date

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference, existing_current_artifact_ids
from romanian_news.analysis.groups.models import GroupAnalysisInput, GroupAnalysisReferenceInput
from romanian_news.analysis.groups.sentiment import sentiment_request_id
from romanian_news.analysis.groups.summary import summary_request_id
from romanian_news.articles.models import ExtractedArticle
from romanian_news.groups import parse_daily_cluster_set
from romanian_news.storage import read_verified_r2_object


def read_pending_group_analysis_references(
    days: set[date] | None = None,
) -> tuple[GroupAnalysisReferenceInput, ...]:
    """Read pending group analysis references without article content."""
    from romanian_news.catalog.cluster_inputs import read_current_cluster_references

    cluster_references = tuple(
        ArtifactReference.model_validate(value.model_dump(), strict=True)
        for value in read_current_cluster_references()
    )
    candidates = []
    requested_ids = []
    for cluster_reference in cluster_references:
        cluster_set = parse_daily_cluster_set(
            read_verified_r2_object(cluster_reference.r2_key, cluster_reference.content_digest)
        )
        if days is not None and cluster_set.day not in days:
            continue
        article_references = _read_article_references(cluster_set.article_version_ids)
        for group in cluster_set.groups:
            articles = tuple(article_references[value] for value in group.article_version_ids)
            summary_id = f"news:summary:{summary_request_id(group)}"
            sentiment_id = f"news:sentiment:{sentiment_request_id(group)}"
            requested_ids.extend((summary_id, sentiment_id))
            candidates.append(
                (cluster_set.day, cluster_reference, group, articles, summary_id, sentiment_id)
            )
    existing = existing_current_artifact_ids(tuple(requested_ids))
    return tuple(
        GroupAnalysisReferenceInput(
            day=day,
            cluster_set=cluster_reference,
            group=group,
            articles=articles,
            summary_needed=summary_id not in existing,
            sentiment_needed=sentiment_id not in existing,
        )
        for day, cluster_reference, group, articles, summary_id, sentiment_id in candidates
        if summary_id not in existing or sentiment_id not in existing
    )


def load_group_analysis_input(value: GroupAnalysisReferenceInput) -> GroupAnalysisInput:
    """Load exact article content for one claimed group analysis job."""
    return GroupAnalysisInput(
        **value.model_dump(exclude={"articles"}),
        articles=tuple((reference, _read_article(reference)) for reference in value.articles),
    )


def read_pending_group_analyses(
    days: set[date] | None = None,
) -> tuple[GroupAnalysisInput, ...]:
    """Read complete pending inputs for the legacy monolithic runner."""
    return tuple(
        load_group_analysis_input(value) for value in read_pending_group_analysis_references(days)
    )


def _read_article_references(version_ids: tuple[Sha256, ...]) -> dict[Sha256, ArtifactReference]:
    from romanian_news.catalog.artifacts import artifact_references_by_version_ids

    catalog_result = artifact_references_by_version_ids(version_ids)
    result = {
        version_id: ArtifactReference.model_validate(value.model_dump(), strict=True)
        for version_id, value in catalog_result.items()
    }
    if set(result) != set(version_ids):
        raise ValueError("Cluster set references unknown article versions")
    return result


def _read_article(reference: ArtifactReference) -> ExtractedArticle:
    return ExtractedArticle.model_validate_json(
        read_verified_r2_object(reference.r2_key, reference.content_digest), strict=True
    )


def _read_articles(
    version_ids: tuple[Sha256, ...],
) -> dict[Sha256, tuple[ArtifactReference, ExtractedArticle]]:
    """Read exact article content for the legacy monolithic runner."""
    return {
        version_id: (reference, _read_article(reference))
        for version_id, reference in _read_article_references(version_ids).items()
    }
