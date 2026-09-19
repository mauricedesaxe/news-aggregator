from __future__ import annotations

from functools import cache
from itertools import groupby
from typing import Annotated

from langfuse import Langfuse
from langfuse.api import ScoreDataType
from pydantic import Field

from romanian_news import NewsModel, Sha256
from romanian_news.catalog import feedback as feedback_catalog
from romanian_news.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)


class NewsFeedbackSyncResult(NewsModel):
    selected_feedback: Annotated[int, Field(ge=0)] = 0
    matched_attempts: Annotated[int, Field(ge=0)] = 0
    completed: Annotated[int, Field(ge=0)] = 0
    failed: Annotated[int, Field(ge=0)] = 0
    unresolved: Annotated[int, Field(ge=0)] = 0


_TRACE_PROVIDER = "langfuse"


def sync_news_feedback(
    manifest_artifact_version_id: Sha256,
    limit: int = 25,
) -> NewsFeedbackSyncResult:
    """Project curated feedback scores for one immutable evaluation manifest."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return NewsFeedbackSyncResult()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Feedback sync limit must be a positive integer")

    routes = feedback_catalog.read_pending_feedback_score_routes(manifest_artifact_version_id)
    selected_feedback = matched_attempts = completed = failed = unresolved = 0
    for _feedback_id, grouped_routes in groupby(routes, key=lambda route: route.feedback_id):
        if selected_feedback == limit:
            break
        selected_feedback += 1
        for route in grouped_routes:
            matched_attempts += 1
            if route.provider == "langsmith":
                feedback_catalog.record_provider_migration_disposition(route.score_id)
                unresolved += 1
                continue
            if (
                route.provider != _TRACE_PROVIDER
                or route.trace_id is None
                or route.observation_id is None
            ):
                unresolved += 1
                continue
            try:
                _create_langfuse_score(route)
            except Exception as error:
                feedback_catalog.record_feedback_sync_attempt(
                    route.score_id, status="failed", error=error
                )
                failed += 1
            else:
                feedback_catalog.record_feedback_sync_attempt(
                    route.score_id, status="completed", error=None
                )
                completed += 1
    return NewsFeedbackSyncResult(
        selected_feedback=selected_feedback,
        matched_attempts=matched_attempts,
        completed=completed,
        failed=failed,
        unresolved=unresolved,
    )


def _create_langfuse_score(route: feedback_catalog.CuratedScoreRoute) -> None:
    response = _langfuse_client().api.scores.create(
        trace_id=route.trace_id,
        observation_id=route.observation_id,
        id=route.score_id,
        name=f"news_reader_{route.concern}_feedback",
        value=1 if route.polarity == "positive" else 0,
        data_type=ScoreDataType.NUMERIC,
        comment=route.note,
        metadata={
            "manifest_artifact_version_id": route.manifest_artifact_version_id,
            "manifest_version": route.manifest_version,
            "feedback_id": route.feedback_id,
            "reader_concern": route.concern,
            "report_version_id": route.report_version_id,
            "model_output_version_id": route.model_output_version_id,
            "model_attempt_id": route.model_attempt_id,
            "curation_rationale": route.rationale,
        },
    )
    if response.id != route.score_id:
        raise ValueError("Langfuse returned a conflicting score identity")


@cache
def _langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        base_url=LANGFUSE_BASE_URL,
    )
