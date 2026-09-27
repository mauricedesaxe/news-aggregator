from __future__ import annotations

from datetime import date, datetime, time, timedelta

from pydantic import model_validator

from romanian_news import BUCHAREST, NewsModel
from romanian_news.analysis.embeddings import embedding_request_id
from romanian_news.analysis.groups.sentiment import sentiment_request_id
from romanian_news.analysis.groups.summary import summary_request_id
from romanian_news.analysis.relevance_v3 import production_relevance_v3_request_id
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import parse_daily_cluster_set
from romanian_news.storage import read_verified_r2_object


class DailyArtifactReferences(NewsModel):
    day: date
    values: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def validate_order(self) -> DailyArtifactReferences:
        order = tuple((value.artifact_id, value.version_id) for value in self.values)
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ValueError("Daily artifact references must be ordered and unique")
        return self


class ExactInputCheck(NewsModel):
    expected_version_ids: tuple[str, ...]
    recorded_version_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.expected_version_ids == self.recorded_version_ids


def bucharest_day_window(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=BUCHAREST)
    return start, datetime.combine(day + timedelta(days=1), time.min, tzinfo=BUCHAREST)


def read_daily_feed_observation_references(day: date) -> DailyArtifactReferences:
    from romanian_news.catalog.daily import (
        read_daily_feed_observation_references as read_references,
    )

    return DailyArtifactReferences(day=day, values=read_references(day))


def read_daily_article_references(day: date) -> DailyArtifactReferences:
    from romanian_news.catalog.daily import read_daily_article_references as read_references

    return DailyArtifactReferences(day=day, values=read_references(day))


def read_daily_relevance_references(day: date) -> DailyArtifactReferences:
    articles = read_daily_article_references(day).values
    return DailyArtifactReferences(
        day=day,
        values=_current_references(
            tuple(
                f"news:relevance:{production_relevance_v3_request_id(value)}" for value in articles
            )
        ),
    )


def read_daily_embedding_references(day: date) -> DailyArtifactReferences:
    from romanian_news.catalog.daily import relevance_is_accepted

    relevance = {value.artifact_id: value for value in read_daily_relevance_references(day).values}
    articles = read_daily_article_references(day).values
    artifact_ids = []
    for article in articles:
        relevance_id = f"news:relevance:{production_relevance_v3_request_id(article)}"
        reference = relevance.get(relevance_id)
        if reference is not None and relevance_is_accepted(reference.version_id):
            artifact_ids.append(f"news:embedding:{embedding_request_id(article, reference)}")
    return DailyArtifactReferences(day=day, values=_current_references(tuple(artifact_ids)))


def read_daily_cluster_reference(day: date) -> ArtifactReference:
    return _required_current_reference(f"news:clusters:{day.isoformat()}")


def read_daily_group_summary_references(day: date) -> DailyArtifactReferences:
    return _read_daily_group_references(day, "summary")


def read_daily_group_sentiment_references(day: date) -> DailyArtifactReferences:
    return _read_daily_group_references(day, "sentiment")


def read_daily_report_reference(day: date) -> ArtifactReference:
    return _required_current_reference(f"news:daily:{day.isoformat()}")


def read_weekly_report_reference(week_start: date) -> ArtifactReference:
    return _required_current_reference(f"news:weekly:{week_start.isoformat()}")


def check_current_artifact_inputs(
    artifact_id: str,
    expected: tuple[ArtifactReference, ...],
) -> ExactInputCheck:
    from romanian_news.catalog.daily import read_current_artifact_input_version_ids

    return ExactInputCheck(
        expected_version_ids=tuple(value.version_id for value in expected),
        recorded_version_ids=read_current_artifact_input_version_ids(artifact_id),
    )


def _read_daily_group_references(day: date, kind: str) -> DailyArtifactReferences:
    cluster_reference = read_daily_cluster_reference(day)
    cluster_set = parse_daily_cluster_set(
        read_verified_r2_object(cluster_reference.r2_key, cluster_reference.content_digest)
    )
    request_id = summary_request_id if kind == "summary" else sentiment_request_id
    artifact_ids = tuple(f"news:{kind}:{request_id(group)}" for group in cluster_set.groups)
    return DailyArtifactReferences(day=day, values=_current_references(artifact_ids))


def _required_current_reference(artifact_id: str) -> ArtifactReference:
    values = _current_references((artifact_id,))
    if len(values) != 1:
        raise ValueError(f"No current artifact exists: {artifact_id}")
    return values[0]


def _current_references(artifact_ids: tuple[str, ...]) -> tuple[ArtifactReference, ...]:
    from romanian_news.catalog.artifacts import current_artifact_references

    return current_artifact_references(artifact_ids)
