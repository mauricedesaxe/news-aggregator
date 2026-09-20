from __future__ import annotations

from datetime import date

from pydantic import model_validator

from romanian_news import NewsModel
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import (
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.articles.models import ExtractedArticle
from romanian_news.derived_binary_protocol import GROUPING_BINARY_QUESTION

GROUPING_ARTICLE_BODY_CHARACTERS = 12_000


class BinaryGroupingCase(NewsModel):
    case_id: str
    control: bool
    day: date
    left_article: ArtifactReference
    left_value: ExtractedArticle
    right_article: ArtifactReference
    right_value: ExtractedArticle
    expected_same_group: bool

    @model_validator(mode="after")
    def require_distinct_articles_from_frozen_day(self) -> BinaryGroupingCase:
        if self.left_article.version_id == self.right_article.version_id:
            raise ValueError("Binary grouping cases require two distinct articles")
        if self.left_value.bucharest_day != self.day or self.right_value.bucharest_day != self.day:
            raise ValueError("Binary grouping articles must belong to the frozen case day")
        return self


def build_grouping_binary_request(case: BinaryGroupingCase) -> BinaryRequest:
    anchors = tuple(
        sorted(
            (
                (case.left_article, case.left_value),
                (case.right_article, case.right_value),
            ),
            key=lambda item: item[0].version_id,
        )
    )
    if any(value.bucharest_day != case.day for _reference, value in anchors):
        raise ValueError("Grouping benchmark articles must belong to the frozen case day")
    state = "\n\n".join(
        _render_article(label, value)
        for label, (_reference, value) in zip(("A", "B"), anchors, strict=True)
    )
    return BinaryRequest(
        question=GROUPING_BINARY_QUESTION,
        state=state,
        state_digest=binary_state_digest(state),
    )


def _render_article(label: str, value: ExtractedArticle) -> str:
    return "\n".join(
        (
            f"Article {label}",
            f"Published: {value.published_at.isoformat()}",
            f"Outlet: {value.outlet_id}",
            f"Title: {value.title}",
            "Body:",
            value.body[:GROUPING_ARTICLE_BODY_CHARACTERS],
        )
    )
