from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall
from romanian_news.articles.models import ExtractedArticle
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup


class GroupAnalysisReferenceInput(NewsModel):
    day: date
    cluster_set: ArtifactReference
    group: NewsGroup
    articles: tuple[ArtifactReference, ...]
    summary_needed: bool
    sentiment_needed: bool


class GroupAnalysisInput(NewsModel):
    day: date
    cluster_set: ArtifactReference
    group: NewsGroup
    articles: tuple[tuple[ArtifactReference, ExtractedArticle], ...]
    summary_needed: bool
    sentiment_needed: bool


class GroupSummary(NewsModel):
    title_ro: Annotated[str, Field(min_length=1, max_length=100)]
    summary_ro: Annotated[str, Field(min_length=1, max_length=500)]
    key_points_ro: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=240)], ...],
        Field(min_length=1, max_length=3),
    ]
    disagreements_ro: tuple[Annotated[str, Field(min_length=1)], ...]
    uncertainty_ro: Annotated[str, Field(min_length=1)] | None = None
    cited_article_version_ids: tuple[Sha256, ...]

    @field_validator("title_ro", mode="before")
    @classmethod
    def strip_news_labels(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return re.sub(
            r"^(?:(?:BREAKING|LIVE|VIDEO|UPDATE)(?=\s|:|-)(?:\s*[:-]\s*|\s+))+",
            "",
            value,
            flags=re.I,
        )


class SentimentAssessment(NewsModel):
    label: Literal["negative", "neutral", "positive", "mixed"]
    score: Annotated[float, Field(ge=-1, le=1)]
    confidence: Annotated[float, Field(ge=0, le=1)]
    rationale_ro: Annotated[str, Field(min_length=1)]


class ArticleSentimentResponse(SentimentAssessment):
    evidence_quote: Annotated[str, Field(min_length=1)]


class ArticleSentiment(SentimentAssessment):
    article_version_id: Sha256
    evidence_quote: Annotated[str, Field(min_length=1)]


class GroupSentiment(NewsModel):
    overall: SentimentAssessment
    articles: tuple[ArticleSentiment, ...]


class GroupSummaryOutput(NewsModel):
    request_id: Sha256
    cluster_set: ArtifactReference
    group_id: Sha256
    articles: tuple[ArtifactReference, ...]
    summary: GroupSummary
    call: ModelCall
    content: bytes


class GroupSentimentOutput(NewsModel):
    request_id: Sha256
    cluster_set: ArtifactReference
    group_id: Sha256
    articles: tuple[ArtifactReference, ...]
    sentiment: GroupSentiment
    call: ModelCall
    content: bytes


class GroupAnalysisOutput(NewsModel):
    summary: GroupSummaryOutput | None
    sentiment: GroupSentimentOutput | None
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_output(self) -> GroupAnalysisOutput:
        if self.summary is None and self.sentiment is None and not self.errors:
            raise ValueError("A group analysis must produce output or an explicit error")
        return self
