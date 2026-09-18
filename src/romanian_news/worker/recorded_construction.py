from __future__ import annotations

from pydantic import model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.groups import EmbeddedArticle, cluster_articles
from romanian_news.reports import (
    DailyReportConstruction,
    DailyReportOutput,
    WeeklyReportConstruction,
    WeeklyReportOutput,
    build_daily_report_from_construction,
    build_weekly_report_from_construction,
)
from romanian_news.themes import DailyThemeOutput


class RecordedConstructionOutputs(NewsModel):
    cluster_content_digest: Sha256
    daily_themes: DailyThemeOutput
    daily_report: DailyReportOutput
    weekly_report: WeeklyReportOutput


class RecordedDailyConstruction(NewsModel):
    cluster_articles: tuple[EmbeddedArticle, ...]
    cluster_threshold: float
    daily_themes: DailyThemeOutput
    daily_report: DailyReportConstruction
    weekly_report: WeeklyReportConstruction
    expected_cluster_content_digest: Sha256
    expected_daily_theme_content_digest: Sha256
    expected_daily_report_content_digest: Sha256
    expected_weekly_report_content_digest: Sha256

    @model_validator(mode="after")
    def validate_days(self) -> RecordedDailyConstruction:
        day = self.daily_report.inputs.day
        if self.daily_themes.theme_set.day != day:
            raise ValueError("The daily themes and report must use the same day")
        if self.daily_report.theme_set != self.daily_themes.theme_set:
            raise ValueError("The daily report must consume the recorded daily themes")
        if self.daily_report.cluster_set.day != day:
            raise ValueError("The cluster set and daily report must use the same day")
        if self.weekly_report.inputs.week_start != day:
            raise ValueError("The construction day must equal the recorded week start")
        if day.weekday() != 0:
            raise ValueError("The recorded construction must start on Monday")
        return self


def build_recorded_construction(
    value: RecordedDailyConstruction,
) -> RecordedConstructionOutputs:
    """Run deterministic construction over recorded immutable inputs."""
    cluster = cluster_articles(
        value.daily_report.inputs.day,
        value.cluster_articles,
        threshold=value.cluster_threshold,
    )
    daily = build_daily_report_from_construction(value.daily_report)
    weekly = build_weekly_report_from_construction(value.weekly_report)
    outputs = RecordedConstructionOutputs(
        cluster_content_digest=cluster.content_digest,
        daily_themes=value.daily_themes,
        daily_report=daily,
        weekly_report=weekly,
    )
    if outputs.cluster_content_digest != value.expected_cluster_content_digest:
        raise ValueError("Recorded cluster construction changed")
    if outputs.daily_themes.content_digest != value.expected_daily_theme_content_digest:
        raise ValueError("Recorded daily theme construction changed")
    if outputs.daily_report.content_digest != value.expected_daily_report_content_digest:
        raise ValueError("Recorded daily report construction changed")
    if outputs.weekly_report.content_digest != value.expected_weekly_report_content_digest:
        raise ValueError("Recorded weekly report construction changed")
    return outputs
