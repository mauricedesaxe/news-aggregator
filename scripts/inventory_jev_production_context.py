#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, cast

from romanian_news import Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import (
    BinaryRequest,
    build_relevance_binary_request,
)
from romanian_news.analysis.binary_grouping import (
    BinaryGroupingCase,
    build_grouping_binary_request,
)
from romanian_news.analysis.binary_ranking import (
    RankingArticleEvidence,
    RankingGroupEvidence,
    build_ranking_binary_request,
)
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.analysis.relevance import ArticleAnalysisInput, RelevanceDecision
from romanian_news.analysis.relevance_v3 import ContextDecision, ImpactDecision
from romanian_news.articles.models import ExtractedArticle
from romanian_news.binary_confidence_evaluation import (
    CONFIDENCE_BENCHMARK,
    BinaryConfidenceCase,
    render_confidence_state,
)
from romanian_news.binary_daily_theme_evaluation import (
    DAILY_THEME_BINARY_BENCHMARK,
    render_daily_theme_pair_state,
)
from romanian_news.binary_tier_evaluation import (
    TIER_DEFINITION,
    BinaryTierCase,
    BinaryTierEventContext,
    BinaryTierState,
    BinaryTierSubjectContext,
    render_binary_tier_state,
)
from romanian_news.catalog.evaluations import (
    NewsEvaluationReportLineage,
    read_evaluation_ranking_decision,
    read_news_evaluation_artifact_references,
    read_news_evaluation_report_lineage,
)
from romanian_news.feedback import list_daily_reports, read_daily_report_version
from romanian_news.groups import DailyClusterSet, NewsGroup, parse_daily_cluster_set
from romanian_news.reports import DailyReport, DailyReportDocument
from romanian_news.storage import read_verified_r2_object
from romanian_news.themes import load_daily_theme_input, parse_daily_theme_set

Concern = Literal["relevance", "grouping", "ranking", "tier", "confidence", "daily_theme"]
CONCERNS: tuple[Concern, ...] = (
    "relevance",
    "grouping",
    "ranking",
    "tier",
    "confidence",
    "daily_theme",
)
UTILIZATION_THRESHOLDS = (50, 75, 90, 100)


class RequestRecord(TypedDict):
    concern: Concern
    day: str
    report_version_id: str
    selection_ids: list[str]
    source_version_ids: list[str]
    question_id: str
    question_digest: str
    state_digest: str
    state_characters: int


class Calibration(TypedDict):
    method: str
    source_request_count: int
    max_observed_input_tokens: int
    max_observed_state_characters: int
    max_observed_input_tokens_per_state_character: float


class Arguments(NamedTuple):
    report_limit: int
    reviewed_calibration: Path
    limits: Path
    output_json: Path
    output_markdown: Path
    output_raw_gzip: Path


def derive_calibrations(reviewed: Mapping[str, object]) -> dict[Concern, Calibration | None]:
    concerns = _list(reviewed.get("concerns"), "reviewed concerns")
    result: dict[Concern, Calibration | None] = {concern: None for concern in CONCERNS}
    for raw_concern in concerns:
        item = _mapping(raw_concern, "reviewed concern")
        concern_name = item.get("concern")
        if concern_name not in CONCERNS:
            continue
        concern = concern_name
        ratios: list[tuple[Fraction, int, int]] = []
        for raw_request in _list(item.get("requests"), f"{concern} requests"):
            request = _mapping(raw_request, f"{concern} request")
            input_tokens = _positive_int(request.get("input_tokens"), "input_tokens")
            state_characters = _positive_int(request.get("state_characters"), "state_characters")
            ratios.append(
                (Fraction(input_tokens, state_characters), input_tokens, state_characters)
            )
        if not ratios:
            continue
        ratio, input_tokens, state_characters = max(ratios, key=lambda value: value[0])
        result[concern] = {
            "method": "maximum observed input_tokens/state_characters in reviewed requests",
            "source_request_count": len(ratios),
            "max_observed_input_tokens": input_tokens,
            "max_observed_state_characters": state_characters,
            "max_observed_input_tokens_per_state_character": round(float(ratio), 12),
        }
    result["relevance"] = None
    return result


def estimate_input_tokens(characters: int, calibration: Calibration | None) -> int | None:
    if calibration is None:
        return None
    numerator = calibration["max_observed_input_tokens"]
    denominator = calibration["max_observed_state_characters"]
    return (characters * numerator + denominator - 1) // denominator


def summarize_concern(
    concern: Concern,
    requests: Sequence[RequestRecord],
    calibration: Calibration | None,
    *,
    approximate_character_limit: int,
    token_limit: int,
) -> dict[str, object]:
    ordered = sorted(requests, key=_request_sort_key)
    character_values = [request["state_characters"] for request in ordered]
    estimated_values = [
        value
        for request in ordered
        if (value := estimate_input_tokens(request["state_characters"], calibration)) is not None
    ]
    by_day: dict[str, list[RequestRecord]] = defaultdict(list)
    for request in ordered:
        by_day[request["day"]].append(request)
    return {
        "concern": concern,
        "scope": _scope(concern),
        "request_count": len(ordered),
        "calibration": calibration,
        "state_character_statistics": _statistics(character_values, approximate_character_limit),
        "estimated_input_token_statistics": (
            _statistics(estimated_values, token_limit) if calibration is not None else None
        ),
        "daily_statistics": [
            {
                "day": day,
                "request_count": len(day_requests),
                "state_character_statistics": _statistics(
                    [request["state_characters"] for request in day_requests],
                    approximate_character_limit,
                ),
                "estimated_input_token_statistics": (
                    _statistics(
                        [
                            cast(
                                int,
                                estimate_input_tokens(request["state_characters"], calibration),
                            )
                            for request in day_requests
                        ],
                        token_limit,
                    )
                    if calibration is not None
                    else None
                ),
            }
            for day, day_requests in sorted(by_day.items())
        ],
        "requests": [
            {
                **request,
                "estimated_input_tokens": estimate_input_tokens(
                    request["state_characters"], calibration
                ),
            }
            for request in ordered
        ],
    }


def build_inventory(
    report_limit: int,
    limits_path: Path,
    calibration_path: Path,
) -> dict[str, object]:
    limits_content = limits_path.read_bytes()
    calibration_content = calibration_path.read_bytes()
    limits = _mapping(cast(object, json.loads(limits_content)), "limit registry")
    documented = _mapping(limits.get("documented_limits"), "documented limits")
    token_limit = _positive_int(
        documented.get("single_question_effective_input_tokens"), "token limit"
    )
    approximate_character_limit = _positive_int(
        documented.get("approximate_english_characters_for_32000_tokens"),
        "approximate character limit",
    )
    reviewed = _mapping(cast(object, json.loads(calibration_content)), "reviewed calibration")
    calibrations = derive_calibrations(reviewed)

    captured = tuple(list_daily_reports(report_limit))
    requests: list[RequestRecord] = []
    reports: list[dict[str, object]] = []
    evidence_gaps: list[dict[str, str]] = []
    for summary in sorted(captured, key=lambda item: (item.day, item.report_version_id)):
        report = read_daily_report_version(summary.report_version_id)
        if report.day != summary.day:
            raise ValueError("Captured daily report day does not match its exact artifact")
        lineage = read_news_evaluation_report_lineage(summary.report_version_id)
        if lineage.report.version_id != summary.report_version_id:
            raise ValueError("Report lineage does not match the captured exact version")
        report_requests, report_record, report_gaps = _inventory_report(report, lineage)
        requests.extend(report_requests)
        reports.append(report_record)
        evidence_gaps.extend(report_gaps)

    by_concern = {
        concern: [request for request in requests if request["concern"] == concern]
        for concern in CONCERNS
    }
    generator_path = Path(__file__).resolve()
    return {
        "schema_version": "jev-production-context-inventory/v1",
        "generated_from": {
            "generation_method": (
                "Capture exact report version IDs once, load immutable lineage and verified objects, "
                "rebuild benchmark states in memory, and retain only identifiers and measurements."
            ),
            "report_limit": report_limit,
            "limit_registry": {
                "path": str(limits_path),
                "sha256": _sha256(limits_content),
            },
            "reviewed_calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_content),
            },
            "generator": {
                "path": str(generator_path),
                "sha256": _sha256(generator_path.read_bytes()),
            },
        },
        "limits": {
            "effective_single_question_input_tokens": token_limit,
            "approximate_english_characters": approximate_character_limit,
        },
        "reports": sorted(reports, key=lambda item: (str(item["day"]), str(item["version_id"]))),
        "concerns": [
            summarize_concern(
                concern,
                by_concern[concern],
                calibrations[concern],
                approximate_character_limit=approximate_character_limit,
                token_limit=token_limit,
            )
            for concern in CONCERNS
        ],
        "evidence_gaps": [
            {
                "concern": "relevance",
                "code": "accepted-only",
                "detail": (
                    "Relevance covers accepted cluster articles only because broader rejected day "
                    "articles cannot be pinned from report lineage."
                ),
            },
            {
                "concern": "all",
                "code": "observed-estimate-not-bound",
                "detail": (
                    "Estimated input tokens use the maximum reviewed input_tokens/state_characters "
                    "ratio. This is a conservative observed estimate, not a formal tokenizer bound."
                ),
            },
            {
                "concern": "relevance",
                "code": "no-reviewed-calibration",
                "detail": "Relevance has no reviewed character calibration, so estimates are null.",
            },
            *sorted(
                evidence_gaps,
                key=lambda item: (
                    item["concern"],
                    item["day"],
                    item["report_version_id"],
                    item["code"],
                ),
            ),
        ],
        "provider_outputs_or_labels_included": False,
        "article_or_state_text_included": False,
    }


def _inventory_report(
    report: DailyReportDocument,
    lineage: NewsEvaluationReportLineage,
) -> tuple[list[RequestRecord], dict[str, object], list[dict[str, str]]]:
    source_references = _deduplicate_references(
        (
            lineage.report,
            *lineage.inputs.themes,
            *lineage.inputs.assessments,
            *lineage.inputs.cluster_set,
            *lineage.inputs.relevance,
            *lineage.inputs.summary,
            *lineage.inputs.sentiment,
        )
    )
    verified = {
        reference.version_id: read_verified_r2_object(reference.r2_key, reference.content_digest)
        for reference in source_references
    }
    cluster_reference = lineage.inputs.cluster_set[0]
    cluster_set = parse_daily_cluster_set(verified[cluster_reference.version_id])
    if cluster_set.day != report.day:
        raise ValueError("Exact cluster set belongs to another report day")

    article_ids = tuple(sorted(cluster_set.article_version_ids))
    analysis_ids = tuple(
        sorted((*cluster_set.article_version_ids, *cluster_set.relevance_version_ids))
    )
    analysis_references = (
        read_news_evaluation_artifact_references(analysis_ids) if analysis_ids else {}
    )
    all_references = _deduplicate_references((*source_references, *analysis_references.values()))
    for reference in analysis_references.values():
        _ = verified.setdefault(
            reference.version_id,
            read_verified_r2_object(reference.r2_key, reference.content_digest),
        )
    articles = {
        article_id: ExtractedArticle.model_validate_json(verified[article_id], strict=True)
        for article_id in article_ids
    }
    relevance_by_article = {
        article_id: analysis_references[relevance_id]
        for article_id, relevance_id in zip(
            cluster_set.article_version_ids,
            cluster_set.relevance_version_ids,
            strict=True,
        )
    }
    day = report.day.isoformat()
    report_version_id = lineage.report.version_id
    common_sources = (report_version_id, cluster_reference.version_id)
    requests: list[RequestRecord] = []

    for article_id in article_ids:
        request = build_relevance_binary_request(
            ArticleAnalysisInput(
                reference=analysis_references[article_id], article=articles[article_id]
            )
        )
        requests.append(
            _request_record(
                "relevance",
                day,
                report_version_id,
                (f"article:{article_id}",),
                (*common_sources, article_id),
                request,
            )
        )

    for left_id, right_id in combinations(article_ids, 2):
        request = build_grouping_binary_request(
            BinaryGroupingCase(
                case_id=f"{day}:{left_id}:{right_id}",
                control=False,
                day=report.day,
                left_article=analysis_references[left_id],
                left_value=articles[left_id],
                right_article=analysis_references[right_id],
                right_value=articles[right_id],
                expected_same_group=False,
            )
        )
        requests.append(
            _request_record(
                "grouping",
                day,
                report_version_id,
                (f"article:{left_id}", f"article:{right_id}"),
                (*common_sources, left_id, right_id),
                request,
            )
        )

    ranking_groups = {
        group.id: _ranking_group(group, articles, analysis_references, relevance_by_article)
        for group in cluster_set.groups
    }
    for left_group, right_group in combinations(sorted(cluster_set.groups, key=lambda g: g.id), 2):
        request = build_ranking_binary_request(
            (ranking_groups[left_group.id], ranking_groups[right_group.id])
        )
        evidence_ids = tuple(
            item
            for group in (left_group, right_group)
            for article_id in group.article_version_ids
            for item in (article_id, relevance_by_article[article_id].version_id)
        )
        requests.append(
            _request_record(
                "ranking",
                day,
                report_version_id,
                (f"group:{left_group.id}", f"group:{right_group.id}"),
                (*common_sources, *evidence_ids),
                request,
            )
        )

    confidence_question = CONFIDENCE_BENCHMARK.questions[0]
    for group in sorted(cluster_set.groups, key=lambda item: item.id):
        case = BinaryConfidenceCase(
            case_id=f"{day}:{group.id}",
            control=False,
            report=lineage.report,
            group_id=group.id,
            articles=tuple(
                (analysis_references[article_id], articles[article_id])
                for article_id in group.article_version_ids
            ),
            expected_sufficient=False,
        )
        state = render_confidence_state(case)
        requests.append(
            _state_record(
                "confidence",
                day,
                report_version_id,
                (f"group:{group.id}",),
                (*common_sources, *group.article_version_ids),
                confidence_question.question_id,
                confidence_question.semantic_digest,
                state,
            )
        )

    theme_input = load_daily_theme_input(cluster_reference, lineage.inputs.summary)
    daily_theme_question = DAILY_THEME_BINARY_BENCHMARK.questions[0]
    summary_by_group = {item.group.id: item.summary.version_id for item in theme_input.groups}
    for left_group, right_group in combinations(sorted(cluster_set.groups, key=lambda g: g.id), 2):
        state = render_daily_theme_pair_state(theme_input, left_group.id, right_group.id)
        requests.append(
            _state_record(
                "daily_theme",
                day,
                report_version_id,
                (f"group:{left_group.id}", f"group:{right_group.id}"),
                (
                    *common_sources,
                    summary_by_group[left_group.id],
                    summary_by_group[right_group.id],
                ),
                daily_theme_question.question_id,
                daily_theme_question.semantic_digest,
                state,
            )
        )

    gaps: list[dict[str, str]] = []
    if isinstance(report, DailyReport):
        subjects = _tier_subjects(cluster_set, lineage, relevance_by_article, verified)
        subject_by_id = {subject.subject_id: subject for subject in subjects}
        if set(subject_by_id) != {section.theme_id for section in report.sections}:
            raise ValueError("DailyReport v3 sections do not match exact theme subjects")
        tier_sources = (
            *common_sources,
            *(reference.version_id for reference in lineage.inputs.themes),
            *(reference.version_id for reference in lineage.inputs.summary),
            *(reference.version_id for reference in relevance_by_article.values()),
        )
        for section in sorted(report.sections, key=lambda item: item.theme_id):
            subject = subject_by_id[section.theme_id]
            case = BinaryTierCase(
                case_id=f"{day}:{section.theme_id}",
                report_version_id=report_version_id,
                group_id=subject.events[0].group_id,
                expected_tier="excluded",
                state=BinaryTierState(
                    day=report.day,
                    target_subject_id=section.theme_id,
                    subjects=subjects,
                ),
            )
            state = render_binary_tier_state(case)
            for question in TIER_DEFINITION.questions:
                requests.append(
                    _state_record(
                        "tier",
                        day,
                        report_version_id,
                        (f"subject:{section.theme_id}",),
                        tier_sources,
                        question.question_id,
                        question.semantic_digest,
                        state,
                    )
                )
    else:
        gaps.append(
            {
                "concern": "tier",
                "code": "older-report-schema",
                "day": day,
                "report_version_id": report_version_id,
                "detail": (
                    "Tier inventory requires DailyReport schema v3 sections and was not guessed "
                    "for this older report schema."
                ),
            }
        )

    report_record: dict[str, object] = {
        "day": day,
        "version_id": report_version_id,
        "content_digest": lineage.report.content_digest,
        "schema_version": getattr(report, "schema_version", None),
        "source_artifacts": [
            _artifact_record(reference)
            for reference in sorted(all_references, key=lambda item: item.version_id)
        ],
    }
    return requests, report_record, gaps


def _tier_subjects(
    cluster_set: DailyClusterSet,
    lineage: NewsEvaluationReportLineage,
    relevance_by_article: Mapping[str, ArtifactReference],
    verified: Mapping[str, bytes],
) -> tuple[BinaryTierSubjectContext, ...]:
    if len(lineage.inputs.themes) != 1:
        raise ValueError("DailyReport v3 tier inventory requires one exact theme artifact")
    summaries = dict(
        _read_summary(verified[reference.version_id]) for reference in lineage.inputs.summary
    )
    evidence = dict(
        _read_relevance_quote(reference, verified[reference.version_id])
        for reference in relevance_by_article.values()
    )
    if set(summaries) != {group.id for group in cluster_set.groups}:
        raise ValueError("Tier summaries do not exactly cover cluster groups")
    if set(evidence) != set(cluster_set.article_version_ids):
        raise ValueError("Tier relevance evidence does not exactly cover cluster articles")
    events = {
        group.id: BinaryTierEventContext(
            group_id=group.id,
            title=summaries[group.id].title_ro,
            summary=summaries[group.id].summary_ro,
            key_points=summaries[group.id].key_points_ro,
            evidence_quotes=tuple(evidence[item] for item in group.article_version_ids),
        )
        for group in cluster_set.groups
    }
    theme_reference = lineage.inputs.themes[0]
    theme_set = parse_daily_theme_set(verified[theme_reference.version_id])
    if theme_set.day != cluster_set.day or theme_set.groups != cluster_set.groups:
        raise ValueError("Tier theme artifact does not match the exact cluster set")
    return tuple(
        BinaryTierSubjectContext(
            subject_id=theme.id,
            title=theme.title,
            summary=theme.summary,
            events=tuple(events[group_id] for group_id in theme.group_ids),
        )
        for theme in theme_set.themes
    )


def _ranking_group(
    group: NewsGroup,
    articles: Mapping[str, ExtractedArticle],
    references: Mapping[str, ArtifactReference],
    relevance_by_article: Mapping[str, ArtifactReference],
) -> RankingGroupEvidence:
    return RankingGroupEvidence(
        group=group,
        articles=tuple(
            RankingArticleEvidence(
                article=references[article_id],
                value=articles[article_id],
                relevance=relevance_by_article[article_id],
                decision=read_evaluation_ranking_decision(
                    relevance_by_article[article_id], article_id
                ),
            )
            for article_id in group.article_version_ids
        ),
    )


def _read_summary(content: bytes) -> tuple[Sha256, GroupSummary]:
    payload = _mapping(cast(object, json.loads(content)), "summary artifact")
    group_id = _sha256_value(payload.get("group_id"), "summary group ID")
    return group_id, GroupSummary.model_validate_json(
        json.dumps(payload.get("summary"), ensure_ascii=False), strict=True
    )


def _read_relevance_quote(reference: ArtifactReference, content: bytes) -> tuple[Sha256, str]:
    payload = _mapping(cast(object, json.loads(content)), "relevance artifact")
    article_id = _sha256_value(payload.get("article_version_id"), "relevance article ID")
    if "decision" in payload:
        decision = RelevanceDecision.model_validate(payload["decision"], strict=True)
    else:
        impact = payload.get("impact")
        context = payload.get("context")
        if isinstance(impact, Mapping) and "decision" in impact:
            decision = ImpactDecision.model_validate(impact["decision"], strict=True)
        elif isinstance(context, Mapping) and "decision" in context:
            decision = ContextDecision.model_validate(context["decision"], strict=True)
        else:
            raise ValueError(f"Relevance artifact has no evidence decision: {reference.version_id}")
    return article_id, decision.evidence_quote


def _request_record(
    concern: Concern,
    day: str,
    report_version_id: str,
    selection_ids: Sequence[str],
    source_version_ids: Sequence[str],
    request: BinaryRequest,
) -> RequestRecord:
    return _state_record(
        concern,
        day,
        report_version_id,
        selection_ids,
        source_version_ids,
        request.question.question_id,
        request.question.semantic_digest,
        request.state,
    )


def _state_record(
    concern: Concern,
    day: str,
    report_version_id: str,
    selection_ids: Sequence[str],
    source_version_ids: Sequence[str],
    question_id: str,
    question_digest: str,
    state: str,
) -> RequestRecord:
    return {
        "concern": concern,
        "day": day,
        "report_version_id": report_version_id,
        "selection_ids": sorted(selection_ids),
        "source_version_ids": sorted(set(source_version_ids)),
        "question_id": question_id,
        "question_digest": question_digest,
        "state_digest": _sha256(state.encode()),
        "state_characters": len(state),
    }


def _statistics(values: Sequence[int], limit: int) -> dict[str, object] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
        "thresholds": [
            {
                "percent": percent,
                "value": math.ceil(limit * percent / 100),
                "count_at_or_above": sum(value >= limit * percent / 100 for value in ordered),
                "share_at_or_above": round(
                    sum(value >= limit * percent / 100 for value in ordered) / len(ordered), 6
                ),
            }
            for percent in UTILIZATION_THRESHOLDS
        ],
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def write_reports(
    report: Mapping[str, object],
    output_json: Path,
    output_markdown: Path,
    output_raw_gzip: Path,
) -> None:
    concerns = _list(report.get("concerns"), "inventory concerns")
    summarized_concerns: list[dict[str, object]] = []
    raw_concerns: list[dict[str, object]] = []
    for raw_concern in concerns:
        concern = _mapping(raw_concern, "inventory concern")
        requests = _list(concern.get("requests"), "inventory requests")
        raw_concerns.append({"concern": concern["concern"], "requests": requests})
        summarized_concerns.append(
            {
                **{key: value for key, value in concern.items() if key != "requests"},
                "request_identity_digest": _sha256(
                    json.dumps(requests, sort_keys=True, separators=(",", ":")).encode()
                ),
            }
        )
    raw_report = {
        "schema_version": "jev-production-context-request-identities/v1",
        "reports": report.get("reports"),
        "concerns": raw_concerns,
    }
    raw_content = gzip.compress(
        (json.dumps(raw_report, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        compresslevel=9,
        mtime=0,
    )
    raw_identity = {
        "path": str(output_raw_gzip),
        "compression": "gzip",
        "sha256": _sha256(raw_content),
    }
    summarized_report = {
        **{key: value for key, value in report.items() if key != "concerns"},
        "concerns": summarized_concerns,
        "raw_request_identities": raw_identity,
    }
    _ = output_raw_gzip.write_bytes(raw_content)
    _ = output_json.write_text(json.dumps(summarized_report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Jev production context inventory",
        "",
        "This read-only inventory rebuilds proposed single-question states from captured exact daily report versions. It stores no article text, state text, provider output, or label.",
        "",
        "| Concern | Scope | Requests | p50 chars | p95 chars | p99 chars | Max chars | >=50% | >=75% | >=90% | >=100% | Estimated max tokens | Estimated >=32k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw_concern in concerns:
        concern = _mapping(raw_concern, "inventory concern")
        character_stats = concern.get("state_character_statistics")
        estimated_stats = concern.get("estimated_input_token_statistics")
        character_values: Mapping[str, object] = (
            _mapping(character_stats, "character statistics") if character_stats else {}
        )
        estimated_values: Mapping[str, object] = (
            _mapping(estimated_stats, "token statistics") if estimated_stats else {}
        )
        thresholds = _list(character_values.get("thresholds", []), "character thresholds")
        estimated_thresholds = _list(estimated_values.get("thresholds", []), "estimated thresholds")
        threshold_counts = (
            [_threshold_display(_mapping(item, "threshold")) for item in thresholds]
            if thresholds
            else ["null"] * len(UTILIZATION_THRESHOLDS)
        )
        if len(threshold_counts) != len(UTILIZATION_THRESHOLDS):
            raise ValueError("Character statistics have incomplete thresholds")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(concern["concern"]),
                    str(concern["scope"]),
                    str(concern["request_count"]),
                    _display(character_values.get("p50")),
                    _display(character_values.get("p95")),
                    _display(character_values.get("p99")),
                    _display(character_values.get("max")),
                    *threshold_counts,
                    _display(estimated_values.get("max")),
                    (
                        _display(
                            _mapping(estimated_thresholds[-1], "estimated threshold").get(
                                "count_at_or_above"
                            )
                        )
                        if estimated_thresholds
                        else "null"
                    ),
                )
            )
            + " |"
        )
    lines.extend(("", "## Evidence gaps", ""))
    for raw_gap in _list(report.get("evidence_gaps"), "evidence gaps"):
        gap = _mapping(raw_gap, "evidence gap")
        lines.append(f'- [{gap["concern"]}/{gap["code"]}] {gap["detail"]}')
    lines.append("")
    _ = output_markdown.write_text("\n".join(lines))


def _scope(concern: Concern) -> str:
    return {
        "relevance": "accepted cluster articles only",
        "grouping": "all unordered accepted cluster article pairs",
        "ranking": "all unordered cluster group pairs with exact relevance evidence",
        "tier": "two registered questions per DailyReport v3 subject",
        "confidence": "each cluster group",
        "daily_theme": "all unordered cluster group pairs",
    }[concern]


def _artifact_record(reference: ArtifactReference) -> dict[str, str]:
    return {
        "artifact_id": reference.artifact_id,
        "version_id": reference.version_id,
        "content_digest": reference.content_digest,
        "r2_key": reference.r2_key,
    }


def _deduplicate_references(
    references: Sequence[ArtifactReference],
) -> tuple[ArtifactReference, ...]:
    result: dict[str, ArtifactReference] = {}
    for reference in references:
        existing = result.get(reference.version_id)
        if existing is not None and existing != reference:
            raise ValueError("Artifact version has conflicting references")
        result[reference.version_id] = reference
    return tuple(result[version_id] for version_id in sorted(result))


def _request_sort_key(request: RequestRecord) -> tuple[object, ...]:
    return (
        request["day"],
        request["report_version_id"],
        tuple(request["selection_ids"]),
        request["question_id"],
        request["state_digest"],
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[object], value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _path(value: object, label: str) -> Path:
    if not isinstance(value, Path):
        raise ValueError(f"{label} must be a path")
    return value


def _sha256_value(value: object, label: str) -> Sha256:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _display(value: object) -> str:
    return "null" if value is None else str(value)


def _threshold_display(value: Mapping[str, object]) -> str:
    count = _display(value.get("count_at_or_above"))
    share = value.get("share_at_or_above")
    return count if share is None else f"{count} ({float(str(share)) * 100:.4f}%)"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(
        description="Inventory Jev context sizes from exact production daily report versions."
    )
    _ = parser.add_argument("--report-limit", required=True, type=int)
    _ = parser.add_argument("--reviewed-calibration", required=True, type=Path)
    _ = parser.add_argument("--limits", required=True, type=Path)
    _ = parser.add_argument("--output-json", required=True, type=Path)
    _ = parser.add_argument("--output-markdown", required=True, type=Path)
    _ = parser.add_argument("--output-raw-gzip", required=True, type=Path)
    values = _mapping(cast(object, vars(parser.parse_args(argv))), "arguments")
    return Arguments(
        report_limit=_positive_int(values.get("report_limit"), "--report-limit"),
        reviewed_calibration=_path(values.get("reviewed_calibration"), "--reviewed-calibration"),
        limits=_path(values.get("limits"), "--limits"),
        output_json=_path(values.get("output_json"), "--output-json"),
        output_markdown=_path(values.get("output_markdown"), "--output-markdown"),
        output_raw_gzip=_path(values.get("output_raw_gzip"), "--output-raw-gzip"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = build_inventory(
        arguments.report_limit,
        arguments.limits,
        arguments.reviewed_calibration,
    )
    write_reports(
        report,
        arguments.output_json,
        arguments.output_markdown,
        arguments.output_raw_gzip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
