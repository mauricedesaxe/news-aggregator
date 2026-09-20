from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, cast

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.binary_evaluation import BinaryRequest, binary_state_digest
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.analysis.relevance_v3 import ContextDecision, ImpactDecision
from romanian_news.binary_benchmark import (
    BinaryBenchmarkCase,
    BinaryBenchmarkEvaluationResult,
    BinaryEvaluator,
    BinaryJudgmentCase,
    BinaryJudgmentResult,
    BinaryTarget,
    TargetId,
    build_execution_identity,
    run_registered_binary_benchmark,
)
from romanian_news.binary_relevance_evaluation import V11_MANIFEST_VERSION, V11_SOURCE_ARTIFACT_ID
from romanian_news.catalog.evaluations import (
    NewsEvaluationReportLineage,
    read_news_evaluation_artifact_references,
    read_news_evaluation_report_lineage,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import (
    NewsEvaluationManifest,
    NewsEvaluationPin,
    ReportEvaluationSpec,
    Tier,
    TierEvaluationSpec,
)
from romanian_news.groups import DailyClusterSet
from romanian_news.storage import read_verified_r2_object
from romanian_news.themes import parse_daily_theme_set

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
ArtifactReader = Callable[[ArtifactReference], bytes]
ReportLineageReader = Callable[[Sha256], NewsEvaluationReportLineage]

TIER_DEFINITION = DERIVED_BINARY_BENCHMARKS["tier"]
_TIER_ORDER: tuple[Tier, ...] = ("main", "worth_knowing", "excluded")
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_SHA256_ADAPTER: TypeAdapter[Sha256] = TypeAdapter(Sha256)


class BinaryTierEventContext(NewsModel):
    group_id: Sha256
    title: NonEmptyText
    summary: NonEmptyText
    key_points: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    evidence_quotes: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]


class BinaryTierSubjectContext(NewsModel):
    subject_id: Sha256
    title: NonEmptyText
    summary: NonEmptyText
    events: Annotated[tuple[BinaryTierEventContext, ...], Field(min_length=1)]


class BinaryTierState(NewsModel):
    day: date
    target_subject_id: Sha256
    subjects: Annotated[tuple[BinaryTierSubjectContext, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_target_subject(self) -> BinaryTierState:
        subject_ids = tuple(subject.subject_id for subject in self.subjects)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("Binary tier subjects must be unique")
        if self.target_subject_id not in subject_ids:
            raise ValueError("Binary tier target must be present in the rendered subjects")
        return self


class BinaryTierCase(NewsModel):
    case_id: NonEmptyText
    report_version_id: Sha256
    group_id: Sha256
    control: bool = False
    expected_tier: Tier
    state: BinaryTierState

    @model_validator(mode="after")
    def require_group_in_target_subject(self) -> BinaryTierCase:
        target = next(
            subject
            for subject in self.state.subjects
            if subject.subject_id == self.state.target_subject_id
        )
        if self.group_id not in {event.group_id for event in target.events}:
            raise ValueError("Binary tier group must belong to the target subject")
        return self

    @property
    def expected_main_subject(self) -> bool:
        return self.expected_tier == "main"

    @property
    def expected_worth_knowing_if_not_main(self) -> bool | None:
        if self.expected_tier == "main":
            return None
        return self.expected_tier == "worth_knowing"


class BinaryTierSource(NewsModel):
    pin: NewsEvaluationPin
    manifest_reference: ArtifactReference
    declared_manifest_version: Literal["news-evaluation-2026-09-11-v11"]
    cases: Annotated[tuple[BinaryTierCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_pinned_v11_source(self) -> BinaryTierSource:
        if (
            self.pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID
            or self.manifest_reference.version_id != V11_SOURCE_ARTIFACT_ID
        ):
            raise ValueError(f"Binary tier source artifact must be {V11_SOURCE_ARTIFACT_ID}")
        if self.declared_manifest_version != V11_MANIFEST_VERSION:
            raise ValueError(f"Binary tier manifest version must be {V11_MANIFEST_VERSION}")
        if len(self.cases) != TIER_DEFINITION.case_count:
            raise ValueError(
                f"Binary tier benchmark requires {TIER_DEFINITION.case_count} cases, received "
                + str(len(self.cases))
            )
        return self


class BinaryTierConfusionRow(NewsModel):
    main: Annotated[int, Field(ge=0)]
    worth_knowing: Annotated[int, Field(ge=0)]
    excluded: Annotated[int, Field(ge=0)]


class BinaryTierConfusionMatrix(NewsModel):
    main: BinaryTierConfusionRow
    worth_knowing: BinaryTierConfusionRow
    excluded: BinaryTierConfusionRow


class BinaryTierRecall(NewsModel):
    main: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    worth_knowing: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    excluded: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]


class BinaryTierCaseResult(NewsModel):
    status: Literal["completed", "failed"]
    case_id: NonEmptyText
    target_id: TargetId
    trial_ref: NonEmptyText
    expected_tier: Tier
    predicted_tier: Tier | None
    main_subject_probability: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None
    worth_knowing_if_not_main_probability: (
        Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)] | None
    )
    main_subject_correct: bool | None
    worth_knowing_if_not_main_correct: bool | None
    exact_match: bool

    @model_validator(mode="after")
    def require_consistent_score(self) -> BinaryTierCaseResult:
        if self.status == "completed":
            if (
                self.predicted_tier is None
                or self.main_subject_probability is None
                or self.worth_knowing_if_not_main_probability is None
                or self.main_subject_correct is None
            ):
                raise ValueError("Completed binary tier cases require both judgments")
            if self.exact_match != (self.predicted_tier == self.expected_tier):
                raise ValueError("Binary tier exact score conflicts with its composed prediction")
            if self.expected_tier == "main":
                if self.worth_knowing_if_not_main_correct is not None:
                    raise ValueError("Main tier cases have no conditional binary ground truth")
            elif self.worth_knowing_if_not_main_correct is None:
                raise ValueError("Non-main tier cases require conditional binary correctness")
        elif (
            self.predicted_tier is not None
            or self.main_subject_probability is not None
            or self.worth_knowing_if_not_main_probability is not None
            or self.main_subject_correct is not None
            or self.worth_knowing_if_not_main_correct is not None
            or self.exact_match
        ):
            raise ValueError("Failed binary tier cases cannot contain a score")
        return self


class BinaryTierMetrics(NewsModel):
    exact_matches: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    total_cases: Annotated[int, Field(ge=0)]
    exact_accuracy: Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
    confusion_matrix: BinaryTierConfusionMatrix
    per_tier_recall: BinaryTierRecall

    @model_validator(mode="after")
    def require_consistent_counts(self) -> BinaryTierMetrics:
        if self.completed_cases + self.failed_cases != self.total_cases:
            raise ValueError("Binary tier completion counts must cover every case")
        if self.exact_matches > self.completed_cases:
            raise ValueError("Binary tier exact matches cannot exceed completed cases")
        return self


class BinaryTierRunResult(NewsModel):
    target: BinaryTarget
    trial_ref: NonEmptyText
    cases: tuple[BinaryTierCaseResult, ...]
    metrics: BinaryTierMetrics

    @model_validator(mode="after")
    def require_consistent_run(self) -> BinaryTierRunResult:
        if any(
            case.target_id != self.target.target_id or case.trial_ref != self.trial_ref
            for case in self.cases
        ):
            raise ValueError("Binary tier cases do not match their run")
        if self.metrics != binary_tier_metrics(self.cases):
            raise ValueError("Binary tier run metrics do not match its cases")
        return self


class BinaryTierEvaluationResult(NewsModel):
    binary_result: BinaryBenchmarkEvaluationResult
    runs: tuple[BinaryTierRunResult, ...]


def load_v11_binary_tier_source(
    pin_content: bytes | str,
    *,
    artifact_reader: ArtifactReader | None = None,
    lineage_reader: ReportLineageReader = read_news_evaluation_report_lineage,
) -> BinaryTierSource:
    pin = NewsEvaluationPin.model_validate_json(pin_content, strict=True)
    if pin.manifest_version_id != V11_SOURCE_ARTIFACT_ID:
        raise ValueError(f"Binary tier source artifact must be {V11_SOURCE_ARTIFACT_ID}")
    manifest_reference = read_news_evaluation_artifact_references((V11_SOURCE_ARTIFACT_ID,))[
        V11_SOURCE_ARTIFACT_ID
    ]
    read_artifact = artifact_reader or _read_artifact
    manifest = NewsEvaluationManifest.model_validate_json(
        read_artifact(manifest_reference), strict=True
    )
    if manifest.version != V11_MANIFEST_VERSION:
        raise ValueError(f"Binary tier manifest version must be {V11_MANIFEST_VERSION}")
    specs = tuple(case for case in manifest.cases if isinstance(case, TierEvaluationSpec))
    if len(specs) != TIER_DEFINITION.case_count:
        raise ValueError(
            f"Binary tier benchmark requires {TIER_DEFINITION.case_count} cases, received "
            + str(len(specs))
        )
    reports = {report.report.version_id: report for report in manifest.reports}
    contexts: dict[Sha256, tuple[BinaryTierSubjectContext, ...]] = {}
    days: dict[Sha256, date] = {}
    for report_version_id in dict.fromkeys(
        cast(Sha256, spec.provenance.report_version_id) for spec in specs
    ):
        report = reports.get(report_version_id)
        if report is None:
            raise ValueError("Binary tier case references an unavailable frozen report")
        day, subjects = _load_report_context(
            report, lineage_reader(report_version_id), read_artifact
        )
        days[report_version_id] = day
        contexts[report_version_id] = subjects
    cases = tuple(_tier_case(spec, days, contexts) for spec in specs)
    return BinaryTierSource(
        pin=pin,
        manifest_reference=manifest_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=cases,
    )


def render_binary_tier_state(case: BinaryTierCase) -> str:
    return json.dumps(
        case.state.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def adapt_binary_tier_case(case: BinaryTierCase) -> BinaryBenchmarkCase:
    state = render_binary_tier_state(case)
    expected_worth = case.expected_worth_knowing_if_not_main
    expected_by_question = {
        "main_subject": case.expected_main_subject,
        # The generic protocol requires a boolean. Tier scoring treats this placeholder as
        # non-applicable for reviewed main cases and never counts it as binary correctness.
        "worth_knowing_if_not_main": expected_worth if expected_worth is not None else False,
    }
    return BinaryBenchmarkCase(
        case_id=case.case_id,
        identity=(
            case.report_version_id,
            case.group_id,
            case.expected_tier,
            "conditional-applicable"
            if expected_worth is not None
            else "conditional-not-applicable",
        ),
        control=case.control,
        judgments=tuple(
            BinaryJudgmentCase(
                judgment_id=question.question_id,
                request=BinaryRequest(
                    question=question,
                    state=state,
                    state_digest=binary_state_digest(state),
                ),
                expected=expected_by_question[question.question_id],
            )
            for question in TIER_DEFINITION.questions
        ),
    )


def adapt_binary_tier_cases(
    cases: tuple[BinaryTierCase, ...],
) -> tuple[BinaryBenchmarkCase, ...]:
    return tuple(adapt_binary_tier_case(case) for case in cases)


def compose_binary_tier(main_subject: Decimal, worth_knowing_if_not_main: Decimal) -> Tier:
    if not (
        Decimal(0) <= main_subject <= Decimal(1)
        and Decimal(0) <= worth_knowing_if_not_main <= Decimal(1)
    ):
        raise ValueError("Binary tier probabilities must be between zero and one")
    thresholds = {
        question.question_id: question.threshold for question in TIER_DEFINITION.questions
    }
    if main_subject >= thresholds["main_subject"]:
        return "main"
    if worth_knowing_if_not_main >= thresholds["worth_knowing_if_not_main"]:
        return "worth_knowing"
    return "excluded"


def run_binary_tier_evaluation(
    source: BinaryTierSource,
    *,
    targets: tuple[BinaryTarget, ...],
    trial_refs: tuple[str, ...],
    evaluators: Mapping[TargetId, BinaryEvaluator],
    execution_ref: str,
    dry_run: bool,
) -> BinaryTierEvaluationResult:
    cases = adapt_binary_tier_cases(source.cases)
    identity = build_execution_identity(
        TIER_DEFINITION,
        source_artifact_id=source.manifest_reference.version_id,
        declared_manifest_version=source.declared_manifest_version,
        cases=cases,
        targets=targets,
        trial_refs=trial_refs,
        execution_mode="dry_run" if dry_run else "live",
        execution_ref=execution_ref,
    )
    binary_result = run_registered_binary_benchmark(
        TIER_DEFINITION,
        identity,
        cases,
        evaluators,
    )
    return score_binary_tier_evaluation(source.cases, binary_result)


def score_binary_tier_evaluation(
    cases: tuple[BinaryTierCase, ...],
    binary_result: BinaryBenchmarkEvaluationResult,
) -> BinaryTierEvaluationResult:
    case_by_id = {case.case_id: case for case in cases}
    result_by_key: dict[tuple[str, str, str, str], BinaryJudgmentResult] = {
        (result.target_id, result.trial_ref, result.case_id, result.question_id): result
        for result in binary_result.results
    }
    runs = tuple(
        _tier_run(
            target,
            trial_ref,
            cases,
            result_by_key,
        )
        for target in binary_result.identity.targets
        for trial_ref in binary_result.identity.trial_refs
    )
    if set(case_by_id) != {identity.case_id for identity in binary_result.identity.cases}:
        raise ValueError("Binary tier cases do not match the generic evaluation identity")
    return BinaryTierEvaluationResult(binary_result=binary_result, runs=runs)


def binary_tier_metrics(cases: tuple[BinaryTierCaseResult, ...]) -> BinaryTierMetrics:
    completed = tuple(case for case in cases if case.status == "completed")
    counts = {
        expected: {
            predicted: sum(
                case.expected_tier == expected and case.predicted_tier == predicted
                for case in completed
            )
            for predicted in _TIER_ORDER
        }
        for expected in _TIER_ORDER
    }
    expected_counts = {
        tier: sum(case.expected_tier == tier for case in completed) for tier in _TIER_ORDER
    }
    exact_matches = sum(case.exact_match for case in completed)
    return BinaryTierMetrics(
        exact_matches=exact_matches,
        completed_cases=len(completed),
        failed_cases=len(cases) - len(completed),
        total_cases=len(cases),
        exact_accuracy=_ratio(exact_matches, len(completed)),
        confusion_matrix=BinaryTierConfusionMatrix(
            **{expected: BinaryTierConfusionRow(**counts[expected]) for expected in _TIER_ORDER}
        ),
        per_tier_recall=BinaryTierRecall(
            **{tier: _ratio(counts[tier][tier], expected_counts[tier]) for tier in _TIER_ORDER}
        ),
    )


def _tier_case(
    spec: TierEvaluationSpec,
    days: Mapping[Sha256, date],
    contexts: Mapping[Sha256, tuple[BinaryTierSubjectContext, ...]],
) -> BinaryTierCase:
    report_version_id = spec.provenance.report_version_id
    group_id = spec.provenance.group_id
    if report_version_id is None or group_id is None:
        raise ValueError("Binary tier evaluation requires a frozen report group")
    subjects = contexts[report_version_id]
    target = next(
        (
            subject
            for subject in subjects
            if group_id in {event.group_id for event in subject.events}
        ),
        None,
    )
    if target is None:
        raise ValueError("Binary tier group is absent from its upstream subject context")
    return BinaryTierCase(
        case_id=spec.case_id,
        report_version_id=report_version_id,
        group_id=group_id,
        control=spec.control,
        expected_tier=spec.expected_tier,
        state=BinaryTierState(
            day=days[report_version_id],
            target_subject_id=target.subject_id,
            subjects=subjects,
        ),
    )


def _load_report_context(
    report: ReportEvaluationSpec,
    lineage: NewsEvaluationReportLineage,
    read_artifact: ArtifactReader,
) -> tuple[date, tuple[BinaryTierSubjectContext, ...]]:
    if lineage.report != report.report:
        raise ValueError("Binary tier lineage does not match the frozen report reference")
    if lineage.inputs.cluster_set != (report.cluster_set,):
        raise ValueError("Binary tier lineage does not match the frozen cluster set")
    expected_themes = (report.themes,) if report.themes is not None else ()
    if lineage.inputs.themes != expected_themes:
        raise ValueError("Binary tier lineage does not match the frozen themes")
    cluster_set = DailyClusterSet.model_validate_json(
        read_artifact(report.cluster_set), strict=True
    )
    summaries = {
        group_id: summary
        for reference in lineage.inputs.summary
        for group_id, summary in (_read_summary(reference, read_artifact),)
    }
    evidence = {
        article_id: quote
        for reference in lineage.inputs.relevance
        for article_id, quote in (_read_relevance_quote(reference, read_artifact),)
    }
    if set(summaries) != {group.id for group in cluster_set.groups}:
        raise ValueError("Binary tier summaries must exactly cover frozen groups")
    if set(evidence) != set(cluster_set.article_version_ids):
        raise ValueError("Binary tier evidence must exactly cover frozen articles")
    events = {
        group.id: BinaryTierEventContext(
            group_id=group.id,
            title=summaries[group.id].title_ro,
            summary=summaries[group.id].summary_ro,
            key_points=summaries[group.id].key_points_ro,
            evidence_quotes=tuple(evidence[article_id] for article_id in group.article_version_ids),
        )
        for group in cluster_set.groups
    }
    if report.themes is None:
        subjects = tuple(
            BinaryTierSubjectContext(
                subject_id=group.id,
                title=summaries[group.id].title_ro,
                summary=summaries[group.id].summary_ro,
                events=(events[group.id],),
            )
            for group in cluster_set.groups
        )
    else:
        theme_set = parse_daily_theme_set(read_artifact(report.themes))
        if theme_set.day != cluster_set.day or theme_set.groups != cluster_set.groups:
            raise ValueError("Binary tier themes do not match the frozen cluster context")
        subjects = tuple(
            BinaryTierSubjectContext(
                subject_id=theme.id,
                title=theme.title,
                summary=theme.summary,
                events=tuple(events[group_id] for group_id in theme.group_ids),
            )
            for theme in theme_set.themes
        )
    return cluster_set.day, subjects


def _read_summary(
    reference: ArtifactReference, read_artifact: ArtifactReader
) -> tuple[Sha256, GroupSummary]:
    payload = _json_object(read_artifact(reference))
    group_id = _SHA256_ADAPTER.validate_python(payload.get("group_id"), strict=True)
    return group_id, GroupSummary.model_validate_json(
        json.dumps(payload.get("summary"), ensure_ascii=False), strict=True
    )


def _read_relevance_quote(
    reference: ArtifactReference, read_artifact: ArtifactReader
) -> tuple[Sha256, str]:
    payload = _json_object(read_artifact(reference))
    article_id = _SHA256_ADAPTER.validate_python(payload.get("article_version_id"), strict=True)
    if "decision" in payload:
        decision = RelevanceDecision.model_validate(payload["decision"], strict=True)
    else:
        impact = payload.get("impact")
        context = payload.get("context")
        if isinstance(impact, dict) and "decision" in impact:
            decision = ImpactDecision.model_validate(impact["decision"], strict=True)
        elif isinstance(context, dict) and "decision" in context:
            decision = ContextDecision.model_validate(context["decision"], strict=True)
        else:
            raise ValueError("Binary tier relevance has no evidence decision")
    return article_id, decision.evidence_quote


def _tier_run(
    target: BinaryTarget,
    trial_ref: str,
    cases: tuple[BinaryTierCase, ...],
    results: Mapping[tuple[str, str, str, str], BinaryJudgmentResult],
) -> BinaryTierRunResult:
    scored = tuple(
        _score_tier_case(
            case,
            target.target_id,
            trial_ref,
            results[(target.target_id, trial_ref, case.case_id, "main_subject")],
            results[
                (
                    target.target_id,
                    trial_ref,
                    case.case_id,
                    "worth_knowing_if_not_main",
                )
            ],
        )
        for case in cases
    )
    return BinaryTierRunResult(
        target=target,
        trial_ref=trial_ref,
        cases=scored,
        metrics=binary_tier_metrics(scored),
    )


def _score_tier_case(
    case: BinaryTierCase,
    target_id: TargetId,
    trial_ref: str,
    main_result: BinaryJudgmentResult,
    worth_result: BinaryJudgmentResult,
) -> BinaryTierCaseResult:
    if main_result.status != "completed" or worth_result.status != "completed":
        return BinaryTierCaseResult(
            status="failed",
            case_id=case.case_id,
            target_id=target_id,
            trial_ref=trial_ref,
            expected_tier=case.expected_tier,
            predicted_tier=None,
            main_subject_probability=None,
            worth_knowing_if_not_main_probability=None,
            main_subject_correct=None,
            worth_knowing_if_not_main_correct=None,
            exact_match=False,
        )
    if main_result.probability is None or worth_result.probability is None:
        raise AssertionError("Completed binary tier judgment omitted its probability")
    predicted = compose_binary_tier(main_result.probability, worth_result.probability)
    expected_worth = case.expected_worth_knowing_if_not_main
    return BinaryTierCaseResult(
        status="completed",
        case_id=case.case_id,
        target_id=target_id,
        trial_ref=trial_ref,
        expected_tier=case.expected_tier,
        predicted_tier=predicted,
        main_subject_probability=main_result.probability,
        worth_knowing_if_not_main_probability=worth_result.probability,
        main_subject_correct=main_result.verdict == case.expected_main_subject,
        worth_knowing_if_not_main_correct=(
            None if expected_worth is None else worth_result.verdict == expected_worth
        ),
        exact_match=predicted == case.expected_tier,
    )


def _read_artifact(reference: ArtifactReference) -> bytes:
    return read_verified_r2_object(reference.r2_key, reference.content_digest)


def _json_object(content: bytes) -> dict[str, object]:
    return _JSON_OBJECT_ADAPTER.validate_json(content, strict=True)


def _ratio(numerator: int, denominator: int) -> Decimal:
    return Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)
