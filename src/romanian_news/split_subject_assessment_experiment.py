from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, cast

import requests
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news import subject_assessments as production
from romanian_news.analysis.binary_evaluation import BinaryAttemptError
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.analysis.jev_relevance import (
    JEV_EXECUTION_POLICY,
    JevExecutionPolicy,
    JevNoulResponse,
)
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.analysis.relevance_v3 import ContextDecision, ImpactDecision
from romanian_news.analysis.tracing import ProviderChatRequest
from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_tier_evaluation import BinaryTierState, compose_binary_tier
from romanian_news.catalog.evaluations import (
    NewsEvaluationReportLineage,
    read_news_evaluation_artifact_references,
    read_news_evaluation_report_lineage,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import NewsEvaluationManifest, ReportEvaluationSpec
from romanian_news.groups import DailyClusterSet
from romanian_news.storage import read_verified_r2_object
from romanian_news.subject_assessments import (
    ASSESSMENT_MAX_TOKENS,
    ASSESSMENT_MODEL,
    ASSESSMENT_PROMPT,
    DailySubjectAssessmentInput,
    DailySubjectAssessmentOutput,
    SubjectAssessment,
    SubjectAssessmentThemeSet,
    SubjectEvidenceInput,
    SubjectSummaryInput,
    Tier,
)
from romanian_news.themes import parse_daily_theme_set

V11_MANIFEST_VERSION_ID = "083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"
V11_REPORT_VERSION_IDS = ("c1369f9a24202a111dfc07e29a7194c73cb22dd491958093932a5d3b9b270d72",)
V11_MANIFEST_ARTIFACT_ID = "news:evaluation-manifest:news-evaluation-2026-09-11-v11"
V11_MANIFEST_CONTENT_DIGEST = "19859f4edb7a7b7f28d02fe22c79e3afd2fdca71d50fec3dc8140a6ee35d1caf"
V11_MANIFEST_R2_KEY = (
    "news/evaluations/manifests/news-evaluation-2026-09-11-v11/"
    f"{V11_MANIFEST_CONTENT_DIGEST}.json"
)
CONTEXT_GUARD_CHARACTERS = 70_000
SPLIT_ASSESSMENT_OPERATION = "news.experiment.split_subject_assessment"
_JSON_OBJECT = TypeAdapter(dict[str, object])
_GROUP_SUMMARY = TypeAdapter(GroupSummary)


class ExperimentAttempt(NewsModel):
    provider: Literal["typesafe", "openrouter"]
    stage: Literal["tier", "ranking", "incumbent"]
    subject_id: Sha256 | None = None
    attempt_number: Annotated[int, Field(gt=0)]
    status: Literal["accepted", "rejected", "retryable_error", "terminal_error"]
    request_id: Sha256
    provider_request_id: str | None
    actual_model: str | None
    input_tokens: Annotated[int, Field(ge=0)] | None
    output_tokens: Annotated[int, Field(ge=0)] | None
    cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)] | None
    latency_ms: Annotated[int, Field(ge=0)]
    error: BinaryAttemptError | None = None

    @property
    def accounting_complete(self) -> bool:
        return all(
            value is not None
            for value in (self.actual_model, self.input_tokens, self.output_tokens, self.cost_usd)
        )


class CompletedArm(NewsModel):
    status: Literal["completed"] = "completed"
    arm: Literal["incumbent", "candidate"]
    report_version_id: Sha256
    wall_latency_ms: Annotated[int, Field(ge=0)]
    attempts: tuple[ExperimentAttempt, ...]
    assessments: tuple[SubjectAssessment, ...]
    accounting_complete: bool

    @model_validator(mode="after")
    def require_complete_accounting(self) -> CompletedArm:
        if self.accounting_complete != all(item.accounting_complete for item in self.attempts):
            raise ValueError("Arm accounting completeness does not match its attempts")
        return self


class FailedArm(NewsModel):
    status: Literal["failed"] = "failed"
    arm: Literal["incumbent", "candidate"]
    report_version_id: Sha256
    wall_latency_ms: Annotated[int, Field(ge=0)]
    attempts: tuple[ExperimentAttempt, ...]
    error: str
    accounting_complete: bool

    @model_validator(mode="after")
    def require_complete_accounting(self) -> FailedArm:
        if self.accounting_complete != all(item.accounting_complete for item in self.attempts):
            raise ValueError("Arm accounting completeness does not match its attempts")
        return self


class UnavailableArm(NewsModel):
    status: Literal["unavailable"] = "unavailable"
    arm: Literal["candidate"] = "candidate"
    report_version_id: Sha256
    wall_latency_ms: Annotated[int, Field(ge=0)]
    attempts: tuple[ExperimentAttempt, ...]
    reason: str
    accounting_complete: bool

    @model_validator(mode="after")
    def require_complete_accounting(self) -> UnavailableArm:
        if self.accounting_complete != all(item.accounting_complete for item in self.attempts):
            raise ValueError("Arm accounting completeness does not match its attempts")
        return self


ArmOutcome = Annotated[CompletedArm | FailedArm | UnavailableArm, Field(discriminator="status")]


class TierObservation(NewsModel):
    subject_id: Sha256
    main_probability: Annotated[Decimal, Field(ge=0, le=1)]
    worth_knowing_probability: Annotated[Decimal, Field(ge=0, le=1)]
    tier: Tier
    attempts: Annotated[tuple[ExperimentAttempt, ...], Field(min_length=1)]


AttemptSink = Callable[[ExperimentAttempt], None]
TierEvaluator = Callable[[BinaryTierState, str, AttemptSink], TierObservation]
RankingCaller = Callable[[ProviderChatRequest], ChatCompletion]
ArtifactReader = Callable[[ArtifactReference], bytes]
VersionReferenceReader = Callable[[tuple[Sha256, ...]], dict[Sha256, ArtifactReference]]


class FrozenV11SourceDescriptor(NewsModel):
    schema_version: Literal["split-subject-assessment-source-v1"] = (
        "split-subject-assessment-source-v1"
    )
    manifest: ArtifactReference
    report_lineages: tuple[NewsEvaluationReportLineage, ...]
    article_references: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def require_exact_frozen_source(self) -> FrozenV11SourceDescriptor:
        if self.manifest.artifact_id != V11_MANIFEST_ARTIFACT_ID:
            raise ValueError("Frozen source has the wrong manifest artifact ID")
        if self.manifest.version_id != V11_MANIFEST_VERSION_ID:
            raise ValueError("Frozen source has the wrong manifest version ID")
        if self.manifest.content_digest != V11_MANIFEST_CONTENT_DIGEST:
            raise ValueError("Frozen source has the wrong manifest content digest")
        if self.manifest.r2_key != V11_MANIFEST_R2_KEY:
            raise ValueError("Frozen source has the wrong manifest object key")
        report_ids = tuple(item.report.version_id for item in self.report_lineages)
        if len(set(report_ids)) != len(report_ids) or set(report_ids) != set(
            V11_REPORT_VERSION_IDS
        ):
            raise ValueError("Frozen source does not exactly cover the v11 reports")
        article_ids = tuple(item.version_id for item in self.article_references)
        if len(set(article_ids)) != len(article_ids):
            raise ValueError("Frozen source contains duplicate article references")
        return self


def load_frozen_v11_source(
    descriptor_source: bytes | Path,
    artifact_reader: ArtifactReader,
) -> tuple[
    FrozenV11SourceDescriptor,
    NewsEvaluationManifest,
    tuple[tuple[Sha256, DailySubjectAssessmentInput], ...],
]:
    descriptor_content = (
        descriptor_source.read_bytes() if isinstance(descriptor_source, Path) else descriptor_source
    )
    descriptor = FrozenV11SourceDescriptor.model_validate_json(descriptor_content, strict=True)
    read_artifact = _verified_cached_reader(artifact_reader)
    manifest = NewsEvaluationManifest.model_validate_json(
        read_artifact(descriptor.manifest), strict=True
    )
    if manifest.version != "news-evaluation-2026-09-11-v11":
        raise ValueError("Frozen source manifest is not the exact v11 manifest")
    reports = {item.report.version_id: item for item in manifest.reports}
    if len(reports) != len(manifest.reports) or not set(V11_REPORT_VERSION_IDS) <= set(reports):
        raise ValueError("Frozen manifest does not contain the v11 report workload")
    lineages = {item.report.version_id: item for item in descriptor.report_lineages}
    articles = {item.version_id: item for item in descriptor.article_references}
    for reference in _descriptor_references(descriptor):
        read_artifact(reference)

    def read_article_references(
        version_ids: tuple[Sha256, ...],
    ) -> dict[Sha256, ArtifactReference]:
        if len(set(version_ids)) != len(version_ids) or set(version_ids) != set(articles):
            raise ValueError("Frozen source does not exactly cover the report articles")
        return {version_id: articles[version_id] for version_id in version_ids}

    inputs = load_frozen_v11_assessment_inputs(
        manifest,
        artifact_reader=read_artifact,
        lineage_reader=lineages.__getitem__,
        version_reference_reader=read_article_references,
    )
    return descriptor, manifest, inputs


def load_frozen_v11_assessment_inputs(
    manifest: NewsEvaluationManifest,
    *,
    artifact_reader: ArtifactReader | None = None,
    lineage_reader: Callable[
        [Sha256], NewsEvaluationReportLineage
    ] = read_news_evaluation_report_lineage,
    version_reference_reader: VersionReferenceReader = read_news_evaluation_artifact_references,
) -> tuple[tuple[Sha256, DailySubjectAssessmentInput], ...]:
    read_artifact = artifact_reader or _read_artifact
    reports = {item.report.version_id: item for item in manifest.reports}
    return tuple(
        (
            report_id,
            _load_report_input(
                reports[report_id],
                lineage_reader(report_id),
                read_artifact,
                version_reference_reader,
            ),
        )
        for report_id in V11_REPORT_VERSION_IDS
    )


def run_candidate_arm(
    report_version_id: Sha256,
    value: DailySubjectAssessmentInput,
    *,
    execution_ref: str,
    tier_evaluator: TierEvaluator | None = None,
    ranking_caller: RankingCaller | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_attempt: AttemptSink | None = None,
) -> ArmOutcome:
    started = clock()
    attempts: list[ExperimentAttempt] = []

    def record(attempt: ExperimentAttempt) -> None:
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)

    evaluator = tier_evaluator or evaluate_jev_tier
    states = _tier_states(value)
    rendered = tuple(_render_state(state) for state in states)
    if any(len(item) > CONTEXT_GUARD_CHARACTERS for item in rendered):
        return _unavailable(
            report_version_id, started, clock, attempts, "Jev context exceeds 70000 characters"
        )
    try:
        observations = tuple(evaluator(state, execution_ref, record) for state in states)
    except Exception as error:
        return _unavailable(
            report_version_id,
            started,
            clock,
            attempts,
            f"Jev subject result unavailable: {type(error).__name__}: {error}",
        )
    if {item.subject_id for item in observations} != {item.target_subject_id for item in states}:
        return _unavailable(
            report_version_id, started, clock, attempts, "Jev omitted a subject result"
        )
    fixed: dict[Sha256, Tier] = {item.subject_id: item.tier for item in observations}
    try:
        assessments = _rank_fixed_tiers(
            value,
            fixed,
            execution_ref=execution_ref,
            caller=ranking_caller or _call_openrouter,
            record=record,
            clock=clock,
        )
    except Exception as error:
        return FailedArm(
            arm="candidate",
            report_version_id=report_version_id,
            wall_latency_ms=_elapsed(started, clock()),
            attempts=tuple(attempts),
            error=f"{type(error).__name__}: {error}",
            accounting_complete=all(item.accounting_complete for item in attempts),
        )
    return CompletedArm(
        arm="candidate",
        report_version_id=report_version_id,
        wall_latency_ms=_elapsed(started, clock()),
        attempts=tuple(attempts),
        assessments=assessments,
        accounting_complete=all(item.accounting_complete for item in attempts),
    )


def run_incumbent_arm(
    report_version_id: Sha256,
    value: DailySubjectAssessmentInput,
    *,
    constructor: Callable[[DailySubjectAssessmentInput], DailySubjectAssessmentOutput]
    | None = None,
    execution_ref: str = "offline-incumbent",
    caller: RankingCaller | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_attempt: AttemptSink | None = None,
) -> ArmOutcome:
    started = clock()
    attempts: list[ExperimentAttempt] = []

    def record(attempt: ExperimentAttempt) -> None:
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt)

    try:
        if constructor is not None:
            output = constructor(value)
            attempts.extend(_incumbent_attempts(output))
            assessments = output.assessment_set.assessments
        else:
            assessments = _rank_fixed_tiers(
                value,
                None,
                execution_ref=execution_ref,
                caller=caller or _call_openrouter,
                record=record,
                clock=clock,
            )
    except Exception as error:
        return FailedArm(
            arm="incumbent",
            report_version_id=report_version_id,
            wall_latency_ms=_elapsed(started, clock()),
            attempts=tuple(attempts),
            error=f"{type(error).__name__}: {error}",
            accounting_complete=all(item.accounting_complete for item in attempts),
        )
    return CompletedArm(
        arm="incumbent",
        report_version_id=report_version_id,
        wall_latency_ms=_elapsed(started, clock()),
        attempts=tuple(attempts),
        assessments=assessments,
        accounting_complete=all(item.accounting_complete for item in attempts),
    )


def evaluate_jev_tier(
    state: BinaryTierState,
    execution_ref: str,
    on_attempt: AttemptSink,
    *,
    policy: JevExecutionPolicy = JEV_EXECUTION_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> TierObservation:
    from romanian_news.config import TYPESAFE_API_KEY

    if not TYPESAFE_API_KEY:
        raise ValueError("TYPESAFE_API_KEY is required")
    rendered = _render_state(state)
    request_id = _digest(
        {
            "state": rendered,
            "execution_ref": execution_ref,
            "policy": policy.model_dump(mode="json"),
        }
    )
    questions = DERIVED_BINARY_BENCHMARKS["tier"].questions
    payload = {
        "state": rendered,
        "model": policy.model,
        "questions": {
            item.question_id: {
                "type": "noul",
                "instructions": item.instructions,
                "criteria": {"true": item.true_criteria, "false": item.false_criteria},
            }
            for item in questions
        },
    }
    headers = {"Authorization": f"Bearer {TYPESAFE_API_KEY}", "Content-Type": "application/json"}
    captured: list[ExperimentAttempt] = []
    for index in range(policy.max_attempts):
        attempt_started = clock()
        response: requests.Response | None = None
        try:
            response = requests.post(
                policy.endpoint, headers=headers, json=payload, timeout=policy.timeout_seconds
            )
            response.raise_for_status()
            parsed = JevNoulResponse.model_validate_json(response.content, strict=True)
            main = parsed.answers["main_subject"].noul
            worth = parsed.answers["worth_knowing_if_not_main"].noul
        except Exception as error:
            status = response.status_code if response is not None else None
            retryable = isinstance(error, requests.RequestException) and (
                status is None or status in (408, 429) or status >= 500
            )
            retry = retryable and index + 1 < policy.max_attempts
            attempt = ExperimentAttempt(
                provider="typesafe",
                stage="tier",
                subject_id=state.target_subject_id,
                attempt_number=index + 1,
                status="retryable_error" if retry else "terminal_error",
                request_id=request_id,
                provider_request_id=_header(response, "x-typesafe-request-id"),
                actual_model=None,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_ms=_elapsed(attempt_started, clock()),
                error=BinaryAttemptError(
                    error_type=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    retryable=retryable,
                ),
            )
            captured.append(attempt)
            on_attempt(attempt)
            if not retry:
                raise
            sleep(min(policy.base_retry_delay_seconds * 2**index, policy.max_retry_delay_seconds))
            continue
        cost = (
            Decimal(parsed.usage.input_tokens)
            / Decimal(1_000_000)
            * Decimal(str(policy.input_cost_per_million_tokens_usd))
        )
        attempt = ExperimentAttempt(
            provider="typesafe",
            stage="tier",
            subject_id=state.target_subject_id,
            attempt_number=index + 1,
            status="accepted",
            request_id=request_id,
            provider_request_id=_header(response, "x-typesafe-request-id"),
            actual_model=parsed.model,
            input_tokens=parsed.usage.input_tokens,
            output_tokens=parsed.usage.output_tokens,
            cost_usd=cost,
            latency_ms=_elapsed(attempt_started, clock()),
            error=None,
        )
        captured.append(attempt)
        on_attempt(attempt)
        return TierObservation(
            subject_id=state.target_subject_id,
            main_probability=main,
            worth_knowing_probability=worth,
            tier=compose_binary_tier(main, worth),
            attempts=tuple(captured),
        )
    raise AssertionError("Jev retry loop exhausted")


def _rank_fixed_tiers(
    value: DailySubjectAssessmentInput,
    fixed: Mapping[Sha256, Tier] | None,
    *,
    execution_ref: str,
    caller: RankingCaller,
    record: AttemptSink,
    clock: Callable[[], float],
) -> tuple[SubjectAssessment, ...]:
    aliases = production._aliases(value)
    by_tier: dict[Tier, tuple[str, ...]] | None = None
    if fixed is None:
        schema = production._response_schema(
            tuple(aliases.subjects), tuple(aliases.version_for_article)
        )
        system_prompt = ASSESSMENT_PROMPT
        response_name = "romanian_news_daily_subject_assessments"
    else:
        by_tier = {
            tier: tuple(
                aliases.subject_for_theme[item.id]
                for item in value.theme_set.themes
                if fixed[item.id] == tier
            )
            for tier in cast(tuple[Tier, ...], ("main", "worth_knowing", "excluded"))
        }
        schema = _fixed_tier_schema(by_tier, tuple(aliases.version_for_article))
        system_prompt = (
            ASSESSMENT_PROMPT + " Tier membership is fixed. Only order within each supplied tier."
        )
        response_name = "split_fixed_tier_assessment"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": production._assessment_context(value)},
    ]
    request_id = _digest(
        {"execution_ref": execution_ref, "context": messages[1]["content"], "fixed": by_tier}
    )
    for index in range(2):
        provider_request: ProviderChatRequest = {
            "model": ASSESSMENT_MODEL,
            "messages": cast(list[ChatCompletionMessageParam], messages),
            "temperature": 0,
            "max_tokens": ASSESSMENT_MAX_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "extra_body": {
                "provider": {"require_parameters": True},
                "reasoning": {"effort": "low"},
            },
        }
        attempt_started = clock()
        try:
            response = caller(provider_request)
        except Exception as error:
            record(
                ExperimentAttempt(
                    provider="openrouter",
                    stage="ranking",
                    attempt_number=index + 1,
                    status="terminal_error",
                    request_id=request_id,
                    provider_request_id=None,
                    actual_model=None,
                    input_tokens=None,
                    output_tokens=None,
                    cost_usd=None,
                    latency_ms=_elapsed(attempt_started, clock()),
                    error=BinaryAttemptError(
                        error_type=type(error).__name__,
                        message=str(error) or type(error).__name__,
                        retryable=False,
                    ),
                )
            )
            raise
        payload = response.model_dump(mode="json")
        content = response.choices[0].message.content or ""
        error_text: str | None = None
        accepted = None
        try:
            accepted = production.parse_subject_assessment_response(content, value)
            if by_tier is not None:
                _require_fixed_tiers(accepted, by_tier)
        except (ValidationError, ValueError) as error:
            error_text = str(error)
        usage = payload.get("usage")
        accounting: Mapping[str, object] = (
            cast(Mapping[str, object], usage) if isinstance(usage, Mapping) else {}
        )
        attempt = ExperimentAttempt(
            provider="openrouter",
            stage="ranking",
            attempt_number=index + 1,
            status="accepted" if error_text is None else "rejected",
            request_id=request_id,
            provider_request_id=str(payload.get("id")) if payload.get("id") else None,
            actual_model=str(payload.get("model")) if payload.get("model") else None,
            input_tokens=_int_or_none(accounting.get("prompt_tokens")),
            output_tokens=_int_or_none(accounting.get("completion_tokens")),
            cost_usd=_decimal_or_none(accounting.get("cost")),
            latency_ms=_elapsed(attempt_started, clock()),
            error=None
            if error_text is None
            else BinaryAttemptError(
                error_type="ValidationError", message=error_text, retryable=False
            ),
        )
        record(attempt)
        if accepted is not None and error_text is None:
            return production._freeze_assessments(accepted, value)
        if index == 1:
            raise ValueError(f"Subject assessment remained invalid after correction: {error_text}")
        messages.extend(
            (
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"Return the complete three-tier response. Validation error: {error_text}",
                },
            )
        )
    raise AssertionError("Ranking correction loop exhausted")


def _fixed_tier_schema(
    by_tier: Mapping[Tier, tuple[str, ...]], article_aliases: tuple[str, ...]
) -> dict[str, object]:
    def tier_schema(subjects: tuple[str, ...]) -> dict[str, object]:
        permitted_subjects = subjects or tuple(
            subject for values in by_tier.values() for subject in values
        )
        item = {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "enum": list(permitted_subjects)},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 300},
                "evidence_articles": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(article_aliases)},
                    "minItems": 1,
                },
            },
            "required": ["subject", "rationale", "evidence_articles"],
            "additionalProperties": False,
        }
        return {
            "type": "array",
            "items": item,
            "minItems": len(subjects),
            "maxItems": len(subjects),
        }

    return {
        "type": "object",
        "properties": {tier: tier_schema(subjects) for tier, subjects in by_tier.items()},
        "required": list(by_tier),
        "additionalProperties": False,
    }


def _require_fixed_tiers(response: object, fixed: Mapping[Tier, tuple[str, ...]]) -> None:
    for tier in cast(tuple[Tier, ...], ("main", "worth_knowing", "excluded")):
        observed = tuple(item.subject for item in getattr(response, tier))
        if set(observed) != set(fixed[tier]) or len(observed) != len(fixed[tier]):
            raise ValueError(f"Ranking response changed fixed {tier} tier membership")


def _tier_states(value: DailySubjectAssessmentInput) -> tuple[BinaryTierState, ...]:
    summaries = {item.group_id: item.summary for item in value.summaries}
    evidence = {item.article.version_id: item.evidence_quote for item in value.evidence}
    subjects = tuple(
        production_binary_subject(theme, value, summaries, evidence)
        for theme in value.theme_set.themes
    )
    return tuple(
        BinaryTierState(day=value.day, target_subject_id=item.subject_id, subjects=subjects)
        for item in subjects
    )


def production_binary_subject(theme, value, summaries, evidence):
    from romanian_news.binary_tier_evaluation import (
        BinaryTierEventContext,
        BinaryTierSubjectContext,
    )

    groups = {item.id: item for item in value.theme_set.groups}
    return BinaryTierSubjectContext(
        subject_id=theme.id,
        title=theme.title,
        summary=theme.summary,
        events=tuple(
            BinaryTierEventContext(
                group_id=group_id,
                title=summaries[group_id].title_ro,
                summary=summaries[group_id].summary_ro,
                key_points=summaries[group_id].key_points_ro,
                evidence_quotes=tuple(
                    evidence[item] for item in groups[group_id].article_version_ids
                ),
            )
            for group_id in theme.group_ids
        ),
    )


def _load_report_input(
    report: ReportEvaluationSpec,
    lineage: NewsEvaluationReportLineage,
    read_artifact: ArtifactReader,
    read_version_references: VersionReferenceReader,
) -> DailySubjectAssessmentInput:
    if lineage.report != report.report or lineage.inputs.cluster_set != (report.cluster_set,):
        raise ValueError("Frozen report lineage does not match manifest")
    if report.themes is None or lineage.inputs.themes != (report.themes,):
        raise ValueError("Frozen report has no exact reader-subject themes")
    theme_set = parse_daily_theme_set(read_artifact(report.themes))
    if not isinstance(theme_set, SubjectAssessmentThemeSet):
        raise ValueError("Frozen report themes cannot be assessed as subjects")
    cluster = DailyClusterSet.model_validate_json(read_artifact(report.cluster_set), strict=True)
    if theme_set.cluster_set != report.cluster_set or theme_set.groups != cluster.groups:
        raise ValueError("Frozen theme and cluster inputs differ")
    summaries_by_id = {
        group_id: (reference, summary)
        for reference in lineage.inputs.summary
        for group_id, summary in (_summary_item(reference, read_artifact),)
    }
    summaries = tuple(
        SubjectSummaryInput(
            group_id=group.id,
            reference=summaries_by_id[group.id][0],
            summary=summaries_by_id[group.id][1],
        )
        for group in cluster.groups
    )
    relevance_by_article = {
        article_id: (reference, quote)
        for reference in lineage.inputs.relevance
        for article_id, quote in (_relevance_item(reference, read_artifact),)
    }
    references = read_version_references(cluster.article_version_ids)
    theme_by_article = {
        article: theme.id for theme in theme_set.themes for article in theme.article_version_ids
    }
    group_by_article = {
        article: group.id for group in cluster.groups for article in group.article_version_ids
    }
    relevance = tuple(relevance_by_article[article][0] for article in cluster.article_version_ids)
    evidence = tuple(
        SubjectEvidenceInput(
            theme_id=theme_by_article[article],
            group_id=group_by_article[article],
            article=references[article],
            relevance=relevance_by_article[article][0],
            summary=summaries_by_id[group_by_article[article]][0],
            evidence_quote=relevance_by_article[article][1],
        )
        for article in cluster.article_version_ids
    )
    return DailySubjectAssessmentInput(
        day=cluster.day,
        themes=report.themes,
        theme_set=theme_set,
        summaries=summaries,
        relevance=relevance,
        evidence=evidence,
    )


def _summary_item(
    reference: ArtifactReference, reader: ArtifactReader
) -> tuple[Sha256, GroupSummary]:
    payload = _JSON_OBJECT.validate_json(reader(reference), strict=True)
    group_id = TypeAdapter(Sha256).validate_python(payload["group_id"], strict=True)
    summary = _GROUP_SUMMARY.validate_json(json.dumps(payload["summary"]), strict=True)
    return group_id, summary


def _relevance_item(reference: ArtifactReference, reader: ArtifactReader) -> tuple[Sha256, str]:
    payload = _JSON_OBJECT.validate_json(reader(reference), strict=True)
    decision: RelevanceDecision | ImpactDecision | ContextDecision
    if "decision" in payload:
        decision = RelevanceDecision.model_validate(payload["decision"], strict=True)
    elif isinstance(payload.get("impact"), Mapping):
        decision = ImpactDecision.model_validate(
            cast(Mapping[str, object], payload["impact"])["decision"], strict=True
        )
    else:
        decision = ContextDecision.model_validate(
            cast(Mapping[str, object], payload["context"])["decision"], strict=True
        )
    return cast(Sha256, payload["article_version_id"]), decision.evidence_quote


def _incumbent_attempts(output: DailySubjectAssessmentOutput) -> tuple[ExperimentAttempt, ...]:
    construction = output.assessment_set.construction
    if not isinstance(construction, production.ModelSubjectAssessmentConstruction):
        return ()
    result: list[ExperimentAttempt] = []
    for index, item in enumerate(construction.attempts, 1):
        usage = item.provider_response.get("usage")
        values: Mapping[str, object] = (
            cast(Mapping[str, object], usage) if isinstance(usage, Mapping) else {}
        )
        result.append(
            ExperimentAttempt(
                provider="openrouter",
                stage="incumbent",
                attempt_number=index,
                status=item.status,
                request_id=construction.request_id,
                provider_request_id=item.response_id,
                actual_model=str(item.provider_response.get("model"))
                if item.provider_response.get("model")
                else None,
                input_tokens=_int_or_none(values.get("prompt_tokens")),
                output_tokens=_int_or_none(values.get("completion_tokens")),
                cost_usd=_decimal_or_none(values.get("cost")),
                latency_ms=construction.call.latency_ms if len(construction.attempts) == 1 else 0,
                error=None
                if item.error is None
                else BinaryAttemptError(
                    error_type="ValidationError", message=item.error, retryable=False
                ),
            )
        )
    return tuple(result)


def _unavailable(
    report_id: Sha256,
    started: float,
    clock: Callable[[], float],
    attempts: list[ExperimentAttempt],
    reason: str,
) -> UnavailableArm:
    return UnavailableArm(
        report_version_id=report_id,
        wall_latency_ms=_elapsed(started, clock()),
        attempts=tuple(attempts),
        reason=reason,
        accounting_complete=all(item.accounting_complete for item in attempts),
    )


def _call_openrouter(request: ProviderChatRequest) -> ChatCompletion:
    return openrouter_client(max_retries=0, timeout_seconds=60).chat.completions.create(**request)


def _read_artifact(reference: ArtifactReference) -> bytes:
    return read_verified_r2_object(reference.r2_key, reference.content_digest)


def _descriptor_references(
    descriptor: FrozenV11SourceDescriptor,
) -> tuple[ArtifactReference, ...]:
    references: list[ArtifactReference] = [descriptor.manifest]
    for lineage in descriptor.report_lineages:
        references.append(lineage.report)
        references.extend(lineage.inputs.themes)
        references.extend(lineage.inputs.assessments)
        references.extend(lineage.inputs.cluster_set)
        references.extend(lineage.inputs.relevance)
        references.extend(lineage.inputs.summary)
        references.extend(lineage.inputs.sentiment)
    references.extend(descriptor.article_references)
    return tuple(references)


def _verified_cached_reader(reader: ArtifactReader) -> ArtifactReader:
    cache: dict[Sha256, tuple[ArtifactReference, bytes]] = {}

    def read(reference: ArtifactReference) -> bytes:
        cached = cache.get(reference.version_id)
        if cached is not None:
            if cached[0] != reference:
                raise ValueError(
                    f"Frozen source has conflicting references for {reference.version_id}"
                )
            return cached[1]
        content = reader(reference)
        if hashlib.sha256(content).hexdigest() != reference.content_digest:
            raise ValueError(f"Frozen source object digest mismatch: {reference.r2_key}")
        cache[reference.version_id] = (reference, content)
        return content

    return read


def _render_state(state: BinaryTierState) -> str:
    return json.dumps(
        state.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _digest(value: object) -> Sha256:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _elapsed(started: float, finished: float) -> int:
    if not all(math.isfinite(item) for item in (started, finished)) or finished < started:
        raise ValueError("Invalid experiment latency clock")
    return round((finished - started) * 1000)


def _header(response: requests.Response | None, name: str) -> str | None:
    return response.headers.get(name) if response is not None else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _decimal_or_none(value: object) -> Decimal | None:
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    return None
