from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from typing import Annotated, Literal

from openai.types.chat import ChatCompletionMessageParam
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.attempts import ModelCall, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.analysis.relevance import RelevanceDecision
from romanian_news.analysis.relevance_v3 import ImpactDecision
from romanian_news.analysis.tracing import ProviderChatRequest, trace_provider_call
from romanian_news.groups import DailyClusterSet
from romanian_news.storage import read_verified_r2_object
from romanian_news.themes import (
    AliasedReaderSubjectThemeSet,
    ReaderSubjectDailyThemeSet,
    parse_daily_theme_set,
)

ASSESSMENT_OPERATION = "news.assess_daily_subjects"
ASSESSMENT_MODEL = "google/gemini-3.8-flash"
ASSESSMENT_PROMPT = (
    "Assess every supplied Romanian news reader subject by national consequence for Romania. "
    "Place each subject exactly once in main, worth_knowing, or excluded. "
    "Order each tier from most to least consequential. "
    "Main is for consequential national developments: government formation and political "
    "crisis, major economic or social shifts, and persistent national financial signals such "
    "as state-bond demand, deposit or market interest rates, stock market movements, and leu "
    "exchange-rate moves. "
    "Worth knowing is for useful low-impact subjects, including low-impact side political "
    "topics, single statistics, and regional news. "
    "Excluded is only for subjects irrelevant to Romanian national economics or politics, and "
    "for opinion or commentary whose underlying event the evidence does not corroborate. "
    "Unconfirmed reporting does not gain rank before corroboration. A disclosed uncertainty "
    "makes the key claim unconfirmed even when a title states it as fact. A subject with an "
    "unconfirmed key claim, such as a single-source rumor, never holds the first main "
    "position: rank it below every corroborated subject in main. "
    "Keep consequence separate from grouping, relevance, confidence, corroboration, and "
    "presentation. "
    "Article count alone does not define importance. Use only supplied evidence. "
    "Give one concise consequence rationale in English and cite one or more supplied article aliases."
)
ASSESSMENT_MAX_TOKENS = 16000
_TIER_ORDER = {"main": 0, "worth_knowing": 1, "excluded": 2}
_SHA256_ADAPTER = TypeAdapter(Sha256)

Tier = Literal["main", "worth_knowing", "excluded"]


class SubjectAssessmentPolicy(NewsModel):
    policy_id: Literal["daily-subject-consequence-v1"]
    model: Literal["google/gemini-3.8-flash"]
    prompt_digest: Sha256
    input_policy: Literal["ordered-schema-v3-themes-summaries-relevance-v1"]
    tier_policy: Literal["ordered-three-tier-complete-partition-v1"]
    correction_policy: Literal["complete-json-once-v1"]
    reasoning_effort: Literal["low"]
    temperature: Literal[0]
    max_tokens: Annotated[int, Field(gt=0)]


PRODUCTION_SUBJECT_ASSESSMENT_POLICY = SubjectAssessmentPolicy(
    policy_id="daily-subject-consequence-v1",
    model=ASSESSMENT_MODEL,
    prompt_digest=hashlib.sha256(ASSESSMENT_PROMPT.encode()).hexdigest(),
    input_policy="ordered-schema-v3-themes-summaries-relevance-v1",
    tier_policy="ordered-three-tier-complete-partition-v1",
    correction_policy="complete-json-once-v1",
    reasoning_effort="low",
    temperature=0,
    max_tokens=ASSESSMENT_MAX_TOKENS,
)


class SubjectSummaryInput(NewsModel):
    group_id: Sha256
    reference: ArtifactReference
    summary: GroupSummary


class SubjectEvidenceInput(NewsModel):
    theme_id: Sha256
    group_id: Sha256
    article: ArtifactReference
    relevance: ArtifactReference
    summary: ArtifactReference
    evidence_quote: Annotated[str, Field(min_length=1)]


class DailySubjectAssessmentInput(NewsModel):
    day: date
    themes: ArtifactReference
    theme_set: ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet
    summaries: tuple[SubjectSummaryInput, ...]
    relevance: tuple[ArtifactReference, ...]
    evidence: tuple[SubjectEvidenceInput, ...]

    @model_validator(mode="after")
    def require_exact_inputs(self) -> DailySubjectAssessmentInput:
        _require_theme_input_alignment(self)
        _require_evidence_coverage(self)
        _require_evidence_membership(self)
        return self


def _require_theme_input_alignment(value: DailySubjectAssessmentInput) -> None:
    if value.theme_set.day != value.day:
        raise ValueError("Subject assessment themes belong to another day")
    if value.theme_set.summary_inputs != tuple(item.reference for item in value.summaries):
        raise ValueError("Subject assessment summaries must preserve exact theme input order")
    if tuple(item.group_id for item in value.summaries) != tuple(
        group.id for group in value.theme_set.groups
    ):
        raise ValueError("Subject assessment summaries must exactly cover theme groups")


def _require_evidence_coverage(value: DailySubjectAssessmentInput) -> None:
    article_ids = tuple(
        item for theme in value.theme_set.themes for item in theme.article_version_ids
    )
    evidence_ids = tuple(item.article.version_id for item in value.evidence)
    if len(evidence_ids) != len(set(evidence_ids)) or set(evidence_ids) != set(article_ids):
        raise ValueError("Subject assessment evidence must exactly cover theme articles")
    if tuple(item.relevance for item in value.evidence) != value.relevance:
        raise ValueError("Subject assessment evidence must preserve relevance input order")


def _require_evidence_membership(value: DailySubjectAssessmentInput) -> None:
    themes = {theme.id: theme for theme in value.theme_set.themes}
    groups = {group.id: group for group in value.theme_set.groups}
    summaries = {item.group_id: item.reference for item in value.summaries}
    for item in value.evidence:
        theme = themes.get(item.theme_id)
        group = groups.get(item.group_id)
        if theme is None or group is None:
            raise ValueError("Subject assessment evidence references an unknown subject or group")
        if item.article.version_id not in theme.article_version_ids:
            raise ValueError(
                "Subject assessment evidence references an article outside its subject"
            )
        if item.article.version_id not in group.article_version_ids:
            raise ValueError("Subject assessment evidence references an article outside its group")
        if summaries[item.group_id] != item.summary:
            raise ValueError("Subject assessment evidence references another group summary")


class SubjectAssessmentMessage(NewsModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SubjectAssessmentAttemptEvidence(NewsModel):
    attempt_id: Sha256
    response_id: Annotated[str, Field(min_length=1)]
    status: Literal["accepted", "rejected"]
    error: str | None
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]


class ModelSubjectAssessmentConstruction(NewsModel):
    kind: Literal["model"] = "model"
    request_id: Sha256
    messages: Annotated[tuple[SubjectAssessmentMessage, ...], Field(min_length=2)]
    input_digest: Sha256
    response_schema_digest: Sha256
    call: ModelCall
    attempts: Annotated[
        tuple[SubjectAssessmentAttemptEvidence, ...], Field(min_length=1, max_length=2)
    ]


class EmptySubjectAssessmentConstruction(NewsModel):
    kind: Literal["empty"] = "empty"


SubjectAssessmentConstruction = Annotated[
    EmptySubjectAssessmentConstruction | ModelSubjectAssessmentConstruction,
    Field(discriminator="kind"),
]


class SubjectAssessmentEvidence(NewsModel):
    group_id: Sha256
    article: ArtifactReference
    relevance: ArtifactReference
    summary: ArtifactReference
    evidence_quote: Annotated[str, Field(min_length=1)]


class SubjectAssessment(NewsModel):
    theme_id: Sha256
    group_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    article_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    tier: Tier
    semantic_rank: Annotated[int, Field(ge=1)]
    rationale: Annotated[str, Field(min_length=1, max_length=300)]
    evidence: Annotated[tuple[SubjectAssessmentEvidence, ...], Field(min_length=1)]


class DailySubjectAssessmentSet(NewsModel):
    schema_version: Literal[1] = 1
    day: date
    request_id: Sha256
    policy: SubjectAssessmentPolicy
    policy_digest: Sha256
    themes: ArtifactReference
    summary_inputs: tuple[ArtifactReference, ...]
    relevance_inputs: tuple[ArtifactReference, ...]
    construction: SubjectAssessmentConstruction
    subject_ids: tuple[Sha256, ...]
    assessments: tuple[SubjectAssessment, ...]

    @model_validator(mode="after")
    def require_complete_partition(self) -> DailySubjectAssessmentSet:
        _require_assessment_identity(self)
        _require_assessment_coverage(self)
        _require_assessment_evidence(self)
        _require_tier_ranks(self)
        _require_construction_match(self)
        return self


def _require_assessment_identity(value: DailySubjectAssessmentSet) -> None:
    if value.policy_digest != subject_assessment_policy_digest(value.policy):
        raise ValueError("Subject assessment policy digest does not match its policy")
    expected_request = _request_identity(
        value.day,
        value.themes,
        value.summary_inputs,
        value.relevance_inputs,
        value.policy,
    )
    if value.request_id != expected_request:
        raise ValueError("Subject assessment request identity does not match exact inputs")


def _require_assessment_coverage(value: DailySubjectAssessmentSet) -> None:
    theme_ids = tuple(item.theme_id for item in value.assessments)
    if len(value.subject_ids) != len(set(value.subject_ids)):
        raise ValueError("Subject assessment source subjects must be unique")
    if set(theme_ids) != set(value.subject_ids):
        raise ValueError("Subject assessments must exactly cover final reader subjects")
    if len(theme_ids) != len(set(theme_ids)):
        raise ValueError("Each reader subject must have exactly one assessment")


def _require_assessment_evidence(value: DailySubjectAssessmentSet) -> None:
    summary_ids = set(value.summary_inputs)
    relevance_ids = set(value.relevance_inputs)
    for assessment in value.assessments:
        if len(assessment.article_version_ids) != len(set(assessment.article_version_ids)):
            raise ValueError("Subject article membership must be unique")
        for evidence in assessment.evidence:
            if evidence.group_id not in assessment.group_ids:
                raise ValueError("Assessment evidence group is outside its subject")
            if evidence.article.version_id not in assessment.article_version_ids:
                raise ValueError("Assessment evidence article is outside its subject")
            if evidence.summary not in summary_ids or evidence.relevance not in relevance_ids:
                raise ValueError("Assessment evidence does not use exact stored inputs")


def _require_tier_ranks(value: DailySubjectAssessmentSet) -> None:
    for tier in _TIER_ORDER:
        ranks = tuple(item.semantic_rank for item in value.assessments if item.tier == tier)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("Subject assessment ranks must follow ordered tier positions")


def _require_construction_match(value: DailySubjectAssessmentSet) -> None:
    if value.assessments and not isinstance(value.construction, ModelSubjectAssessmentConstruction):
        raise ValueError("Non-empty subject assessments require model evidence")
    if not value.assessments and not isinstance(
        value.construction, EmptySubjectAssessmentConstruction
    ):
        raise ValueError("Empty subject assessments cannot have model evidence")


class DailySubjectAssessmentOutput(NewsModel):
    assessment_set: DailySubjectAssessmentSet
    content_digest: Sha256
    content: bytes


class _ProposedSubjectAssessment(NewsModel):
    subject: Annotated[str, Field(pattern=r"^subject_[0-9]{2,}$")]
    rationale: Annotated[str, Field(min_length=1, max_length=300)]
    evidence_articles: Annotated[
        tuple[Annotated[str, Field(pattern=r"^article_[0-9]{2,}$")], ...],
        Field(min_length=1),
    ]


class _SubjectAssessmentResponse(NewsModel):
    main: tuple[_ProposedSubjectAssessment, ...]
    worth_knowing: tuple[_ProposedSubjectAssessment, ...]
    excluded: tuple[_ProposedSubjectAssessment, ...]


class SubjectAssessmentCorrectionExhausted(ValueError):
    """Both subject assessment responses failed validation."""


def subject_assessment_policy_digest(policy: SubjectAssessmentPolicy) -> Sha256:
    """Identify every assessment generation and acceptance choice."""
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def subject_assessment_request_id(
    value: DailySubjectAssessmentInput,
    policy: SubjectAssessmentPolicy = PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
) -> Sha256:
    """Identify one assessment request from exact ordered immutable inputs."""
    return _request_identity(
        value.day,
        value.themes,
        tuple(item.reference for item in value.summaries),
        value.relevance,
        policy,
    )


def subject_assessment_run_id(request_id: Sha256, implementation_ref: str) -> Sha256:
    """Identify one production attempt by request and implementation."""
    if not implementation_ref:
        raise ValueError("Implementation reference is required")
    return _sha256(f"{request_id}\0{implementation_ref}".encode())


def read_daily_subject_assessment_input(day: date) -> DailySubjectAssessmentInput:
    """Load exact current schema-v3 themes, summaries, and relevance evidence."""
    from romanian_news.catalog.artifacts import artifact_references_by_version_ids
    from romanian_news.catalog.themes import read_daily_theme_reference

    themes = read_daily_theme_reference(day)
    theme_set = parse_daily_theme_set(read_verified_r2_object(themes.r2_key, themes.content_digest))
    if not isinstance(theme_set, ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet):
        raise ValueError("Subject assessment requires reader subjects from schema version 3 or 4")
    cluster_set = DailyClusterSet.model_validate_json(
        read_verified_r2_object(
            theme_set.cluster_set.r2_key,
            theme_set.cluster_set.content_digest,
        ),
        strict=True,
    )
    references = artifact_references_by_version_ids(
        (*cluster_set.article_version_ids, *cluster_set.relevance_version_ids)
    )
    summaries = tuple(
        SubjectSummaryInput(
            group_id=group.id,
            reference=reference,
            summary=_read_summary(reference, group.id),
        )
        for group, reference in zip(theme_set.groups, theme_set.summary_inputs, strict=True)
    )
    theme_by_article = {
        article_id: theme.id
        for theme in theme_set.themes
        for article_id in theme.article_version_ids
    }
    group_by_article = {
        article_id: group.id
        for group in theme_set.groups
        for article_id in group.article_version_ids
    }
    summary_by_group = {item.group_id: item.reference for item in summaries}
    relevance = tuple(
        _artifact_reference(references[version_id])
        for version_id in cluster_set.relevance_version_ids
    )
    evidence = tuple(
        SubjectEvidenceInput(
            theme_id=theme_by_article[article_id],
            group_id=group_by_article[article_id],
            article=_artifact_reference(references[article_id]),
            relevance=relevance_reference,
            summary=summary_by_group[group_by_article[article_id]],
            evidence_quote=_read_relevance_quote(relevance_reference, article_id),
        )
        for article_id, relevance_reference in zip(
            cluster_set.article_version_ids, relevance, strict=True
        )
    )
    return DailySubjectAssessmentInput(
        day=day,
        themes=themes,
        theme_set=theme_set,
        summaries=summaries,
        relevance=relevance,
        evidence=evidence,
    )


def construct_daily_subject_assessments(
    value: DailySubjectAssessmentInput,
    policy: SubjectAssessmentPolicy = PRODUCTION_SUBJECT_ASSESSMENT_POLICY,
) -> DailySubjectAssessmentOutput:
    """Assess one complete day of final reader subjects."""
    request_id = subject_assessment_request_id(value, policy)
    policy_digest = subject_assessment_policy_digest(policy)
    if not value.theme_set.themes:
        return _output(
            DailySubjectAssessmentSet(
                day=value.day,
                request_id=request_id,
                policy=policy,
                policy_digest=policy_digest,
                themes=value.themes,
                summary_inputs=tuple(item.reference for item in value.summaries),
                relevance_inputs=value.relevance,
                construction=EmptySubjectAssessmentConstruction(),
                subject_ids=(),
                assessments=(),
            )
        )
    aliases = _aliases(value)
    response_schema = _response_schema(tuple(aliases.subjects), tuple(aliases.version_for_article))
    messages = [
        SubjectAssessmentMessage(role="system", content=ASSESSMENT_PROMPT),
        SubjectAssessmentMessage(role="user", content=_assessment_context(value)),
    ]
    attempts: list[SubjectAssessmentAttemptEvidence] = []
    responses = []
    accepted: _SubjectAssessmentResponse | None = None
    started = time.monotonic()
    model_request_id = _model_request_id(request_id, tuple(messages), response_schema, policy)
    for attempt_index in range(2):
        provider_inputs: ProviderChatRequest = {
            "model": policy.model,
            "messages": [_message_param(message) for message in messages],
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "romanian_news_daily_subject_assessments",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "extra_body": {
                "provider": {"require_parameters": True},
                "reasoning": {"effort": policy.reasoning_effort},
            },
        }
        attempt_started = time.monotonic()
        provider_call = trace_provider_call(
            ASSESSMENT_OPERATION,
            model_request_id,
            provider_inputs,
            lambda provider_inputs=provider_inputs: (
                openrouter_client().chat.completions.create(**provider_inputs)
            ),
        )
        response = provider_call.response
        responses.append(response)
        content = response.choices[0].message.content or ""
        error_text = None
        rejection: ValidationError | ValueError | None = None
        try:
            accepted = parse_subject_assessment_response(content, value)
        except (ValidationError, ValueError) as error:
            rejection = error
            error_text = str(error)
        status: Literal["accepted", "rejected"] = "accepted" if error_text is None else "rejected"
        recorded = record_model_attempt(
            response,
            request_id=model_request_id,
            operation_key=ASSESSMENT_OPERATION,
            attempt_index=attempt_index,
            latency_ms=round((time.monotonic() - attempt_started) * 1000),
            status=status,
            error=error_text,
            fallback_response_id=str(provider_call.call_id),
            trace=provider_call.trace,
        )
        attempts.append(
            SubjectAssessmentAttemptEvidence(
                attempt_id=recorded.attempt_id,
                response_id=recorded.response_id,
                status=status,
                error=error_text,
                response_content=content,
                response_content_digest=_sha256(content.encode()),
                provider_response=response.model_dump(mode="json"),
            )
        )
        if accepted is not None and error_text is None:
            break
        if attempt_index == 1:
            assert rejection is not None
            raise SubjectAssessmentCorrectionExhausted(
                f"Subject assessment remained invalid after correction: {error_text}"
            ) from rejection
        messages.extend(
            (
                SubjectAssessmentMessage(role="assistant", content=content),
                SubjectAssessmentMessage(
                    role="user",
                    content=(
                        "The response was invalid. Return the complete three-tier assessment. "
                        f"Validation error: {error_text}"
                    ),
                ),
            )
        )
    if accepted is None:
        raise RuntimeError("Subject assessment correction loop did not return")
    construction = ModelSubjectAssessmentConstruction(
        request_id=model_request_id,
        messages=tuple(messages),
        input_digest=_messages_digest(tuple(messages)),
        response_schema_digest=_sha256(_canonical_json(response_schema)),
        call=ModelCall(
            response_id=attempts[-1].response_id,
            model=str(responses[-1].model),
            input_tokens=sum(item.usage.prompt_tokens if item.usage else 0 for item in responses),
            output_tokens=sum(
                item.usage.completion_tokens if item.usage else 0 for item in responses
            ),
            latency_ms=round((time.monotonic() - started) * 1000),
        ),
        attempts=tuple(attempts),
    )
    return _output(
        DailySubjectAssessmentSet(
            day=value.day,
            request_id=request_id,
            policy=policy,
            policy_digest=policy_digest,
            themes=value.themes,
            summary_inputs=tuple(item.reference for item in value.summaries),
            relevance_inputs=value.relevance,
            construction=construction,
            subject_ids=tuple(theme.id for theme in value.theme_set.themes),
            assessments=_freeze_assessments(accepted, value),
        )
    )


def parse_subject_assessment_response(
    content: str,
    value: DailySubjectAssessmentInput,
) -> _SubjectAssessmentResponse:
    """Parse and validate one model response against exact supplied subjects and evidence."""
    response = _SubjectAssessmentResponse.model_validate_json(content, strict=True)
    proposed = tuple(
        item for tier in (response.main, response.worth_knowing, response.excluded) for item in tier
    )
    aliases = _aliases(value)
    observed_subjects = tuple(item.subject for item in proposed)
    if len(observed_subjects) != len(set(observed_subjects)) or set(observed_subjects) != set(
        aliases.subjects
    ):
        raise ValueError("Assessment response must contain every supplied subject exactly once")
    article_subject = {
        aliases.article_for_version[item.article.version_id]: aliases.subject_for_theme[
            item.theme_id
        ]
        for item in value.evidence
    }
    for item in proposed:
        if len(item.evidence_articles) != len(set(item.evidence_articles)):
            raise ValueError("Assessment evidence article aliases must be unique")
        for article_alias in item.evidence_articles:
            if article_subject.get(article_alias) != item.subject:
                raise ValueError("Assessment evidence must belong to the assessed subject")
    return response


def parse_daily_subject_assessment_set(content: bytes) -> DailySubjectAssessmentSet:
    """Parse one durable subject assessment artifact at its owning boundary."""
    return DailySubjectAssessmentSet.model_validate_json(content, strict=True)


def subject_order_key(assessment: SubjectAssessment) -> tuple[int, int]:
    """Return the complete semantic order without a technical identifier tie-break."""
    return _TIER_ORDER[assessment.tier], assessment.semantic_rank


def resolve_group_anchor(
    theme_set: ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet,
    group_id: Sha256,
) -> Sha256:
    """Resolve one stable group anchor to its trial-specific final reader subject."""
    matches = tuple(theme.id for theme in theme_set.themes if group_id in theme.group_ids)
    if len(matches) != 1:
        raise ValueError(f"Group anchor does not resolve to one final subject: {group_id}")
    return matches[0]


def compare_subject_anchors(
    assessment_set: DailySubjectAssessmentSet,
    theme_set: ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet,
    higher_group_id: Sha256,
    lower_group_id: Sha256,
) -> bool:
    """Compare stable anchors after final subject construction."""
    higher_theme_id = resolve_group_anchor(theme_set, higher_group_id)
    lower_theme_id = resolve_group_anchor(theme_set, lower_group_id)
    if higher_theme_id == lower_theme_id:
        raise ValueError(
            "Ranking anchors resolve to one final subject: "
            f"{higher_group_id}, {lower_group_id}, subject={higher_theme_id}"
        )
    assessments = {item.theme_id: item for item in assessment_set.assessments}
    try:
        higher_key = subject_order_key(assessments[higher_theme_id])
        lower_key = subject_order_key(assessments[lower_theme_id])
    except KeyError as error:
        raise ValueError("Ranking comparison subject has no assessment") from error
    if higher_key == lower_key:
        raise ValueError(
            "Compared subjects tie before the technical ID: "
            f"{higher_theme_id}, {lower_theme_id}, key={higher_key}"
        )
    return higher_key < lower_key


def _freeze_assessments(
    response: _SubjectAssessmentResponse,
    value: DailySubjectAssessmentInput,
) -> tuple[SubjectAssessment, ...]:
    evidence = {item.article.version_id: item for item in value.evidence}
    themes = {item.id: item for item in value.theme_set.themes}
    aliases = _aliases(value)
    result = []
    tier_lists: tuple[tuple[Tier, tuple[_ProposedSubjectAssessment, ...]], ...] = (
        ("main", response.main),
        ("worth_knowing", response.worth_knowing),
        ("excluded", response.excluded),
    )
    for tier, items in tier_lists:
        for rank, item in enumerate(items, start=1):
            theme_id = aliases.theme_for_subject[item.subject]
            result.append(
                SubjectAssessment(
                    theme_id=theme_id,
                    group_ids=themes[theme_id].group_ids,
                    article_version_ids=themes[theme_id].article_version_ids,
                    tier=tier,
                    semantic_rank=rank,
                    rationale=" ".join(item.rationale.split()),
                    evidence=tuple(
                        SubjectAssessmentEvidence(
                            group_id=evidence[article_id].group_id,
                            article=evidence[article_id].article,
                            relevance=evidence[article_id].relevance,
                            summary=evidence[article_id].summary,
                            evidence_quote=evidence[article_id].evidence_quote,
                        )
                        for article_id in (
                            aliases.version_for_article[article_alias]
                            for article_alias in item.evidence_articles
                        )
                    ),
                )
            )
    return tuple(result)


def _request_identity(
    day: date,
    themes: ArtifactReference,
    summaries: tuple[ArtifactReference, ...],
    relevance: tuple[ArtifactReference, ...],
    policy: SubjectAssessmentPolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "operation": ASSESSMENT_OPERATION,
                "day": day.isoformat(),
                "theme_version_id": themes.version_id,
                "ordered_summary_version_ids": [item.version_id for item in summaries],
                "ordered_relevance_version_ids": [item.version_id for item in relevance],
                "policy_digest": subject_assessment_policy_digest(policy),
                "response_schema_digest": _sha256(
                    _canonical_json(_SubjectAssessmentResponse.model_json_schema())
                ),
                "assessment_set_schema_digest": _sha256(
                    _canonical_json(DailySubjectAssessmentSet.model_json_schema())
                ),
            }
        )
    )


def _model_request_id(
    parent_request_id: Sha256,
    messages: tuple[SubjectAssessmentMessage, ...],
    response_schema: dict[str, object],
    policy: SubjectAssessmentPolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "parent_request_id": parent_request_id,
                "messages_digest": _messages_digest(messages),
                "response_schema_digest": _sha256(_canonical_json(response_schema)),
                "provider_request": {
                    "model": policy.model,
                    "temperature": policy.temperature,
                    "max_tokens": policy.max_tokens,
                    "reasoning_effort": policy.reasoning_effort,
                },
            }
        )
    )


class _AssessmentAliases(NewsModel):
    subjects: tuple[str, ...]
    subject_for_theme: dict[Sha256, str]
    theme_for_subject: dict[str, Sha256]
    event_for_group: dict[Sha256, str]
    article_for_version: dict[Sha256, str]
    version_for_article: dict[str, Sha256]


def _aliases(value: DailySubjectAssessmentInput) -> _AssessmentAliases:
    subject_width = max(2, len(str(len(value.theme_set.themes))))
    article_width = max(2, len(str(len(value.evidence))))
    subjects = tuple(
        f"subject_{position:0{subject_width}d}"
        for position, _ in enumerate(value.theme_set.themes, 1)
    )
    subject_for_theme = dict(
        zip((theme.id for theme in value.theme_set.themes), subjects, strict=True)
    )
    articles = tuple(
        f"article_{position:0{article_width}d}" for position, _ in enumerate(value.evidence, 1)
    )
    article_for_version = dict(
        zip((item.article.version_id for item in value.evidence), articles, strict=True)
    )
    return _AssessmentAliases(
        subjects=subjects,
        subject_for_theme=subject_for_theme,
        theme_for_subject={alias: theme_id for theme_id, alias in subject_for_theme.items()},
        event_for_group={
            group.id: f"event_{position:0{max(2, len(str(len(value.theme_set.groups))))}d}"
            for position, group in enumerate(value.theme_set.groups, 1)
        },
        article_for_version=article_for_version,
        version_for_article={
            alias: version_id for version_id, alias in article_for_version.items()
        },
    )


def _assessment_context(value: DailySubjectAssessmentInput) -> str:
    summaries = {item.group_id: item.summary for item in value.summaries}
    evidence = {item.article.version_id: item for item in value.evidence}
    groups = {group.id: group for group in value.theme_set.groups}
    aliases = _aliases(value)
    return json.dumps(
        {
            "day": value.day.isoformat(),
            "subjects": [
                {
                    "subject": aliases.subject_for_theme[theme.id],
                    "title": theme.title,
                    "summary": theme.summary,
                    "events": [
                        {
                            "event": aliases.event_for_group[group_id],
                            "title": summaries[group_id].title_ro,
                            "summary": summaries[group_id].summary_ro,
                            "key_points": summaries[group_id].key_points_ro,
                            "uncertainty": summaries[group_id].uncertainty_ro,
                            "articles": [
                                {
                                    "article": aliases.article_for_version[article_id],
                                    "relevance_evidence_quote": evidence[article_id].evidence_quote,
                                }
                                for article_id in groups[group_id].article_version_ids
                            ],
                        }
                        for group_id in theme.group_ids
                    ],
                }
                for theme in value.theme_set.themes
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _response_schema(
    subject_aliases: tuple[str, ...], article_aliases: tuple[str, ...]
) -> dict[str, object]:
    item = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "enum": list(subject_aliases)},
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
        "type": "object",
        "properties": {tier: {"type": "array", "items": item} for tier in _TIER_ORDER},
        "required": list(_TIER_ORDER),
        "additionalProperties": False,
    }


def _read_summary(reference: ArtifactReference, group_id: Sha256) -> GroupSummary:
    payload = json.loads(read_verified_r2_object(reference.r2_key, reference.content_digest))
    if payload.get("group_id") != group_id:
        raise ValueError("Subject assessment summary references another group")
    return GroupSummary.model_validate_json(
        json.dumps(payload.get("summary"), ensure_ascii=False), strict=True
    )


def _read_relevance_quote(reference: ArtifactReference, article_id: Sha256) -> str:
    payload = json.loads(read_verified_r2_object(reference.r2_key, reference.content_digest))
    if payload.get("article_version_id") != article_id:
        raise ValueError("Subject assessment relevance references another article")
    if "decision" in payload:
        decision = RelevanceDecision.model_validate(payload["decision"], strict=True)
    else:
        impact = payload.get("impact")
        if not isinstance(impact, dict) or "decision" not in impact:
            raise ValueError("Subject assessment relevance has no impact evidence")
        decision = ImpactDecision.model_validate(impact["decision"], strict=True)
    return decision.evidence_quote


def _artifact_reference(value: object) -> ArtifactReference:
    return ArtifactReference.model_validate(value, from_attributes=True)


def _message_param(message: SubjectAssessmentMessage) -> ChatCompletionMessageParam:
    if message.role == "system":
        return {"role": "system", "content": message.content}
    if message.role == "assistant":
        return {"role": "assistant", "content": message.content}
    return {"role": "user", "content": message.content}


def _messages_digest(messages: tuple[SubjectAssessmentMessage, ...]) -> Sha256:
    return _sha256(_canonical_json([item.model_dump(mode="json") for item in messages]))


def _output(value: DailySubjectAssessmentSet) -> DailySubjectAssessmentOutput:
    content = _canonical_json(value.model_dump(mode="json"))
    return DailySubjectAssessmentOutput(
        assessment_set=value,
        content_digest=_sha256(content),
        content=content,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
