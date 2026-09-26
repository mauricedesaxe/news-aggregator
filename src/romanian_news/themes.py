from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Annotated, Literal, TypeVar

from pydantic import Field, TypeAdapter, model_validator

from romanian_news import GROUP_ANALYSIS_MODEL, NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.corrected_structured import (
    StructuredMessage,
    run_corrected_structured_openrouter,
)
from romanian_news.analysis.groups.models import GroupSummary
from romanian_news.artifacts import ArtifactReference
from romanian_news.groups import NewsGroup, parse_daily_cluster_set
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256
from romanian_news.storage import read_verified_r2_object

THEME_PROMPT = (
    "Organize all supplied Romanian news event groups into coherent daily reader themes. "
    "Write the title and summary in English. Return every group ID exactly once. "
    "Combine groups only when they cover the same event or a direct development. "
    "A shared category such as politics, economics, health, or infrastructure is not enough. "
    "When in doubt, keep groups separate. "
    "Preserve the supplied order when listing group IDs. "
    "Use a concise title and a one or two sentence summary."
)
THEME_ASSIGNMENT_PROMPT = (
    "Assign each supplied Romanian news event group to an integer theme label. "
    "Use the same label only when groups cover the same event or a direct development. "
    "A shared category such as politics, economics, health, or infrastructure is not enough. "
    "When in doubt, use separate labels. Return only assignments keyed by supplied group ID."
)
READER_SUBJECT_ASSIGNMENT_PROMPT = (
    "Assign each supplied Romanian news event group to an integer reader-subject label. "
    "Use the same label when groups are meaningfully related parts of one specific subject that "
    "a reader benefits from understanding together, even when they describe distinct events. "
    "The relationship may connect institutions, policies, public services, or financial signals, "
    "and it does not imply one incident or a causal chain. For example, parallel financial strain "
    "at two public transport operators in the same city is one public-transport subject. Several "
    "stock-index movements and exchange developments are one capital-market subject. Tax policy "
    "and tax-enforcement developments can form one fiscal-policy subject. Fuel and electricity "
    "price pressures can form one household-energy-cost subject. Economic contraction and public "
    "debt can form one macro-fiscal-pressure subject. European defense policy and Romania's US "
    "security partnership can form one international-defense-posture subject. A broad category "
    "such as politics, economics, transport, energy, or health is not enough by itself. Keep a "
    "political appointment separate from a currency move, and keep a commute survey separate from "
    "operator finances. Keep labor-cost changes separate from public debt and broad macroeconomic "
    "signals unless one stated policy directly connects them. Preserve exact event boundaries. "
    "Return only assignments keyed by supplied group ID."
)
THEME_PROSE_PROMPT = (
    "Write the title and summary in English for each supplied merged daily theme. "
    "Use only the supplied event information and cover every event in the theme. "
    "Do not add facts. Return exactly the supplied theme keys with only title and summary. "
    "Use a concise title and a one or two sentence summary."
)
THEME_MAX_TOKENS = 4000
THEME_INPUT_POLICY = "ordered-group-summaries-and-article-ids-v1"
THEME_ASSIGNMENT_POLICY = "exact-partition-ordered-v1"
THEME_CORRECTION_POLICY = "complete-json-once-v1"
THEME_OPERATION = "news.construct_daily_themes"
THEME_ASSIGNMENT_OPERATION = f"{THEME_OPERATION}.assignment"
THEME_PROSE_OPERATION = f"{THEME_OPERATION}.merged_prose"
THEME_MODEL = "google/gemini-3.8-flash"
_ParsedStage = TypeVar("_ParsedStage")


class DailyThemeCorrectionExhaustedError(ValueError):
    """Report that both model responses failed daily theme validation."""


class ThemePolicy(NewsModel):
    policy_id: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    prompt_digest: Sha256
    input_policy: Literal["ordered-group-summaries-and-article-ids-v1"]
    assignment_policy: Literal["exact-partition-ordered-v1"]
    correction_policy: Literal["complete-json-once-v1"]
    temperature: Literal[0]
    max_tokens: Annotated[int, Field(gt=0)]


LEGACY_THEME_POLICY = ThemePolicy(
    policy_id="daily-reader-themes-v1",
    model=GROUP_ANALYSIS_MODEL,
    prompt_digest=hashlib.sha256(THEME_PROMPT.encode()).hexdigest(),
    input_policy=THEME_INPUT_POLICY,
    assignment_policy=THEME_ASSIGNMENT_POLICY,
    correction_policy=THEME_CORRECTION_POLICY,
    temperature=0,
    max_tokens=THEME_MAX_TOKENS,
)


class SparseThemePolicy(NewsModel):
    policy_id: Literal["daily-reader-themes-sparse-v2"]
    model: Literal["google/gemini-3.8-flash"]
    assignment_prompt_digest: Sha256
    prose_prompt_digest: Sha256
    input_policy: Literal["ordered-group-summaries-and-article-ids-v1"]
    assignment_policy: Literal["integer-equivalence-first-occurrence-v1"]
    singleton_policy: Literal["source-title-first-two-sentences-v1"]
    prose_policy: Literal["merged-only-exact-key-prose-v1"]
    correction_policy: Literal["complete-json-once-v1"]
    reasoning_effort: Literal["low"]
    temperature: Literal[0]
    max_tokens: Annotated[int, Field(gt=0)]


class ReaderSubjectThemePolicy(NewsModel):
    policy_id: Literal["daily-reader-subjects-v3"]
    model: Literal["google/gemini-3.8-flash"]
    assignment_prompt_digest: Sha256
    prose_prompt_digest: Sha256
    input_policy: Literal["ordered-group-summaries-and-article-ids-v1"]
    assignment_policy: Literal["reader-subject-first-occurrence-v1"]
    singleton_policy: Literal["source-title-first-two-sentences-v1"]
    prose_policy: Literal["merged-only-exact-key-prose-v1"]
    correction_policy: Literal["complete-json-once-v1"]
    reasoning_effort: Literal["low"]
    temperature: Literal[0]
    max_tokens: Annotated[int, Field(gt=0)]


ExecutableSparseThemePolicy = SparseThemePolicy | ReaderSubjectThemePolicy


class SparseThemePolicyDefinition(NewsModel):
    policy: ExecutableSparseThemePolicy
    assignment_prompt: Annotated[str, Field(min_length=1)]
    prose_prompt: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def require_prompt_digests(self) -> SparseThemePolicyDefinition:
        if (
            hashlib.sha256(self.assignment_prompt.encode()).hexdigest()
            != self.policy.assignment_prompt_digest
        ):
            raise ValueError("Sparse assignment prompt digest does not match its prompt")
        if (
            hashlib.sha256(self.prose_prompt.encode()).hexdigest()
            != self.policy.prose_prompt_digest
        ):
            raise ValueError("Sparse prose prompt digest does not match its prompt")
        return self


LEGACY_SPARSE_THEME_POLICY = SparseThemePolicy(
    policy_id="daily-reader-themes-sparse-v2",
    model=THEME_MODEL,
    assignment_prompt_digest=hashlib.sha256(THEME_ASSIGNMENT_PROMPT.encode()).hexdigest(),
    prose_prompt_digest=hashlib.sha256(THEME_PROSE_PROMPT.encode()).hexdigest(),
    input_policy=THEME_INPUT_POLICY,
    assignment_policy="integer-equivalence-first-occurrence-v1",
    singleton_policy="source-title-first-two-sentences-v1",
    prose_policy="merged-only-exact-key-prose-v1",
    correction_policy=THEME_CORRECTION_POLICY,
    reasoning_effort="low",
    temperature=0,
    max_tokens=THEME_MAX_TOKENS,
)
LEGACY_SPARSE_THEME_DEFINITION = SparseThemePolicyDefinition(
    policy=LEGACY_SPARSE_THEME_POLICY,
    assignment_prompt=THEME_ASSIGNMENT_PROMPT,
    prose_prompt=THEME_PROSE_PROMPT,
)
PRODUCTION_THEME_POLICY = ReaderSubjectThemePolicy(
    policy_id="daily-reader-subjects-v3",
    model=THEME_MODEL,
    assignment_prompt_digest=hashlib.sha256(READER_SUBJECT_ASSIGNMENT_PROMPT.encode()).hexdigest(),
    prose_prompt_digest=hashlib.sha256(THEME_PROSE_PROMPT.encode()).hexdigest(),
    input_policy=THEME_INPUT_POLICY,
    assignment_policy="reader-subject-first-occurrence-v1",
    singleton_policy="source-title-first-two-sentences-v1",
    prose_policy="merged-only-exact-key-prose-v1",
    correction_policy=THEME_CORRECTION_POLICY,
    reasoning_effort="low",
    temperature=0,
    max_tokens=THEME_MAX_TOKENS,
)
PRODUCTION_THEME_DEFINITION = SparseThemePolicyDefinition(
    policy=PRODUCTION_THEME_POLICY,
    assignment_prompt=READER_SUBJECT_ASSIGNMENT_PROMPT,
    prose_prompt=THEME_PROSE_PROMPT,
)


class ThemeGroupInput(NewsModel):
    group: NewsGroup
    summary: ArtifactReference
    value: GroupSummary


class DailyThemeInput(NewsModel):
    day: date
    cluster_set: ArtifactReference
    groups: tuple[ThemeGroupInput, ...]


class ThemeModelMessage(NewsModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ThemeModelAttemptEvidence(NewsModel):
    attempt_id: Sha256
    response_id: Annotated[str, Field(min_length=1)]
    status: Literal["accepted", "rejected"]
    error: str | None
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]


class ThemeStageEvidence(NewsModel):
    request_id: Sha256
    messages: Annotated[tuple[ThemeModelMessage, ...], Field(min_length=2)]
    input_digest: Sha256
    response_schema_digest: Sha256
    call: ModelCall
    attempts: Annotated[tuple[ThemeModelAttemptEvidence, ...], Field(min_length=1, max_length=2)]


class SparseThemeConstruction(NewsModel):
    kind: Literal["sparse"] = "sparse"
    assignment: ThemeStageEvidence
    merged_prose: ThemeStageEvidence | None


class EmptyThemeConstruction(NewsModel):
    kind: Literal["empty"] = "empty"


class ModelThemeConstruction(NewsModel):
    kind: Literal["model"] = "model"
    messages: Annotated[tuple[ThemeModelMessage, ...], Field(min_length=2)]
    input_digest: Sha256
    response_schema_digest: Sha256
    call: ModelCall
    attempts: Annotated[tuple[ThemeModelAttemptEvidence, ...], Field(min_length=1)]


ThemeConstructionEvidence = Annotated[
    EmptyThemeConstruction | ModelThemeConstruction,
    Field(discriminator="kind"),
]


class DailyTheme(NewsModel):
    id: Sha256
    title: Annotated[str, Field(min_length=1, max_length=100)]
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    group_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]
    article_version_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]


class DailyThemeSet(NewsModel):
    schema_version: Literal[1] = 1
    day: date
    request_id: Sha256
    policy: ThemePolicy
    policy_digest: Sha256
    cluster_set: ArtifactReference
    groups: tuple[NewsGroup, ...]
    summary_inputs: tuple[ArtifactReference, ...]
    construction: ThemeConstructionEvidence
    themes: tuple[DailyTheme, ...]

    @model_validator(mode="after")
    def require_exact_partition(self) -> DailyThemeSet:
        _validate_theme_set_identity(self)
        _validate_construction_evidence(self.groups, self.construction)
        return self


class SparseDailyThemeSet(NewsModel):
    schema_version: Literal[2] = 2
    day: date
    request_id: Sha256
    policy: SparseThemePolicy
    policy_digest: Sha256
    cluster_set: ArtifactReference
    groups: tuple[NewsGroup, ...]
    summary_inputs: tuple[ArtifactReference, ...]
    construction: EmptyThemeConstruction | SparseThemeConstruction
    themes: tuple[DailyTheme, ...]

    @model_validator(mode="after")
    def require_exact_sparse_evidence(self) -> SparseDailyThemeSet:
        _validate_sparse_theme_set(self)
        return self


class ReaderSubjectDailyThemeSet(NewsModel):
    schema_version: Literal[3] = 3
    day: date
    request_id: Sha256
    policy: ReaderSubjectThemePolicy
    policy_digest: Sha256
    cluster_set: ArtifactReference
    groups: tuple[NewsGroup, ...]
    summary_inputs: tuple[ArtifactReference, ...]
    construction: EmptyThemeConstruction | SparseThemeConstruction
    themes: tuple[DailyTheme, ...]

    @model_validator(mode="after")
    def require_exact_sparse_evidence(self) -> ReaderSubjectDailyThemeSet:
        _validate_reader_subject_theme_set(self)
        return self


class AliasedReaderSubjectThemeSet(NewsModel):
    """Reader-subject theme set whose assignment schema uses group aliases.

    Google AI Studio rejects strict structured-output schemas larger than
    roughly 16 KB, which 64-hex-character group property names exceed once
    a day grows past about 75 groups. Version 4 aliases each group as
    g1..gN in the assignment schema and context, keeping the request far
    below that limit while memberships resolve to the same real ids.
    """

    schema_version: Literal[4] = 4
    day: date
    request_id: Sha256
    policy: ReaderSubjectThemePolicy
    policy_digest: Sha256
    cluster_set: ArtifactReference
    groups: tuple[NewsGroup, ...]
    summary_inputs: tuple[ArtifactReference, ...]
    construction: EmptyThemeConstruction | SparseThemeConstruction
    themes: tuple[DailyTheme, ...]

    @model_validator(mode="after")
    def require_exact_aliased_evidence(self) -> AliasedReaderSubjectThemeSet:
        _validate_reader_subject_theme_set(self)
        return self


class DailyThemeOutput(NewsModel):
    theme_set: (
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
    )
    content_digest: Sha256
    content: bytes


class _ProposedTheme(NewsModel):
    title: Annotated[str, Field(min_length=1, max_length=100)]
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    group_ids: Annotated[tuple[Sha256, ...], Field(min_length=1)]


class _ThemeAssignmentResponse(NewsModel):
    themes: Annotated[tuple[_ProposedTheme, ...], Field(min_length=1)]


class _ThemeProse(NewsModel):
    title: Annotated[str, Field(min_length=1, max_length=100)]
    summary: Annotated[str, Field(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def require_at_most_two_sentences(self) -> _ThemeProse:
        if _first_two_sentences(self.summary) != self.summary.strip():
            raise ValueError("Merged theme summary must contain at most two sentences")
        return self


class _FrozenThemeMembership(NewsModel):
    key: Annotated[str, Field(pattern=r"^theme_[0-9]+$")]
    groups: Annotated[tuple[ThemeGroupInput, ...], Field(min_length=1)]


def theme_policy_digest(policy: ThemePolicy) -> Sha256:
    """Identify every value that can change theme generation or acceptance."""
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def daily_theme_request_id(value: DailyThemeInput, policy: ThemePolicy) -> Sha256:
    """Identify one complete daily theme request from exact immutable inputs."""
    return _daily_theme_request_identity(
        value.day,
        value.cluster_set,
        tuple((item.group, item.summary) for item in value.groups),
        policy,
    )


def _daily_theme_request_identity(
    day: date,
    cluster_set: ArtifactReference,
    groups: tuple[tuple[NewsGroup, ArtifactReference], ...],
    policy: ThemePolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "operation": THEME_OPERATION,
                "day": day.isoformat(),
                "cluster_set_version_id": cluster_set.version_id,
                "ordered_groups": [
                    {
                        "group_id": group.id,
                        "article_version_ids": list(group.article_version_ids),
                        "summary_version_id": summary.version_id,
                    }
                    for group, summary in groups
                ],
                "policy_digest": theme_policy_digest(policy),
                "response_schema_digest": _response_schema_digest(),
                "daily_theme_set_schema_digest": _daily_theme_set_schema_digest(),
            }
        )
    )


def daily_theme_id(
    day: date,
    ordered_group_ids: tuple[Sha256, ...],
    policy_digest: Sha256,
) -> Sha256:
    """Identify accepted theme membership and order under one policy."""
    return _sha256(
        _canonical_json(
            {
                "day": day.isoformat(),
                "group_ids": list(ordered_group_ids),
                "policy_digest": policy_digest,
            }
        )
    )


def daily_theme_run_id(request_id: Sha256, accepted_response_ids: str | tuple[str, ...]) -> Sha256:
    """Identify one observed provider run while retaining version 1 identities."""
    if isinstance(accepted_response_ids, str):
        if not accepted_response_ids:
            raise ValueError("Accepted response ID is required")
        return _sha256(f"{request_id}\0{accepted_response_ids}".encode())
    if not accepted_response_ids or any(not item for item in accepted_response_ids):
        raise ValueError("Accepted response IDs are required")
    return _sha256(
        _canonical_json(
            {
                "request_id": request_id,
                "accepted_response_ids": list(accepted_response_ids),
            }
        )
    )


def sparse_theme_policy_digest(policy: ExecutableSparseThemePolicy) -> Sha256:
    """Identify every sparse construction and acceptance choice."""
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def _sparse_daily_theme_request_id(
    value: DailyThemeInput, policy: ExecutableSparseThemePolicy
) -> Sha256:
    if isinstance(policy, ReaderSubjectThemePolicy):
        return _reader_subject_request_identity_from_parts(
            value.day,
            value.cluster_set,
            tuple((item.group, item.summary) for item in value.groups),
            policy,
            schema_version=4,
        )
    return _sparse_request_identity_from_parts(
        value.day,
        value.cluster_set,
        tuple((item.group, item.summary) for item in value.groups),
        policy,
    )


def _stage_request_id(
    parent_request_id: Sha256,
    operation: str,
    initial_messages: tuple[ThemeModelMessage, ThemeModelMessage],
    response_schema: dict[str, object],
    schema_name: str,
    policy: ExecutableSparseThemePolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "parent_request_id": parent_request_id,
                "operation": operation,
                "initial_message_digest": _messages_digest(initial_messages),
                "response_schema_digest": _sha256(_canonical_json(response_schema)),
                "provider_request": {
                    "model": policy.model,
                    "temperature": policy.temperature,
                    "max_tokens": policy.max_tokens,
                    "schema_name": schema_name,
                    "require_parameters": True,
                    "reasoning_effort": policy.reasoning_effort,
                },
            }
        )
    )


def read_daily_theme_input(day: date) -> DailyThemeInput:
    """Select the current cluster set and exact current summary for every group."""
    cluster = _current_reference(f"news:clusters:{day.isoformat()}", "news_clusters")
    cluster_set = parse_daily_cluster_set(
        read_verified_r2_object(cluster.r2_key, cluster.content_digest)
    )
    summaries = tuple(
        _current_reference(f"news:summary:{_summary_request_id(group)}", "news_summary")
        for group in cluster_set.groups
    )
    return load_daily_theme_input(cluster, summaries)


def read_recorded_daily_theme_input(day: date) -> DailyThemeInput:
    """Reconstruct exact inputs from the current daily theme run."""
    from romanian_news.catalog.theme_inputs import read_recorded_daily_theme_references

    references = read_recorded_daily_theme_references(day)
    if not references:
        raise ValueError(f"Daily themes are unavailable for {day.isoformat()}")
    return load_daily_theme_input(references[0], references[1:])


def load_daily_theme_input(
    cluster_set: ArtifactReference,
    summaries: tuple[ArtifactReference, ...],
) -> DailyThemeInput:
    """Parse and validate immutable inputs at the daily theme boundary."""
    clusters = parse_daily_cluster_set(
        read_verified_r2_object(cluster_set.r2_key, cluster_set.content_digest)
    )
    if len(summaries) != len(clusters.groups):
        raise ValueError("Daily theme summaries must exactly cover cluster groups")
    values: list[ThemeGroupInput] = []
    for group, reference in zip(clusters.groups, summaries, strict=True):
        payload = _analysis_payload(reference)
        if payload.get("group_id") != group.id:
            raise ValueError("Daily theme summary references another group")
        summary = GroupSummary.model_validate_json(
            json.dumps(payload.get("summary"), ensure_ascii=False), strict=True
        )
        if not summary.cited_article_version_ids:
            raise ValueError("Daily theme summary must cite an article")
        if not set(summary.cited_article_version_ids) <= set(group.article_version_ids):
            raise ValueError("Daily theme summary cites an article outside its group")
        values.append(ThemeGroupInput(group=group, summary=reference, value=summary))
    return DailyThemeInput(day=clusters.day, cluster_set=cluster_set, groups=tuple(values))


def construct_daily_themes(
    value: DailyThemeInput,
    definition: SparseThemePolicyDefinition = PRODUCTION_THEME_DEFINITION,
) -> DailyThemeOutput:
    """Construct sparse daily themes while preserving exact source membership."""
    policy = definition.policy
    request_id = _sparse_daily_theme_request_id(value, policy)
    policy_digest = sparse_theme_policy_digest(policy)
    if not value.groups:
        return _theme_output(
            _sparse_theme_set(
                policy=policy,
                day=value.day,
                request_id=request_id,
                policy_digest=policy_digest,
                cluster_set=value.cluster_set,
                groups=(),
                summary_inputs=(),
                construction=EmptyThemeConstruction(),
                themes=(),
            )
        )

    group_ids = tuple(item.group.id for item in value.groups)
    aliased = isinstance(policy, ReaderSubjectThemePolicy)
    assignment_schema = _assignment_response_schema(group_ids, aliased=aliased)
    assignments, assignment_evidence = _run_theme_stage(
        policy=policy,
        parent_request_id=request_id,
        operation=THEME_ASSIGNMENT_OPERATION,
        prompt=definition.assignment_prompt,
        context=_theme_context(value, aliased=aliased),
        response_schema=assignment_schema,
        schema_name="romanian_news_daily_theme_assignments",
        parse=lambda content: _parse_assignment_response(content, group_ids, aliased=aliased),
        correction_instruction="Return only the complete assignments object.",
        error_type=DailyThemeCorrectionExhaustedError,
    )
    plan = _freeze_theme_memberships(value, assignments)
    merged = tuple(membership for membership in plan if len(membership.groups) > 1)
    prose: dict[str, _ThemeProse] = {}
    prose_evidence = None
    if merged:
        prose_schema = _prose_response_schema(tuple(item.key for item in merged))
        prose, prose_evidence = _run_theme_stage(
            policy=policy,
            parent_request_id=request_id,
            operation=THEME_PROSE_OPERATION,
            prompt=definition.prose_prompt,
            context=_merged_theme_context(value.day, merged),
            response_schema=prose_schema,
            schema_name="romanian_news_daily_theme_merged_prose",
            parse=lambda content: _parse_prose_response(
                content, tuple(item.key for item in merged)
            ),
            correction_instruction="Return only every supplied theme key with title and summary.",
            error_type=DailyThemeCorrectionExhaustedError,
        )
    themes = _assemble_sparse_themes(value.day, plan, prose, policy_digest)
    return _theme_output(
        _sparse_theme_set(
            policy=policy,
            day=value.day,
            request_id=request_id,
            policy_digest=policy_digest,
            cluster_set=value.cluster_set,
            groups=tuple(item.group for item in value.groups),
            summary_inputs=tuple(item.summary for item in value.groups),
            construction=SparseThemeConstruction(
                assignment=assignment_evidence,
                merged_prose=prose_evidence,
            ),
            themes=themes,
        )
    )


def _sparse_theme_set(
    *,
    policy: ExecutableSparseThemePolicy,
    day: date,
    request_id: Sha256,
    policy_digest: Sha256,
    cluster_set: ArtifactReference,
    groups: tuple[NewsGroup, ...],
    summary_inputs: tuple[ArtifactReference, ...],
    construction: EmptyThemeConstruction | SparseThemeConstruction,
    themes: tuple[DailyTheme, ...],
) -> SparseDailyThemeSet | AliasedReaderSubjectThemeSet:
    if isinstance(policy, ReaderSubjectThemePolicy):
        return AliasedReaderSubjectThemeSet(
            day=day,
            request_id=request_id,
            policy=policy,
            policy_digest=policy_digest,
            cluster_set=cluster_set,
            groups=groups,
            summary_inputs=summary_inputs,
            construction=construction,
            themes=themes,
        )
    return SparseDailyThemeSet(
        day=day,
        request_id=request_id,
        policy=policy,
        policy_digest=policy_digest,
        cluster_set=cluster_set,
        groups=groups,
        summary_inputs=summary_inputs,
        construction=construction,
        themes=themes,
    )


def _run_theme_stage(
    *,
    policy: ExecutableSparseThemePolicy,
    parent_request_id: Sha256,
    operation: str,
    prompt: str,
    context: str,
    response_schema: dict[str, object],
    schema_name: str,
    parse: Callable[[str], _ParsedStage],
    correction_instruction: str,
    error_type: type[DailyThemeCorrectionExhaustedError],
) -> tuple[_ParsedStage, ThemeStageEvidence]:
    initial_messages = (
        ThemeModelMessage(role="system", content=prompt),
        ThemeModelMessage(role="user", content=context),
    )
    request_id = _stage_request_id(
        parent_request_id,
        operation,
        initial_messages,
        response_schema,
        schema_name,
        policy,
    )
    started = time.monotonic()
    stage = "assignment" if operation == THEME_ASSIGNMENT_OPERATION else "merged prose"
    run = run_corrected_structured_openrouter(
        operation=operation,
        request_id=request_id,
        model=policy.model,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        reasoning_effort=policy.reasoning_effort,
        schema_name=schema_name,
        response_schema=response_schema,
        initial_messages=(
            StructuredMessage(role=initial_messages[0].role, content=initial_messages[0].content),
            StructuredMessage(role=initial_messages[1].role, content=initial_messages[1].content),
        ),
        parse=parse,
        correction_message=lambda error: (
            f"The response was invalid. {correction_instruction} Validation error: {error}"
        ),
        exhausted_error=lambda error: error_type(
            f"Daily theme {stage} stage remained invalid after correction: {error}"
        ),
        unreachable_error="Daily theme correction loop did not return",
        started_at=started,
    )
    messages = tuple(
        ThemeModelMessage(role=message.role, content=message.content) for message in run.messages
    )
    return run.value, ThemeStageEvidence(
        request_id=request_id,
        messages=messages,
        input_digest=_messages_digest(messages),
        response_schema_digest=_sha256(_canonical_json(response_schema)),
        call=run.call,
        attempts=tuple(
            ThemeModelAttemptEvidence(
                attempt_id=attempt.attempt_id,
                response_id=attempt.response_id,
                status=attempt.status,
                error=attempt.error,
                response_content=attempt.response_content,
                response_content_digest=attempt.response_content_digest,
                provider_response=attempt.provider_response,
            )
            for attempt in run.attempts
        ),
    )


def _freeze_theme_memberships(
    value: DailyThemeInput,
    assignments: dict[Sha256, int],
) -> tuple[_FrozenThemeMembership, ...]:
    normalized_labels: dict[int, int] = {}
    grouped: dict[int, list[ThemeGroupInput]] = {}
    for item in value.groups:
        source_label = assignments[item.group.id]
        label = normalized_labels.setdefault(source_label, len(normalized_labels) + 1)
        grouped.setdefault(label, []).append(item)
    width = max(2, len(str(len(grouped))))
    return tuple(
        _FrozenThemeMembership(
            key=f"theme_{label:0{width}d}",
            groups=tuple(grouped[label]),
        )
        for label in range(1, len(grouped) + 1)
    )


def _assemble_sparse_themes(
    day: date,
    plan: tuple[_FrozenThemeMembership, ...],
    prose: dict[str, _ThemeProse],
    policy_digest: Sha256,
) -> tuple[DailyTheme, ...]:
    themes: list[DailyTheme] = []
    for membership in plan:
        group_ids = tuple(item.group.id for item in membership.groups)
        if len(membership.groups) == 1:
            source = membership.groups[0].value
            title = source.title_ro
            summary = _first_two_sentences(source.summary_ro)
        else:
            generated = prose[membership.key]
            title = generated.title
            summary = generated.summary
        themes.append(
            DailyTheme(
                id=daily_theme_id(day, group_ids, policy_digest),
                title=title,
                summary=summary,
                group_ids=group_ids,
                article_version_ids=tuple(
                    dict.fromkeys(
                        article_id
                        for item in membership.groups
                        for article_id in item.group.article_version_ids
                    )
                ),
            )
        )
    return tuple(themes)


def _first_two_sentences(value: str) -> str:
    text = value.strip()
    boundaries = 0
    for match in re.finditer(r'[.!?…]+["»”’]?', text):
        if match.end() < len(text) and not text[match.end()].isspace():
            continue
        preceding = text[: match.start()]
        token_match = re.search(r"([\w.]+)$", preceding, flags=re.UNICODE)
        token = token_match.group(1) if token_match else ""
        normalized = token.casefold().rstrip(".")
        remainder = text[match.end() :].lstrip()
        next_character = remainder[:1]
        if match.group(0).startswith(".") and (
            normalized
            in {
                "dl",
                "dna",
                "dr",
                "ing",
                "jud",
                "prof",
            }
            or (len(token) == 1 and token.isupper())
            or (token.count(".") >= 1 and next_character.islower())
            or (
                normalized in {"art", "cca", "conf", "etc", "mun", "nr", "str"}
                and (next_character.islower() or next_character.isdigit())
            )
        ):
            continue
        boundaries += 1
        if boundaries == 2:
            return text[: match.end()].strip()
    return text


def parse_daily_theme_set(
    content: bytes,
) -> (
    DailyThemeSet | SparseDailyThemeSet | ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet
):
    """Parse one durable daily theme set at its owning boundary."""
    return TypeAdapter(
        DailyThemeSet
        | SparseDailyThemeSet
        | ReaderSubjectDailyThemeSet
        | AliasedReaderSubjectThemeSet
    ).validate_json(content, strict=True)


def _assignment_group_keys(group_ids: tuple[Sha256, ...]) -> dict[str, Sha256]:
    """Map short model-facing aliases to cluster group ids.

    Gemini's strict structured output rejects schemas with too many long
    property names, so the assignment stage speaks in g1..gN aliases and
    the parse maps them back to real ids.
    """
    return {f"g{index}": group_id for index, group_id in enumerate(group_ids, start=1)}


def _assignment_response_schema(
    group_ids: tuple[Sha256, ...],
    *,
    aliased: bool = False,
) -> dict[str, object]:
    if not group_ids:
        raise ValueError("Theme assignment schema requires groups")
    if aliased:
        property_names = tuple(_assignment_group_keys(group_ids))
    else:
        property_names = group_ids
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": len(group_ids),
                    }
                    for name in property_names
                },
                "required": list(property_names),
                "additionalProperties": False,
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }


def _parse_assignment_response(
    content: str,
    group_ids: tuple[Sha256, ...],
    *,
    aliased: bool = False,
) -> dict[Sha256, int]:
    try:
        raw_document: object = json.loads(content, object_pairs_hook=_object_from_unique_pairs)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Theme assignment response is invalid JSON: {error}") from error
    document = _require_string_object(raw_document, "Theme assignment response must be an object")
    if set(document) != {"assignments"}:
        raise ValueError("Theme assignment response must contain only assignments")
    assignments = _require_string_object(
        document["assignments"], "Theme assignments must be an object"
    )
    if aliased:
        supplied = _assignment_group_keys(group_ids)
    else:
        supplied = {group_id: group_id for group_id in group_ids}
    assigned = set(assignments)
    missing = tuple(name for name in supplied if name not in assigned)
    unknown = tuple(name for name in assignments if name not in supplied)
    if missing or unknown:
        raise ValueError(
            "Theme assignments must contain every supplied group and no others; "
            f"missing: {', '.join(missing) or 'none'}; "
            f"unknown: {', '.join(unknown) or 'none'}"
        )
    parsed: dict[Sha256, int] = {}
    for name, group_id in supplied.items():
        assignment = assignments[name]
        if (
            not isinstance(assignment, int)
            or isinstance(assignment, bool)
            or not 1 <= assignment <= len(group_ids)
        ):
            raise ValueError(
                f"Theme assignment for {name} must be an integer from 1 through "
                f"{len(group_ids)}"
            )
        parsed[group_id] = assignment
    return parsed


def _object_from_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Theme response repeats property: {key}")
        value[key] = item
    return value


def _require_string_object(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(message)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(message)
        result[key] = item
    return result


def _prose_response_schema(keys: tuple[str, ...]) -> dict[str, object]:
    prose = _ThemeProse.model_json_schema()
    prose["required"] = list(prose["properties"])
    themes = {
        "type": "object",
        "properties": dict.fromkeys(keys, prose),
        "required": list(keys),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"themes": themes},
        "required": ["themes"],
        "additionalProperties": False,
    }


def _parse_prose_response(content: str, keys: tuple[str, ...]) -> dict[str, _ThemeProse]:
    try:
        raw_document: object = json.loads(content, object_pairs_hook=_object_from_unique_pairs)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Merged prose response is invalid JSON: {error}") from error
    document = _require_string_object(raw_document, "Merged prose response must be an object")
    if set(document) != {"themes"}:
        raise ValueError("Merged prose response must contain only themes")
    themes = _require_string_object(document["themes"], "Merged prose themes must be an object")
    if set(themes) != set(keys):
        missing = tuple(key for key in keys if key not in themes)
        unknown = tuple(key for key in themes if key not in set(keys))
        raise ValueError(
            "Merged prose must contain exactly the supplied theme keys; "
            f"missing: {', '.join(missing) or 'none'}; "
            f"unknown: {', '.join(unknown) or 'none'}"
        )
    return {key: _ThemeProse.model_validate(themes[key]) for key in keys}


def _merged_theme_context(
    day: date,
    plan: tuple[_FrozenThemeMembership, ...],
) -> str:
    return json.dumps(
        {
            "day": day.isoformat(),
            "themes": {
                membership.key: [
                    {
                        "event_title": item.value.title_ro,
                        "event_summary": item.value.summary_ro,
                        "key_points": list(item.value.key_points_ro),
                    }
                    for item in membership.groups
                ]
                for membership in plan
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _messages_digest(messages: tuple[ThemeModelMessage, ...]) -> Sha256:
    return _sha256(_canonical_json([message.model_dump(mode="json") for message in messages]))


def _sparse_daily_theme_set_schema_digest() -> Sha256:
    return _sha256(_canonical_json(SparseDailyThemeSet.model_json_schema()))


def _reader_subject_daily_theme_set_schema_digest() -> Sha256:
    return _sha256(_canonical_json(ReaderSubjectDailyThemeSet.model_json_schema()))


def _aliased_reader_subject_theme_set_schema_digest() -> Sha256:
    return _sha256(_canonical_json(AliasedReaderSubjectThemeSet.model_json_schema()))


def _validate_sparse_theme_set(theme_set: SparseDailyThemeSet) -> None:
    _validate_theme_partition(theme_set.groups, theme_set.themes)
    if len(theme_set.summary_inputs) != len(theme_set.groups):
        raise ValueError("Daily theme summaries must exactly cover groups")
    if theme_set.policy_digest != sparse_theme_policy_digest(theme_set.policy):
        raise ValueError("Sparse theme policy digest does not match its policy")
    expected_request_id = _sparse_request_identity_from_parts(
        theme_set.day,
        theme_set.cluster_set,
        tuple(zip(theme_set.groups, theme_set.summary_inputs, strict=True)),
        theme_set.policy,
    )
    if theme_set.request_id != expected_request_id:
        raise ValueError("Sparse theme request identity does not match its exact inputs")
    for theme in theme_set.themes:
        if theme.id != daily_theme_id(theme_set.day, theme.group_ids, theme_set.policy_digest):
            raise ValueError("Daily theme identity does not match its membership")
    if not theme_set.groups:
        if not isinstance(theme_set.construction, EmptyThemeConstruction):
            raise ValueError("Empty sparse themes cannot have model evidence")
        return
    if not isinstance(theme_set.construction, SparseThemeConstruction):
        raise ValueError("Non-empty sparse themes require sparse construction evidence")
    _validate_sparse_evidence(theme_set)


def _sparse_request_identity_from_parts(
    day: date,
    cluster_set: ArtifactReference,
    groups: tuple[tuple[NewsGroup, ArtifactReference], ...],
    policy: SparseThemePolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "operation": THEME_OPERATION,
                "schema_version": 2,
                "day": day.isoformat(),
                "cluster_set_version_id": cluster_set.version_id,
                "ordered_groups": [
                    {
                        "group_id": group.id,
                        "article_version_ids": list(group.article_version_ids),
                        "summary_version_id": summary.version_id,
                    }
                    for group, summary in groups
                ],
                "policy_digest": sparse_theme_policy_digest(policy),
                "daily_theme_set_schema_digest": _sparse_daily_theme_set_schema_digest(),
            }
        )
    )


def _reader_subject_request_identity_from_parts(
    day: date,
    cluster_set: ArtifactReference,
    groups: tuple[tuple[NewsGroup, ArtifactReference], ...],
    policy: ReaderSubjectThemePolicy,
    *,
    schema_version: int = 3,
) -> Sha256:
    schema_digest = (
        _aliased_reader_subject_theme_set_schema_digest()
        if schema_version == 4
        else _reader_subject_daily_theme_set_schema_digest()
    )
    return _sha256(
        _canonical_json(
            {
                "operation": THEME_OPERATION,
                "schema_version": schema_version,
                "day": day.isoformat(),
                "cluster_set_version_id": cluster_set.version_id,
                "ordered_groups": [
                    {
                        "group_id": group.id,
                        "article_version_ids": list(group.article_version_ids),
                        "summary_version_id": summary.version_id,
                    }
                    for group, summary in groups
                ],
                "policy_digest": sparse_theme_policy_digest(policy),
                "daily_theme_set_schema_digest": schema_digest,
            }
        )
    )


def _validate_reader_subject_theme_set(
    theme_set: ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet,
) -> None:
    _validate_theme_partition(theme_set.groups, theme_set.themes)
    if len(theme_set.summary_inputs) != len(theme_set.groups):
        raise ValueError("Daily theme summaries must exactly cover groups")
    if theme_set.policy_digest != sparse_theme_policy_digest(theme_set.policy):
        raise ValueError("Sparse theme policy digest does not match its policy")
    expected_request_id = _reader_subject_request_identity_from_parts(
        theme_set.day,
        theme_set.cluster_set,
        tuple(zip(theme_set.groups, theme_set.summary_inputs, strict=True)),
        theme_set.policy,
        schema_version=theme_set.schema_version,
    )
    if theme_set.request_id != expected_request_id:
        raise ValueError("Sparse theme request identity does not match its exact inputs")
    for theme in theme_set.themes:
        if theme.id != daily_theme_id(theme_set.day, theme.group_ids, theme_set.policy_digest):
            raise ValueError("Daily theme identity does not match its membership")
    if not theme_set.groups:
        if not isinstance(theme_set.construction, EmptyThemeConstruction):
            raise ValueError("Empty sparse themes cannot have model evidence")
        return
    if not isinstance(theme_set.construction, SparseThemeConstruction):
        raise ValueError("Non-empty sparse themes require sparse construction evidence")
    assignment_prompt = theme_set.construction.assignment.messages[0].content
    if _sha256(assignment_prompt.encode()) != theme_set.policy.assignment_prompt_digest:
        raise ValueError("Sparse assignment evidence does not match its prompt digest")
    prose_evidence = theme_set.construction.merged_prose
    prose_prompt = (
        prose_evidence.messages[0].content if prose_evidence is not None else THEME_PROSE_PROMPT
    )
    if _sha256(prose_prompt.encode()) != theme_set.policy.prose_prompt_digest:
        raise ValueError("Sparse prose evidence does not match its prompt digest")
    _validate_sparse_evidence(theme_set, assignment_prompt, prose_prompt)


def _validate_sparse_evidence(
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet,
    assignment_prompt: str = THEME_ASSIGNMENT_PROMPT,
    prose_prompt: str = THEME_PROSE_PROMPT,
    *,
    aliased: bool = False,
) -> None:
    aliased = theme_set.schema_version >= 4
    assert isinstance(theme_set.construction, SparseThemeConstruction)
    construction = theme_set.construction
    group_ids = tuple(group.id for group in theme_set.groups)
    assignment_schema = _assignment_response_schema(group_ids, aliased=aliased)
    assignment_context = _validated_assignment_context(
        construction.assignment.messages[1].content, theme_set
    )
    assignment_messages = (
        ThemeModelMessage(role="system", content=assignment_prompt),
        ThemeModelMessage(role="user", content=assignment_context),
    )
    _validate_stage_evidence(
        construction.assignment,
        theme_set.request_id,
        THEME_ASSIGNMENT_OPERATION,
        assignment_schema,
        assignment_messages,
        "romanian_news_daily_theme_assignments",
        "Return only the complete assignments object.",
        theme_set.policy,
    )
    assignment_attempt = construction.assignment.attempts[-1]
    assignments = _parse_assignment_response(
        assignment_attempt.response_content, group_ids, aliased=aliased
    )
    expected_memberships = _normalized_group_memberships(group_ids, assignments)
    if tuple(theme.group_ids for theme in theme_set.themes) != expected_memberships:
        raise ValueError("Final theme memberships do not match accepted assignment evidence")
    _validate_singleton_prose(theme_set.themes, assignment_context, group_ids)
    width = max(2, len(str(len(expected_memberships))))
    merged_keys = tuple(
        f"theme_{index:0{width}d}"
        for index, membership in enumerate(expected_memberships, start=1)
        if len(membership) > 1
    )
    if not merged_keys:
        if construction.merged_prose is not None:
            raise ValueError("Singleton-only themes cannot have merged prose evidence")
        return
    if construction.merged_prose is None:
        raise ValueError("Merged themes require merged prose evidence")
    prose_schema = _prose_response_schema(merged_keys)
    prose_messages = (
        ThemeModelMessage(role="system", content=prose_prompt),
        ThemeModelMessage(
            role="user",
            content=_merged_context_from_assignment_context(
                assignment_context, expected_memberships, group_ids
            ),
        ),
    )
    _validate_stage_evidence(
        construction.merged_prose,
        theme_set.request_id,
        THEME_PROSE_OPERATION,
        prose_schema,
        prose_messages,
        "romanian_news_daily_theme_merged_prose",
        "Return only every supplied theme key with title and summary.",
        theme_set.policy,
    )
    prose = _parse_prose_response(
        construction.merged_prose.attempts[-1].response_content, merged_keys
    )
    merged_themes = tuple(theme for theme in theme_set.themes if len(theme.group_ids) > 1)
    for key, theme in zip(merged_keys, merged_themes, strict=True):
        if (theme.title, theme.summary) != (prose[key].title, prose[key].summary):
            raise ValueError("Merged theme prose does not match accepted prose evidence")


def _validate_stage_evidence(
    evidence: ThemeStageEvidence,
    parent_request_id: Sha256,
    operation: str,
    response_schema: dict[str, object],
    initial_messages: tuple[ThemeModelMessage, ThemeModelMessage],
    schema_name: str,
    correction_instruction: str,
    policy: ExecutableSparseThemePolicy,
) -> None:
    expected_request_id = _stage_request_id(
        parent_request_id,
        operation,
        initial_messages,
        response_schema,
        schema_name,
        policy,
    )
    if evidence.request_id != expected_request_id:
        raise ValueError("Theme stage request identity does not match exact inputs")
    if evidence.response_schema_digest != _sha256(_canonical_json(response_schema)):
        raise ValueError("Theme stage response schema digest does not match")
    expected_messages = list(initial_messages)
    if len(evidence.attempts) == 2:
        rejected = evidence.attempts[0]
        if rejected.status != "rejected":
            raise ValueError("Theme correction requires an initial rejected attempt")
        expected_messages.extend(
            (
                ThemeModelMessage(role="assistant", content=rejected.response_content),
                ThemeModelMessage(
                    role="user",
                    content=(
                        f"The response was invalid. {correction_instruction} "
                        f"Validation error: {rejected.error}"
                    ),
                ),
            )
        )
    if evidence.messages != tuple(expected_messages):
        raise ValueError("Theme stage messages do not match exact request and correction history")
    if evidence.input_digest != _messages_digest(evidence.messages):
        raise ValueError("Theme stage input digest does not match its messages")
    if evidence.call.model != policy.model:
        raise ValueError("Theme stage model does not match the sparse policy")
    for attempt in evidence.attempts:
        _validate_model_attempt(attempt)
        if attempt.provider_response.get("model") != policy.model:
            raise ValueError("Theme stage provider model does not match the sparse policy")
    _validate_accepted_attempt(evidence)


def _validated_assignment_context(
    content: str,
    theme_set: SparseDailyThemeSet | ReaderSubjectDailyThemeSet | AliasedReaderSubjectThemeSet,
) -> str:
    aliased = theme_set.schema_version >= 4
    try:
        raw: object = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Theme assignment context is invalid JSON") from error
    document = _require_string_object(raw, "Theme assignment context must be an object")
    if set(document) != {"day", "groups"} or document["day"] != theme_set.day.isoformat():
        raise ValueError("Theme assignment context does not match the theme day")
    groups = document["groups"]
    if not isinstance(groups, list) or len(groups) != len(theme_set.groups):
        raise ValueError("Theme assignment context does not exactly cover groups")
    aliased = theme_set.schema_version >= 4
    if aliased:
        expected_names = _assignment_group_keys(tuple(group.id for group in theme_set.groups))
    else:
        expected_names = {group.id: group.id for group in theme_set.groups}
    for raw_group, expected_name, group in zip(
        groups, expected_names, theme_set.groups, strict=True
    ):
        context_group = _require_string_object(
            raw_group, "Theme assignment context group must be an object"
        )
        if set(context_group) != {
            "group_id",
            "article_version_ids",
            "event_title",
            "event_summary",
            "key_points",
        }:
            raise ValueError("Theme assignment context group has unexpected fields")
        if context_group["group_id"] != expected_name:
            raise ValueError("Theme assignment context group order does not match")
        if context_group["article_version_ids"] != list(group.article_version_ids):
            raise ValueError("Theme assignment context articles do not match")
        if not isinstance(context_group["event_title"], str) or not isinstance(
            context_group["event_summary"], str
        ):
            raise ValueError("Theme assignment context prose must be text")
        key_points = context_group["key_points"]
        if not isinstance(key_points, list) or not all(
            isinstance(item, str) for item in key_points
        ):
            raise ValueError("Theme assignment context key points must be text")
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if content != canonical:
        raise ValueError("Theme assignment context must use canonical JSON")
    return canonical


def _merged_context_from_assignment_context(
    assignment_context: str,
    memberships: tuple[tuple[Sha256, ...], ...],
    group_ids: tuple[Sha256, ...],
) -> str:
    document = _require_string_object(
        json.loads(assignment_context), "Theme assignment context must be an object"
    )
    groups = document["groups"]
    assert isinstance(groups, list)
    assert len(groups) == len(group_ids)
    groups_by_id = {group_id: group for group_id, group in zip(group_ids, groups, strict=True)}
    width = max(2, len(str(len(memberships))))
    return json.dumps(
        {
            "day": document["day"],
            "themes": {
                f"theme_{index:0{width}d}": [
                    {
                        "event_title": groups_by_id[group_id]["event_title"],
                        "event_summary": groups_by_id[group_id]["event_summary"],
                        "key_points": groups_by_id[group_id]["key_points"],
                    }
                    for group_id in membership
                ]
                for index, membership in enumerate(memberships, start=1)
                if len(membership) > 1
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_singleton_prose(
    themes: tuple[DailyTheme, ...],
    assignment_context: str,
    group_ids: tuple[Sha256, ...],
) -> None:
    document = _require_string_object(
        json.loads(assignment_context), "Theme assignment context must be an object"
    )
    groups = document["groups"]
    assert isinstance(groups, list)
    groups_by_id = {
        group_id: group
        for group_id, group in zip(group_ids, groups, strict=True)
        if isinstance(group, dict)
    }
    for theme in themes:
        if len(theme.group_ids) != 1:
            continue
        source = groups_by_id[theme.group_ids[0]]
        expected = (
            source["event_title"],
            _first_two_sentences(str(source["event_summary"])),
        )
        if (theme.title, theme.summary) != expected:
            raise ValueError("Singleton theme prose does not match its source summary evidence")


def _normalized_group_memberships(
    group_ids: tuple[Sha256, ...], assignments: dict[Sha256, int]
) -> tuple[tuple[Sha256, ...], ...]:
    normalized_labels: dict[int, int] = {}
    memberships: dict[int, list[Sha256]] = {}
    for group_id in group_ids:
        source_label = assignments[group_id]
        label = normalized_labels.setdefault(source_label, len(normalized_labels) + 1)
        memberships.setdefault(label, []).append(group_id)
    return tuple(tuple(memberships[label]) for label in range(1, len(memberships) + 1))


def _validate_theme_set_identity(theme_set: DailyThemeSet) -> None:
    _validate_theme_partition(theme_set.groups, theme_set.themes)
    if len(theme_set.summary_inputs) != len(theme_set.groups):
        raise ValueError("Daily theme summaries must exactly cover groups")
    expected_policy_digest = theme_policy_digest(theme_set.policy)
    if theme_set.policy_digest != expected_policy_digest:
        raise ValueError("Daily theme policy digest does not match its policy")
    expected_request_id = _daily_theme_request_identity(
        theme_set.day,
        theme_set.cluster_set,
        tuple(zip(theme_set.groups, theme_set.summary_inputs, strict=True)),
        theme_set.policy,
    )
    if theme_set.request_id != expected_request_id:
        raise ValueError("Daily theme request identity does not match its exact inputs")
    for theme in theme_set.themes:
        if theme.id != daily_theme_id(
            theme_set.day,
            theme.group_ids,
            theme_set.policy_digest,
        ):
            raise ValueError("Daily theme identity does not match its membership")


def _validate_construction_evidence(
    groups: tuple[NewsGroup, ...],
    construction: ThemeConstructionEvidence,
) -> None:
    if isinstance(construction, EmptyThemeConstruction) != (not groups):
        raise ValueError("Only an empty group set can omit model construction evidence")
    if not isinstance(construction, ModelThemeConstruction):
        return
    expected_input_digest = _sha256(
        _canonical_json([message.model_dump(mode="json") for message in construction.messages])
    )
    if construction.input_digest != expected_input_digest:
        raise ValueError("Daily theme model input digest does not match its messages")
    if construction.response_schema_digest != _response_schema_digest():
        raise ValueError("Daily theme response schema digest does not match")
    for attempt in construction.attempts:
        _validate_model_attempt(attempt)
    _validate_accepted_attempt(construction)


def _validate_model_attempt(attempt: ThemeModelAttemptEvidence) -> None:
    if attempt.response_content_digest != _sha256(attempt.response_content.encode()):
        raise ValueError("Daily theme attempt content digest does not match")
    if attempt.status == "accepted" and attempt.error is not None:
        raise ValueError("An accepted model attempt cannot have an error")
    if attempt.status == "rejected" and not attempt.error:
        raise ValueError("A rejected model attempt requires an error")
    provider_content = _provider_response_content(attempt.provider_response)
    if provider_content != attempt.response_content:
        raise ValueError("Daily theme provider response content does not match")
    provider_response_id = attempt.provider_response.get("id")
    if provider_response_id not in (None, "") and str(provider_response_id) != attempt.response_id:
        raise ValueError("Daily theme provider response ID does not match")


def _validate_accepted_attempt(
    construction: ModelThemeConstruction | ThemeStageEvidence,
) -> None:
    accepted_attempts = tuple(
        attempt for attempt in construction.attempts if attempt.status == "accepted"
    )
    if len(accepted_attempts) != 1:
        raise ValueError("Daily theme construction must have exactly one accepted attempt")
    accepted_attempt = accepted_attempts[0]
    if accepted_attempt != construction.attempts[-1]:
        raise ValueError("The accepted model attempt must be final")
    if construction.call.response_id != accepted_attempt.response_id:
        raise ValueError("Daily theme selected response ID does not match its accepted attempt")


def _validate_theme_partition(
    groups: tuple[NewsGroup, ...], themes: tuple[DailyTheme, ...]
) -> None:
    if len({theme.id for theme in themes}) != len(themes):
        raise ValueError("Daily theme IDs must be unique")
    supplied = tuple(group.id for group in groups)
    assigned = tuple(group_id for theme in themes for group_id in theme.group_ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(supplied):
        raise ValueError("Daily themes must exactly partition all groups")
    _validate_source_relative_group_order(
        supplied,
        tuple(theme.group_ids for theme in themes),
    )
    groups_by_id = {group.id: group for group in groups}
    for theme in themes:
        expected = tuple(
            dict.fromkeys(
                article_id
                for group_id in theme.group_ids
                for article_id in groups_by_id[group_id].article_version_ids
            )
        )
        if theme.article_version_ids != expected:
            raise ValueError("Daily theme article union does not match its groups")


def _validate_source_relative_group_order(
    supplied: tuple[Sha256, ...],
    theme_group_ids: tuple[tuple[Sha256, ...], ...],
) -> None:
    positions = {group_id: position for position, group_id in enumerate(supplied)}
    if any(
        group_ids != tuple(sorted(group_ids, key=positions.__getitem__))
        for group_ids in theme_group_ids
    ):
        raise ValueError("Daily theme group IDs must preserve source-relative order")


def _provider_response_content(provider_response: dict[str, object]) -> str:
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Daily theme provider response must contain choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Daily theme provider response choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Daily theme provider response choice must contain a message")
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("Daily theme provider response message content must be text")
    return content


def _theme_output(
    theme_set: DailyThemeSet
    | SparseDailyThemeSet
    | ReaderSubjectDailyThemeSet
    | AliasedReaderSubjectThemeSet,
) -> DailyThemeOutput:
    content = _canonical_json(theme_set.model_dump(mode="json"))
    return DailyThemeOutput(
        theme_set=theme_set,
        content_digest=_sha256(content),
        content=content,
    )


def _theme_context(value: DailyThemeInput, *, aliased: bool = False) -> str:
    if aliased:
        group_keys = _assignment_group_keys(tuple(item.group.id for item in value.groups))
    else:
        group_keys = {item.group.id: item.group.id for item in value.groups}
    return json.dumps(
        {
            "day": value.day.isoformat(),
            "groups": [
                {
                    "group_id": next(
                        alias for alias, group_id in group_keys.items() if group_id == item.group.id
                    ),
                    "article_version_ids": list(item.group.article_version_ids),
                    "event_title": item.value.title_ro,
                    "event_summary": item.value.summary_ro,
                    "key_points": list(item.value.key_points_ro),
                }
                for item in value.groups
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _response_schema() -> dict[str, object]:
    schema = _ThemeAssignmentResponse.model_json_schema()
    schema["required"] = list(schema["properties"])
    proposed = schema["$defs"]["_ProposedTheme"]
    proposed["required"] = list(proposed["properties"])
    return schema


def _daily_theme_set_schema_digest() -> Sha256:
    return _sha256(_canonical_json(DailyThemeSet.model_json_schema()))


def _response_schema_digest() -> Sha256:
    return _sha256(_canonical_json(_response_schema()))


def _analysis_payload(reference: ArtifactReference) -> dict[str, object]:
    payload = json.loads(read_verified_r2_object(reference.r2_key, reference.content_digest))
    if not isinstance(payload, dict):
        raise ValueError("Daily theme summary payload must be an object")
    return payload


def _current_reference(artifact_id: str, kind: str) -> ArtifactReference:
    from romanian_news.catalog.artifacts import current_artifact_reference

    reference = current_artifact_reference(artifact_id, kind)
    if reference is None:
        raise ValueError(f"Required theme input is unavailable: {artifact_id}")
    return reference


def _summary_request_id(group: NewsGroup) -> Sha256:
    from romanian_news.analysis.groups.summary import summary_request_id

    return summary_request_id(group)
