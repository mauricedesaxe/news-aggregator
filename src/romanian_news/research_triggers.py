from __future__ import annotations

import hashlib
import time
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.corrected_structured import (
    StructuredMessage,
    run_corrected_structured_openrouter,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.identity import canonical_json as _canonical_json
from romanian_news.identity import sha256 as _sha256
from romanian_news.reports import ArchivedDailyReport, parse_daily_report
from romanian_news.storage import read_verified_r2_object

TRIGGER_OPERATION = "news.trigger_daily_research"
TRIGGER_MODEL = "google/gemini-3.8-flash"
TRIGGER_PROMPT = (
    "You triage subjects in a daily report on Romanian political, economic, and social news. "
    "For each supplied subject, rate whether one background research round would materially "
    "improve the subject for the reader beyond what the report already says. "
    "Rate every subject exactly once on a gap_strength scale from 0.0 to 1.0. "
    "Signals that typically create a gap: "
    "a level statistic (wage, pension, price, index, rate, output) presented without its trend; "
    "a market or macro move without comparative context; "
    "a lesser-known institution, company, or person whose significance the reader cannot be "
    "expected to know; "
    "a surprising or single-source claim; "
    "an ongoing situation whose consequences or background are unclear. "
    "A complete subject (fully contextualized political development, opinion-framed item, "
    "minor roundup) scores near zero. "
    "When gap_strength is at least 0.3, write exactly one specific research question in English naming "
    "the subject, the comparator or series or source, and the period, and list one or more "
    "evidence types. Otherwise leave the question null and list no evidence types. "
    "Evidence types are: entity_context, historical_data, financial_data, "
    "external_corroboration, coverage_review, investigative_report. "
    "Use only the supplied subject content."
)
TRIGGER_MAX_TOKENS = 4000

EvidenceType = Literal[
    "entity_context",
    "historical_data",
    "financial_data",
    "external_corroboration",
    "coverage_review",
    "investigative_report",
]


class ResearchTriggerPolicy(NewsModel):
    policy_id: Literal["daily-research-flags-v1"]
    model: Literal["google/gemini-3.8-flash"]
    prompt_digest: Sha256
    input_policy: Literal["schema-v3-report-sections-v1"]
    strength_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    correction_policy: Literal["complete-json-once-v1"]
    temperature: Literal[0]
    max_tokens: Annotated[int, Field(gt=0)]


PRODUCTION_RESEARCH_TRIGGER_POLICY = ResearchTriggerPolicy(
    policy_id="daily-research-flags-v1",
    model=TRIGGER_MODEL,
    prompt_digest=hashlib.sha256(TRIGGER_PROMPT.encode()).hexdigest(),
    input_policy="schema-v3-report-sections-v1",
    strength_threshold=0.3,
    correction_policy="complete-json-once-v1",
    temperature=0,
    max_tokens=TRIGGER_MAX_TOKENS,
)


class SubjectTriggerInput(NewsModel):
    alias: Annotated[str, Field(pattern=r"^subject_[0-9]{2}$")]
    theme_id: Sha256
    title: Annotated[str, Field(min_length=1)]
    summary: Annotated[str, Field(min_length=1)]
    event_count: Annotated[int, Field(ge=1)]
    article_count: Annotated[int, Field(ge=1)]


class DailyResearchTriggerInput(NewsModel):
    day: date
    report: ArtifactReference
    subjects: tuple[SubjectTriggerInput, ...]

    @model_validator(mode="after")
    def require_exact_subjects(self) -> DailyResearchTriggerInput:
        aliases = tuple(item.alias for item in self.subjects)
        theme_ids = tuple(item.theme_id for item in self.subjects)
        if len(aliases) != len(set(aliases)) or len(theme_ids) != len(set(theme_ids)):
            raise ValueError("Research trigger subjects must be unique")
        return self


class TriggerMessage(NewsModel):
    role: Literal["system", "assistant", "user"]
    content: str


class EmptyResearchTriggerConstruction(NewsModel):
    kind: Literal["empty"] = "empty"


class TriggerAttemptEvidence(NewsModel):
    attempt_id: Sha256
    response_id: str
    status: Literal["accepted", "rejected"]
    error: str | None
    response_content: str
    response_content_digest: Sha256
    provider_response: dict[str, object]


class ModelResearchTriggerConstruction(NewsModel):
    kind: Literal["model"] = "model"
    request_id: Sha256
    messages: tuple[TriggerMessage, ...]
    input_digest: Sha256
    response_schema_digest: Sha256
    call: ModelCall
    attempts: tuple[TriggerAttemptEvidence, ...]


ResearchTriggerConstruction = Annotated[
    ModelResearchTriggerConstruction | EmptyResearchTriggerConstruction,
    Field(discriminator="kind"),
]


class SubjectResearchFlag(NewsModel):
    theme_id: Sha256
    gap_strength: Annotated[float, Field(ge=0.0, le=1.0)]
    flagged: bool
    question: Annotated[str | None, Field(min_length=1)] = None
    evidence_types: tuple[EvidenceType, ...] = ()

    @model_validator(mode="after")
    def require_question_when_flagged(self) -> SubjectResearchFlag:
        if self.flagged and (self.question is None or not self.evidence_types):
            raise ValueError(
                f"Flagged subject {self.theme_id} requires a question and evidence types"
            )
        if not self.flagged and (self.question is not None or self.evidence_types):
            raise ValueError(
                f"Unflagged subject {self.theme_id} must not carry a question or evidence types"
            )
        return self


class DailyResearchTriggerSet(NewsModel):
    day: date
    request_id: Sha256
    policy: ResearchTriggerPolicy
    policy_digest: Sha256
    report: ArtifactReference
    construction: ResearchTriggerConstruction
    triggers: tuple[SubjectResearchFlag, ...]

    @model_validator(mode="after")
    def require_complete_partition(self) -> DailyResearchTriggerSet:
        theme_ids = tuple(item.theme_id for item in self.triggers)
        if len(theme_ids) != len(set(theme_ids)):
            raise ValueError("Research trigger flags must reference each subject once")
        return self


class DailyResearchTriggerOutput(NewsModel):
    trigger_set: DailyResearchTriggerSet
    content_digest: Sha256
    content: bytes


class _ProposedSubjectTrigger(NewsModel):
    subject: Annotated[str, Field(pattern=r"^subject_[0-9]{2}$")]
    gap_strength: Annotated[float, Field(ge=0.0, le=1.0)]
    question: Annotated[str | None, Field(min_length=1)] = None
    evidence_types: tuple[EvidenceType, ...] = ()


class _TriggerResponse(NewsModel):
    subjects: tuple[_ProposedSubjectTrigger, ...]


class ResearchTriggerCorrectionExhausted(ValueError):
    """Both research trigger responses failed validation."""


def read_daily_research_trigger_input(day: date) -> DailyResearchTriggerInput:
    """Build one trigger input from the current frozen daily report."""
    from romanian_news.daily import read_daily_report_reference

    reference = read_daily_report_reference(day)
    document = parse_daily_report(
        read_verified_r2_object(reference.r2_key, reference.content_digest)
    )
    if isinstance(document, ArchivedDailyReport):
        raise ValueError("Archived reports carry no research trigger subjects")
    subjects = tuple(
        SubjectTriggerInput(
            alias=f"subject_{position:02d}",
            theme_id=section.theme_id,
            title=section.title,
            summary=section.summary,
            event_count=len(section.events),
            article_count=sum(len(event.articles) for event in section.events),
        )
        for position, section in enumerate(document.sections, start=1)
    )
    return DailyResearchTriggerInput(day=day, report=reference, subjects=subjects)


def research_trigger_request_id(
    value: DailyResearchTriggerInput, policy: ResearchTriggerPolicy
) -> Sha256:
    """Identify one trigger request across exact inputs and every acceptance choice."""
    return _sha256(
        _canonical_json(
            {
                "operation": TRIGGER_OPERATION,
                "day": value.day.isoformat(),
                "report_version_id": value.report.version_id,
                "policy_digest": research_trigger_policy_digest(policy),
                "response_schema_digest": _sha256(
                    _canonical_json(_TriggerResponse.model_json_schema())
                ),
                "trigger_set_schema_digest": _sha256(
                    _canonical_json(DailyResearchTriggerSet.model_json_schema())
                ),
            }
        )
    )


def research_trigger_policy_digest(policy: ResearchTriggerPolicy) -> Sha256:
    """Identify every trigger generation and acceptance choice."""
    return _sha256(_canonical_json(policy.model_dump(mode="json")))


def research_trigger_run_id(request_id: Sha256, implementation_ref: str) -> Sha256:
    """Identify one publishing run of a trigger request."""
    return _sha256(
        _canonical_json({"request_id": request_id, "implementation_ref": implementation_ref})
    )


def construct_daily_research_triggers(
    value: DailyResearchTriggerInput,
    policy: ResearchTriggerPolicy = PRODUCTION_RESEARCH_TRIGGER_POLICY,
) -> DailyResearchTriggerOutput:
    """Rate one frozen report's subjects for background research need."""
    if not value.subjects:
        return _output(
            DailyResearchTriggerSet(
                day=value.day,
                request_id=research_trigger_request_id(value, policy),
                policy=policy,
                policy_digest=research_trigger_policy_digest(policy),
                report=value.report,
                construction=EmptyResearchTriggerConstruction(),
                triggers=(),
            )
        )
    response_schema = _response_schema()
    messages = (
        TriggerMessage(role="system", content=TRIGGER_PROMPT),
        TriggerMessage(role="user", content=_trigger_context(value)),
    )
    started = time.monotonic()
    model_request_id = _model_request_id(
        research_trigger_request_id(value, policy), messages, response_schema, policy
    )
    run = run_corrected_structured_openrouter(
        operation=TRIGGER_OPERATION,
        request_id=model_request_id,
        model=policy.model,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        reasoning_effort="low",
        schema_name="romanian_news_daily_research_triggers",
        response_schema=response_schema,
        initial_messages=(
            StructuredMessage(role=messages[0].role, content=messages[0].content),
            StructuredMessage(role=messages[1].role, content=messages[1].content),
        ),
        parse=lambda content: parse_research_trigger_response(content, value),
        correction_message=lambda error: (
            "The response was invalid. Rate every supplied subject exactly once. "
            f"Validation error: {error}"
        ),
        exhausted_error=lambda error: ResearchTriggerCorrectionExhausted(
            f"Research trigger response remained invalid after correction: {error}"
        ),
        unreachable_error="Research trigger correction loop did not return",
        started_at=started,
    )
    construction_messages = tuple(
        TriggerMessage(role=message.role, content=message.content) for message in run.messages
    )
    construction = ModelResearchTriggerConstruction(
        request_id=model_request_id,
        messages=construction_messages,
        input_digest=_messages_digest(construction_messages),
        response_schema_digest=_sha256(_canonical_json(response_schema)),
        call=run.call,
        attempts=tuple(
            TriggerAttemptEvidence(
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
    return _output(
        DailyResearchTriggerSet(
            day=value.day,
            request_id=research_trigger_request_id(value, policy),
            policy=policy,
            policy_digest=research_trigger_policy_digest(policy),
            report=value.report,
            construction=construction,
            triggers=_freeze_triggers(run.value, value, policy),
        )
    )


def parse_research_trigger_response(
    content: str,
    value: DailyResearchTriggerInput,
) -> _TriggerResponse:
    """Parse and validate one model response against exact supplied subjects."""
    response = _TriggerResponse.model_validate_json(content, strict=True)
    observed = tuple(item.subject for item in response.subjects)
    expected = tuple(item.alias for item in value.subjects)
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("Trigger response must rate every supplied subject exactly once")
    return response


def parse_daily_research_trigger_set(content: bytes) -> DailyResearchTriggerSet:
    """Parse one durable research trigger artifact at its owning boundary."""
    return DailyResearchTriggerSet.model_validate_json(content, strict=True)


def _report_day(artifact_id: str) -> date:
    prefix = "news:daily:"
    if not artifact_id.startswith(prefix):
        raise ValueError(f"Invalid daily report artifact identity: {artifact_id}")
    return date.fromisoformat(artifact_id.removeprefix(prefix))


def _freeze_triggers(
    response: _TriggerResponse,
    value: DailyResearchTriggerInput,
    policy: ResearchTriggerPolicy,
) -> tuple[SubjectResearchFlag, ...]:
    themes = {item.theme_id: item for item in value.subjects}
    by_alias = {item.alias: item for item in value.subjects}
    result = []
    for item in response.subjects:
        subject = by_alias[item.subject]
        flagged = item.gap_strength >= policy.strength_threshold
        question = " ".join(item.question.split()) if item.question else None
        result.append(
            SubjectResearchFlag(
                theme_id=subject.theme_id,
                gap_strength=item.gap_strength,
                flagged=flagged,
                question=question if flagged else None,
                evidence_types=tuple(item.evidence_types) if flagged else (),
            )
        )
    missing = set(themes) - {item.theme_id for item in result}
    if missing:
        raise ValueError(f"Trigger response missed subjects: {sorted(missing)}")
    return tuple(result)


def _trigger_context(value: DailyResearchTriggerInput) -> str:
    lines = [f"Report day: {value.day.isoformat()}"]
    for item in value.subjects:
        lines.extend(
            (
                f"Subject {item.alias}: {item.title}",
                f"Event groups: {item.event_count}; articles: {item.article_count}",
                f"Summary: {item.summary}",
            )
        )
    return "\n".join(lines)


def _response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["subjects"],
        "properties": {
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subject", "gap_strength", "question", "evidence_types"],
                    "properties": {
                        "subject": {"type": "string", "pattern": "^subject_[0-9]{2}$"},
                        "gap_strength": {"type": "number", "minimum": 0, "maximum": 1},
                        "question": {"type": ["string", "null"], "minLength": 1},
                        "evidence_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "entity_context",
                                    "historical_data",
                                    "financial_data",
                                    "external_corroboration",
                                    "coverage_review",
                                    "investigative_report",
                                ],
                            },
                        },
                    },
                },
            }
        },
    }


def _model_request_id(
    parent_request_id: Sha256,
    messages: tuple[TriggerMessage, ...],
    response_schema: dict[str, object],
    policy: ResearchTriggerPolicy,
) -> Sha256:
    return _sha256(
        _canonical_json(
            {
                "parent_request_id": parent_request_id,
                "messages": [message.model_dump(mode="json") for message in messages],
                "response_schema_digest": _sha256(_canonical_json(response_schema)),
                "policy": policy.model_dump(mode="json"),
            }
        )
    )


def _messages_digest(messages: tuple[TriggerMessage, ...]) -> Sha256:
    return _sha256(_canonical_json([message.model_dump(mode="json") for message in messages]))


def _output(trigger_set: DailyResearchTriggerSet) -> DailyResearchTriggerOutput:
    content = _canonical_json(trigger_set.model_dump(mode="json"))
    return DailyResearchTriggerOutput(
        trigger_set=trigger_set,
        content_digest=_sha256(content),
        content=content,
    )
