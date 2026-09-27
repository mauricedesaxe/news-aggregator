from __future__ import annotations

import math
import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Generic, Literal, TypeVar

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import Field, model_validator

from romanian_news import NewsModel, Sha256
from romanian_news.analysis.attempts import ModelCall, ProviderResponse, record_model_attempt
from romanian_news.analysis.client import openrouter_client
from romanian_news.analysis.relevance import ArticleAnalysisInput
from romanian_news.analysis.tracing import (
    ModelTraceReference,
    ProviderCallResult,
    trace_provider_call,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.identity import canonical_json, sha256

CONTEXT_OPERATION_KEY = "news.relevance.v3.context"
IMPACT_OPERATION_KEY = "news.relevance.v3.impact"

CONTEXT_PROMPT = (
    "Assess only the article's Romanian context. Return subject_role as principal when Romania is "
    "the main claim, secondary when a material Romanian claim is one part of the article, incidental "
    "for a passing mention, or absent when Romania is not a subject. Return news_cycle as "
    "current_cycle when the main claim reports a current event or present consequence. Return "
    "historical_retrospective only when the main claim looks back at history without a current "
    "consequence. Background chronology inside current news remains current_cycle. Return "
    "romanian_consequence as direct for a stated Romanian effect, quoted for an attributed claim, "
    "possible for a conditional or uncertain effect, or absent when no Romanian effect is stated. "
    "Quoted means the Romanian consequence exists only as an attributed claimed benefit or harm. "
    "An attributed action in an active Romanian government-formation process is direct, not quoted. "
    "Use certainty clear only when the article supports the classification without material doubt. "
    "Use uncertain near a boundary so a relevant article is not rejected. Return one exact quote "
    "from the supplied article and a short reason in Romanian. Do not assess impact or decide whether "
    "the article should be accepted. Anchor the article's main claim to its title and opening "
    "statement. For a roundup, listicle, or commentary whose title and opening say it analyzes "
    "communication, rhetoric, body language, or reactions, classify that analysis itself. The "
    "underlying events are source material. A political scenario mentioned inside one item does not "
    "become the article's stated Romanian consequence."
)
IMPACT_PROMPT = (
    "Assess only the Romanian consequence stated by the article. Return consequence_status as "
    "realized for an effect that happened, committed for an enacted or binding decision, proposed "
    "for a concrete proposal, forecast for a concrete prediction, or hypothetical for a conditional "
    "possibility or counterfactual upside. Return effect_basis as actual_consequence for an effect "
    "that occurs or would occur under the reported status. Return foregone_opportunity when the article "
    "values an unrealized project or benefit. A project that never secured funding is a foregone "
    "opportunity even when its exclusion is final. Measure the reported consequence, not the "
    "counterfactual value of an unrealized project. Keep effect_scope independent from the article's "
    "subject. Choose individual, organization, sector, broad_population, national_market, "
    "local_public_finances, public_finances, or several_systems for who or what receives the effect. "
    "Public_finances means Romania's national public finances. City and county budgets are "
    "local_public_finances. Return quantified as true only when the article states a numeric amount or "
    "quantity for the Romanian consequence. Return magnitude as narrow, routine, or major. Strong "
    "political relevance requires national government power, legislation, policy change, or operation "
    "of national institutions. A newly published status report about ongoing prison or police "
    "conditions is not strong by itself. Strong economic relevance requires material sector-wide or "
    "national conditions. Assess the article's main economic consequence, not an opening anecdote. "
    "Use certainty "
    "clear only when the article supports the classification without material doubt. Use uncertain "
    "near a boundary so a relevant article is not rejected. Return one exact quote from the supplied "
    "article and a short reason in Romanian. Do not assess Romanian subject centrality or decide "
    "whether the article should be accepted. Keep social, health, education, crime, and human-rights "
    "harm separate from political and economic relevance. Existing law, an old bill, institutional "
    "tolerance, or calls for reform do not make a continuing social condition strongly political. "
    "Limited education or work prospects for affected individuals do not make it strongly economic. "
    "Use strong only when the article's main claim reports a current change to national power, law, "
    "policy, institutional operation, a material economic sector, or the national economy."
)
CORRECTION_INSTRUCTION = (
    "The response was invalid. Return complete JSON and copy evidence_quote verbatim from the "
    "article title or body. Validation error: {error}"
)

SubjectRole = Literal["principal", "secondary", "incidental", "absent"]
NewsCycle = Literal["current_cycle", "historical_retrospective"]
RomanianConsequence = Literal["direct", "quoted", "possible", "absent"]
ConsequenceStatus = Literal["realized", "committed", "proposed", "forecast", "hypothetical"]
EffectBasis = Literal["actual_consequence", "foregone_opportunity"]
EffectScope = Literal[
    "individual",
    "organization",
    "sector",
    "broad_population",
    "national_market",
    "local_public_finances",
    "public_finances",
    "several_systems",
]
Magnitude = Literal["narrow", "routine", "major"]
RelevanceStrength = Literal["none", "weak", "strong"]
Certainty = Literal["clear", "uncertain"]
ExecutionMode = Literal["full_evaluation", "production_early_exit"]


class ContextDecision(NewsModel):
    subject_role: SubjectRole
    news_cycle: NewsCycle
    romanian_consequence: RomanianConsequence
    certainty: Certainty
    evidence_quote: Annotated[str, Field(min_length=1)]
    reason_ro: Annotated[str, Field(min_length=1)]


class ImpactDecision(NewsModel):
    consequence_status: ConsequenceStatus
    effect_basis: EffectBasis
    effect_scope: EffectScope
    magnitude: Magnitude
    political_relevance: RelevanceStrength
    economic_relevance: RelevanceStrength
    quantified: bool
    certainty: Certainty
    evidence_quote: Annotated[str, Field(min_length=1)]
    reason_ro: Annotated[str, Field(min_length=1)]


class GatePolicy(NewsModel):
    prompt: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    temperature: float
    max_tokens: Annotated[int, Field(gt=0)]
    response_schema_name: Annotated[str, Field(min_length=1)]
    semantic_correction_attempts: Annotated[int, Field(ge=0)]
    correction_instruction: Annotated[str, Field(min_length=1)]
    correction_fallback: Literal["article_title"]
    evidence_policy: Annotated[str, Field(min_length=1)]
    text_policy: Annotated[str, Field(min_length=1)]
    max_body_characters: Annotated[int, Field(gt=0)]
    provider_require_parameters: bool


class ContextAcceptancePolicy(NewsModel):
    reject_clear_subject_roles: tuple[SubjectRole, ...]
    reject_clear_news_cycles: tuple[NewsCycle, ...]
    reject_clear_consequences: tuple[RomanianConsequence, ...]
    reject_clear_incidental_without_direct_consequence: bool
    uncertain_passes: bool


class ImpactAcceptancePolicy(NewsModel):
    uncertain_passes: bool
    accepted_effect_scopes: tuple[EffectScope, ...]
    accepted_magnitudes: tuple[Magnitude, ...]
    accepted_relevance: tuple[RelevanceStrength, ...]
    accepted_concrete_statuses: tuple[ConsequenceStatus, ...]
    accept_realized_public_finances: bool


class CombinedAcceptancePolicy(NewsModel):
    accepted_effect_bases: tuple[EffectBasis, ...]
    principal_current_subject_roles: tuple[SubjectRole, ...]
    principal_current_news_cycles: tuple[NewsCycle, ...]
    principal_current_accepted_statuses: tuple[ConsequenceStatus, ...]
    principal_current_accepted_scopes: tuple[EffectScope, ...]
    default_rejected_subject_roles: tuple[SubjectRole, ...]
    secondary_exception_consequences: tuple[RomanianConsequence, ...]
    secondary_exception_effect_bases: tuple[EffectBasis, ...]
    secondary_exception_statuses: tuple[ConsequenceStatus, ...]
    secondary_exception_scopes: tuple[EffectScope, ...]
    secondary_exception_magnitudes: tuple[Magnitude, ...]
    secondary_exception_relevance: tuple[RelevanceStrength, ...]
    rejected_quoted_statuses: tuple[ConsequenceStatus, ...]
    foregone_public_finance_effect_bases: tuple[EffectBasis, ...]
    foregone_public_finance_statuses: tuple[ConsequenceStatus, ...]
    foregone_public_finance_scopes: tuple[EffectScope, ...]
    foregone_public_finance_magnitudes: tuple[Magnitude, ...]
    quantified_actual_market_effect_bases: tuple[EffectBasis, ...]
    quantified_actual_market_scopes: tuple[EffectScope, ...]
    quantified_actual_market_magnitudes: tuple[Magnitude, ...]


class AcceptancePolicy(NewsModel):
    acceptance_algorithm_id: Annotated[str, Field(min_length=1)]
    context: ContextAcceptancePolicy
    impact: ImpactAcceptancePolicy
    combined: CombinedAcceptancePolicy


class RelevanceV3Policy(NewsModel):
    policy_id: Annotated[str, Field(min_length=1)]
    context: GatePolicy
    impact: GatePolicy
    acceptance: AcceptancePolicy

    @model_validator(mode="after")
    def require_identical_article_input(self) -> RelevanceV3Policy:
        context_input = (self.context.text_policy, self.context.max_body_characters)
        impact_input = (self.impact.text_policy, self.impact.max_body_characters)
        if context_input != impact_input:
            raise ValueError("V3 gates must receive the same article input")
        return self


def _gate_policy(prompt: str, schema_name: str) -> GatePolicy:
    return GatePolicy(
        prompt=prompt,
        model="google/gemini-2.5-flash",
        temperature=0,
        max_tokens=800,
        response_schema_name=schema_name,
        semantic_correction_attempts=1,
        correction_instruction=CORRECTION_INSTRUCTION,
        correction_fallback="article_title",
        evidence_policy="verbatim-then-article-title-v1",
        text_policy="published-title-body-24000-v1",
        max_body_characters=24000,
        provider_require_parameters=True,
    )


RELEVANCE_V3_POLICY = RelevanceV3Policy(
    policy_id="relevance-v3-main-claim-domain-guard",
    context=_gate_policy(CONTEXT_PROMPT, "romanian_news_relevance_v3_context"),
    impact=_gate_policy(IMPACT_PROMPT, "romanian_news_relevance_v3_impact").model_copy(
        update={"max_tokens": 2048}
    ),
    acceptance=AcceptancePolicy(
        acceptance_algorithm_id="recall-first-combined-v3-attempt-3-material-v3",
        context=ContextAcceptancePolicy(
            reject_clear_subject_roles=("absent",),
            reject_clear_news_cycles=("historical_retrospective",),
            reject_clear_consequences=("absent",),
            reject_clear_incidental_without_direct_consequence=True,
            uncertain_passes=True,
        ),
        impact=ImpactAcceptancePolicy(
            uncertain_passes=True,
            accepted_effect_scopes=(
                "sector",
                "broad_population",
                "national_market",
                "public_finances",
                "several_systems",
            ),
            accepted_magnitudes=("routine", "major"),
            accepted_relevance=("strong",),
            accepted_concrete_statuses=("realized", "committed", "proposed", "forecast"),
            accept_realized_public_finances=True,
        ),
        combined=CombinedAcceptancePolicy(
            accepted_effect_bases=("actual_consequence",),
            principal_current_subject_roles=("principal",),
            principal_current_news_cycles=("current_cycle",),
            principal_current_accepted_statuses=("hypothetical",),
            principal_current_accepted_scopes=("organization",),
            default_rejected_subject_roles=("secondary",),
            secondary_exception_consequences=("direct", "quoted"),
            secondary_exception_effect_bases=("actual_consequence",),
            secondary_exception_statuses=("forecast",),
            secondary_exception_scopes=("national_market",),
            secondary_exception_magnitudes=("routine", "major"),
            secondary_exception_relevance=("strong",),
            rejected_quoted_statuses=("committed",),
            foregone_public_finance_effect_bases=("foregone_opportunity",),
            foregone_public_finance_statuses=("realized",),
            foregone_public_finance_scopes=("public_finances",),
            foregone_public_finance_magnitudes=("routine", "major"),
            quantified_actual_market_effect_bases=("actual_consequence",),
            quantified_actual_market_scopes=("national_market",),
            quantified_actual_market_magnitudes=("routine", "major"),
        ),
    ),
)


class GateTraceReference(NewsModel):
    trace_id: Annotated[str, Field(min_length=1)]
    observation_id: Annotated[str, Field(min_length=1)]


class GateCall(NewsModel):
    request_id: Sha256
    call: ModelCall
    cost_usd: Annotated[float, Field(ge=0)]
    response_count: Annotated[int, Field(gt=0)]
    traces: tuple[GateTraceReference, ...]
    accounting_complete: bool

    @property
    def observability_complete(self) -> bool:
        return self.accounting_complete and len(self.traces) == self.response_count


class ContextGateResult(NewsModel):
    decision: ContextDecision
    provider: GateCall


class ImpactGateResult(NewsModel):
    decision: ImpactDecision
    provider: GateCall


class RelevanceV3Output(NewsModel):
    request_id: Sha256
    policy: RelevanceV3Policy
    mode: ExecutionMode
    execution_ref: str | None
    article: ArtifactReference
    context: ContextGateResult
    impact: ImpactGateResult | None
    accepted: bool
    content: bytes

    @property
    def policy_digest(self) -> Sha256:
        return relevance_v3_policy_digest(self.policy)

    @property
    def context_early_exit(self) -> bool:
        return not context_is_accepted(self.context.decision, self.policy.acceptance.context)

    @property
    def input_tokens(self) -> int:
        return self.context.provider.call.input_tokens + (
            self.impact.provider.call.input_tokens if self.impact else 0
        )

    @property
    def output_tokens(self) -> int:
        return self.context.provider.call.output_tokens + (
            self.impact.provider.call.output_tokens if self.impact else 0
        )

    @property
    def cost_usd(self) -> float:
        return self.context.provider.cost_usd + (
            self.impact.provider.cost_usd if self.impact else 0
        )

    @property
    def observability_complete(self) -> bool:
        return self.context.provider.observability_complete and (
            self.impact is None or self.impact.provider.observability_complete
        )

    @model_validator(mode="after")
    def require_mode_result(self) -> RelevanceV3Output:
        context_accepted = context_is_accepted(
            self.context.decision, self.policy.acceptance.context
        )
        if self.mode == "full_evaluation" and not self.execution_ref:
            raise ValueError("Full V3 evaluation requires an execution reference")
        if self.mode == "full_evaluation" and self.impact is None:
            raise ValueError("Full V3 evaluation requires both gate results")
        if self.impact is None and context_accepted:
            raise ValueError("V3 impact can be omitted only after a context rejection")
        expected = self.impact is not None and relevance_v3_is_accepted(
            self.context.decision,
            self.impact.decision,
            self.policy.acceptance,
        )
        if self.accepted != expected:
            raise ValueError("V3 acceptance conflicts with the gate decisions")
        return self


def analyze_relevance_v3(
    value: ArticleAnalysisInput,
    *,
    mode: ExecutionMode,
    policy: RelevanceV3Policy = RELEVANCE_V3_POLICY,
    execution_ref: str | None = None,
) -> RelevanceV3Output:
    request_id = relevance_v3_request_id(
        value.reference,
        mode=mode,
        policy=policy,
        execution_ref=execution_ref,
    )
    context_execution = _run_gate(
        value,
        request_id=request_id,
        operation_key=CONTEXT_OPERATION_KEY,
        policy=policy.context,
        decision_type=ContextDecision,
        execution_ref=execution_ref,
    )
    context = ContextGateResult(
        decision=ContextDecision.model_validate(context_execution.decision),
        provider=context_execution.provider,
    )
    context_accepted = context_is_accepted(context.decision, policy.acceptance.context)
    impact = None
    impact_responses: tuple[ChatCompletion, ...] = ()
    if mode == "full_evaluation" or context_accepted:
        impact_execution = _run_gate(
            value,
            request_id=request_id,
            operation_key=IMPACT_OPERATION_KEY,
            policy=policy.impact,
            decision_type=ImpactDecision,
            execution_ref=execution_ref,
        )
        impact = ImpactGateResult(
            decision=ImpactDecision.model_validate(impact_execution.decision),
            provider=impact_execution.provider,
        )
        impact_responses = impact_execution.responses
    accepted = impact is not None and relevance_v3_is_accepted(
        context.decision,
        impact.decision,
        policy.acceptance,
    )
    payload = canonical_json(
        {
            "accepted": accepted,
            "article_version_id": value.reference.version_id,
            "context": context.model_dump(mode="json"),
            "context_provider_responses": [
                response.model_dump(mode="json") for response in context_execution.responses
            ],
            "impact": impact.model_dump(mode="json") if impact else None,
            "impact_provider_responses": [
                response.model_dump(mode="json") for response in impact_responses
            ],
            "mode": mode,
            "execution_ref": execution_ref,
            "policy": relevance_v3_policy_payload(policy),
            "policy_digest": relevance_v3_policy_digest(policy),
            "request_id": request_id,
        }
    )
    return RelevanceV3Output(
        request_id=request_id,
        policy=policy,
        mode=mode,
        execution_ref=execution_ref,
        article=value.reference,
        context=context,
        impact=impact,
        accepted=accepted,
        content=payload,
    )


def context_is_accepted(decision: ContextDecision, policy: ContextAcceptancePolicy) -> bool:
    if decision.certainty == "uncertain":
        return policy.uncertain_passes
    incidental_without_direct = (
        policy.reject_clear_incidental_without_direct_consequence
        and decision.subject_role == "incidental"
        and decision.romanian_consequence != "direct"
    )
    return (
        decision.subject_role not in policy.reject_clear_subject_roles
        and decision.news_cycle not in policy.reject_clear_news_cycles
        and decision.romanian_consequence not in policy.reject_clear_consequences
        and not incidental_without_direct
    )


def relevance_v3_is_accepted(
    context: ContextDecision,
    impact: ImpactDecision,
    policy: AcceptancePolicy,
) -> bool:
    if not context_is_accepted(context, policy.context):
        return False
    if context.certainty == "uncertain" or impact.certainty == "uncertain":
        return policy.context.uncertain_passes and policy.impact.uncertain_passes
    if context.subject_role in policy.combined.default_rejected_subject_roles:
        return _accepts_secondary_forecast(context, impact, policy.combined)
    if _accepts_foregone_public_finance_loss(impact, policy.combined):
        return True
    if impact.effect_basis not in policy.combined.accepted_effect_bases:
        return False
    if (
        context.romanian_consequence == "quoted"
        and impact.consequence_status in policy.combined.rejected_quoted_statuses
    ):
        return False
    principal_current = (
        context.subject_role in policy.combined.principal_current_subject_roles
        and context.news_cycle in policy.combined.principal_current_news_cycles
        and context.romanian_consequence != "absent"
    )
    if (
        principal_current
        and impact.consequence_status in policy.combined.principal_current_accepted_statuses
        and _has_accepted_materiality(impact, policy.impact)
    ):
        return True
    if (
        principal_current
        and impact.effect_scope in policy.combined.principal_current_accepted_scopes
        and _has_accepted_materiality(impact, policy.impact)
    ):
        return True
    if _accepts_quantified_actual_market_consequence(impact, policy.combined):
        return True
    return _impact_is_accepted(impact, policy.impact)


def _accepts_secondary_forecast(
    context: ContextDecision,
    impact: ImpactDecision,
    policy: CombinedAcceptancePolicy,
) -> bool:
    return (
        context.romanian_consequence in policy.secondary_exception_consequences
        and impact.effect_basis in policy.secondary_exception_effect_bases
        and impact.consequence_status in policy.secondary_exception_statuses
        and impact.effect_scope in policy.secondary_exception_scopes
        and impact.magnitude in policy.secondary_exception_magnitudes
        and (
            impact.political_relevance in policy.secondary_exception_relevance
            or impact.economic_relevance in policy.secondary_exception_relevance
        )
    )


def _accepts_foregone_public_finance_loss(
    impact: ImpactDecision,
    policy: CombinedAcceptancePolicy,
) -> bool:
    return (
        impact.effect_basis in policy.foregone_public_finance_effect_bases
        and impact.consequence_status in policy.foregone_public_finance_statuses
        and impact.effect_scope in policy.foregone_public_finance_scopes
        and impact.magnitude in policy.foregone_public_finance_magnitudes
        and impact.quantified
    )


def _accepts_quantified_actual_market_consequence(
    impact: ImpactDecision,
    policy: CombinedAcceptancePolicy,
) -> bool:
    return (
        impact.effect_basis in policy.quantified_actual_market_effect_bases
        and impact.effect_scope in policy.quantified_actual_market_scopes
        and impact.magnitude in policy.quantified_actual_market_magnitudes
        and impact.quantified
    )


def _has_accepted_materiality(
    impact: ImpactDecision,
    policy: ImpactAcceptancePolicy,
) -> bool:
    return impact.magnitude in policy.accepted_magnitudes and (
        impact.political_relevance in policy.accepted_relevance
        or impact.economic_relevance in policy.accepted_relevance
    )


def _impact_is_accepted(decision: ImpactDecision, policy: ImpactAcceptancePolicy) -> bool:
    if decision.certainty == "uncertain":
        return policy.uncertain_passes
    if (
        policy.accept_realized_public_finances
        and decision.consequence_status == "realized"
        and decision.effect_scope == "public_finances"
        and decision.quantified
        and decision.magnitude in policy.accepted_magnitudes
    ):
        return True
    return (
        decision.consequence_status in policy.accepted_concrete_statuses
        and decision.effect_scope in policy.accepted_effect_scopes
        and decision.magnitude in policy.accepted_magnitudes
        and (
            decision.political_relevance in policy.accepted_relevance
            or decision.economic_relevance in policy.accepted_relevance
        )
    )


def relevance_v3_policy_payload(policy: RelevanceV3Policy) -> dict[str, object]:
    return {
        **policy.model_dump(mode="json"),
        "context_response_schema": ContextDecision.model_json_schema(),
        "impact_response_schema": ImpactDecision.model_json_schema(),
    }


def relevance_v3_policy_digest(policy: RelevanceV3Policy = RELEVANCE_V3_POLICY) -> Sha256:
    return sha256(canonical_json(relevance_v3_policy_payload(policy)))


def production_relevance_v3_request_id(
    article: ArtifactReference,
    policy: RelevanceV3Policy = RELEVANCE_V3_POLICY,
) -> Sha256:
    """Identify one production V3 relevance analysis."""
    return relevance_v3_request_id(article, mode="production_early_exit", policy=policy)


def relevance_v3_request_id(
    article: ArtifactReference,
    *,
    mode: ExecutionMode,
    policy: RelevanceV3Policy = RELEVANCE_V3_POLICY,
    execution_ref: str | None = None,
) -> Sha256:
    if mode == "full_evaluation" and not execution_ref:
        raise ValueError("Full V3 evaluation requires an execution reference")
    return sha256(
        canonical_json(
            {
                "article_version_id": article.version_id,
                "execution_mode": mode,
                "execution_ref": execution_ref,
                "policy": relevance_v3_policy_payload(policy),
                "policy_digest": relevance_v3_policy_digest(policy),
                "purpose": "news.relevance.v3",
            }
        )
    )


DecisionT = TypeVar("DecisionT", ContextDecision, ImpactDecision)


@dataclass(frozen=True)
class _AttemptResponse:
    payload: dict[str, object]

    def model_dump(self, *, mode: str) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True)
class _GateExecution(Generic[DecisionT]):
    decision: DecisionT
    provider: GateCall
    responses: tuple[ChatCompletion, ...]


def _run_gate(
    value: ArticleAnalysisInput,
    *,
    request_id: Sha256,
    operation_key: str,
    policy: GatePolicy,
    decision_type: type[DecisionT],
    execution_ref: str | None,
) -> _GateExecution[DecisionT]:
    gate_request_id = _gate_request_id(request_id, operation_key, execution_ref)
    started = time.monotonic()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": policy.prompt},
        {
            "role": "user",
            "content": relevance_v3_article_text(value, policy.max_body_characters),
        },
    ]
    responses: list[ChatCompletion] = []
    traces: list[GateTraceReference] = []
    for attempt in range(policy.semantic_correction_attempts + 1):
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": policy.response_schema_name,
                "strict": True,
                "schema": decision_type.model_json_schema(),
            },
        }
        trace_inputs: dict[str, object] = {
            "model": policy.model,
            "messages": messages,
            "temperature": policy.temperature,
            "max_tokens": policy.max_tokens,
            "response_format": response_format,
            "extra_body": {"provider": {"require_parameters": policy.provider_require_parameters}},
        }

        attempt_started = time.monotonic()
        provider_call = trace_provider_call(
            operation_key,
            gate_request_id,
            trace_inputs,
            partial(_create_completion, policy, messages, response_format),
        )
        response = provider_call.response
        responses.append(response)
        latency_ms = round((time.monotonic() - attempt_started) * 1000)
        content = response.choices[0].message.content
        decision = None
        try:
            if not content:
                raise ValueError("Relevance gate returned no content")
            decision = decision_type.model_validate_json(content)
            _require_evidence(decision.evidence_quote, value)
        except ValueError as error:
            if attempt == policy.semantic_correction_attempts and decision is not None:
                decision = decision.model_copy(update={"evidence_quote": value.article.title})
                trace = _record_attempt(
                    response,
                    gate_request_id,
                    operation_key,
                    attempt,
                    latency_ms,
                    "accepted",
                    None,
                    provider_call,
                )
                if trace is not None:
                    traces.append(trace)
                break
            trace = _record_attempt(
                response,
                gate_request_id,
                operation_key,
                attempt,
                latency_ms,
                "rejected",
                str(error),
                provider_call,
            )
            if trace is not None:
                traces.append(trace)
            if attempt == policy.semantic_correction_attempts:
                raise ValueError(
                    f"{operation_key} remained invalid after correction: {error}"
                ) from error
            messages.extend(
                (
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": policy.correction_instruction.format(error=error),
                    },
                )
            )
            continue
        trace = _record_attempt(
            response,
            gate_request_id,
            operation_key,
            attempt,
            latency_ms,
            "accepted",
            None,
            provider_call,
        )
        if trace is not None:
            traces.append(trace)
        break
    else:
        raise RuntimeError("Relevance V3 correction loop did not return")
    return _GateExecution(
        decision=decision,
        provider=GateCall(
            request_id=gate_request_id,
            call=_model_call(responses, round((time.monotonic() - started) * 1000)),
            cost_usd=sum(_response_cost(response) for response in responses),
            response_count=len(responses),
            traces=tuple(traces),
            accounting_complete=all(
                _response_accounting_complete(response) for response in responses
            ),
        ),
        responses=tuple(responses),
    )


def _record_attempt(
    response: ChatCompletion,
    request_id: Sha256,
    operation_key: str,
    attempt_index: int,
    latency_ms: int,
    status: Literal["accepted", "rejected"],
    error: str | None,
    provider_call: ProviderCallResult[ChatCompletion],
) -> GateTraceReference | None:
    record_model_attempt(
        _attempt_response(response),
        request_id=request_id,
        operation_key=operation_key,
        attempt_index=attempt_index,
        latency_ms=latency_ms,
        status=status,
        error=error,
        fallback_response_id=str(provider_call.call_id),
        trace=provider_call.trace,
    )
    if provider_call.trace is None:
        return None
    return _gate_trace_reference(provider_call.trace)


def _gate_trace_reference(trace: ModelTraceReference) -> GateTraceReference:
    return GateTraceReference(
        trace_id=trace.trace_id,
        observation_id=trace.observation_id,
    )


def _create_completion(
    policy: GatePolicy,
    messages: list[ChatCompletionMessageParam],
    response_format: ResponseFormatJSONSchema,
) -> ChatCompletion:
    return openrouter_client().chat.completions.create(
        model=policy.model,
        messages=messages,
        temperature=policy.temperature,
        max_tokens=policy.max_tokens,
        response_format=response_format,
        extra_body={"provider": {"require_parameters": policy.provider_require_parameters}},
    )


def relevance_v3_article_text(value: ArticleAnalysisInput, max_body_characters: int) -> str:
    return (
        f"Publicat: {value.article.published_at.isoformat()}\n"
        f"Titlu: {value.article.title}\n\n"
        f"Articol:\n{value.article.body[:max_body_characters]}"
    )


def _require_evidence(evidence_quote: str, value: ArticleAnalysisInput) -> None:
    source_text = f"{value.article.title}\n{value.article.body}"
    evidence = _normalize_evidence_quote(evidence_quote)
    if not evidence or evidence not in _normalize_evidence_quote(source_text):
        raise ValueError("Relevance evidence quote is not present in the article")


def _normalize_evidence_quote(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized))


def _gate_request_id(
    request_id: Sha256,
    operation_key: str,
    execution_ref: str | None,
) -> Sha256:
    return sha256(
        canonical_json(
            {
                "execution_ref": execution_ref,
                "operation_key": operation_key,
                "request_id": request_id,
            }
        )
    )


def _model_call(responses: list[ChatCompletion], latency_ms: int) -> ModelCall:
    response = responses[-1]
    return ModelCall(
        response_id=response.id,
        model=response.model,
        input_tokens=sum(_response_token_count(item, "prompt_tokens") for item in responses),
        output_tokens=sum(_response_token_count(item, "completion_tokens") for item in responses),
        latency_ms=latency_ms,
    )


def _attempt_response(response: ChatCompletion) -> ProviderResponse:
    if _response_accounting_complete(response):
        return response
    payload = response.model_dump(mode="json")
    usage = payload.get("usage")
    values = usage if isinstance(usage, Mapping) else {}
    return _AttemptResponse(
        payload={
            **payload,
            "usage": {
                "prompt_tokens": _storage_token_count(values.get("prompt_tokens")),
                "completion_tokens": _storage_token_count(values.get("completion_tokens")),
                "cost": _storage_cost(values.get("cost")),
            },
        }
    )


def _storage_token_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _storage_cost(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        return 0.0
    return float(value)


def _response_accounting_complete(response: ChatCompletion) -> bool:
    payload = response.model_dump(mode="json")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return False
    return (
        _is_non_negative_number(usage.get("prompt_tokens"), integer=True)
        and _is_non_negative_number(usage.get("completion_tokens"), integer=True)
        and _is_non_negative_number(usage.get("cost"), integer=False)
    )


def _response_token_count(
    response: ChatCompletion,
    field: Literal["prompt_tokens", "completion_tokens"],
) -> int:
    payload = response.model_dump(mode="json")
    usage = payload.get("usage")
    value = usage.get(field) if isinstance(usage, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _is_non_negative_number(value: object, *, integer: bool) -> bool:
    if isinstance(value, bool):
        return False
    if integer:
        return isinstance(value, int) and value >= 0
    return isinstance(value, int | float) and math.isfinite(value) and value >= 0


def _response_cost(response: ChatCompletion) -> float:
    payload = response.model_dump(mode="json")
    usage = payload.get("usage")
    cost = usage.get("cost") if isinstance(usage, Mapping) else None
    return _storage_cost(cost)
