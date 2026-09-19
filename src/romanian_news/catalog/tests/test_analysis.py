from romanian_news.analysis.artifacts import ArtifactReference
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.groups.models import GroupSummaryOutput
from romanian_news.analysis.relevance import (
    RELEVANCE_POLICY_V1,
    RelevanceDecision,
    RelevanceOutput,
)
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ContextGateResult,
    GateCall,
    ImpactDecision,
    ImpactGateResult,
    RelevanceV3Output,
)
from romanian_news.catalog.analysis import (
    _analysis_catalog_statements,
    _analysis_file,
    _analysis_run_id,
)


def test_group_analysis_catalog_guards_cluster_input_before_articles() -> None:
    cluster = _reference("clusters", "c")
    article = _reference("article", "a")
    output = GroupSummaryOutput.model_construct(
        request_id="r" * 64,
        cluster_set=cluster,
        group_id="g" * 64,
        articles=(article,),
        call=ModelCall(
            response_id="response-1",
            model="model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        content=b'{"provider_response":null}',
    )
    run_id = _analysis_run_id(output)
    file = _analysis_file(output, run_id)

    statements = _analysis_catalog_statements(output, file, run_id, "implementation")

    inputs = [params for sql, params in statements if "INTO run_inputs" in sql]
    assert [(params[2], params[3]) for params in inputs] == [
        (cluster.version_id, "cluster_set"),
        (article.version_id, "article"),
    ]
    guarded_updates = [
        (sql, params) for sql, params in statements if "SET current_version_id" in sql
    ]
    assert len(guarded_updates) == 1
    assert "FROM run_inputs input" in guarded_updates[0][0]
    assert guarded_updates[0][1] == [
        file.version_id,
        run_id,
        file.artifact_id,
        run_id,
        file.version_id,
        run_id,
    ]


def test_relevance_catalog_records_derived_acceptance() -> None:
    article = _reference("article", "a")
    decision = RelevanceDecision(
        national_reach="nationwide",
        consequence_magnitude="major",
        political_relevance="strong",
        economic_relevance="none",
        romania_relevance="strong",
        confidence=0.9,
        evidence_quote="Dovadă",
        reason_ro="Relevant",
    )
    output = RelevanceOutput.model_construct(
        request_id="r" * 64,
        policy=RELEVANCE_POLICY_V1,
        article=article,
        decision=decision,
        call=ModelCall(
            response_id="response-1",
            model="model",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        content=b'{"provider_responses":[]}',
    )
    run_id = _analysis_run_id(output)
    file = _analysis_file(output, run_id)

    statements = _analysis_catalog_statements(output, file, run_id, "implementation")

    metadata = next(params for sql, params in statements if "news_relevance_versions" in sql)
    assert metadata == [file.version_id, 1]


def test_v3_relevance_catalog_records_both_gate_calls() -> None:
    article = _reference("article", "a")
    context_call = ModelCall(
        response_id="context-response",
        model="model",
        input_tokens=12,
        output_tokens=7,
        latency_ms=10,
    )
    impact_call = ModelCall(
        response_id="impact-response",
        model="model",
        input_tokens=18,
        output_tokens=9,
        latency_ms=20,
    )
    output = RelevanceV3Output.model_construct(
        request_id="r" * 64,
        policy=RELEVANCE_V3_POLICY,
        mode="production_early_exit",
        execution_ref=None,
        article=article,
        context=ContextGateResult(
            decision=ContextDecision(
                subject_role="principal",
                news_cycle="current_cycle",
                romanian_consequence="direct",
                certainty="clear",
                evidence_quote="Dovadă",
                reason_ro="Relevant",
            ),
            provider=GateCall(
                request_id="c" * 64,
                call=context_call,
                cost_usd=0.003,
                response_count=1,
                traces=(),
                accounting_complete=True,
            ),
        ),
        impact=ImpactGateResult(
            decision=ImpactDecision(
                consequence_status="realized",
                effect_basis="actual_consequence",
                effect_scope="broad_population",
                magnitude="major",
                political_relevance="strong",
                economic_relevance="none",
                quantified=True,
                certainty="clear",
                evidence_quote="Dovadă",
                reason_ro="Relevant",
            ),
            provider=GateCall(
                request_id="d" * 64,
                call=impact_call,
                cost_usd=0.004,
                response_count=1,
                traces=(),
                accounting_complete=True,
            ),
        ),
        accepted=True,
        content=(
            b'{"accepted":true,'
            b'"context_provider_responses":[{"usage":{"cost":0.003}}],'
            b'"impact_provider_responses":[{"usage":{"cost":0.004}}]}'
        ),
    )
    run_id = _analysis_run_id(output)
    file = _analysis_file(output, run_id)

    statements = _analysis_catalog_statements(output, file, run_id, "implementation")

    run = next(params for sql, params in statements if "INSERT INTO runs" in sql)
    metadata = next(params for sql, params in statements if "news_relevance_versions" in sql)
    model_call = next(params for sql, params in statements if "news_model_calls" in sql)
    assert run[1] == "news.relevance.v3"
    assert metadata == [file.version_id, 1]
    assert model_call == [
        file.version_id,
        "news.relevance.v3",
        "model",
        30,
        16,
        0.007,
        30,
        2,
    ]


def _reference(name: str, digest_character: str) -> ArtifactReference:
    digest = digest_character * 64
    return ArtifactReference(
        artifact_id=f"news:{name}",
        version_id=digest,
        content_digest=digest,
        r2_key=f"news/{name}.json",
    )
