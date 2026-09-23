from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parents[3]))

import pytest
from openai.types.chat import ChatCompletion

from romanian_news.artifacts import ArtifactReference
from romanian_news.binary_tier_evaluation import BinaryTierState, BinaryTierSubjectContext
from romanian_news.catalog.evaluations import NewsEvaluationReportLineage
from romanian_news.evaluation import (
    NewsEvaluationManifest,
    ReportEvaluationSpec,
    ReportInputReferences,
)
from romanian_news.groups import DailyClusterSet
from romanian_news.split_subject_assessment_experiment import (
    V11_MANIFEST_ARTIFACT_ID,
    V11_MANIFEST_CONTENT_DIGEST,
    V11_MANIFEST_R2_KEY,
    V11_MANIFEST_VERSION_ID,
    V11_REPORT_VERSION_IDS,
    ExperimentAttempt,
    FailedArm,
    FrozenV11SourceDescriptor,
    TierObservation,
    UnavailableArm,
    _fixed_tier_schema,
    load_frozen_v11_assessment_inputs,
    load_frozen_v11_source,
    run_candidate_arm,
)
from romanian_news.tests.evaluation_factories import relevance_decision
from romanian_news.tests.test_subject_assessments import _input, _sparse_input
from scripts.run_split_subject_assessment_experiment import (
    GEMINI_REQUEST_RESERVE_USD,
    JEV_REQUEST_RESERVE_USD,
    InFlightArm,
    RunnerArguments,
    SplitExperimentCheckpoint,
    _arm_reserve,
    _candidate_ranking_policy_digest,
    _experiment_artifact_reader,
    _identity,
    _require_arm_budget,
    execute,
)

REPORT_ID = "c1369f9a24202a111dfc07e29a7194c73cb22dd491958093932a5d3b9b270d72"


def _reference(
    version_id: str,
    content: bytes = b"content",
    *,
    artifact_id: str = "artifact",
    r2_key: str | None = None,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        version_id=version_id,
        content_digest=hashlib.sha256(content).hexdigest(),
        r2_key=r2_key or f"objects/{version_id}.json",
    )


def _manifest_reference() -> ArtifactReference:
    return ArtifactReference(
        artifact_id=V11_MANIFEST_ARTIFACT_ID,
        version_id=V11_MANIFEST_VERSION_ID,
        content_digest=V11_MANIFEST_CONTENT_DIGEST,
        r2_key=V11_MANIFEST_R2_KEY,
    )


def _source_descriptor(
    *, article_references: tuple[ArtifactReference, ...] = ()
) -> FrozenV11SourceDescriptor:
    report = _reference(REPORT_ID)
    return FrozenV11SourceDescriptor(
        manifest=_manifest_reference(),
        report_lineages=(
            NewsEvaluationReportLineage(
                report=report,
                inputs=ReportInputReferences(cluster_set=(), summary=(), sentiment=()),
            ),
        ),
        article_references=article_references,
    )


def _attempt(subject_id: str) -> ExperimentAttempt:
    return ExperimentAttempt(
        provider="typesafe",
        stage="tier",
        subject_id=subject_id,
        attempt_number=1,
        status="accepted",
        request_id="d" * 64,
        provider_request_id="jev-request",
        actual_model="jev-1.13.0",
        input_tokens=10,
        output_tokens=2,
        cost_usd=Decimal("0.00000042"),
        latency_ms=3,
    )


def _completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "gemini-request",
            "object": "chat.completion",
            "created": 1,
            "model": "google/gemini-3.8-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost": 0.01,
            },
        }
    )


def test_fixed_tier_schema_locks_membership_and_cardinality() -> None:
    schema = _fixed_tier_schema(
        {"main": ("subject_01",), "worth_knowing": (), "excluded": ("subject_02",)},
        ("article_01", "article_02"),
    )
    properties = cast(Mapping[str, object], schema["properties"])
    main = cast(Mapping[str, object], properties["main"])
    item = cast(Mapping[str, object], main["items"])
    item_properties = cast(Mapping[str, object], item["properties"])
    subject = cast(Mapping[str, object], item_properties["subject"])
    worth = cast(Mapping[str, object], properties["worth_knowing"])
    worth_item = cast(Mapping[str, object], worth["items"])
    worth_properties = cast(Mapping[str, object], worth_item["properties"])
    worth_subject = cast(Mapping[str, object], worth_properties["subject"])
    assert subject["enum"] == ["subject_01"]
    assert main["minItems"] == main["maxItems"] == 1
    assert worth["maxItems"] == 0
    assert worth_subject["enum"] == ["subject_01", "subject_02"]


def test_candidate_batches_both_tier_questions_and_freezes_whole_day() -> None:
    value = _input()
    seen_states: list[BinaryTierState] = []

    def tier(state: BinaryTierState, _execution_ref: str, record) -> TierObservation:
        seen_states.append(state)
        attempt = _attempt(state.target_subject_id)
        record(attempt)
        is_first = state.target_subject_id == value.theme_set.themes[0].id
        return TierObservation(
            subject_id=state.target_subject_id,
            main_probability=Decimal("0.8") if is_first else Decimal("0.1"),
            worth_knowing_probability=Decimal("0.2") if is_first else Decimal("0.8"),
            tier="main" if is_first else "worth_knowing",
            attempts=(attempt,),
        )

    content = json.dumps(
        {
            "main": [
                {
                    "subject": "subject_01",
                    "rationale": "National impact.",
                    "evidence_articles": ["article_01"],
                }
            ],
            "worth_knowing": [
                {
                    "subject": "subject_02",
                    "rationale": "Useful context.",
                    "evidence_articles": ["article_02"],
                }
            ],
            "excluded": [],
        }
    )
    outcome = run_candidate_arm(
        REPORT_ID,
        value,
        execution_ref="test",
        tier_evaluator=tier,
        ranking_caller=lambda _request: _completion(content),
    )

    assert outcome.status == "completed"
    assert len(seen_states) == len(value.theme_set.themes)
    assert all(state.subjects == seen_states[0].subjects for state in seen_states)
    assert [item.tier for item in outcome.assessments] == ["main", "worth_knowing"]
    assert len(outcome.attempts) == 3
    assert outcome.accounting_complete


def test_frozen_v11_loader_reconstructs_exact_assessment_inputs(monkeypatch) -> None:
    value = _sparse_input(monkeypatch)
    cluster = DailyClusterSet(
        day=value.day,
        algorithm="test",
        threshold=0,
        embedding_model="test",
        article_version_ids=tuple(item.article.version_id for item in value.evidence),
        relevance_version_ids=tuple(item.version_id for item in value.relevance),
        embedding_version_ids=tuple("e" * 64 for _item in value.evidence),
        merges=(),
        groups=value.theme_set.groups,
    )
    reports = tuple(
        ReportEvaluationSpec(
            report=ArtifactReference(
                artifact_id=f"news:report:{index}",
                version_id=report_id,
                content_digest=f"{index + 40:064x}",
                r2_key=f"news/reports/{report_id}.json",
            ),
            themes=value.themes,
            cluster_set=value.theme_set.cluster_set,
        )
        for index, report_id in enumerate(V11_REPORT_VERSION_IDS)
    )
    manifest = NewsEvaluationManifest.model_construct(reports=reports)
    lineages = {
        item.report.version_id: NewsEvaluationReportLineage(
            report=item.report,
            inputs=ReportInputReferences(
                themes=(value.themes,),
                cluster_set=(value.theme_set.cluster_set,),
                relevance=value.relevance,
                summary=tuple(item.reference for item in value.summaries),
                sentiment=(),
            ),
        )
        for item in reports
    }
    articles = {item.article.version_id: item.article for item in value.evidence}
    artifacts = {
        value.themes.version_id: value.theme_set.model_dump_json().encode(),
        value.theme_set.cluster_set.version_id: cluster.model_dump_json().encode(),
    }
    for item in value.summaries:
        artifacts[item.reference.version_id] = json.dumps(
            {
                "group_id": item.group_id,
                "summary": item.summary.model_dump(mode="json"),
            }
        ).encode()
    for item in value.evidence:
        decision = relevance_decision(accepted=True).model_copy(
            update={"evidence_quote": item.evidence_quote}
        )
        artifacts[item.relevance.version_id] = json.dumps(
            {
                "article_version_id": item.article.version_id,
                "decision": decision.model_dump(mode="json"),
            }
        ).encode()
    loaded = load_frozen_v11_assessment_inputs(
        manifest,
        artifact_reader=lambda reference: artifacts[reference.version_id],
        lineage_reader=lineages.__getitem__,
        version_reference_reader=lambda _article_ids: articles,
    )

    assert tuple(item[0] for item in loaded) == V11_REPORT_VERSION_IDS
    assert all(item[1] == value for item in loaded)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_id", "wrong", "artifact ID"),
        ("version_id", "f" * 64, "version ID"),
        ("content_digest", "f" * 64, "content digest"),
        ("r2_key", "wrong.json", "object key"),
    ),
)
def test_frozen_source_descriptor_rejects_manifest_identity_drift(
    field: str, value: str, message: str
) -> None:
    descriptor = _source_descriptor().model_dump(mode="json")
    descriptor["manifest"][field] = value

    with pytest.raises(ValueError, match=message):
        FrozenV11SourceDescriptor.model_validate_json(json.dumps(descriptor), strict=True)


def test_frozen_source_descriptor_requires_exact_report_coverage() -> None:
    descriptor = _source_descriptor().model_dump(mode="json")
    descriptor["report_lineages"] = []

    with pytest.raises(ValueError, match="exactly cover the v11 reports"):
        FrozenV11SourceDescriptor.model_validate_json(json.dumps(descriptor), strict=True)


def test_frozen_source_loader_rejects_object_digest_mismatch() -> None:
    descriptor = _source_descriptor()

    with pytest.raises(ValueError, match="object digest mismatch"):
        load_frozen_v11_source(descriptor.model_dump_json().encode(), lambda _reference: b"wrong")


def test_frozen_source_loader_requires_exact_article_coverage(monkeypatch) -> None:
    manifest_content = json.dumps(
        {
            "version": "news-evaluation-2026-09-11-v11",
            "reviewed_at": "2026-09-11T00:00:00Z",
            "issue_url": "https://example.test/issue",
            "prior_manifest": None,
            "source_feedback_ids": [],
            "reports": [
                {
                    "report": _reference(REPORT_ID).model_dump(mode="json"),
                    "themes": None,
                    "cluster_set": _reference("b" * 64).model_dump(mode="json"),
                }
            ],
            "cases": [],
        }
    ).encode()
    manifest_reference = _manifest_reference().model_copy(
        update={"content_digest": hashlib.sha256(manifest_content).hexdigest()}
    )
    monkeypatch.setattr(
        "romanian_news.split_subject_assessment_experiment.V11_MANIFEST_CONTENT_DIGEST",
        manifest_reference.content_digest,
    )
    report = _reference(REPORT_ID)
    cluster = _reference("b" * 64)
    descriptor = FrozenV11SourceDescriptor(
        manifest=manifest_reference,
        report_lineages=(
            NewsEvaluationReportLineage(
                report=report,
                inputs=ReportInputReferences(cluster_set=(cluster,), summary=(), sentiment=()),
            ),
        ),
        article_references=(),
    )
    content = {
        manifest_reference.version_id: manifest_content,
        report.version_id: b"content",
        cluster.version_id: b"content",
    }

    def load_inputs(_manifest, **kwargs):
        kwargs["version_reference_reader"](("c" * 64,))
        raise AssertionError("article coverage check should fail")

    monkeypatch.setattr(
        "romanian_news.split_subject_assessment_experiment.load_frozen_v11_assessment_inputs",
        load_inputs,
    )

    with pytest.raises(ValueError, match="exactly cover the report articles"):
        load_frozen_v11_source(
            descriptor.model_dump_json().encode(),
            lambda reference: content[reference.version_id],
        )


def test_proxy_artifact_reader_verifies_status_and_digest(monkeypatch) -> None:
    content = b"verified"
    reference = _reference("d" * 64, content)
    calls = []

    class Response:
        status_code = 200
        content = b"verified"

    monkeypatch.setenv("NEWS_R2_READ_PROXY_URL", "https://proxy.example.test/read")
    monkeypatch.setattr(
        "scripts.run_split_subject_assessment_experiment.requests.get",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )

    assert _experiment_artifact_reader()(reference) == content
    assert calls == [
        (
            "https://proxy.example.test/read",
            {"params": {"key": reference.r2_key}, "timeout": 30},
        )
    ]
    bad_reference = reference.model_copy(update={"content_digest": "f" * 64})
    with pytest.raises(ValueError, match="proxy digest mismatch"):
        _experiment_artifact_reader()(bad_reference)

    Response.status_code = 503
    with pytest.raises(RuntimeError, match="returned 503"):
        _experiment_artifact_reader()(reference)


def test_candidate_rejects_fixed_tier_change_after_one_correction() -> None:
    value = _input()

    def tier(state: BinaryTierState, _execution_ref: str, record) -> TierObservation:
        attempt = _attempt(state.target_subject_id)
        record(attempt)
        return TierObservation(
            subject_id=state.target_subject_id,
            main_probability=Decimal("0.8"),
            worth_knowing_probability=Decimal("0.1"),
            tier="main",
            attempts=(attempt,),
        )

    invalid = json.dumps(
        {
            "main": [
                {"subject": "subject_01", "rationale": "One.", "evidence_articles": ["article_01"]}
            ],
            "worth_knowing": [
                {
                    "subject": "subject_02",
                    "rationale": "Moved.",
                    "evidence_articles": ["article_02"],
                }
            ],
            "excluded": [],
        }
    )
    calls = 0

    def ranking(_request):
        nonlocal calls
        calls += 1
        return _completion(invalid)

    outcome = run_candidate_arm(
        REPORT_ID, value, execution_ref="test", tier_evaluator=tier, ranking_caller=ranking
    )
    assert outcome.status == "failed"
    assert calls == 2
    assert [item.status for item in outcome.attempts[-2:]] == ["rejected", "rejected"]


def test_candidate_rejects_evidence_owned_by_another_subject() -> None:
    value = _input()

    def tier(state: BinaryTierState, _execution_ref: str, record) -> TierObservation:
        attempt = _attempt(state.target_subject_id)
        record(attempt)
        return TierObservation(
            subject_id=state.target_subject_id,
            main_probability=Decimal("0.8"),
            worth_knowing_probability=Decimal("0.1"),
            tier="main",
            attempts=(attempt,),
        )

    invalid = json.dumps(
        {
            "main": [
                {"subject": "subject_01", "rationale": "One.", "evidence_articles": ["article_02"]},
                {"subject": "subject_02", "rationale": "Two.", "evidence_articles": ["article_02"]},
            ],
            "worth_knowing": [],
            "excluded": [],
        }
    )
    outcome = run_candidate_arm(
        REPORT_ID,
        value,
        execution_ref="test",
        tier_evaluator=tier,
        ranking_caller=lambda _request: _completion(invalid),
    )
    assert outcome.status == "failed"
    assert "evidence must belong" in outcome.error


def test_candidate_refuses_over_guard_without_provider_call(monkeypatch) -> None:
    value = _input()
    subject = BinaryTierSubjectContext.model_construct(
        subject_id="a" * 64,
        title="x" * 70_001,
        summary="summary",
        events=(),
    )
    state = BinaryTierState.model_construct(
        day=value.day, target_subject_id=subject.subject_id, subjects=(subject,)
    )
    monkeypatch.setattr(
        "romanian_news.split_subject_assessment_experiment._tier_states", lambda _value: (state,)
    )
    called = False

    def tier(_state, _execution_ref, _record):
        nonlocal called
        called = True
        raise AssertionError

    outcome = run_candidate_arm(REPORT_ID, value, execution_ref="test", tier_evaluator=tier)
    assert outcome.status == "unavailable"
    assert not called
    assert outcome.attempts == ()


def test_dry_run_checkpoint_identity_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    manifest = NewsEvaluationManifest.model_construct(version="news-evaluation-2026-09-11-v11")
    inputs = ((REPORT_ID, _input()),)
    arguments = RunnerArguments(
        execution_ref="checkpoint-test",
        output=output,
        dry_run=True,
        resume=False,
        retry_failed=False,
    )
    first = execute(arguments, manifest=manifest, inputs=inputs)
    resumed = execute(
        RunnerArguments(
            execution_ref=arguments.execution_ref,
            output=arguments.output,
            dry_run=arguments.dry_run,
            resume=True,
            retry_failed=False,
        ),
        manifest=manifest,
        inputs=inputs,
    )
    assert first == resumed

    with pytest.raises(ValueError, match="identity"):
        _ = execute(
            RunnerArguments(
                execution_ref="different",
                output=output,
                dry_run=True,
                resume=True,
                retry_failed=False,
            ),
            manifest=manifest,
            inputs=inputs,
        )


def test_resume_refuses_in_flight_without_explicit_retry(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    identity = _identity("resume-test", True)
    in_flight = InFlightArm(
        trial_ref="split-assessment-v1:trial-001",
        arm="candidate",
        report_version_id=REPORT_ID,
        attempts=(_attempt("a" * 64),),
    )
    output.write_text(
        SplitExperimentCheckpoint(
            identity_digest=identity,
            execution_ref="resume-test",
            mode="dry_run",
            in_flight=in_flight,
        ).model_dump_json()
    )
    manifest = NewsEvaluationManifest.model_construct(version="news-evaluation-2026-09-11-v11")
    inputs = ((REPORT_ID, _input()),)

    with pytest.raises(ValueError, match="unknown spend"):
        _ = execute(
            RunnerArguments("resume-test", output, True, True, False),
            manifest=manifest,
            inputs=inputs,
        )

    result = execute(
        RunnerArguments("resume-test", output, True, True, True),
        manifest=manifest,
        inputs=inputs,
    )
    assert result.unknown_accounting
    assert result.unknown_in_flight == (in_flight,)
    assert not result.accounting_complete
    assert result.known_spend_usd == in_flight.attempts[0].cost_usd


def test_arm_reserves_include_all_possible_attempts() -> None:
    assert _arm_reserve("incumbent", 99) == Decimal(2) * GEMINI_REQUEST_RESERVE_USD
    assert _arm_reserve("candidate", 2) == (
        Decimal(2 * 4) * JEV_REQUEST_RESERVE_USD + Decimal(2) * GEMINI_REQUEST_RESERVE_USD
    )
    _require_arm_budget(Decimal("4.896"), Decimal("0.104"))
    with pytest.raises(RuntimeError, match="cannot admit"):
        _require_arm_budget(Decimal("4.8960001"), Decimal("0.104"))


def test_identity_changes_when_candidate_ranking_policy_drifts(monkeypatch) -> None:
    original = _identity("drift-test", True)
    assert len(_candidate_ranking_policy_digest()) == 64
    monkeypatch.setattr(
        "scripts.run_split_subject_assessment_experiment._candidate_ranking_policy_digest",
        lambda: "f" * 64,
    )
    assert _identity("drift-test", True) != original


def test_identity_changes_when_frozen_protocol_content_drifts(monkeypatch) -> None:
    original = _identity("drift-test", True)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"changed protocol")
    assert _identity("drift-test", True) != original


def test_identity_changes_when_frozen_source_descriptor_drifts() -> None:
    original = _identity("drift-test", True)
    assert _identity("drift-test", True, source_descriptor_digest="f" * 64) != original


def test_terminal_outcomes_reject_mismatched_accounting_flag() -> None:
    incomplete = _attempt("a" * 64).model_copy(update={"cost_usd": None})
    with pytest.raises(ValueError, match="accounting completeness"):
        FailedArm(
            arm="candidate",
            report_version_id=REPORT_ID,
            wall_latency_ms=1,
            attempts=(incomplete,),
            error="failed",
            accounting_complete=True,
        )
    with pytest.raises(ValueError, match="accounting completeness"):
        UnavailableArm(
            report_version_id=REPORT_ID,
            wall_latency_ms=1,
            attempts=(incomplete,),
            reason="unavailable",
            accounting_complete=True,
        )
