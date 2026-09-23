from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from romanian_news.analysis.binary_ranking import (
    RankingArticleEvidence,
    RankingGroupEvidence,
    analyze_binary_ranking,
    build_ranking_binary_request,
)
from romanian_news.binary_benchmark import (
    TYPESAFE_JEV_TARGET,
    build_binary_evaluators,
    build_execution_identity,
    run_registered_binary_benchmark,
)
from romanian_news.binary_ranking_evaluation import (
    V11_BASELINE_ARTIFACT_ID,
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRankingSource,
    BinaryRankingSourceCase,
    build_ranking_benchmark_cases,
    load_v11_binary_ranking_source,
)
from romanian_news.derived_binary_protocol import DERIVED_BINARY_BENCHMARKS
from romanian_news.evaluation import (
    NewsEvaluationPin,
    RankingEvaluationCase,
    RankingEvaluationSpec,
)
from romanian_news.groups import NewsGroup
from romanian_news.tests import evaluation_factories


def test_renderer_exposes_only_neutral_aliases_and_frozen_evidence() -> None:
    source = _source()
    request = build_ranking_binary_request(source.cases[0].groups)

    assert "Group A" in request.state
    assert "Group B" in request.state
    assert "Synthetic article 1" in request.state
    assert "Synthetic body." in request.state
    assert '"evidence_quote":"Synthetic evidence"' in request.state
    for forbidden in (
        "higher",
        "lower",
        "observed",
        "report order",
        "tier",
        "theme",
        "expected",
        "control",
        "feedback",
    ):
        assert forbidden not in request.state.casefold()


def test_alias_orientation_is_label_independent_and_expected_is_derived_after_aliasing() -> None:
    source = _source()
    original = source.cases[0]
    reversed_groups = original.model_copy(update={"groups": tuple(reversed(original.groups))})
    flipped_label = original.model_copy(
        update={
            "higher_group_id": original.lower_group_id,
            "lower_group_id": original.higher_group_id,
        }
    )

    original_case = build_ranking_benchmark_cases(
        source.model_copy(update={"cases": (original, *source.cases[1:])})
    )[0]
    reversed_case = build_ranking_benchmark_cases(
        source.model_copy(update={"cases": (reversed_groups, *source.cases[1:])})
    )[0]
    flipped_case = build_ranking_benchmark_cases(
        source.model_copy(update={"cases": (flipped_label, *source.cases[1:])})
    )[0]

    assert original_case.judgments[0].request == reversed_case.judgments[0].request
    assert original_case.judgments[0].expected == reversed_case.judgments[0].expected
    assert original_case.judgments[0].request == flipped_case.judgments[0].request
    assert original_case.judgments[0].expected is not flipped_case.judgments[0].expected
    assert original_case.identity == reversed_case.identity
    assert original_case.identity != flipped_case.identity


def test_source_requires_exact_pinned_28_case_workload() -> None:
    source = _source()

    assert len(source.cases) == 28
    with pytest.raises(ValidationError, match="exactly 28 unique cases"):
        _source(source.cases[:-1])
    with pytest.raises(ValidationError, match="exactly 28 unique cases"):
        _source((*source.cases, source.cases[-1].model_copy(update={"case_id": "extra"})))
    with pytest.raises(ValidationError, match="source artifact"):
        BinaryRankingSource(
            pin=source.pin.model_copy(update={"manifest_version_id": "f" * 64}),
            manifest_reference=source.manifest_reference,
            baseline_reference=source.baseline_reference,
            declared_manifest_version=source.declared_manifest_version,
            cases=source.cases,
        )
    with pytest.raises(ValidationError, match="baseline artifact"):
        BinaryRankingSource(
            pin=source.pin.model_copy(update={"baseline_version_id": "f" * 64}),
            manifest_reference=source.manifest_reference,
            baseline_reference=source.baseline_reference,
            declared_manifest_version=source.declared_manifest_version,
            cases=source.cases,
        )


def test_adapter_preserves_immutable_artifact_identities() -> None:
    source = _source()
    adapted = build_ranking_benchmark_cases(source)[0]
    first_article = source.cases[0].groups[0].articles[0]

    assert any(first_article.article.version_id in value for value in adapted.identity)
    assert any(first_article.relevance.version_id in value for value in adapted.identity)
    with pytest.raises(ValidationError, match="frozen"):
        source.cases[0].case_id = "mutated"


def test_loader_reuses_exact_pinned_hydrated_release(monkeypatch) -> None:
    manifest = evaluation_factories.synthetic_manifest()
    dataset = evaluation_factories.synthetic_dataset()
    ranking_spec = next(case for case in manifest.cases if isinstance(case, RankingEvaluationSpec))
    ranking_case = next(case for case in dataset.cases if isinstance(case, RankingEvaluationCase))
    specs = tuple(
        ranking_spec.model_copy(update={"case_id": f"ranking-{index:02d}"}) for index in range(28)
    )
    cases = tuple(ranking_case.model_copy(update={"case_id": spec.case_id}) for spec in specs)
    manifest = manifest.model_copy(update={"version": V11_MANIFEST_VERSION, "cases": specs})
    dataset = dataset.model_copy(update={"version": V11_MANIFEST_VERSION, "cases": cases})
    pin = NewsEvaluationPin(
        manifest_version_id=V11_SOURCE_ARTIFACT_ID,
        baseline_version_id=V11_BASELINE_ARTIFACT_ID,
    )
    manifest_reference = evaluation_factories.reference(900, "manifest").model_copy(
        update={"version_id": V11_SOURCE_ARTIFACT_ID}
    )
    baseline_reference = evaluation_factories.reference(901, "baseline").model_copy(
        update={"version_id": V11_BASELINE_ARTIFACT_ID}
    )
    release = SimpleNamespace(
        pin=pin,
        manifest_reference=manifest_reference,
        baseline_reference=baseline_reference,
        manifest=manifest,
        dataset=dataset,
    )
    loaded_pins = []
    articles = {
        evaluation_factories.embedded_article(
            index
        ).article.r2_key: evaluation_factories.embedded_article(index).value
        for index in (1, 2)
    }
    monkeypatch.setattr(
        "romanian_news.binary_ranking_evaluation.load_news_evaluation_release",
        lambda content: loaded_pins.append(content) or release,
    )
    monkeypatch.setattr(
        "romanian_news.binary_ranking_evaluation.read_verified_r2_object",
        lambda key, _digest: articles[key].model_dump_json().encode(),
    )

    loaded = load_v11_binary_ranking_source(b"exact-pin")

    assert loaded_pins == [b"exact-pin"]
    assert loaded.pin == pin
    assert loaded.manifest_reference == manifest_reference
    assert loaded.baseline_reference == baseline_reference
    assert len(loaded.cases) == 28


def test_generic_core_executes_all_28_cases_in_dry_run_without_provider_calls() -> None:
    source = _source()
    definition = DERIVED_BINARY_BENCHMARKS["ranking"]
    cases = build_ranking_benchmark_cases(source)
    identity = build_execution_identity(
        definition,
        source_artifact_id=source.manifest_reference.version_id,
        declared_manifest_version=source.declared_manifest_version,
        cases=cases,
        targets=(TYPESAFE_JEV_TARGET,),
        trial_refs=("dry-ranking-001",),
        execution_mode="dry_run",
        execution_ref="git:test-ranking",
    )

    result = run_registered_binary_benchmark(
        definition,
        identity,
        cases,
        build_binary_evaluators(identity.targets, dry_run=True),
    )

    assert len(result.results) == 28
    assert all(item.status == "completed" for item in result.results)
    assert all(
        item.provider_request_id is not None and item.provider_request_id.startswith("dry-run:")
        for item in result.results
    )
    report = analyze_binary_ranking(result)
    trial = report["trials"][0]
    assert trial["inversion_count"] == trial["completed"] - trial["correct"]


def _source(
    cases: tuple[BinaryRankingSourceCase, ...] | None = None,
) -> BinaryRankingSource:
    first = evaluation_factories.embedded_article(1)
    second = evaluation_factories.embedded_article(2)
    first_group = RankingGroupEvidence(
        group=NewsGroup(
            id=evaluation_factories.reference(10, "group").version_id,
            article_version_ids=(first.article.version_id,),
        ),
        articles=(
            RankingArticleEvidence(
                article=first.article,
                value=first.value,
                relevance=first.relevance,
                decision=evaluation_factories.relevance_decision(accepted=True, major=True),
            ),
        ),
    )
    second_group = RankingGroupEvidence(
        group=NewsGroup(
            id=evaluation_factories.reference(11, "group").version_id,
            article_version_ids=(second.article.version_id,),
        ),
        articles=(
            RankingArticleEvidence(
                article=second.article,
                value=second.value,
                relevance=second.relevance,
                decision=evaluation_factories.relevance_decision(accepted=False),
            ),
        ),
    )
    selected_cases = cases or tuple(
        BinaryRankingSourceCase(
            case_id=f"ranking-{index:02d}",
            control=index == 0,
            groups=(first_group, second_group),
            higher_group_id=(first_group.group.id if index % 2 == 0 else second_group.group.id),
            lower_group_id=(second_group.group.id if index % 2 == 0 else first_group.group.id),
        )
        for index in range(28)
    )
    manifest_reference = evaluation_factories.reference(990, "manifest").model_copy(
        update={"version_id": V11_SOURCE_ARTIFACT_ID}
    )
    baseline_reference = evaluation_factories.reference(991, "baseline").model_copy(
        update={"version_id": V11_BASELINE_ARTIFACT_ID}
    )
    return BinaryRankingSource(
        pin=NewsEvaluationPin(
            manifest_version_id=V11_SOURCE_ARTIFACT_ID,
            baseline_version_id=baseline_reference.version_id,
        ),
        manifest_reference=manifest_reference,
        baseline_reference=baseline_reference,
        declared_manifest_version=V11_MANIFEST_VERSION,
        cases=selected_cases,
    )
