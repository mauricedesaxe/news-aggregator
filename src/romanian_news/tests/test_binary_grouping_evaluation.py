from __future__ import annotations

from decimal import Decimal

from romanian_news.analysis.binary_evaluation import BinaryProbabilityObservation
from romanian_news.analysis.binary_grouping import GROUPING_BINARY_QUESTION, BinaryGroupingCase
from romanian_news.binary_grouping_evaluation import (
    BinaryGroupingSource,
    adapt_binary_grouping_cases,
    run_binary_grouping_evaluation,
)
from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_25_TARGET,
    TYPESAFE_JEV_TARGET,
)
from romanian_news.evaluation import GroupingEvaluationCase, NewsEvaluationPin
from romanian_news.tests.evaluation_factories import reference, synthetic_dataset


def _source() -> BinaryGroupingSource:
    case = next(
        item for item in synthetic_dataset().cases if isinstance(item, GroupingEvaluationCase)
    )
    articles = {item.article.version_id: item for item in case.articles}
    binary_case = BinaryGroupingCase(
        case_id=case.case_id,
        control=case.control,
        day=case.day,
        left_article=articles[case.left_article_version_id].article,
        left_value=articles[case.left_article_version_id].value,
        right_article=articles[case.right_article_version_id].article,
        right_value=articles[case.right_article_version_id].value,
        expected_same_group=case.expected_same_group,
    )
    return BinaryGroupingSource(
        pin=NewsEvaluationPin(
            manifest_version_id="083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a",
            baseline_version_id="9" * 64,
        ),
        manifest_reference=reference(99, "manifest").model_copy(
            update={
                "version_id": "083d5ae5ba73686511f5eb01ce770dc2e5826de22cf27df76e205bd9b946216a"
            }
        ),
        declared_manifest_version="news-evaluation-2026-09-11-v11",
        cases=(binary_case,),
    )


def test_grouping_evaluation_covers_targets_trials_and_pair_identities() -> None:
    source = _source()
    case = source.cases[0]
    targets = (OPENROUTER_GEMINI_25_TARGET, TYPESAFE_JEV_TARGET)

    def evaluate(request, trial_ref, **_kwargs):
        return BinaryProbabilityObservation(
            request_id=("1" if trial_ref.endswith("1") else "2") * 64,
            provider_request_id="provider-request",
            model=(
                "google/gemini-2.5-flash"
                if request.question.question_id == GROUPING_BINARY_QUESTION.question_id
                else "unexpected"
            ),
            probability=Decimal("0.75"),
            predicted_accepted=True,
            input_tokens=10,
            output_tokens=1,
            latency_ms=5,
            estimated_cost_usd=Decimal("0.001"),
        )

    result = run_binary_grouping_evaluation(
        source,
        targets=targets,
        trial_refs=("trial-1", "trial-2"),
        evaluators={target.target_id: evaluate for target in targets},
    )

    assert len(result.runs) == 4
    assert result.case_ids == (case.case_id,)
    assert all(run.metrics.accuracy == Decimal(1) for run in result.runs)
    assert {
        (item.left_article_version_id, item.right_article_version_id)
        for run in result.runs
        for item in run.cases
    } == {(case.left_article.version_id, case.right_article.version_id)}


def test_grouping_cases_adapt_without_changing_the_registered_request() -> None:
    source = _source()
    adapted = adapt_binary_grouping_cases(source.cases)

    assert len(adapted) == 1
    assert adapted[0].control == source.cases[0].control
    assert adapted[0].judgments[0].request.question == GROUPING_BINARY_QUESTION
    assert adapted[0].judgments[0].expected == source.cases[0].expected_same_group
