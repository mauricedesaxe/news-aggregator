from __future__ import annotations

from romanian_news.analysis.binary_grouping import BinaryGroupingCase, build_grouping_binary_request
from romanian_news.evaluation import GroupingEvaluationCase
from romanian_news.tests.evaluation_factories import synthetic_dataset


def test_grouping_request_is_symmetric_and_contains_only_anchor_articles() -> None:
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

    request = build_grouping_binary_request(binary_case)
    reversed_request = build_grouping_binary_request(
        binary_case.model_copy(
            update={
                "left_article": binary_case.right_article,
                "left_value": binary_case.right_value,
                "right_article": binary_case.left_article,
                "right_value": binary_case.left_value,
            }
        )
    )

    assert request == reversed_request
    assert request.question.question_id == "same_news_event"
    assert request.state.count("Article A") == 1
    assert request.state.count("Article B") == 1
    assert binary_case.case_id not in request.state
    assert str(binary_case.expected_same_group).lower() not in request.state.lower()
    assert "vector" not in request.state.lower()
