from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from romanian_news.binary_benchmark import BINARY_TARGET_REGISTRY
from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_25_TARGET,
    OPENROUTER_GEMINI_38_TARGET,
    TYPESAFE_JEV_TARGET,
)
from romanian_news.derived_binary_protocol import (
    DERIVED_BINARY_CONCERNS,
    TIER_COMPOSITION_DIGEST,
)


def _protocol() -> dict[str, object]:
    path = (
        Path(__file__).parents[1] / "src" / "romanian_news" / "jev_aspect_evaluation_protocol.json"
    )
    return cast(dict[str, object], json.loads(path.read_text()))


def test_protocol_freezes_executable_concerns_and_targets() -> None:
    protocol = _protocol()
    concerns = protocol["concerns"]
    targets = protocol["targets"]
    assert isinstance(concerns, list)
    assert isinstance(targets, list)

    assert concerns == [
        {
            "concern": concern.concern,
            "case_count": concern.case_count,
            "scored_unit_count": concern.scored_unit_count,
            "calls_per_model_trial": concern.calls_per_model_trial,
            "state_format": concern.state_format,
            "question_digests": [question.semantic_digest for question in concern.questions],
            **(
                {"composition_digest": TIER_COMPOSITION_DIGEST} if concern.concern == "tier" else {}
            ),
        }
        for concern in DERIVED_BINARY_CONCERNS
    ]
    assert targets == [
        target.model_dump(mode="json")
        for target in (
            TYPESAFE_JEV_TARGET,
            OPENROUTER_GEMINI_25_TARGET,
            OPENROUTER_GEMINI_38_TARGET,
        )
    ]


def test_protocol_call_forecast_and_ceiling_cover_all_attempts() -> None:
    protocol = _protocol()
    execution = protocol["execution"]
    sentiment = protocol["sentiment"]
    assert isinstance(execution, dict)
    assert isinstance(sentiment, dict)

    calls_per_model_trial = sum(
        concern.calls_per_model_trial for concern in DERIVED_BINARY_CONCERNS
    )
    assert calls_per_model_trial == 132
    assert execution["planned_requests_per_model_trial"] == calls_per_model_trial
    assert execution["planned_provider_requests"] == calls_per_model_trial * 3 * 3
    assert execution["maximum_provider_attempts"] == 1188 * 3
    assert execution["spend_ceiling_usd"] == "10.00"
    assert execution["per_request_reserve_usd"] == {
        target_id: str(registration.maximum_request_cost_usd)
        for target_id, registration in BINARY_TARGET_REGISTRY.items()
    }
    assert sentiment["planned_provider_requests"] == 0
