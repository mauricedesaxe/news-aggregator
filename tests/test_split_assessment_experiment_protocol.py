from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import cast

from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_38_TARGET,
    TYPESAFE_JEV_TARGET,
)
from romanian_news.derived_binary_protocol import (
    DERIVED_BINARY_BENCHMARKS,
    TIER_COMPOSITION_DIGEST,
)
from romanian_news.split_subject_assessment_experiment import CONTEXT_GUARD_CHARACTERS
from scripts.run_split_subject_assessment_experiment import (
    ARM_ORDER,
    SPEND_CEILING_USD,
    TRIAL_REFS,
)


def _protocol() -> dict[str, object]:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "romanian_news"
        / "split_assessment_experiment_protocol.json"
    )
    return cast(dict[str, object], json.loads(path.read_text()))


def test_protocol_freezes_source_models_and_tier_semantics() -> None:
    protocol = _protocol()
    source = cast(dict[str, object], protocol["source"])
    arms = cast(dict[str, dict[str, object]], protocol["arms"])
    tier = cast(dict[str, object], protocol["tier_contract"])
    definition = DERIVED_BINARY_BENCHMARKS["tier"]

    assert source["declared_manifest_version"] == "news-evaluation-2026-09-11-v11"
    assert source["tier_case_count"] == definition.case_count
    assert source["ranking_case_count"] == DERIVED_BINARY_BENCHMARKS["ranking"].case_count
    assert arms["incumbent"]["model"] == OPENROUTER_GEMINI_38_TARGET.requested_model
    assert arms["candidate"]["tier_model"] == TYPESAFE_JEV_TARGET.requested_model
    assert arms["candidate"]["ranking_model"] == OPENROUTER_GEMINI_38_TARGET.requested_model
    assert tier["question_digests"] == [
        question.semantic_digest for question in definition.questions
    ]
    assert tier["composition_digest"] == TIER_COMPOSITION_DIGEST
    assert tier["context_guard_characters"] == CONTEXT_GUARD_CHARACTERS


def test_protocol_matches_runtime_execution_contract() -> None:
    protocol = _protocol()
    execution = cast(dict[str, object], protocol["execution"])

    assert execution["trial_count"] == len(TRIAL_REFS)
    assert execution["trial_refs"] == list(TRIAL_REFS)
    assert execution["arm_order"] == [list(arms) for arms in ARM_ORDER]
    assert Decimal(cast(str, execution["spend_ceiling_usd"])) == SPEND_CEILING_USD
