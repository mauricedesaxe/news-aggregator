from __future__ import annotations

import json
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
    assert tier["context_guard_characters"] == 70_000


def test_protocol_measures_complete_runs_without_fallback() -> None:
    protocol = _protocol()
    execution = cast(dict[str, object], protocol["execution"])
    quality = cast(dict[str, object], protocol["quality"])
    tier = cast(dict[str, object], protocol["tier_contract"])

    assert execution["trial_count"] == 3
    assert execution["arm_order"] == [
        ["incumbent", "candidate"],
        ["candidate", "incumbent"],
        ["incumbent", "candidate"],
    ]
    assert execution["provider_reuse"] == (
        "Historical Jev results may be reported as context but are forbidden in measured "
        "latency and cost trials. Resume may reuse only exact terminal results from this "
        "experiment identity."
    )
    assert execution["cost_policy"] == (
        "Include every accepted, rejected, corrected, retried, and failed provider attempt. "
        "Missing usage makes cost a lower bound, never zero."
    )
    assert execution["latency_policy"] == (
        "Measure wall time from arm entry through validated assessment output. Report summed "
        "provider-attempt latency separately for diagnosis."
    )
    assert tier["context_policy"] == (
        "Refuse an over-guard candidate arm. Never clip, summarize, chunk, or silently "
        "substitute the incumbent."
    )
    assert quality["primary_unit"] == "Complete validated day-level subject assessment."
