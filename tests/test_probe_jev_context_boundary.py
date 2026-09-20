# pyright: reportAny=false
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/probe_jev_context_boundary.py"
SPEC = importlib.util.spec_from_file_location("probe_jev_context_boundary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_registered_states_have_exact_lengths_digests_and_final_markers() -> None:
    protocols = tuple(probe.build_protocol(version) for version in ("v1", "v2", "v3"))

    for protocol in protocols:
        for case in protocol["cases"]:
            state = probe.build_state(case["state_characters"])
            assert len(state) == case["state_characters"]
            assert state.endswith(f"{probe.MARKER}\n")
            assert probe.binary_state_digest(state) == case["state_digest"]


def test_protocol_has_separate_bounded_execution_policy() -> None:
    protocol = probe.build_protocol()

    assert protocol["execution"] == {
        "maximum_provider_requests": 3,
        "attempts_per_case": 1,
        "timeout_seconds": 90,
        "spend_ceiling_usd": "0.05",
        "input_cost_per_million_tokens_usd": "0.042",
        "stop_before_next_case_if_spend_ceiling_reached": True,
    }


def test_follow_up_protocol_is_bound_to_first_result() -> None:
    protocol = probe.build_protocol("v2")

    assert [case["state_characters"] for case in protocol["cases"]] == [
        80_000,
        100_000,
        110_000,
    ]
    assert protocol["prior_evidence"]["sha256"] == probe.V1_RESULT_SHA256

    final_protocol = probe.build_protocol("v3")
    assert [case["state_characters"] for case in final_protocol["cases"]] == [
        40_000,
        60_000,
        70_000,
    ]
    assert final_protocol["prior_evidence"]["sha256"] == probe.V2_RESULT_SHA256


def test_registered_protocol_artifacts_match_the_generator() -> None:
    for version in ("v1", "v2", "v3"):
        artifact = ROOT / f"artifacts/jev-aspect-v1/jev-context-boundary-protocol-{version}.json"
        assert json.loads(artifact.read_bytes()) == probe.build_protocol(version)
