#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple, cast

import requests

from romanian_news.analysis.binary_evaluation import (
    BinaryQuestion,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.analysis.jev_relevance import (
    JEV_INPUT_COST_PER_MILLION_TOKENS_USD,
    JEV_RELEVANCE_ENDPOINT,
    JevNoulResponse,
)
from romanian_news.config import TYPESAFE_API_KEY

MODEL = "jev-1.13.0"
PROTOCOL_CASE_CHARACTERS = {
    "v1": (120_000, 150_000, 180_000),
    "v2": (80_000, 100_000, 110_000),
    "v3": (40_000, 60_000, 70_000),
}
V1_RESULT_SHA256 = "72b1c22be77fc844bd11802f7c8fdafb3aab7109a26b5f8e57c182e133e71ed3"
V2_RESULT_SHA256 = "269061a4c0f452993d3fb6e5de47e356f0378dc11ce2672886f5dcd74d919000"
MARKER = "BOUNDARY_MARKER_PRESENT"
SPEND_CEILING_USD = Decimal("0.05")
TIMEOUT_SECONDS = 90
QUESTION = BinaryQuestion(
    question_id="boundary_marker_present",
    instructions=f"Determine whether the exact text {MARKER!r} appears anywhere in the state.",
    true_criteria=f"The state contains the exact text {MARKER!r}.",
    false_criteria=f"The state does not contain the exact text {MARKER!r}.",
    threshold=Decimal("0.5"),
)


class Arguments(NamedTuple):
    protocol: Path
    output: Path | None
    write_protocol: bool
    protocol_version: str


def build_state(character_count: int) -> str:
    prefix = "Deterministic context-boundary probe. Only the final marker matters.\n"
    suffix = f"\n{MARKER}\n"
    filler_length = character_count - len(prefix) - len(suffix)
    if filler_length < 0:
        raise ValueError("Probe character count is too small")
    unit = "neutral filler text 0123456789 abcdefghijklmnopqrstuvwxyz\n"
    filler = (unit * ((filler_length + len(unit) - 1) // len(unit)))[:filler_length]
    state = prefix + filler + suffix
    if len(state) != character_count:
        raise AssertionError("Boundary probe state has the wrong character count")
    return state


def build_protocol(version: str = "v1") -> dict[str, object]:
    try:
        character_counts = PROTOCOL_CASE_CHARACTERS[version]
    except KeyError as error:
        raise ValueError(f"Unknown boundary protocol version: {version}") from error
    cases: list[dict[str, object]] = []
    for character_count in character_counts:
        state = build_state(character_count)
        cases.append(
            {
                "case_id": f"ascii-{character_count}",
                "state_characters": character_count,
                "state_digest": binary_state_digest(state),
                "marker_at_end": state.endswith(f"{MARKER}\n"),
            }
        )
    protocol: dict[str, object] = {
        "schema_version": f"jev-context-boundary-protocol/{version}",
        "purpose": (
            "Observe Jev acceptance, provider-reported input tokens, and rejection behavior "
            "around and beyond the documented approximate 150000-character boundary."
        ),
        "provider": "typesafe",
        "endpoint": JEV_RELEVANCE_ENDPOINT,
        "requested_model": MODEL,
        "question": QUESTION.model_dump(mode="json"),
        "question_digest": QUESTION.semantic_digest,
        "cases": cases,
        "execution": {
            "maximum_provider_requests": len(character_counts),
            "attempts_per_case": 1,
            "timeout_seconds": TIMEOUT_SECONDS,
            "spend_ceiling_usd": str(SPEND_CEILING_USD),
            "input_cost_per_million_tokens_usd": str(
                Decimal(str(JEV_INPUT_COST_PER_MILLION_TOKENS_USD))
            ),
            "stop_before_next_case_if_spend_ceiling_reached": True,
        },
        "interpretation": {
            "primary_evidence": (
                "HTTP outcome and provider-reported usage.input_tokens; accepted responses above "
                "32000 or a plateau at 32000 distinguish behavior better than character estimates."
            ),
            "secondary_evidence": (
                "The marker is at the final bytes of every state. Its probability is suggestive, "
                "not proof, because model output is nondeterministic."
            ),
            "not_an_accuracy_evaluation": True,
        },
    }
    if version == "v2":
        protocol["purpose"] = (
            "Bracket the Jev acceptance boundary below 120000 characters after every v1 case "
            "at 120000 characters or larger was rejected with max_tokens_exceeded."
        )
        protocol["prior_evidence"] = {
            "path": "artifacts/jev-aspect-v1/jev-context-boundary-result-v1.json",
            "sha256": V1_RESULT_SHA256,
        }
    elif version == "v3":
        protocol["purpose"] = (
            "Bracket the Jev acceptance boundary below 80000 characters after every v2 case "
            "at 80000 characters or larger was rejected with max_tokens_exceeded."
        )
        protocol["prior_evidence"] = {
            "path": "artifacts/jev-aspect-v1/jev-context-boundary-result-v2.json",
            "sha256": V2_RESULT_SHA256,
        }
    return protocol


def run_probe(
    protocol: Mapping[str, object], *, protocol_artifact_sha256: str
) -> dict[str, object]:
    schema_version = protocol.get("schema_version")
    if not isinstance(schema_version, str) or "/" not in schema_version:
        raise ValueError("Boundary protocol has an invalid schema version")
    expected = build_protocol(schema_version.rsplit("/", 1)[-1])
    if protocol != expected:
        raise ValueError("Boundary protocol does not match the deterministic registered protocol")
    if not TYPESAFE_API_KEY:
        raise ValueError("TYPESAFE_API_KEY is required for the Jev boundary probe")

    results: list[dict[str, object]] = []
    actual_spend = Decimal(0)
    cases = _list(protocol.get("cases"), "protocol cases")
    for raw_case in cases:
        if actual_spend >= SPEND_CEILING_USD:
            break
        case = cast(Mapping[str, object], raw_case)
        character_count = int(cast(int, case["state_characters"]))
        state = build_state(character_count)
        request = BinaryRequest(
            question=QUESTION,
            state=state,
            state_digest=cast(str, case["state_digest"]),
        )
        result = _request_case(cast(str, case["case_id"]), request)
        result_cost = Decimal(cast(str, result["estimated_cost_usd"]))
        actual_spend += result_cost
        if actual_spend > SPEND_CEILING_USD:
            raise RuntimeError("Jev boundary probe exceeded its preregistered spend ceiling")
        results.append(result)

    return {
        "schema_version": "jev-context-boundary-result/v1",
        "protocol_artifact_sha256": protocol_artifact_sha256,
        "protocol_semantic_sha256": _sha256(_canonical_json(protocol)),
        "requested_cases": len(cases),
        "completed_cases": len(results),
        "actual_spend_usd": str(actual_spend),
        "results": results,
    }


def _request_case(case_id: str, request: BinaryRequest) -> dict[str, object]:
    payload = {
        "state": request.state,
        "model": MODEL,
        "questions": {
            QUESTION.question_id: {
                "type": "noul",
                "instructions": QUESTION.instructions,
                "criteria": {
                    "true": QUESTION.true_criteria,
                    "false": QUESTION.false_criteria,
                },
            }
        },
    }
    started = time.monotonic()
    response = requests.post(
        JEV_RELEVANCE_ENDPOINT,
        headers={
            "Authorization": f"Bearer {TYPESAFE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    latency_ms = max(0, round((time.monotonic() - started) * 1000))
    response_digest = _sha256(response.content)
    provider_request_id = response.headers.get("x-request-id")
    if not response.ok:
        return {
            "case_id": case_id,
            "state_digest": request.state_digest,
            "state_characters": len(request.state),
            "outcome": "rejected",
            "http_status": response.status_code,
            "provider_request_id": provider_request_id,
            "response_sha256": response_digest,
            "response_body": _safe_response_body(response),
            "input_tokens": None,
            "output_tokens": None,
            "marker_probability": None,
            "latency_ms": latency_ms,
            "estimated_cost_usd": "0",
        }

    validated = JevNoulResponse.model_validate_json(response.content, strict=True)
    answer = validated.answers[QUESTION.question_id]
    cost = (
        Decimal(validated.usage.input_tokens)
        * Decimal(str(JEV_INPUT_COST_PER_MILLION_TOKENS_USD))
        / Decimal(1_000_000)
    )
    return {
        "case_id": case_id,
        "state_digest": request.state_digest,
        "state_characters": len(request.state),
        "outcome": "accepted",
        "http_status": response.status_code,
        "provider_request_id": provider_request_id,
        "response_sha256": response_digest,
        "response_body": None,
        "actual_model": validated.model,
        "input_tokens": validated.usage.input_tokens,
        "output_tokens": validated.usage.output_tokens,
        "marker_probability": str(answer.noul),
        "latency_ms": latency_ms,
        "estimated_cost_usd": str(cost),
    }


def _safe_response_body(response: requests.Response) -> object:
    try:
        return cast(object, response.json())
    except requests.exceptions.JSONDecodeError:
        return response.text[:2000]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[object], value)


def _parse_arguments(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description="Preregister or execute the Jev context probe")
    _ = parser.add_argument("--protocol", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path)
    _ = parser.add_argument("--write-protocol", action="store_true")
    _ = parser.add_argument(
        "--protocol-version", choices=tuple(PROTOCOL_CASE_CHARACTERS), default="v1"
    )
    parsed = _mapping(cast(object, vars(parser.parse_args(argv))), "arguments")
    protocol = parsed.get("protocol")
    output = parsed.get("output")
    write_protocol = parsed.get("write_protocol")
    protocol_version = parsed.get("protocol_version")
    if not isinstance(protocol, Path):
        parser.error("--protocol must be a path")
    if output is not None and not isinstance(output, Path):
        parser.error("--output must be a path")
    if not isinstance(write_protocol, bool):
        parser.error("--write-protocol must be a boolean")
    if not isinstance(protocol_version, str):
        parser.error("--protocol-version must be a string")
    if write_protocol == (output is not None):
        parser.error("choose exactly one of --write-protocol or --output")
    return Arguments(
        protocol=protocol,
        output=output,
        write_protocol=write_protocol,
        protocol_version=protocol_version,
    )


def main() -> None:
    arguments = _parse_arguments()
    if arguments.write_protocol:
        content = (
            json.dumps(build_protocol(arguments.protocol_version), indent=2, sort_keys=True) + "\n"
        )
        _ = arguments.protocol.write_text(content)
        return
    protocol_content = arguments.protocol.read_bytes()
    protocol = _mapping(cast(object, json.loads(protocol_content)), "boundary protocol")
    result = run_probe(protocol, protocol_artifact_sha256=_sha256(protocol_content))
    if arguments.output is None:
        raise AssertionError("An execution output path is required")
    _ = arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
