#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/jev-aspect-v1"
OUTPUT_JSON = ARTIFACTS / "jev-context-safety-decision-v1.json"
OUTPUT_MARKDOWN = ARTIFACTS / "jev-context-safety-decision-v1.md"
REFUSAL_BOUNDARY_CHARACTERS = 70_000
TOKEN_LIMIT = 32_000
ROLE_ORDER = ("relevance", "grouping", "ranking", "tier", "confidence", "daily_theme")
STATUSES = {
    "relevance": "conditional",
    "grouping": "safe",
    "ranking": "conditional",
    "tier": "unsafe",
    "confidence": "safe",
    "daily_theme": "safe",
}
SOURCE_PATHS = (
    ROOT / "src/romanian_news/jev_context_limits.json",
    ARTIFACTS / "jev-context-headroom-reviewed-v1.json",
    ARTIFACTS / "jev-production-context-latest-7-v1.json",
    ARTIFACTS / "jev-production-context-latest-7-v1-requests.json.gz",
    ARTIFACTS / "jev-context-boundary-result-v2.json",
    ARTIFACTS / "jev-context-boundary-result-v3.json",
    ARTIFACTS / "jev-batched-tier-cost-v1.json",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected object list for {label}")
    return cast(list[dict[str, object]], value)


def _boundary_case(result: dict[str, object], characters: int) -> dict[str, object]:
    matches = [
        item
        for item in _list(result.get("results"), "boundary results")
        if item.get("state_characters") == characters
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {characters}-character boundary result")
    return matches[0]


def build_decision(root: Path = ROOT) -> dict[str, object]:
    artifacts = root / "artifacts/jev-aspect-v1"
    limits = _load_json(root / "src/romanian_news/jev_context_limits.json")
    documented = cast(dict[str, object], limits["documented_limits"])
    if documented.get("single_question_effective_input_tokens") != TOKEN_LIMIT:
        raise ValueError("The registered Jev single-question limit changed")

    reviewed = _load_json(artifacts / "jev-context-headroom-reviewed-v1.json")
    reviewed_by_role = {
        cast(str, item["concern"]): item
        for item in _list(reviewed.get("concerns"), "reviewed concerns")
    }
    production = _load_json(artifacts / "jev-production-context-latest-7-v1.json")
    production_by_role = {
        cast(str, item["concern"]): item
        for item in _list(production.get("concerns"), "production concerns")
    }
    with gzip.open(
        artifacts / "jev-production-context-latest-7-v1-requests.json.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        raw = cast(dict[str, object], json.load(stream))
    requests_by_role = {
        cast(str, item["concern"]): _list(item.get("requests"), "production requests")
        for item in _list(raw.get("concerns"), "raw production concerns")
    }

    accepted = _boundary_case(_load_json(artifacts / "jev-context-boundary-result-v3.json"), 70_000)
    rejected = _boundary_case(_load_json(artifacts / "jev-context-boundary-result-v2.json"), 80_000)
    rejected_body = cast(dict[str, object], rejected.get("response_body"))
    rejected_detail = cast(dict[str, object], rejected_body.get("detail"))
    if (
        accepted.get("outcome") != "accepted"
        or accepted.get("input_tokens") != 30_489
        or rejected.get("http_status") != 400
        or rejected_detail.get("error_type") != "max_tokens_exceeded"
    ):
        raise ValueError("The registered Jev boundary observations changed")

    batched = _load_json(artifacts / "jev-batched-tier-cost-v1.json")
    if batched.get("purpose") != (
        "Cost and transport evidence only; not a quality-equivalence evaluation."
    ):
        raise ValueError("The tier batching evidence changed purpose")
    batch_comparison = cast(dict[str, object], batched["comparison"])

    roles: list[dict[str, object]] = []
    for role in ROLE_ORDER:
        requests = requests_by_role[role]
        sizes = [cast(int, request["state_characters"]) for request in requests]
        maximum = max(sizes)
        affected = sum(size > REFUSAL_BOUNDARY_CHARACTERS for size in sizes)
        provider_requests = len(requests)
        affected_provider_requests = affected
        if role == "tier":
            grouped: dict[tuple[str, tuple[str, ...], str], list[dict[str, object]]] = defaultdict(
                list
            )
            for request in requests:
                key = (
                    cast(str, request["report_version_id"]),
                    tuple(cast(list[str], request["selection_ids"])),
                    cast(str, request["state_digest"]),
                )
                grouped[key].append(request)
            expected_questions = {"main_subject", "worth_knowing_if_not_main"}
            if any(
                {cast(str, item["question_id"]) for item in items} != expected_questions
                for items in grouped.values()
            ):
                raise ValueError("A tier state does not have the two registered questions")
            provider_requests = len(grouped)
            affected_provider_requests = sum(
                cast(int, items[0]["state_characters"]) > REFUSAL_BOUNDARY_CHARACTERS
                for items in grouped.values()
            )

        reviewed_role = reviewed_by_role.get(role)
        observed_headroom = None
        reviewed_affected = None
        routing_delta = None
        if reviewed_role is not None:
            reviewed_stats = cast(dict[str, object], reviewed_role["request_statistics"])
            observed_headroom = cast(int, reviewed_stats["minimum_headroom_tokens"])
            reviewed_requests = _list(reviewed_role.get("requests"), "reviewed requests")
            reviewed_affected = sum(
                cast(int, request["state_characters"]) > REFUSAL_BOUNDARY_CHARACTERS
                for request in reviewed_requests
            )
            if reviewed_affected == 0:
                routing_delta = "0.000000"

        production_role = production_by_role[role]
        estimated_stats = production_role.get("estimated_input_token_statistics")
        estimated_headroom = None
        if isinstance(estimated_stats, dict):
            estimated_headroom = TOKEN_LIMIT - cast(int, estimated_stats["max"])

        status = STATUSES[role]
        fallback = {
            "preflight": (
                "always_use_incumbent_with_full_state"
                if status == "unsafe"
                else "use_incumbent_with_full_state_when_state_characters_exceed_70000"
            ),
            "provider_rejection": "use_incumbent_with_full_state",
            "retry_unchanged_jev_request": False,
            "clip_or_chunk_state": False,
        }
        roles.append(
            {
                "role": role,
                "status": status,
                "production": {
                    "question_requests": len(requests),
                    "affected_question_requests": affected,
                    "provider_requests": provider_requests,
                    "affected_provider_requests": affected_provider_requests,
                    "maximum_state_characters": maximum,
                    "minimum_character_headroom_at_guard": (REFUSAL_BOUNDARY_CHARACTERS - maximum),
                    "maximum_naive_tail_clip_characters": max(
                        0, maximum - REFUSAL_BOUNDARY_CHARACTERS
                    ),
                    "maximum_naive_tail_clip_fraction": round(
                        max(0, maximum - REFUSAL_BOUNDARY_CHARACTERS) / maximum, 6
                    ),
                    "clip_measurement": "character loss only; semantic evidence loss not measured",
                },
                "minimum_observed_reviewed_token_headroom": observed_headroom,
                "minimum_estimated_production_token_headroom": estimated_headroom,
                "estimate_is_formal_tokenizer_bound": False,
                "reviewed_routing_effect": {
                    "affected_requests": reviewed_affected,
                    "accuracy_delta_percentage_points": routing_delta,
                    "affected_production_quality": "not_estimable",
                },
                "fallback": fallback,
            }
        )

    source_paths = tuple(
        root / path.relative_to(ROOT) if root != ROOT else path for path in SOURCE_PATHS
    )
    return {
        "schema_version": "jev-context-safety-decision/v1",
        "issue_id": "news-aggregator-vt6.6.3",
        "decision_scope": "context routing only; no production migration approval",
        "policy": {
            "state_character_guard": REFUSAL_BOUNDARY_CHARACTERS,
            "measurement": "Python len(state) Unicode code points",
            "basis": "70,000-character deterministic ASCII state accepted; 80,000 rejected",
            "formal_tokenizer_bound": False,
            "documented_state_plus_longest_question_tokens": TOKEN_LIMIT,
        },
        "roles": roles,
        "tier_batching": {
            **batch_comparison,
            "sample_count": 1,
            "quality_equivalence_evaluated": False,
            "state_size_reduced": False,
        },
        "sources": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path.read_bytes()),
            }
            for path in source_paths
        ],
        "article_or_state_text_included": False,
        "full_provider_output_included": False,
    }


def _markdown(report: dict[str, object]) -> str:
    roles = _list(report["roles"], "decision roles")
    lines = [
        "# Jev context-safety decision",
        "",
        "This decision covers context routing only. It does not approve a production migration.",
        "",
        "The local guard uses `len(state) <= 70,000`. The value comes from one deterministic ASCII probe: 70,000 characters was accepted at 30,489 input tokens, while 80,000 characters returned HTTP 400 `max_tokens_exceeded`. This is not a formal tokenizer bound.",
        "",
        "| Role | Status | Affected production requests | Provider requests | Max chars | Character headroom | Observed token headroom | Estimated token headroom | Max naive clip | Reviewed routing delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role in roles:
        production = cast(dict[str, object], role["production"])
        reviewed = cast(dict[str, object], role["reviewed_routing_effect"])
        lines.append(
            "| "
            + " | ".join(
                (
                    str(role["role"]),
                    str(role["status"]),
                    f'{production["affected_question_requests"]}/{production["question_requests"]}',
                    f'{production["affected_provider_requests"]}/{production["provider_requests"]}',
                    str(production["maximum_state_characters"]),
                    str(production["minimum_character_headroom_at_guard"]),
                    _display(role["minimum_observed_reviewed_token_headroom"]),
                    _display(role["minimum_estimated_production_token_headroom"]),
                    str(production["maximum_naive_tail_clip_characters"]),
                    _display(reviewed["accuracy_delta_percentage_points"]),
                )
            )
            + " |"
        )
    batch = cast(dict[str, object], report["tier_batching"])
    lines.extend(
        (
            "",
            "## Decision",
            "",
            "- `safe` means context-safe only with the local guard and full-state incumbent fallback.",
            "- `conditional` means coverage or calibration is incomplete, or sampled requests need fallback.",
            "- `unsafe` means Jev is not the production default. Tier always stays on the incumbent.",
            "- No role clips, chunks, summarizes, or drops evidence. A local refusal or provider rejection routes the unchanged full state to the incumbent.",
            "- Do not retry an unchanged `max_tokens_exceeded` request.",
            "",
            "## Truncation evidence",
            "",
            "The 70,000-character guard affects no reviewed request, so it changes zero reviewed routing decisions for grouping, ranking, tier, confidence, and daily theme. Relevance has no reviewed headroom sample. Quality on affected production ranking and tier requests is not estimable because the reviewed corpus has no matching over-guard states.",
            "",
            "Naive clipping would remove as many as 70,493 ranking characters and 40,825 tier characters. Character loss is not a measure of semantic evidence loss, so clipping is rejected.",
            "",
            "## Tier batching",
            "",
            f'One exact tier state used {batch["separate_input_tokens"]} input tokens in two separate requests and {batch["batched_input_tokens"]} in one batched request. The measured reduction was {float(cast(float, batch["input_token_reduction_fraction"])) * 100:.4f}%. Both answer IDs returned. The probe did not evaluate quality equivalence, and batching does not reduce state size.',
            "",
        )
    )
    return "\n".join(lines)


def _display(value: object) -> str:
    return "N/A" if value is None else str(value)


def write_reports(
    report: dict[str, object],
    json_path: Path = OUTPUT_JSON,
    markdown_path: Path = OUTPUT_MARKDOWN,
) -> None:
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    markdown_path.write_text(_markdown(report))


def main() -> None:
    write_reports(build_decision())


if __name__ == "__main__":
    main()
