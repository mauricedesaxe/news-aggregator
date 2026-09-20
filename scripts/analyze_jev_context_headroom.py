#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from romanian_news.binary_benchmark import BinaryBenchmarkEvaluationResult
from romanian_news.derived_binary_benchmark import load_v11_derived_binary_workload

JEV_TARGET_ID = "typesafe-jev"
UTILIZATION_THRESHOLDS = (50, 75, 90, 100)


class RequestEvidence(TypedDict):
    case_id: str
    judgment_id: str
    state_digest: str
    day: str | None
    input_tokens: int
    state_characters: int | None


def analyze(
    limits_path: Path, artifacts: Sequence[Path], *, pin_path: Path | None = None
) -> dict[str, object]:
    limits = json.loads(limits_path.read_text())
    token_limit = int(limits["documented_limits"]["single_question_effective_input_tokens"])
    pin_content = pin_path.read_bytes() if pin_path is not None else None
    concerns = [
        _analyze_artifact(
            path,
            token_limit,
            _state_character_counts(path, pin_content) if pin_content is not None else {},
        )
        for path in sorted(artifacts)
    ]
    evidence_gaps = [
        "The reviewed v11 artifacts are not a recent-production-day sample.",
        "Provider token counts exist only for requests that TypeSafe accepted.",
        "Pairwise benchmark requests do not establish full-pipeline context fit.",
    ]
    if pin_content is None:
        evidence_gaps.insert(
            0,
            "Completed benchmark artifacts do not retain rendered state text, so character counts require rebuilding frozen requests.",
        )
    return {
        "schema_version": "jev-context-headroom/v1",
        "generated_from": {
            "limit_registry": str(limits_path),
            "limit_registry_sha256": _sha256(limits_path.read_bytes()),
            "target_id": JEV_TARGET_ID,
            "counting_method": "TypeSafe response usage.input_tokens",
            "deduplication": "case_id, judgment_id, and state_digest across repeated trials",
        },
        "effective_single_question_limit_tokens": token_limit,
        "concerns": concerns,
        "evidence_gaps": evidence_gaps,
    }


def _state_character_counts(path: Path, pin_content: bytes) -> dict[tuple[str, str, str], int]:
    artifact = BinaryBenchmarkEvaluationResult.model_validate_json(path.read_bytes(), strict=True)
    workload = load_v11_derived_binary_workload(artifact.identity.benchmark, pin_content)
    return {
        (case.case_id, judgment.judgment_id, judgment.request.state_digest): len(
            judgment.request.state
        )
        for case in workload.cases
        for judgment in case.judgments
    }


def _analyze_artifact(
    path: Path,
    token_limit: int,
    state_character_counts: dict[tuple[str, str, str], int],
) -> dict[str, object]:
    result = BinaryBenchmarkEvaluationResult.model_validate_json(path.read_bytes(), strict=True)
    days = {
        case.case_id: next(
            (value.removeprefix("day:") for value in case.identity if value.startswith("day:")),
            None,
        )
        for case in result.identity.cases
    }
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    latencies: list[int] = []
    total_cost = 0.0
    for item in result.results:
        if item.target_id != JEV_TARGET_ID:
            continue
        if item.status != "completed":
            raise ValueError(f"Jev result is not complete: {item.request_id}")
        final = item.attempts[-1]
        if final.input_tokens is None or final.cost_usd is None:
            raise ValueError(f"Jev result lacks provider accounting: {item.request_id}")
        grouped[(item.case_id, item.judgment_id, item.state_digest)].append(final.input_tokens)
        latencies.append(final.latency_ms)
        total_cost += float(final.cost_usd)

    if state_character_counts and set(state_character_counts) != set(grouped):
        raise ValueError(f"Rebuilt request identities do not match the artifact: {path}")
    requests: list[RequestEvidence] = []
    for (case_id, judgment_id, state_digest), observed in sorted(grouped.items()):
        if len(observed) != len(result.identity.trial_refs) or len(set(observed)) != 1:
            raise ValueError(
                f"Jev token count changed across trials for {case_id}/{judgment_id}: {observed}"
            )
        requests.append(
            {
                "case_id": case_id,
                "judgment_id": judgment_id,
                "state_digest": state_digest,
                "day": days[case_id],
                "input_tokens": observed[0],
                "state_characters": state_character_counts.get(
                    (case_id, judgment_id, state_digest)
                ),
            }
        )

    by_day: dict[str, list[int]] = defaultdict(list)
    for request in requests:
        by_day[request["day"] or "unknown"].append(request["input_tokens"])
    token_counts = [request["input_tokens"] for request in requests]
    character_counts = [
        value for request in requests if (value := request["state_characters"]) is not None
    ]
    return {
        "concern": result.identity.benchmark,
        "source_artifact": str(path),
        "source_artifact_sha256": _sha256(path.read_bytes()),
        "execution_identity_digest": result.identity_digest,
        "declared_manifest_version": result.identity.declared_manifest_version,
        "unique_request_count": len(requests),
        "trial_count": len(result.identity.trial_refs),
        "request_statistics": _statistics(token_counts, token_limit),
        "state_character_statistics": (
            _percentiles(character_counts) if character_counts else None
        ),
        "daily_statistics": [
            {"day": day, **_statistics(values, token_limit)}
            for day, values in sorted(by_day.items())
        ],
        "all_trial_latency_ms": _percentiles(latencies),
        "all_trial_cost_usd": f"{total_cost:.9f}",
        "requests": requests,
    }


def _statistics(values: Sequence[int], token_limit: int) -> dict[str, object]:
    if not values:
        raise ValueError("Context statistics require at least one request")
    maximum = max(values)
    return {
        "count": len(values),
        **_percentiles(values),
        "max_utilization_percent": round(maximum * 100 / token_limit, 6),
        "minimum_headroom_tokens": token_limit - maximum,
        "thresholds": [
            {
                "percent": percent,
                "tokens": token_limit * percent // 100,
                "count_at_or_above": sum(value >= token_limit * percent / 100 for value in values),
                "share_at_or_above": round(
                    sum(value >= token_limit * percent / 100 for value in values) / len(values), 6
                ),
            }
            for percent in UTILIZATION_THRESHOLDS
        ],
    }


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def write_reports(report: dict[str, object], output_json: Path, output_markdown: Path) -> None:
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    concerns = report["concerns"]
    assert isinstance(concerns, list)
    lines = [
        "# Jev context headroom on reviewed requests",
        "",
        "TypeSafe reports token use after each accepted request. The table deduplicates identical requests across the three trials and compares them with the documented 32,000-token single-question limit.",
        "",
        "| Concern | Requests | p50 | p95 | p99 | Max | Max use | Minimum headroom | >=50% | >=75% | >=90% | >=100% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for concern in concerns:
        assert isinstance(concern, dict)
        statistics = concern["request_statistics"]
        assert isinstance(statistics, dict)
        thresholds = statistics["thresholds"]
        assert isinstance(thresholds, list)
        threshold_counts = [str(item["count_at_or_above"]) for item in thresholds]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(concern["concern"]),
                    str(statistics["count"]),
                    str(statistics["p50"]),
                    str(statistics["p95"]),
                    str(statistics["p99"]),
                    str(statistics["max"]),
                    f'{statistics["max_utilization_percent"]}%',
                    str(statistics["minimum_headroom_tokens"]),
                    *threshold_counts,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Limits of this evidence",
            "",
            *[f"- {gap}" for gap in cast(list[str], report["evidence_gaps"])],
            "",
        )
    )
    output_markdown.write_text("\n".join(lines))


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Jev token headroom in live artifacts.")
    parser.add_argument("--limits", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--pin", type=Path)
    parser.add_argument("artifacts", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = analyze(arguments.limits, arguments.artifacts, pin_path=arguments.pin)
    write_reports(report, arguments.output_json, arguments.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
