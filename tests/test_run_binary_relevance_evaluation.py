from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import TypeAdapter

from romanian_news.analysis.binary_evaluation import (
    RELEVANCE_BINARY_QUESTION,
    BinaryProbabilityObservation,
)
from romanian_news.binary_relevance_evaluation import (
    OPENROUTER_GEMINI_25_TARGET,
    OPENROUTER_GEMINI_38_TARGET,
    TYPESAFE_JEV_TARGET,
    V11_MANIFEST_VERSION,
    V11_SOURCE_ARTIFACT_ID,
    BinaryRelevanceEvaluationResult,
)
from romanian_news.tests import evaluation_factories
from romanian_news.tests.test_binary_relevance_evaluation import _source

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_binary_relevance_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("run_binary_relevance_evaluation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


class _RegisteredHeldOutSource(TypedDict):
    declared_manifest_version: str
    source_artifact_id: str


class _RegisteredMonolithicPolicy(TypedDict):
    question_id: str
    question_digest: str
    threshold: str
    execution_policy_digests: dict[str, str]


class _RegisteredAtomicPolicy(TypedDict):
    architecture_contract: str
    architecture_contract_digest: str


class _Preregistration(TypedDict):
    held_out_source: _RegisteredHeldOutSource
    monolithic_policy: _RegisteredMonolithicPolicy
    atomic_policy: _RegisteredAtomicPolicy


def test_preregistration_matches_frozen_monolithic_policy() -> None:
    preregistration = TypeAdapter(_Preregistration).validate_json(
        (
            Path(__file__).parents[1]
            / "src"
            / "romanian_news"
            / "binary_relevance_preregistration.json"
        ).read_bytes()
    )
    held_out = preregistration["held_out_source"]
    assert held_out["declared_manifest_version"] == V11_MANIFEST_VERSION
    assert held_out["source_artifact_id"] == V11_SOURCE_ARTIFACT_ID
    policy = preregistration["monolithic_policy"]
    assert policy["question_id"] == RELEVANCE_BINARY_QUESTION.question_id
    assert policy["question_digest"] == RELEVANCE_BINARY_QUESTION.semantic_digest
    assert policy["threshold"] == str(RELEVANCE_BINARY_QUESTION.threshold)
    assert policy["execution_policy_digests"] == {
        target.target_id: target.execution_policy_digest
        for target in (
            TYPESAFE_JEV_TARGET,
            OPENROUTER_GEMINI_25_TARGET,
            OPENROUTER_GEMINI_38_TARGET,
        )
    }
    atomic_policy = preregistration["atomic_policy"]
    assert (
        hashlib.sha256(atomic_policy["architecture_contract"].encode()).hexdigest()
        == (atomic_policy["architecture_contract_digest"])
    )


def test_parser_requires_execution_ref_and_output_and_defaults_to_three_trials() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([])

    arguments = runner.parse_args(
        ["--execution-ref", "git:abc", "--output", "result.json", "--dry-run"]
    )

    assert arguments.execution_ref == "git:abc"
    assert arguments.output == Path("result.json")
    assert arguments.trials == 3
    assert arguments.dry_run is True
    assert arguments.resume is False
    assert arguments.retry_errors is False


def test_execution_plan_rotates_providers_by_trial() -> None:
    targets = (
        OPENROUTER_GEMINI_25_TARGET,
        OPENROUTER_GEMINI_38_TARGET,
        TYPESAFE_JEV_TARGET,
    )

    plan = runner.rotated_execution_plan(targets, ("run:trial-001", "run:trial-002"), ("a", "b"))

    assert tuple(item.target_id for item in plan[:6:2]) == (
        "openrouter-gemini-2.5-flash",
        "openrouter-gemini-3.8-flash",
        "typesafe-jev",
    )
    assert tuple(item.target_id for item in plan[6:12:2]) == (
        "openrouter-gemini-3.8-flash",
        "typesafe-jev",
        "openrouter-gemini-2.5-flash",
    )
    assert len(set(plan)) == 12


def test_dry_run_covers_every_target_trial_and_case_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source()
    _install_articles(monkeypatch, source)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    output = tmp_path / "dry-run.json"

    result = runner.execute(
        runner.parse_args(
            [
                "--execution-ref",
                "git:dry-run",
                "--output",
                str(output),
                "--trials",
                "2",
                "--dry-run",
            ]
        ),
        source=source,
    )

    parsed = BinaryRelevanceEvaluationResult.model_validate_json(output.read_bytes(), strict=True)
    assert parsed == result
    assert len(result.runs) == 6
    assert sum(len(run.cases) for run in result.runs) == 12
    assert {run.target.requested_model for run in result.runs} == {
        "jev-1.13.0",
        "google/gemini-2.5-flash",
        "google/gemini-3.8-flash",
    }
    assert all(case.status == "completed" for run in result.runs for case in run.cases)
    assert {
        tuple(sorted(case.model_dump().keys())) for run in result.runs for case in run.cases
    } == {tuple(sorted(result.runs[0].cases[0].model_dump().keys()))}


def test_resume_rejects_checkpoint_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source(one_case=True)
    _install_articles(monkeypatch, source)
    output = tmp_path / "checkpoint.json"
    arguments = runner.parse_args(
        [
            "--execution-ref",
            "git:first",
            "--output",
            str(output),
            "--trials",
            "1",
            "--dry-run",
        ]
    )
    runner.execute(arguments, source=source)

    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.execute(
            runner.parse_args(
                [
                    "--execution-ref",
                    "git:other",
                    "--output",
                    str(output),
                    "--trials",
                    "1",
                    "--dry-run",
                    "--resume",
                ]
            ),
            source=source,
        )


def test_interruption_then_resume_does_not_repeat_completed_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source()
    _install_articles(monkeypatch, source)
    output = tmp_path / "checkpoint.json"
    calls: Counter[tuple[str, str, str]] = Counter()
    evaluators = _counting_evaluators(calls)
    real_write = runner._write_checkpoint
    writes = 0

    def interrupt_after_second_write(*args, **kwargs) -> None:
        nonlocal writes
        real_write(*args, **kwargs)
        writes += 1
        if writes == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_write_checkpoint", interrupt_after_second_write)
    arguments = runner.parse_args(
        ["--execution-ref", "git:resume", "--output", str(output), "--trials", "1"]
    )
    with pytest.raises(KeyboardInterrupt):
        runner.execute(arguments, source=source, evaluators=evaluators)

    monkeypatch.setattr(runner, "_write_checkpoint", real_write)
    runner.execute(
        runner.parse_args(
            [
                "--execution-ref",
                "git:resume",
                "--output",
                str(output),
                "--trials",
                "1",
                "--resume",
            ]
        ),
        source=source,
        evaluators=evaluators,
    )

    assert len(calls) == 3
    assert set(calls.values()) == {2}


def test_resume_reuses_failures_unless_retry_errors_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source(one_case=True)
    _install_articles(monkeypatch, source)
    output = tmp_path / "checkpoint.json"
    calls: Counter[tuple[str, str, str]] = Counter()
    evaluators = _counting_evaluators(calls, fail_target="typesafe-jev")
    base = ["--execution-ref", "git:retry", "--output", str(output), "--trials", "1"]

    first = runner.execute(runner.parse_args(base), source=source, evaluators=evaluators)
    assert first.runs[-1].cases[0].status == "failed"
    runner.execute(runner.parse_args([*base, "--resume"]), source=source, evaluators=evaluators)
    assert sum(calls.values()) == 3

    runner.execute(
        runner.parse_args([*base, "--resume", "--retry-errors"]),
        source=source,
        evaluators=evaluators,
    )
    assert sum(calls.values()) == 4
    assert sum(count for key, count in calls.items() if key[0] == "typesafe-jev") == 2


def test_final_artifact_order_is_stable_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source()
    _install_articles(monkeypatch, source)
    output = tmp_path / "result.json"

    result = runner.execute(
        runner.parse_args(
            [
                "--execution-ref",
                "git:stable",
                "--output",
                str(output),
                "--trials",
                "2",
                "--dry-run",
            ]
        ),
        source=source,
    )

    assert tuple((run.target.target_id, run.trial_ref) for run in result.runs) == tuple(
        (target.target_id, trial_ref)
        for target in sorted(result.targets, key=lambda item: item.target_id)
        for trial_ref in sorted(result.trial_refs)
    )
    assert all(tuple(case.case_id for case in run.cases) == result.case_ids for run in result.runs)
    assert json.loads(output.read_text()) == result.model_dump(mode="json")


def _install_articles(monkeypatch: pytest.MonkeyPatch, source) -> None:
    articles = {
        case.article.r2_key: evaluation_factories.embedded_article(index).value
        for index, case in enumerate(source.manifest.cases, start=1)
        if case.concern == "relevance"
    }
    monkeypatch.setattr(
        "romanian_news.binary_relevance_evaluation.read_verified_r2_object",
        lambda key, _digest: articles[key].model_dump_json().encode(),
    )


def _counting_evaluators(
    calls: Counter[tuple[str, str, str]],
    *,
    fail_target: str | None = None,
):
    targets = (
        OPENROUTER_GEMINI_25_TARGET,
        OPENROUTER_GEMINI_38_TARGET,
        TYPESAFE_JEV_TARGET,
    )

    def build(target):
        def evaluate(request, trial_ref, *, on_attempt=None):
            key = (target.target_id, trial_ref, request.state_digest)
            calls[key] += 1
            if target.target_id == fail_target:
                raise RuntimeError("expected failure")
            probability = Decimal("0.75")
            return BinaryProbabilityObservation(
                request_id=("1" if target.target_id == "typesafe-jev" else "2") * 64,
                provider_request_id=f"fake:{target.target_id}:{request.state_digest}",
                model=target.requested_model,
                probability=probability,
                predicted_accepted=True,
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
                estimated_cost_usd=Decimal(0),
            )

        return evaluate

    return {target.target_id: build(target) for target in targets}
