import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from langfuse.api.commons.types.dataset_item import DatasetItem
from langfuse.api.commons.types.dataset_status import DatasetStatus
from langfuse.api.core.api_error import ApiError

from romanian_news import evaluation_projection
from romanian_news.analysis.attempts import ModelCall
from romanian_news.analysis.relevance import (
    RELEVANCE_POLICY_V1,
    RELEVANCE_POLICY_V2,
    RelevanceOutput,
    relevance_policy_digest,
    relevance_request_id,
)
from romanian_news.analysis.relevance_v3 import (
    RELEVANCE_V3_POLICY,
    ContextDecision,
    ImpactDecision,
    relevance_v3_policy_digest,
    relevance_v3_request_id,
)
from romanian_news.artifacts import ArtifactReference
from romanian_news.catalog import evaluations
from romanian_news.catalog.evaluations import LoadedNewsEvaluationRelease
from romanian_news.evaluation import (
    ConfidenceEvaluationSpec,
    EvaluationSpecProvenance,
    NewsEvaluationPin,
    ThemeEvaluationDaySpec,
    ThemePairExpectation,
    TierEvaluationSpec,
    build_news_evaluation_baseline,
    evaluate_news_dataset,
)
from romanian_news.tests.evaluation_factories import (
    embedded_article,
    relevance_decision,
    synthetic_dataset,
    synthetic_manifest,
)

MANIFEST_VERSION_ID = "a" * 64
MANIFEST_REFERENCE = ArtifactReference(
    artifact_id="news:evaluation-manifest:test",
    version_id=MANIFEST_VERSION_ID,
    content_digest="b" * 64,
    r2_key="news/evaluations/manifests/test.json",
)


class FakeLangfuseClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._dataset: SimpleNamespace | None = None
        self.items: dict[str, DatasetItem] = {}
        self.runs = {}
        self.run_calls = []
        self.fail_experiment_receipts = 0
        self.hidden_run_reads = 0
        self.post_run_hidden_reads = 0
        self.hidden_item_reads = 0
        self.post_run_hidden_item_reads = 0
        self.blank_item_reads = 0
        self.post_run_blank_item_reads = 0
        self.api = SimpleNamespace(
            experiments=SimpleNamespace(
                list=self.list_experiments,
                list_items=self.list_experiment_items,
            ),
        )

    @property
    def dataset(self) -> SimpleNamespace:
        if self._dataset is None:
            raise AssertionError("dataset was not created")
        return self._dataset

    def get_dataset(self, name):
        if self._dataset is None or self._dataset.name != name:
            raise ApiError(status_code=404, body={})
        return self.dataset

    def create_dataset(self, *, name, metadata, **_arguments):
        self.events.append("create-dataset")
        self._dataset = SimpleNamespace(
            id="dataset-1",
            name=name,
            metadata=metadata,
            items=[],
        )
        return self.dataset

    def create_dataset_item(self, **arguments):
        self.events.append("create-item")
        item_id = arguments["id"]
        item = DatasetItem(
            id=item_id,
            status=DatasetStatus.ACTIVE,
            input=arguments["input"],
            expected_output=arguments["expected_output"],
            metadata=arguments["metadata"],
            dataset_id=self.dataset.id,
            dataset_name=self.dataset.name,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            media_references=[],
        )
        self.items[item_id] = item
        self.dataset.items = list(self.items.values())
        return item

    def replace_item(self, item: DatasetItem) -> None:
        self.items[item.id] = item
        self.dataset.items = list(self.items.values())

    def run_experiment(self, **arguments):
        self.events.append("run-experiment")
        data = tuple(arguments["data"])
        self.run_calls.append({**arguments, "data": data})
        self.hidden_run_reads = self.post_run_hidden_reads
        self.hidden_item_reads = self.post_run_hidden_item_reads
        self.blank_item_reads = self.post_run_blank_item_reads
        run_name = arguments["run_name"]
        run = self.runs.setdefault(
            run_name,
            {
                "id": f"run-{len(self.runs) + 1}",
                "metadata": arguments["metadata"],
                "items": {},
                "end_time": None,
                "item_count": 0,
            },
        )
        item_results = []
        for item in data:
            output = arguments["task"](item=item)
            arguments["evaluators"][0](
                input=item.input,
                output=output,
                expected_output=item.expected_output,
                metadata=item.metadata,
            )
            run["items"][item.id] = output
            item_results.append(SimpleNamespace(item=item, output=output))
        run["end_time"] = "complete"
        run["item_count"] = len(run["items"])
        return SimpleNamespace(
            dataset_run_id=run["id"],
            experiment_id=run["id"],
            dataset_run_url=f"https://example.test/runs/{run['id']}",
            item_results=item_results,
        )

    def list_experiments(self, *, dataset_id, **_arguments):
        if self.hidden_run_reads:
            self.hidden_run_reads -= 1
            return SimpleNamespace(data=[], meta=SimpleNamespace(cursor=None))
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id=run["id"],
                    name="",
                    dataset_id=dataset_id,
                    metadata=evaluation_projection._flatten_metadata(run["metadata"]),
                    end_time=run.get("end_time", "complete"),
                    item_count=run.get("item_count", len(run["items"])),
                )
                for run_name, run in self.runs.items()
            ],
            meta=SimpleNamespace(cursor=None),
        )

    def list_experiment_items(self, *, experiment_id, **_arguments):
        run = next(run for run in self.runs.values() if run["id"] == experiment_id)
        outputs_visible = self.hidden_item_reads == 0
        outputs_blank = outputs_visible and self.blank_item_reads > 0
        if self.hidden_item_reads:
            self.hidden_item_reads -= 1
        elif self.blank_item_reads:
            self.blank_item_reads -= 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    experiment_id=run["id"],
                    experiment_item_id=item_id,
                    end_time="complete",
                    level="DEFAULT",
                    output=(
                        None if not outputs_visible else "" if outputs_blank else json.dumps(output)
                    ),
                )
                for item_id, output in run["items"].items()
            ],
            meta=SimpleNamespace(cursor=None),
        )


def test_projection_creates_reference_only_items_and_reuses_exact_retry(monkeypatch) -> None:
    release = _release()
    client, events, receipts = _patch_boundaries(monkeypatch)

    first = evaluation_projection.project_news_evaluation_release(release)
    first_ids = tuple(client.items)
    second = evaluation_projection.project_news_evaluation_release(release)

    assert first.created_examples == 5
    assert first.reused_examples == 0
    assert second.created_examples == 0
    assert second.reused_examples == 5
    assert tuple(client.items) == first_ids
    expected_first_id = str(
        uuid5(
            NAMESPACE_URL,
            f"chartly:news-evaluation:{MANIFEST_VERSION_ID}:case:"
            f"{release.manifest.cases[0].case_id}",
        )
    )
    assert first_ids[0] == expected_first_id
    assert sum(item.metadata["split"] == "executable" for item in client.items.values()) == 5
    assert [event for event in events if event == "receipt:dataset"] == ["receipt:dataset"]
    assert events[-1] == "receipt:dataset"
    receipt = receipts[("langfuse", MANIFEST_VERSION_ID, "dataset", "")]
    assert receipt["dataset_name"] == f"chartly-news-evaluation-{MANIFEST_VERSION_ID}"

    projected = {
        "inputs": [item.input for item in client.items.values()],
        "outputs": [item.expected_output for item in client.items.values()],
        "metadata": [item.metadata for item in client.items.values()],
    }
    keys = _all_keys(projected)
    assert not keys.intersection(
        {
            "article",
            "artifact",
            "embedding",
            "feedback_note",
            "model_response",
            "note",
            "provider_response",
            "vector",
        }
    )
    assert keys >= {"artifact_references", "manifest", "feedback_ids", "group_ids"}


def test_projection_never_overwrites_conflicting_or_unrelated_items(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    evaluation_projection.project_news_evaluation_release(release)
    first = next(iter(client.items.values()))
    conflicts = (
        ("input", {**first.input, "case_id": "changed"}),
        ("expected_output", {"changed": True}),
        ("metadata", {**first.metadata, "control": not first.metadata["control"]}),
        ("metadata", {**first.metadata, "split": "future-theme"}),
    )
    for field, value in conflicts:
        client.replace_item(first.model_copy(update={field: value}))
        with pytest.raises(ValueError, match="item identity conflict"):
            evaluation_projection.project_news_evaluation_release(release)
        client.replace_item(first)

    client.replace_item(
        first.model_copy(
            update={"id": "unrelated", "input": {}, "expected_output": {}, "metadata": {}}
        )
    )
    with pytest.raises(ValueError, match="unrelated item IDs"):
        evaluation_projection.project_news_evaluation_release(release)


def test_experiment_reuses_complete_receipt_and_keeps_implementation_identity(monkeypatch) -> None:
    release = _release()
    client, events, receipts = _patch_boundaries(monkeypatch)

    first = evaluation_projection.run_news_evaluation_experiment(release, "git:test")
    retry = evaluation_projection.run_news_evaluation_experiment(release, "git:test")
    second = evaluation_projection.run_news_evaluation_experiment(release, "git:second")

    assert retry == first
    assert second.experiment_id != first.experiment_id
    assert len(client.run_calls) == 2
    assert [len(call["data"]) for call in client.run_calls] == [5, 5]
    assert all(
        item.metadata["concern"] != "daily_theme"
        for call in client.run_calls
        for item in call["data"]
    )
    assert events[-2:] == ["run-experiment", "receipt:experiment"]
    assert (
        "langfuse",
        MANIFEST_VERSION_ID,
        "experiment",
        "git:test",
    ) in receipts


def test_deterministic_langfuse_experiment_excludes_projected_theme_cases(
    monkeypatch,
) -> None:
    release = _release_with_theme_spec()
    client, _events, _receipts = _patch_boundaries(monkeypatch)

    evaluation_projection.run_news_evaluation_experiment(release, "git:deterministic")

    assert any(item.metadata["concern"] == "daily_theme" for item in client.items.values())
    assert all(item.metadata["concern"] != "daily_theme" for item in client.run_calls[0]["data"])


def test_projection_expands_executable_theme_expectations_as_grouping_items(
    monkeypatch,
) -> None:
    release = _release_with_theme_spec()
    theme = release.manifest.cases[-1]
    assert isinstance(theme, ThemeEvaluationDaySpec)
    report = release.manifest.reports[0].model_copy(update={"themes": MANIFEST_REFERENCE})
    executable = theme.model_copy(
        update={
            "model_output": MANIFEST_REFERENCE,
            "provenance": theme.provenance.model_copy(
                update={"report_version_id": report.report.version_id}
            ),
        }
    )
    release = release.model_copy(
        update={
            "manifest": release.manifest.model_copy(
                update={"cases": (*release.manifest.cases[:-1], executable)}
                | {"reports": (report,)}
            )
        }
    )
    client, _events, _receipts = _patch_boundaries(monkeypatch)

    projection = evaluation_projection.project_news_evaluation_release(release)

    assert projection.created_examples == 6
    grouping = tuple(
        item
        for item in client.items.values()
        if item.metadata["concern"] == "grouping" and item.input["case_id"] == "daily-theme-pair"
    )
    assert len(grouping) == 1
    assert grouping[0].expected_output == {"expected_same_group": True}
    references = grouping[0].input["artifact_references"]
    assert MANIFEST_REFERENCE.version_id in {item["version_id"] for item in references}


def test_projection_includes_tier_and_confidence_expectations(monkeypatch) -> None:
    release = _release()
    report = release.manifest.reports[0]
    group_id = release.dataset.reports[0].report_group_ids[0]
    provenance = EvaluationSpecProvenance(
        feedback_ids=(),
        report_version_id=report.report.version_id,
        group_id=group_id,
    )
    cases = (
        *release.manifest.cases,
        TierEvaluationSpec(
            case_id="tier-case",
            provenance=provenance,
            expected_tier="worth_knowing",
        ),
        ConfidenceEvaluationSpec(
            case_id="confidence-case",
            provenance=provenance,
            expected_sufficient=False,
        ),
    )
    release = release.model_copy(
        update={"manifest": release.manifest.model_copy(update={"cases": cases})}
    )
    client, _events, _receipts = _patch_boundaries(monkeypatch)

    projection = evaluation_projection.project_news_evaluation_release(release)

    assert projection.created_examples == 7
    by_case = {item.input["case_id"]: item for item in client.items.values()}
    assert by_case["tier-case"].expected_output == {"expected_tier": "worth_knowing"}
    assert by_case["confidence-case"].expected_output == {"expected_sufficient": False}


def test_experiment_adopts_complete_remote_run_after_receipt_failure(monkeypatch) -> None:
    release = _release()
    client, _events, receipts = _patch_boundaries(monkeypatch)
    client.fail_experiment_receipts = 1

    with pytest.raises(RuntimeError, match="catalog receipt failed"):
        evaluation_projection.run_news_evaluation_experiment(release, "git:crash")

    result = evaluation_projection.run_news_evaluation_experiment(release, "git:crash")

    assert len(client.run_calls) == 1
    assert result.experiment_id == "run-1"
    assert ("langfuse", MANIFEST_VERSION_ID, "experiment", "git:crash") in receipts


def test_experiment_records_receipt_after_remote_results_are_visible(monkeypatch) -> None:
    release = _release()
    client, events, receipts = _patch_boundaries(monkeypatch)
    delays = []
    client.post_run_hidden_reads = 2
    monkeypatch.setattr(evaluation_projection, "_sleep", delays.append)

    result = evaluation_projection.run_news_evaluation_experiment(release, "git:delayed")

    assert result.experiment_id == "run-1"
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 0.5, 1.0]
    assert events[-1] == "receipt:experiment"
    assert ("langfuse", MANIFEST_VERSION_ID, "experiment", "git:delayed") in receipts


def test_experiment_rejects_persistent_visibility_failure_before_receipt(monkeypatch) -> None:
    release = _release()
    client, events, receipts = _patch_boundaries(monkeypatch)
    delays = []
    client.post_run_hidden_reads = 100
    monkeypatch.setattr(evaluation_projection, "_sleep", delays.append)

    with pytest.raises(ValueError, match="missing 5 of 5 items"):
        evaluation_projection.run_news_evaluation_experiment(release, "git:lagged")

    assert delays == [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        15.0,
        30.0,
        60.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        15.0,
        30.0,
        60.0,
    ]
    assert events[-1] == "run-experiment"
    assert ("langfuse", MANIFEST_VERSION_ID, "experiment", "git:lagged") not in receipts


def test_experiment_rejects_conflicting_deterministic_output_without_rerun(monkeypatch) -> None:
    release = _release()
    client, _events, receipts = _patch_boundaries(monkeypatch)
    evaluation_projection.project_news_evaluation_release(release)
    items = tuple(item for item in client.dataset.items if item.metadata["split"] == "executable")
    item = items[0]
    expected = evaluation_projection._expected_run_outputs(
        items, evaluation_projection._results_by_case(release)
    )[item.id]
    name = evaluation_projection._experiment_name(MANIFEST_VERSION_ID, "git:conflict")
    client.runs[name] = {
        "id": "run-conflict",
        "metadata": evaluation_projection._experiment_metadata(release, "git:conflict"),
        "items": {
            item.id: {
                "case_id": expected["case_id"],
                "passed": not expected["passed"],
            }
        },
    }

    with pytest.raises(ValueError, match="Experiment output conflicts"):
        evaluation_projection.run_news_evaluation_experiment(release, "git:conflict")

    assert client.run_calls == []
    assert ("langfuse", MANIFEST_VERSION_ID, "experiment", "git:conflict") not in receipts


def test_remote_experiment_treats_blank_output_as_not_yet_visible(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    result = evaluation_projection.run_news_evaluation_experiment(release, "git:blank")
    client.blank_item_reads = 1

    remote = evaluation_projection._read_remote_experiment(
        evaluation_projection._langfuse_client(),
        result.dataset_name,
        result.experiment_name,
        evaluation_projection._experiment_metadata(release, "git:blank"),
    )

    assert remote is not None
    assert remote.outputs == {}


@pytest.mark.parametrize("failed_first", [True, False])
def test_remote_experiment_ignores_failed_retry_rows_in_any_order(
    monkeypatch, failed_first
) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    result = evaluation_projection.run_news_evaluation_experiment(release, "git:retried-item")
    run = client.runs[result.experiment_name]
    item_id, output = next(iter(run["items"].items()))
    successful = SimpleNamespace(
        experiment_id=run["id"],
        experiment_item_id=item_id,
        end_time="complete",
        level="DEFAULT",
        output=json.dumps(output),
    )
    failed = SimpleNamespace(
        experiment_id=run["id"],
        experiment_item_id=item_id,
        end_time="complete",
        level="ERROR",
        output=None,
    )
    rows = [failed, successful] if failed_first else [successful, failed]
    monkeypatch.setattr(
        client.api.experiments,
        "list_items",
        lambda **_arguments: SimpleNamespace(
            data=rows,
            meta=SimpleNamespace(cursor=None),
        ),
    )

    remote = evaluation_projection._read_remote_experiment(
        evaluation_projection._langfuse_client(),
        result.dataset_name,
        result.experiment_name,
        evaluation_projection._experiment_metadata(release, "git:retried-item"),
    )

    assert remote is not None
    assert remote.outputs == {item_id: output}


def test_remote_experiment_rejects_two_successful_rows(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    result = evaluation_projection.run_news_evaluation_experiment(release, "git:duplicate-item")
    run = client.runs[result.experiment_name]
    item_id, output = next(iter(run["items"].items()))
    successful = SimpleNamespace(
        experiment_id=run["id"],
        experiment_item_id=item_id,
        end_time="complete",
        level="DEFAULT",
        output=json.dumps(output),
    )
    monkeypatch.setattr(
        client.api.experiments,
        "list_items",
        lambda **_arguments: SimpleNamespace(
            data=[successful, successful],
            meta=SimpleNamespace(cursor=None),
        ),
    )

    with pytest.raises(ValueError, match="Experiment contains duplicate item"):
        evaluation_projection._read_remote_experiment(
            evaluation_projection._langfuse_client(),
            result.dataset_name,
            result.experiment_name,
            evaluation_projection._experiment_metadata(release, "git:duplicate-item"),
        )


def test_experiment_output_keeps_non_empty_malformed_json_strict() -> None:
    with pytest.raises(json.JSONDecodeError):
        evaluation_projection._experiment_output("not-json")


def test_existing_active_experiment_waits_for_completion_without_new_work(
    monkeypatch,
) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    items = tuple(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["split"] == "executable"
    )
    outputs = evaluation_projection._expected_run_outputs(
        items, evaluation_projection._results_by_case(release)
    )
    name = evaluation_projection._experiment_name(MANIFEST_VERSION_ID, "git:active-complete")
    client.runs[name] = {
        "id": "run-active-complete",
        "metadata": evaluation_projection._experiment_metadata(release, "git:active-complete"),
        "items": outputs,
        "end_time": None,
        "item_count": len(items),
    }
    delays = []

    def complete_after_wait(delay):
        delays.append(delay)
        client.runs[name]["end_time"] = "complete"

    monkeypatch.setattr(evaluation_projection, "_sleep", complete_after_wait)

    result = evaluation_projection.run_news_evaluation_experiment(release, "git:active-complete")

    assert result.experiment_id == "run-active-complete"
    assert client.run_calls == []
    assert delays == [30.0]


def test_existing_active_experiment_never_starts_duplicate_work(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    items = tuple(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["split"] == "executable"
    )
    outputs = evaluation_projection._expected_run_outputs(
        items, evaluation_projection._results_by_case(release)
    )
    name = evaluation_projection._experiment_name(MANIFEST_VERSION_ID, "git:active")
    client.runs[name] = {
        "id": "run-active",
        "metadata": evaluation_projection._experiment_metadata(release, "git:active"),
        "items": {items[0].id: outputs[items[0].id]},
        "end_time": None,
        "item_count": 1,
    }
    delays = []
    monkeypatch.setattr(evaluation_projection, "_ACTIVE_RUN_RETRY_DELAYS", (30.0, 30.0))
    monkeypatch.setattr(evaluation_projection, "_sleep", delays.append)

    with pytest.raises(ValueError, match="remained active"):
        evaluation_projection.run_news_evaluation_experiment(release, "git:active")

    assert client.run_calls == []
    assert delays == [30.0, 30.0]


def test_existing_incomplete_experiment_waits_for_outputs_before_model_work(
    monkeypatch,
) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    items = tuple(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["split"] == "executable"
    )
    outputs = evaluation_projection._expected_run_outputs(
        items, evaluation_projection._results_by_case(release)
    )
    name = evaluation_projection._experiment_name(MANIFEST_VERSION_ID, "git:eventual")
    client.runs[name] = {
        "id": "run-eventual",
        "metadata": evaluation_projection._experiment_metadata(release, "git:eventual"),
        "items": outputs,
    }
    client.hidden_item_reads = 2
    delays = []
    monkeypatch.setattr(evaluation_projection, "_sleep", delays.append)

    result = evaluation_projection.run_news_evaluation_experiment(release, "git:eventual")

    assert result.experiment_id == "run-eventual"
    assert client.run_calls == []
    assert delays == [0.5, 1.0]


def test_experiment_resumes_only_missing_items(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    items = tuple(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["split"] == "executable"
    )
    outputs = evaluation_projection._expected_run_outputs(
        items, evaluation_projection._results_by_case(release)
    )
    name = evaluation_projection._experiment_name(MANIFEST_VERSION_ID, "git:partial")
    client.runs[name] = {
        "id": "run-partial",
        "metadata": evaluation_projection._experiment_metadata(release, "git:partial"),
        "items": {item.id: outputs[item.id] for item in items[:2]},
    }

    result = evaluation_projection.run_news_evaluation_experiment(release, "git:partial")

    assert result.experiment_id == "run-partial"
    assert [item.id for item in client.run_calls[0]["data"]] == [item.id for item in items[2:]]


def test_fresh_relevance_experiment_uses_exact_article_and_does_no_work_on_retry(
    monkeypatch,
) -> None:
    release = _release()
    client, _events, receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, digest: calls.append(("read", key, digest))
        or article.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance",
        lambda value, policy: calls.append(("analyze", value.reference, policy.policy_id))
        or _relevance_output(value.reference, policy, accepted=True),
    )

    first = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V1, "git:fresh"
    )
    calls_after_first = tuple(calls)
    retry = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V1, "git:fresh"
    )

    relevance_case = next(case for case in release.manifest.cases if case.concern == "relevance")
    assert retry == first
    assert tuple(calls) == calls_after_first
    assert calls[0] == (
        "read",
        relevance_case.article.r2_key,
        relevance_case.article.content_digest,
    )
    assert calls[1] == ("analyze", relevance_case.article, RELEVANCE_POLICY_V1.policy_id)
    assert len(client.run_calls) == 1
    assert first.case_results[0].article_version_id == relevance_case.article.version_id
    assert first.case_results[0].policy_digest == relevance_policy_digest(RELEVANCE_POLICY_V1)
    assert (
        "langfuse",
        MANIFEST_VERSION_ID,
        "relevance",
        relevance_policy_digest(RELEVANCE_POLICY_V1),
        "git:fresh",
    ) in receipts


def test_candidate_policy_has_its_own_receipt_and_rejects_conflicting_receipt(monkeypatch) -> None:
    release = _release()
    client, _events, receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, digest: calls.append(("read", key, digest))
        or article.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance",
        lambda value, policy: calls.append(("analyze", policy.policy_id))
        or _relevance_output(value.reference, policy, accepted=True),
    )

    baseline = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V1, "git:compare"
    )
    candidate = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V2, "git:compare"
    )
    candidate_key = (
        "langfuse",
        MANIFEST_VERSION_ID,
        "relevance",
        relevance_policy_digest(RELEVANCE_POLICY_V2),
        "git:compare",
    )

    assert baseline.experiment_id != candidate.experiment_id
    assert candidate.policy_digest == relevance_policy_digest(RELEVANCE_POLICY_V2)
    assert receipts[candidate_key] == {
        "provider": "langfuse",
        "manifest_artifact_version_id": MANIFEST_VERSION_ID,
        "policy_digest": relevance_policy_digest(RELEVANCE_POLICY_V2),
        "implementation_ref": "git:compare",
        "policy_id": RELEVANCE_POLICY_V2.policy_id,
        "dataset_id": candidate.dataset_id,
        "dataset_name": candidate.dataset_name,
        "experiment_id": candidate.experiment_id,
        "experiment_name": candidate.experiment_name,
        "experiment_url": candidate.experiment_url,
    }
    calls_before_conflict = tuple(calls)
    receipts[candidate_key]["policy_id"] = "conflicting-policy"
    with pytest.raises(ValueError, match="receipt conflicts"):
        evaluation_projection._run_relevance_policy_experiment(
            release, RELEVANCE_POLICY_V2, "git:compare"
        )
    assert tuple(calls) == calls_before_conflict
    assert len(client.run_calls) == 2


def test_fresh_relevance_experiment_resumes_only_missing_cases(monkeypatch) -> None:
    release = _release_with_two_relevance_cases()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    items = tuple(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["concern"] == "relevance"
    )
    expected = evaluation_projection._fresh_relevance_expected(release, items, RELEVANCE_POLICY_V1)
    name = evaluation_projection._relevance_experiment_name(
        MANIFEST_VERSION_ID,
        relevance_policy_digest(RELEVANCE_POLICY_V1),
        "git:partial-fresh",
    )
    client.runs[name] = {
        "id": "run-partial-relevance",
        "metadata": evaluation_projection._relevance_experiment_metadata(
            release, RELEVANCE_POLICY_V1, "git:partial-fresh"
        ),
        "items": {
            items[0].id: _fresh_run_output(
                expected[items[0].id], RELEVANCE_POLICY_V1, accepted=True
            )
        },
    }
    article = embedded_article(1).value
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, digest: calls.append(("read", key, digest))
        or article.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance",
        lambda value, policy: calls.append(("analyze", value.reference.version_id))
        or _relevance_output(value.reference, policy, accepted=True),
    )

    result = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V1, "git:partial-fresh"
    )

    assert [item.id for item in client.run_calls[0]["data"]] == [items[1].id]
    assert [call[0] for call in calls] == ["read", "analyze"]
    assert len(result.case_results) == 2
    assert result.experiment_id == "run-partial-relevance"


def test_fresh_relevance_experiment_waits_for_delayed_item_output(monkeypatch) -> None:
    release = _release()
    client, events, receipts = _patch_boundaries(monkeypatch)
    client.post_run_hidden_item_reads = 2
    delays = []
    article = embedded_article(1).value
    monkeypatch.setattr(evaluation_projection, "_sleep", delays.append)
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda *_args: article.model_dump_json().encode(),
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance",
        lambda value, policy: _relevance_output(value.reference, policy, accepted=True),
    )

    result = evaluation_projection._run_relevance_policy_experiment(
        release, RELEVANCE_POLICY_V1, "git:delayed-output"
    )

    assert result.case_results[0].accepted is True
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 0.5, 1.0]
    assert events[-1] == "receipt:relevance"
    assert (
        "langfuse",
        MANIFEST_VERSION_ID,
        "relevance",
        relevance_policy_digest(RELEVANCE_POLICY_V1),
        "git:delayed-output",
    ) in receipts


def test_fresh_relevance_run_rejects_conflicting_identity(monkeypatch) -> None:
    release = _release()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    dataset = evaluation_projection.project_news_evaluation_release(release)
    item = next(
        item
        for item in client.get_dataset(dataset.dataset_name).items
        if item.metadata["concern"] == "relevance"
    )
    policy_digest = relevance_policy_digest(RELEVANCE_POLICY_V1)
    name = evaluation_projection._relevance_experiment_name(
        MANIFEST_VERSION_ID, policy_digest, "git:conflict"
    )
    expected = evaluation_projection._fresh_relevance_expected(
        release, (item,), RELEVANCE_POLICY_V1
    )[item.id]
    output = _fresh_run_output(expected, RELEVANCE_POLICY_V1, accepted=True)
    output["request_id"] = "f" * 64
    client.runs[name] = {
        "id": "run-conflict",
        "metadata": evaluation_projection._relevance_experiment_metadata(
            release, RELEVANCE_POLICY_V1, "git:conflict"
        ),
        "items": {item.id: output},
    }

    with pytest.raises(ValueError, match="identity conflicts"):
        evaluation_projection._run_relevance_policy_experiment(
            release, RELEVANCE_POLICY_V1, "git:conflict"
        )


def test_v3_experiment_reports_manifest_slices_accounting_and_resumes(monkeypatch) -> None:
    release, prior_content = _release_with_prior_and_two_relevance_cases()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    prior = release.manifest.prior_manifest
    assert prior is not None
    article_payload = article.model_dump_json().encode()
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, _digest: prior_content if key == prior.r2_key else article_payload,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance_v3",
        lambda value, *, mode, policy, execution_ref: calls.append(
            (value.reference.version_id, mode, execution_ref)
        )
        or _v3_output(value.reference, policy, execution_ref),
    )

    first = evaluation_projection._run_relevance_v3_experiment(release, "git:v3")
    retry = evaluation_projection._run_relevance_v3_experiment(release, "git:v3")

    assert retry == first
    assert len(client.run_calls) == 1
    assert calls == [
        (case.article.version_id, "full_evaluation", "git:v3")
        for case in release.manifest.cases
        if case.concern == "relevance"
    ]
    assert first.policy_id == RELEVANCE_V3_POLICY.policy_id
    assert first.policy_digest == relevance_v3_policy_digest()
    assert first.full_metrics.total_cases == 2
    assert first.prior_metrics.total_cases == 1
    assert first.new_metrics.total_cases == 1
    assert first.context_early_exit_rate == 0
    assert first.input_tokens == 6
    assert first.output_tokens == 10
    assert first.cost_usd == pytest.approx(0.02)


def test_v3_incomplete_observability_persists_without_repeating_paid_calls(
    monkeypatch,
) -> None:
    release, prior_content = _release_with_prior_and_two_relevance_cases()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    prior = release.manifest.prior_manifest
    assert prior is not None
    article_payload = article.model_dump_json().encode()
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, _digest: prior_content if key == prior.r2_key else article_payload,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance_v3",
        lambda value, *, mode, policy, execution_ref: calls.append(value.reference.version_id)
        or _v3_output(
            value.reference,
            policy,
            execution_ref,
            observability_complete=False,
        ),
    )

    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:v3-observability")
    first = evaluation_projection._run_relevance_v3_experiment(release, plan.implementation_refs[0])
    retry = evaluation_projection._run_relevance_v3_experiment(release, plan.implementation_refs[0])

    assert retry == first
    assert len(calls) == 2
    assert first.observability_complete is False
    assert all(not result.observability_complete for result in first.case_results)
    projection = evaluation_projection.RelevanceV3EvaluationProjection(
        manifest_artifact_version_id=first.manifest_artifact_version_id,
        plan=plan,
        runs=(
            first,
            *(_v3_projection(ref, 0.90, 0.90, ()) for ref in plan.implementation_refs[1:]),
        ),
        false_negative_ids=(),
        threshold_failures=("run 1 has incomplete observability",),
        passed=False,
    )
    assert projection.threshold_failures == ("run 1 has incomplete observability",)
    assert len(client.run_calls) == 1


def test_v3_remote_output_is_strict_and_binds_the_execution_reference(monkeypatch) -> None:
    release, prior_content = _release_with_prior_and_two_relevance_cases()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    prior = release.manifest.prior_manifest
    assert prior is not None
    article_payload = article.model_dump_json().encode()
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, _digest: prior_content if key == prior.r2_key else article_payload,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance_v3",
        lambda value, *, mode, policy, execution_ref: _v3_output(
            value.reference, policy, execution_ref
        ),
    )
    result = evaluation_projection._run_relevance_v3_experiment(release, "git:v3-strict")
    remote = client.runs[result.experiment_name]
    output = next(iter(remote["items"].values()))

    output["accepted"] = 1
    with pytest.raises(ValueError, match="invalid decision"):
        evaluation_projection._run_relevance_v3_experiment(release, "git:v3-strict")

    output["accepted"] = True
    output["execution_ref"] = "git:other"
    with pytest.raises(ValueError, match="identity conflicts"):
        evaluation_projection._run_relevance_v3_experiment(release, "git:v3-strict")

    assert len(client.run_calls) == 1


def test_fresh_evaluation_plan_validates_and_derives_stable_refs() -> None:
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:test", trial_count=4)

    assert plan.implementation_refs == (
        "git:test:fresh-trial-1-of-4",
        "git:test:fresh-trial-2-of-4",
        "git:test:fresh-trial-3-of-4",
        "git:test:fresh-trial-4-of-4",
    )
    assert evaluation_projection.FreshEvaluationPlan(implementation_ref="git:test").trial_count == 3
    assert (
        evaluation_projection.FreshEvaluationPlan(
            implementation_ref="  git:test  "
        ).implementation_ref
        == "git:test"
    )
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        evaluation_projection.FreshEvaluationPlan(implementation_ref="git:test", trial_count=2)
    with pytest.raises(ValueError, match="at least 1 character"):
        evaluation_projection.FreshEvaluationPlan(implementation_ref="   ")


def test_relevance_comparison_runs_paired_trials_and_one_failure_blocks_promotion(
    monkeypatch,
) -> None:
    release = _release()
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:compare")

    def run(_release, policy, implementation_ref):
        baseline, candidate = _promotion_pair(implementation_ref=implementation_ref)
        if policy == RELEVANCE_POLICY_V1:
            return baseline
        if implementation_ref.endswith("2-of-3"):
            return candidate.model_copy(update={"metrics": baseline.metrics})
        return candidate

    monkeypatch.setattr(evaluation_projection, "_run_relevance_policy_experiment", run)

    result = evaluation_projection.compare_relevance_policies(release, plan)

    assert tuple(trial.implementation_ref for trial in result.trials) == plan.implementation_refs
    assert [trial.passed for trial in result.trials] == [True, False, True]
    assert result.promoted is False
    assert all(failure.startswith("trial 2 of 3: ") for failure in result.promotion_failures)

    with pytest.raises(ValueError, match="Grouped verdict"):
        evaluation_projection.RelevanceComparisonResult(
            manifest_artifact_version_id=result.manifest_artifact_version_id,
            plan=plan,
            trials=result.trials,
            promotion_failures=result.promotion_failures,
            promoted=True,
        )


def test_v3_grouped_evaluation_uses_plan_refs_and_resumes(monkeypatch) -> None:
    release, prior_content = _release_with_prior_and_two_relevance_cases()
    client, _events, _receipts = _patch_boundaries(monkeypatch)
    article = embedded_article(1).value
    prior = release.manifest.prior_manifest
    assert prior is not None
    article_payload = article.model_dump_json().encode()
    monkeypatch.setattr(
        evaluation_projection,
        "read_verified_r2_object",
        lambda key, _digest: prior_content if key == prior.r2_key else article_payload,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "analyze_relevance_v3",
        lambda value, *, mode, policy, execution_ref: _v3_output(
            value.reference, policy, execution_ref
        ),
    )
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:v3")

    first = evaluation_projection.run_relevance_v3_evaluation(release, plan)
    retry = evaluation_projection.run_relevance_v3_evaluation(release, plan)

    assert retry == first
    assert [run.implementation_ref for run in first.runs] == list(plan.implementation_refs)
    assert len({run.experiment_id for run in first.runs}) == 3
    assert len({run.case_results[0].request_id for run in first.runs}) == 3
    assert [run.case_results[0].execution_ref for run in first.runs] == list(
        plan.implementation_refs
    )
    assert len(client.run_calls) == 3
    assert first.false_negative_ids == ()
    assert first.threshold_failures == ()
    assert first.passed is True


def test_v3_grouped_evaluation_reports_each_trial_failure_and_false_negative(
    monkeypatch,
) -> None:
    release = _release()
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:v3")
    runs = iter(
        (
            _v3_projection(plan.implementation_refs[0], 0.89, 0.95, ("case-a",)),
            _v3_projection(plan.implementation_refs[1], 0.95, 0.89, ("case-b",)),
            _v3_projection(plan.implementation_refs[2], 0.90, 0.90, ("case-a",)),
        )
    )
    monkeypatch.setattr(
        evaluation_projection,
        "_run_relevance_v3_experiment",
        lambda *_args: next(runs),
    )

    result = evaluation_projection.run_relevance_v3_evaluation(release, plan)

    assert result.false_negative_ids == ("case-a", "case-b")
    assert result.threshold_failures == (
        "run 1 precision below 0.90: 0.89",
        "run 2 recall below 0.90: 0.89",
    )
    assert result.passed is False


def test_v3_grouped_projection_rejects_missing_or_misordered_refs() -> None:
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:v3")
    runs = tuple(_v3_projection(ref, 0.90, 0.90, ()) for ref in plan.implementation_refs)

    for invalid_runs in (runs[:2], tuple(reversed(runs))):
        with pytest.raises(ValueError, match="exact reference coverage"):
            evaluation_projection.RelevanceV3EvaluationProjection(
                manifest_artifact_version_id=MANIFEST_VERSION_ID,
                plan=plan,
                runs=invalid_runs,
                false_negative_ids=(),
                threshold_failures=(),
                passed=True,
            )


def test_v3_grouped_projection_rejects_mixed_manifests_false_negatives_and_verdicts() -> None:
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:v3")
    runs = tuple(_v3_projection(ref, 0.90, 0.90, ()) for ref in plan.implementation_refs)

    with pytest.raises(ValueError, match="V3 run manifests must match the group"):
        evaluation_projection.RelevanceV3EvaluationProjection(
            manifest_artifact_version_id="9" * 64,
            plan=plan,
            runs=runs,
            false_negative_ids=(),
            threshold_failures=(),
            passed=True,
        )

    with pytest.raises(ValueError, match="V3 false negatives do not match its runs"):
        evaluation_projection.RelevanceV3EvaluationProjection(
            manifest_artifact_version_id=MANIFEST_VERSION_ID,
            plan=plan,
            runs=runs,
            false_negative_ids=("case-a",),
            threshold_failures=(),
            passed=True,
        )

    with pytest.raises(ValueError, match="V3 threshold failures do not match its runs"):
        evaluation_projection.RelevanceV3EvaluationProjection(
            manifest_artifact_version_id=MANIFEST_VERSION_ID,
            plan=plan,
            runs=runs,
            false_negative_ids=(),
            threshold_failures=("run 1 precision below 0.90: 0.89",),
            passed=False,
        )

    with pytest.raises(ValueError, match="V3 verdict does not match its threshold failures"):
        evaluation_projection.RelevanceV3EvaluationProjection(
            manifest_artifact_version_id=MANIFEST_VERSION_ID,
            plan=plan,
            runs=runs,
            false_negative_ids=(),
            threshold_failures=(),
            passed=False,
        )


def test_comparison_trial_rejects_experiments_and_claims_from_other_trials() -> None:
    baseline, candidate = _promotion_pair()

    with pytest.raises(ValueError, match="trial implementation reference"):
        evaluation_projection.RelevanceComparisonTrial(
            ordinal=1,
            implementation_ref="git:other",
            baseline=baseline,
            candidate=candidate,
            promotion_failures=(),
            passed=True,
        )

    with pytest.raises(ValueError, match="Trial promotion failures do not match"):
        evaluation_projection.RelevanceComparisonTrial(
            ordinal=1,
            implementation_ref="git:test",
            baseline=baseline,
            candidate=candidate,
            promotion_failures=("candidate recall below gate: 0.81",),
            passed=True,
        )

    with pytest.raises(ValueError, match="Trial verdict does not match"):
        evaluation_projection.RelevanceComparisonTrial(
            ordinal=1,
            implementation_ref="git:test",
            baseline=baseline,
            candidate=candidate,
            promotion_failures=(),
            passed=False,
        )


def test_comparison_result_rejects_mispaired_mixed_and_misclaimed_trials() -> None:
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:compare")
    refs = plan.implementation_refs

    def trial(ordinal, implementation_ref):
        baseline, candidate = _promotion_pair(implementation_ref=implementation_ref)
        return evaluation_projection.RelevanceComparisonTrial(
            ordinal=ordinal,
            implementation_ref=implementation_ref,
            baseline=baseline,
            candidate=candidate,
            promotion_failures=(),
            passed=True,
        )

    with pytest.raises(ValueError, match="exact reference coverage"):
        evaluation_projection.RelevanceComparisonResult(
            manifest_artifact_version_id=MANIFEST_VERSION_ID,
            plan=plan,
            trials=(trial(1, refs[1]), trial(2, refs[0]), trial(3, refs[2])),
            promotion_failures=(),
            promoted=True,
        )

    valid_trials = (trial(1, refs[0]), trial(2, refs[1]), trial(3, refs[2]))

    with pytest.raises(ValueError, match="trial manifests must match the group"):
        evaluation_projection.RelevanceComparisonResult(
            manifest_artifact_version_id="9" * 64,
            plan=plan,
            trials=valid_trials,
            promotion_failures=(),
            promoted=True,
        )

    with pytest.raises(ValueError, match="Grouped promotion failures do not match"):
        evaluation_projection.RelevanceComparisonResult(
            manifest_artifact_version_id=MANIFEST_VERSION_ID,
            plan=plan,
            trials=valid_trials,
            promotion_failures=("trial 1 of 3: candidate recall below gate: 0.81",),
            promoted=False,
        )


def test_relevance_promotion_gate_accepts_only_a_strict_safe_improvement() -> None:
    baseline, candidate = _promotion_pair()

    assert evaluation_projection.relevance_promotion_failures(baseline, candidate) == ()

    negative_failure = candidate.model_copy(
        update={
            "case_results": (
                candidate.case_results[0],
                _case_result("negative", False, False, True, RELEVANCE_POLICY_V2),
            ),
            "metrics": baseline.metrics,
        }
    )
    negative_failures = evaluation_projection.relevance_promotion_failures(
        baseline, negative_failure
    )
    assert any("expected-negative case accepted" in failure for failure in negative_failures)
    assert any("precision did not strictly improve" in failure for failure in negative_failures)

    control_failure = candidate.model_copy(
        update={
            "case_results": (
                _case_result("positive-control", True, True, False, RELEVANCE_POLICY_V2),
                candidate.case_results[1],
            ),
            "metrics": candidate.metrics.model_copy(
                update={"recall": 0.0, "positive_control_preservation": 0.0}
            ),
        }
    )
    control_failures = evaluation_projection.relevance_promotion_failures(baseline, control_failure)
    assert any("positive control rejected" in failure for failure in control_failures)
    assert any("candidate recall regressed" in failure for failure in control_failures)
    assert any("positive-control preservation regressed" in failure for failure in control_failures)

    equal_precision = candidate.model_copy(
        update={"metrics": candidate.metrics.model_copy(update={"precision": 0.5})}
    )
    assert any(
        "precision did not strictly improve" in failure
        for failure in evaluation_projection.relevance_promotion_failures(baseline, equal_precision)
    )


@pytest.mark.parametrize(
    ("changes", "failure"),
    (
        (
            {"manifest_artifact_version_id": "b" * 64},
            "baseline and candidate manifest identities differ",
        ),
        (
            {"implementation_ref": "git:other"},
            "baseline and candidate implementation references differ",
        ),
        ({"dataset_id": "other-dataset"}, "baseline and candidate dataset identities differ"),
        ({"dataset_name": "other"}, "baseline and candidate dataset identities differ"),
    ),
)
def test_relevance_promotion_rejects_experiment_identity_mismatch(changes, failure) -> None:
    baseline, candidate = _promotion_pair()

    failures = evaluation_projection.relevance_promotion_failures(
        baseline, candidate.model_copy(update=changes)
    )

    assert failures == (failure,)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("case_id", "other-case"),
        ("article_version_id", "3" * 64),
        ("control", False),
        ("expected_accepted", False),
    ),
)
def test_relevance_promotion_rejects_case_identity_mismatch(field, value) -> None:
    baseline, candidate = _promotion_pair()
    changed_case = candidate.case_results[0].model_copy(update={field: value})
    changed = candidate.model_copy(
        update={"case_results": (changed_case, *candidate.case_results[1:])}
    )

    failures = evaluation_projection.relevance_promotion_failures(baseline, changed)

    assert failures == ("baseline and candidate case identities differ",)


def test_explicit_projection_requires_langfuse_credentials(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_projection, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(evaluation_projection, "LANGFUSE_SECRET_KEY", None)
    monkeypatch.setattr(
        evaluation_projection,
        "_langfuse_client",
        lambda: (_ for _ in ()).throw(AssertionError("client constructed")),
    )

    with pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY"):
        evaluation_projection.project_news_evaluation_release(_release())


def test_cli_projects_the_pin_and_runs_an_explicit_experiment(monkeypatch, capsys) -> None:
    release = _release()
    calls = []
    dataset_projection = evaluation_projection.NewsEvaluationDatasetProjection(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        dataset_id="dataset-10",
        dataset_name="synthetic",
        created_examples=5,
        reused_examples=0,
    )
    experiment_projection = evaluation_projection.NewsEvaluationExperimentProjection(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        dataset_id=dataset_projection.dataset_id,
        dataset_name=dataset_projection.dataset_name,
        experiment_id="experiment-11",
        experiment_name="synthetic-run",
    )
    monkeypatch.setattr(
        evaluation_projection,
        "load_news_evaluation_release",
        lambda content: calls.append(("load", content)) or release,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "project_news_evaluation_release",
        lambda loaded: calls.append(("project", loaded)) or dataset_projection,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "run_news_evaluation_experiment",
        lambda loaded, implementation_ref: calls.append(("experiment", loaded, implementation_ref))
        or experiment_projection,
    )

    assert evaluation_projection.main(("--implementation-ref", "git:abc123")) == 0
    assert calls[0] == ("load", evaluation_projection.PIN_PATH.read_bytes())
    assert calls[1] == ("project", release)
    assert calls[2][2:] == ("git:abc123",)
    assert "synthetic-run" in capsys.readouterr().out


def test_cli_runs_relevance_comparison_with_a_fresh_plan(monkeypatch, capsys) -> None:
    release = _release()
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:abc123", trial_count=4)
    comparison = _comparison_result(plan)
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "load_news_evaluation_release",
        lambda content: calls.append(("load", content)) or release,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "compare_relevance_policies",
        lambda loaded, received_plan: calls.append(("compare", loaded, received_plan))
        or comparison,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "project_news_evaluation_release",
        lambda *_args: (_ for _ in ()).throw(AssertionError("deterministic path ran")),
    )

    result = evaluation_projection.main(
        (
            "--implementation-ref",
            "git:abc123",
            "--compare-relevance",
            "--trial-count",
            "4",
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload == comparison.model_dump(mode="json")
    assert calls == [
        ("load", evaluation_projection.PIN_PATH.read_bytes()),
        ("compare", release, plan),
    ]


def test_cli_runs_v3_evaluation_with_a_fresh_plan(monkeypatch, capsys) -> None:
    release = _release()
    plan = evaluation_projection.FreshEvaluationPlan(implementation_ref="git:abc123", trial_count=4)
    runs = tuple(_v3_projection(ref, 0.90, 0.90, ()) for ref in plan.implementation_refs)
    evaluation = evaluation_projection.RelevanceV3EvaluationProjection(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        plan=plan,
        runs=runs,
        false_negative_ids=(),
        threshold_failures=(),
        passed=True,
    )
    calls = []
    monkeypatch.setattr(
        evaluation_projection,
        "load_news_evaluation_release",
        lambda _content: release,
    )
    monkeypatch.setattr(
        evaluation_projection,
        "run_relevance_v3_evaluation",
        lambda loaded, received_plan: calls.append((loaded, received_plan)) or evaluation,
    )

    result = evaluation_projection.main(
        (
            "--implementation-ref",
            "git:abc123",
            "--evaluate-relevance-v3",
            "--trial-count",
            "4",
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == evaluation.model_dump(mode="json")
    assert calls == [(release, plan)]


def test_cli_rejects_trial_count_for_deterministic_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_projection,
        "load_news_evaluation_release",
        lambda _content: _release(),
    )

    with pytest.raises(SystemExit):
        evaluation_projection.main(("--implementation-ref", "git:abc123", "--trial-count", "3"))


def test_cli_rejects_an_empty_implementation_reference() -> None:
    with pytest.raises(SystemExit):
        evaluation_projection.main(("--implementation-ref", "   "))


def _comparison_result(
    plan: evaluation_projection.FreshEvaluationPlan,
) -> evaluation_projection.RelevanceComparisonResult:
    trials = []
    for ordinal, implementation_ref in enumerate(plan.implementation_refs, start=1):
        baseline, candidate = _promotion_pair(implementation_ref=implementation_ref)
        trials.append(
            evaluation_projection.RelevanceComparisonTrial(
                ordinal=ordinal,
                implementation_ref=implementation_ref,
                baseline=baseline,
                candidate=candidate,
                promotion_failures=(),
                passed=True,
            )
        )
    return evaluation_projection.RelevanceComparisonResult(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        plan=plan,
        trials=tuple(trials),
        promotion_failures=(),
        promoted=True,
    )


def _patch_boundaries(monkeypatch):
    events = []
    receipts = {}
    client = FakeLangfuseClient(events)

    def query(sql, params):
        if "news_relevance_experiments" in sql:
            key = (params[0], params[1], "relevance", params[2], params[3])
        else:
            key = tuple(params)
        receipt = receipts.get(key)
        return [receipt] if receipt is not None else []

    def batch(statements):
        assert len(statements) == 1
        sql, values = statements[0]
        if "news_relevance_experiments" in sql:
            events.append("receipt:relevance")
            if client.fail_experiment_receipts:
                client.fail_experiment_receipts -= 1
                raise RuntimeError("catalog receipt failed")
            key = (values[0], values[1], "relevance", values[2], values[3])
            receipts[key] = {
                "provider": values[0],
                "manifest_artifact_version_id": values[1],
                "policy_digest": values[2],
                "implementation_ref": values[3],
                "policy_id": values[4],
                "dataset_id": values[5],
                "dataset_name": values[6],
                "experiment_id": values[7],
                "experiment_name": values[8],
                "experiment_url": values[9],
            }
            return
        kind = values[2]
        events.append(f"receipt:{kind}")
        if kind == "experiment" and client.fail_experiment_receipts:
            client.fail_experiment_receipts -= 1
            raise RuntimeError("catalog receipt failed")
        key = (values[0], values[1], kind, values[8])
        receipts[key] = {
            "provider": values[0],
            "manifest_artifact_version_id": values[1],
            "projection_kind": kind,
            "dataset_id": values[3],
            "dataset_name": values[4],
            "experiment_id": values[5],
            "experiment_name": values[6],
            "experiment_url": values[7],
            "implementation_ref": values[8],
        }

    monkeypatch.setattr(evaluation_projection, "LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setattr(evaluation_projection, "LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setattr(evaluation_projection, "_langfuse_client", lambda: client)
    monkeypatch.setattr(evaluations, "catalog_query", query)
    monkeypatch.setattr(evaluations, "catalog_batch", batch)
    monkeypatch.setattr(evaluation_projection, "_sleep", lambda _delay: None)
    return client, events, receipts


def _release() -> LoadedNewsEvaluationRelease:
    dataset = synthetic_dataset()
    baseline = build_news_evaluation_baseline(
        evaluate_news_dataset(dataset),
        MANIFEST_REFERENCE,
    )
    return LoadedNewsEvaluationRelease.model_construct(
        pin=NewsEvaluationPin(
            manifest_version_id=MANIFEST_VERSION_ID,
            baseline_version_id="c" * 64,
        ),
        manifest=synthetic_manifest(),
        manifest_reference=MANIFEST_REFERENCE,
        baseline_reference=ArtifactReference(
            artifact_id="news:evaluation-baseline:test",
            version_id="c" * 64,
            content_digest="d" * 64,
            r2_key="news/evaluations/baselines/test.json",
        ),
        baseline=baseline,
        dataset=dataset,
    )


def _release_with_theme_spec() -> LoadedNewsEvaluationRelease:
    release = _release()
    report = release.manifest.reports[0]
    theme = ThemeEvaluationDaySpec(
        case_id="daily-theme-day",
        provenance=EvaluationSpecProvenance(feedback_ids=()),
        day=release.manifest.reviewed_at.date(),
        source_report=report.report,
        cluster_set=report.cluster_set,
        summaries=(MANIFEST_REFERENCE,),
        expectations=(
            ThemePairExpectation(
                case_id="daily-theme-pair",
                feedback_ids=(),
                left_group_id="1" * 64,
                right_group_id="2" * 64,
                expected_same_theme=True,
                rationale="Reviewed theme pair.",
            ),
        ),
    )
    return release.model_copy(
        update={
            "manifest": release.manifest.model_copy(
                update={"cases": (*release.manifest.cases, theme)}
            )
        }
    )


def _release_with_two_relevance_cases() -> LoadedNewsEvaluationRelease:
    release = _release()
    first = next(case for case in release.manifest.cases if case.concern == "relevance")
    second = first.model_copy(update={"case_id": "relevance-second"})
    manifest = release.manifest.model_copy(update={"cases": (first, second)})
    return release.model_copy(update={"manifest": manifest})


def _release_with_prior_and_two_relevance_cases():
    release = _release_with_two_relevance_cases()
    prior = _release().manifest
    prior_reference = ArtifactReference(
        artifact_id="news:evaluation-manifest:prior",
        version_id="9" * 64,
        content_digest="8" * 64,
        r2_key="prior-manifest.json",
    )
    return (
        release.model_copy(
            update={
                "manifest": release.manifest.model_copy(update={"prior_manifest": prior_reference})
            }
        ),
        prior.model_dump_json().encode(),
    )


def _v3_output(article, policy, execution_ref, *, observability_complete=True):
    context = ContextDecision(
        subject_role="principal",
        news_cycle="current_cycle",
        romanian_consequence="direct",
        certainty="clear",
        evidence_quote="Dovada",
        reason_ro="Motiv",
    )
    impact = ImpactDecision(
        consequence_status="realized",
        effect_basis="actual_consequence",
        effect_scope="sector",
        magnitude="routine",
        political_relevance="none",
        economic_relevance="strong",
        quantified=True,
        certainty="clear",
        evidence_quote="Dovada",
        reason_ro="Motiv",
    )
    return SimpleNamespace(
        policy=policy,
        policy_digest=relevance_v3_policy_digest(policy),
        request_id=relevance_v3_request_id(
            article,
            mode="full_evaluation",
            policy=policy,
            execution_ref=execution_ref,
        ),
        execution_ref=execution_ref,
        accepted=True,
        context=SimpleNamespace(decision=context),
        impact=SimpleNamespace(decision=impact),
        context_early_exit=False,
        input_tokens=3,
        output_tokens=5,
        cost_usd=0.01,
        observability_complete=observability_complete,
    )


def _v3_projection(name, precision, recall, false_negative_ids, *, observability_complete=True):
    metrics = evaluation_projection.RelevanceV3SliceMetrics(
        passed_cases=1,
        total_cases=1,
        precision=precision,
        recall=recall,
    )
    return evaluation_projection.RelevanceV3ExperimentProjection(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        dataset_id="dataset-v3",
        dataset_name="v3",
        experiment_id=f"experiment-{name}",
        experiment_name=name,
        implementation_ref=name,
        policy_id=RELEVANCE_V3_POLICY.policy_id,
        policy_digest=relevance_v3_policy_digest(),
        case_results=(),
        full_metrics=metrics,
        prior_metrics=metrics,
        new_metrics=metrics,
        false_negative_ids=false_negative_ids,
        context_early_exit_rate=0,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        observability_complete=observability_complete,
    )


def _relevance_output(article, policy, *, accepted):
    decision = relevance_decision(accepted=accepted)
    return RelevanceOutput(
        request_id=relevance_request_id(article, policy),
        policy=policy,
        article=article,
        decision=decision,
        call=ModelCall(
            response_id=f"response-{policy.policy_id}",
            model=policy.model,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        ),
        content=b'{"provider_responses":[]}',
    )


def _fresh_run_output(expected, policy, *, accepted):
    output = _relevance_output(expected.article, policy, accepted=accepted)
    return {
        "case_id": expected.case_id,
        "article_version_id": expected.article.version_id,
        "policy_id": output.policy.policy_id,
        "policy_digest": output.policy_digest,
        "request_id": output.request_id,
        "accepted": output.accepted,
        "decision": output.decision.model_dump(mode="json"),
    }


def _promotion_pair(*, implementation_ref="git:test"):
    baseline = _policy_projection(
        RELEVANCE_POLICY_V1,
        (
            _case_result("positive-control", True, True, True, RELEVANCE_POLICY_V1),
            _case_result("negative", False, False, True, RELEVANCE_POLICY_V1),
        ),
    )
    candidate = _policy_projection(
        RELEVANCE_POLICY_V2,
        (
            _case_result("positive-control", True, True, True, RELEVANCE_POLICY_V2),
            _case_result("negative", False, False, False, RELEVANCE_POLICY_V2),
        ),
    )
    return (
        baseline.model_copy(update={"implementation_ref": implementation_ref}),
        candidate.model_copy(update={"implementation_ref": implementation_ref}),
    )


def _case_result(case_id, expected, control, accepted, policy):
    return evaluation_projection.FreshRelevanceCaseResult(
        case_id=case_id,
        article_version_id="1" * 64,
        policy_id=policy.policy_id,
        policy_digest=relevance_policy_digest(policy),
        request_id="2" * 64,
        expected_accepted=expected,
        control=control,
        accepted=accepted,
        passed=accepted == expected,
        decision=relevance_decision(accepted=accepted),
    )


def _policy_projection(policy, results):
    return evaluation_projection.RelevancePolicyExperimentProjection(
        manifest_artifact_version_id=MANIFEST_VERSION_ID,
        dataset_id="dataset-20",
        dataset_name="synthetic",
        experiment_id=f"experiment-{policy.policy_id}",
        experiment_name=policy.policy_id,
        implementation_ref="git:test",
        policy_id=policy.policy_id,
        policy_digest=relevance_policy_digest(policy),
        case_results=results,
        metrics=evaluation_projection._fresh_relevance_metrics(results),
    )


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()
