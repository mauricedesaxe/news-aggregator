from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from romanian_news.analysis import tracing
from romanian_news.catalog import archive_model_spend


class _Response:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        return {
            "id": "response-1",
            "mode": mode,
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.003},
        }


class _Observation:
    def __init__(self) -> None:
        self.trace_id = "trace-1"
        self.id = "observation-1"
        self.values = {}
        self.ended = False
        self.fail_update = False

    def update(self, **values) -> None:
        if self.fail_update:
            raise RuntimeError("update failed")
        self.values.update(values)

    def end(self) -> None:
        self.ended = True


class _Client:
    def __init__(self) -> None:
        self.observations = []
        self.authenticated = True
        self.fail_start = False
        self.fail_flush = False
        self.start_arguments = None

    def auth_check(self) -> bool:
        if isinstance(self.authenticated, Exception):
            raise self.authenticated
        return self.authenticated

    def start_observation(self, **values):
        if self.fail_start:
            raise RuntimeError("start failed")
        self.start_arguments = values
        observation = _Observation()
        self.observations.append(observation)
        return observation

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace-for-{seed}"

    def flush(self) -> None:
        if self.fail_flush:
            raise RuntimeError("flush failed")


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch):
    client = _Client()
    monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setattr(tracing, "LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setattr(tracing, "LANGFUSE_PROJECT_ID", "test-project")
    monkeypatch.setattr(tracing, "_langfuse_client", lambda: client)
    tracing.langfuse_tracing_available.cache_clear()
    return client


def test_tracing_disabled_calls_provider_without_observation(monkeypatch, _reset_tracing) -> None:
    monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", None)
    response = _Response()

    result = tracing.trace_provider_call(
        "news.relevance", "request-1", {"model": "test/model"}, lambda: response
    )

    assert result.response is response
    assert result.trace is None
    assert _reset_tracing.observations == []


def test_archive_call_reserves_before_provider_and_settles_response(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", None)
    events = []
    monkeypatch.setattr(
        archive_model_spend,
        "reserve_archive_spend",
        lambda day, reservation_id, operation, request, **kwargs: events.append(
            ("reserve", day, reservation_id, operation, request, kwargs["limit_usd"])
        ),
    )
    monkeypatch.setattr(
        archive_model_spend,
        "settle_archive_spend",
        lambda reservation_id, cost: events.append(("settle", reservation_id, cost)),
    )
    monkeypatch.setattr(
        archive_model_spend,
        "read_archive_spend",
        lambda _day: archive_model_spend.ArchiveSpend(Decimal("0.003"), Decimal(0), 1),
    )
    with tracing.archive_model_day(date(2025, 9, 27)):
        result = tracing.trace_provider_call(
            "news.relevance", "request-1", {}, lambda: (events.append(("call",)), _Response())[1]
        )

    assert result.response.model_dump(mode="json")["id"] == "response-1"
    assert [value[0] for value in events] == ["reserve", "call", "settle"]
    assert events[0][2] == result.call_id == events[2][1]
    assert events[2][2] == Decimal("0.003")


def test_archive_limit_refusal_never_calls_provider(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(
        archive_model_spend,
        "reserve_archive_spend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            archive_model_spend.ArchiveSpendLimitReached("full")
        ),
    )
    with tracing.archive_model_day(date(2025, 9, 27)):
        with pytest.raises(archive_model_spend.ArchiveSpendLimitReached, match="full"):
            tracing.trace_provider_call(
                "news.relevance", "request-1", {}, lambda: pytest.fail("provider called")
            )


def test_successful_trace_records_full_observation(_reset_tracing) -> None:
    response = _Response()
    inputs = {"model": "test/model", "messages": [{"role": "user", "content": "full"}]}

    result = tracing.trace_provider_call("news.relevance", "request-1", inputs, lambda: response)

    reference = result.trace
    observation = _reset_tracing.observations[0]
    assert result.response is response
    assert reference is not None
    assert reference.provider == "langfuse"
    assert reference.trace_id == "trace-1"
    assert reference.observation_id == "observation-1"
    assert reference.project_ref == "test-project"
    assert _reset_tracing.start_arguments == {
        "trace_context": {"trace_id": f"trace-for-{result.call_id}"},
        "name": "news.relevance",
        "as_type": "generation",
        "input": inputs,
        "metadata": {"operation_key": "news.relevance", "request_id": "request-1"},
        "model": "test/model",
    }
    assert observation.values == {
        "output": {
            "provider_response": {
                "id": "response-1",
                "mode": "json",
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.003},
            }
        },
        "usage_details": {"input": 12, "output": 4},
        "cost_details": {"total": 0.003},
    }
    assert observation.ended


def test_trace_setup_failure_does_not_block_provider(_reset_tracing) -> None:
    _reset_tracing.fail_start = True
    calls = 0

    def call() -> _Response:
        nonlocal calls
        calls += 1
        return _Response()

    result = tracing.trace_provider_call("news.embed", "request-1", {}, call)

    assert isinstance(result.response, _Response)
    assert result.trace is None
    assert calls == 1


def test_invalid_trace_credentials_disable_tracing_without_blocking_provider(
    _reset_tracing,
) -> None:
    _reset_tracing.authenticated = RuntimeError("forbidden")

    result = tracing.trace_provider_call("news.relevance", "request-1", {}, _Response)

    assert isinstance(result.response, _Response)
    assert result.trace is None
    assert _reset_tracing.observations == []


def test_trace_update_failure_does_not_block_provider_result(_reset_tracing) -> None:
    observation = _Observation()
    observation.fail_update = True
    _reset_tracing.start_observation = lambda **_values: observation

    result = tracing.trace_provider_call("news.embed", "request-1", {}, _Response)

    assert isinstance(result.response, _Response)
    assert result.trace is not None
    assert observation.ended


def test_provider_error_finishes_trace_and_preserves_error(_reset_tracing) -> None:
    error = RuntimeError("provider failed")

    def call() -> _Response:
        raise error

    with pytest.raises(RuntimeError) as raised:
        tracing.trace_provider_call("news.summarize_group", "request-1", {}, call)

    observation = _reset_tracing.observations[0]
    assert raised.value is error
    assert observation.values == {
        "output": {"error": "RuntimeError: provider failed"},
        "level": "ERROR",
        "status_message": "provider failed",
    }
    assert observation.ended


def test_flush_failure_is_best_effort(_reset_tracing) -> None:
    _reset_tracing.fail_flush = True

    tracing.flush_langfuse_traces()


def test_failed_delivery_does_not_create_a_durable_trace_reference(_reset_tracing) -> None:
    _reset_tracing.fail_flush = True

    result = tracing.trace_provider_call("news.embed", "request-1", {}, _Response)

    assert result.trace is None


def test_trace_reference_accepts_opaque_remote_ids() -> None:
    value = tracing.ModelTraceReference(
        provider="langfuse",
        trace_id="trace-provider-value",
        observation_id="observation-provider-value",
        project_ref="project",
        recorded_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert value.observation_id == "observation-provider-value"
