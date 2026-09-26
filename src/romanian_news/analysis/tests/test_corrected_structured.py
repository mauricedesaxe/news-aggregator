import json
from types import SimpleNamespace

import pytest

from romanian_news.analysis import corrected_structured
from romanian_news.analysis.corrected_structured import (
    StructuredMessage,
    run_corrected_structured_openrouter,
)


def test_corrected_structured_call_records_rejection_and_acceptance(monkeypatch) -> None:
    calls, records = _provider(monkeypatch, ("{", '{"answer": 42}'))

    result = _run(lambda content: json.loads(content))

    assert result.value == {"answer": 42}
    assert [attempt.status for attempt in result.attempts] == ["rejected", "accepted"]
    assert [record["status"] for record in records] == ["rejected", "accepted"]
    assert result.call.input_tokens == 20
    assert result.call.output_tokens == 10
    assert result.call.response_id == "response-2"
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": "{"}
    assert calls[1]["messages"][-1]["role"] == "user"
    assert "Validation error:" in calls[1]["messages"][-1]["content"]
    assert calls[0]["response_format"]["json_schema"] == {
        "name": "test_response",
        "strict": True,
        "schema": {"type": "object"},
    }
    assert calls[0]["extra_body"] == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "low"},
    }


def test_corrected_structured_call_raises_supplied_error_after_two_rejections(
    monkeypatch,
) -> None:
    calls, records = _provider(monkeypatch, ("{", "{"))

    with pytest.raises(ValueError, match="still invalid") as raised:
        _run(lambda content: json.loads(content))

    assert isinstance(raised.value.__cause__, ValueError)
    assert len(calls) == len(records) == 2


def test_corrected_structured_call_does_not_retry_provider_failures(monkeypatch) -> None:
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))),
    )

    with pytest.raises(ConnectionError, match="provider unavailable"):
        _run(lambda content: json.loads(content))

    assert len(calls) == 1


def test_corrected_structured_call_accepts_none_as_a_parsed_value(monkeypatch) -> None:
    _provider(monkeypatch, ('{"answer": null}',))

    result = _run(lambda _content: None)

    assert result.value is None


def _run(parse):
    return run_corrected_structured_openrouter(
        operation="news.test",
        request_id="a" * 64,
        model="provider/model",
        temperature=0,
        max_tokens=100,
        reasoning_effort="low",
        schema_name="test_response",
        response_schema={"type": "object"},
        initial_messages=(
            StructuredMessage(role="system", content="system"),
            StructuredMessage(role="user", content="user"),
        ),
        parse=parse,
        correction_message=lambda error: f"Validation error: {error}",
        exhausted_error=lambda error: ValueError(f"still invalid: {error}"),
        unreachable_error="unreachable",
        started_at=0.0,
    )


def _provider(monkeypatch, contents):
    calls = []
    records = []
    responses = iter(
        _response(content, f"response-{index}") for index, content in enumerate(contents, start=1)
    )
    monkeypatch.setattr(corrected_structured.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        corrected_structured,
        "openrouter_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: calls.append(kwargs) or next(responses)
                )
            )
        ),
    )

    def record(response, **kwargs):
        records.append(kwargs)
        return SimpleNamespace(attempt_id=str(len(records)) * 64, response_id=response.id)

    monkeypatch.setattr(corrected_structured, "record_model_attempt", record)
    return calls, records


def _response(content: str, response_id: str):
    payload = {
        "id": response_id,
        "model": "provider/model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
        "choices": [{"message": {"content": content}}],
    }
    return SimpleNamespace(
        id=response_id,
        model="provider/model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=(SimpleNamespace(message=SimpleNamespace(content=content)),),
        model_dump=lambda *, mode: payload,
    )
