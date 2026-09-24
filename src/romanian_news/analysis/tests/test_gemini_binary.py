import ast
import hashlib
import inspect
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import OpenAI

from romanian_news.analysis import gemini_binary
from romanian_news.analysis.binary_evaluation import (
    BinaryAttemptEvidence,
    BinaryQuestion,
    BinaryRequest,
    binary_state_digest,
)
from romanian_news.analysis.gemini_binary import (
    GEMINI_BINARY_EXECUTION_POLICY,
    evaluate_gemini_binary,
    gemini_binary_execution_policy_digest,
    gemini_binary_request_id,
)


def test_gemini_binary_sends_only_canonical_semantics_and_normalizes_accounting(
    monkeypatch,
) -> None:
    request = _request()
    sent: list[object] = []

    def handle(transport_request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(transport_request.content))
        return httpx.Response(
            200,
            json={
                "id": "openrouter-request-1",
                "object": "chat.completion",
                "created": 1_760_000_000,
                "model": "google/gemini-2.5-flash-001",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "report_probability",
                                        "arguments": json.dumps({"probability": 0.73}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 7,
                    "total_tokens": 108,
                    "cost": 0.00042,
                },
            },
            request=transport_request,
        )

    def client(*, max_retries: int, timeout_seconds: float) -> OpenAI:
        del max_retries, timeout_seconds
        return OpenAI(
            base_url="http://openrouter.test/v1",
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    monkeypatch.setattr(gemini_binary, "openrouter_client", client)

    result = evaluate_gemini_binary(
        request,
        execution_ref="git:test:trial-1",
        clock=iter((10.0, 10.125)).__next__,
    )

    (body,) = sent
    assert body == {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "system",
                "content": request.question.instructions,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "true_criteria": request.question.true_criteria,
                        "false_criteria": request.question.false_criteria,
                        "state": request.state,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "report_probability",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "probability": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["probability"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "report_probability"},
        },
        "provider": {"require_parameters": True},
    }
    assert "threshold" not in json.dumps(body)
    assert result.request_id == gemini_binary_request_id(request, "git:test:trial-1")
    assert result.provider_request_id == "openrouter-request-1"
    assert result.model == "google/gemini-2.5-flash-001"
    assert result.probability == Decimal("0.73")
    assert result.predicted_accepted is False
    assert result.input_tokens == 101
    assert result.output_tokens == 7
    assert result.latency_ms == 125
    assert result.estimated_cost_usd == Decimal("0.00042")


def test_gemini_binary_execution_policy_and_request_identity_are_stable() -> None:
    request = _request()

    assert (
        gemini_binary_execution_policy_digest()
        == "046e423c0156e2fe311050511f3d13b853d61075587028dbfbda566af326e641"
    )
    first = gemini_binary_request_id(request, "trial-1")
    expected = hashlib.sha256(
        json.dumps(
            {
                "semantic_digest": request.question.semantic_digest,
                "state_digest": request.state_digest,
                "execution_ref": "trial-1",
                "execution_policy_digest": gemini_binary_execution_policy_digest(),
                "purpose": "news.binary.gemini.evaluation",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert first == expected
    assert first == gemini_binary_request_id(request, "trial-1")
    assert first != gemini_binary_request_id(request, "trial-2")
    assert first != gemini_binary_request_id(
        request.model_copy(update={"state": "Other", "state_digest": binary_state_digest("Other")}),
        "trial-1",
    )
    assert first != gemini_binary_request_id(
        request.model_copy(
            update={
                "question": request.question.model_copy(
                    update={"instructions": "Different instructions"}
                )
            }
        ),
        "trial-1",
    )
    assert first != gemini_binary_request_id(
        request,
        "trial-1",
        GEMINI_BINARY_EXECUTION_POLICY.model_copy(update={"max_tokens": 65}),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("id", "", "provider request ID"),
        ("id", None, "provider request ID"),
        ("model", "", "response model"),
        ("model", 1, "response model"),
        ("prompt_tokens", None, "prompt_tokens"),
        ("prompt_tokens", True, "prompt_tokens"),
        ("prompt_tokens", -1, "prompt_tokens"),
        ("completion_tokens", "7", "completion_tokens"),
        ("cost", None, "cost"),
        ("cost", float("nan"), "cost"),
        ("cost", -0.01, "cost"),
        ("cost", "0.01", "cost"),
    ),
)
def test_gemini_binary_rejects_missing_or_invalid_identity_and_accounting(
    monkeypatch,
    field: str,
    value: object,
    message: str,
) -> None:
    response = _response(**{field: value})
    create = Mock(return_value=response)
    monkeypatch.setattr(gemini_binary, "openrouter_client", lambda **_kwargs: _client(create))

    with pytest.raises(ValueError, match=message):
        evaluate_gemini_binary(_request(), execution_ref="trial-1")


def test_gemini_binary_rejects_wrong_tool_without_semantic_retry(monkeypatch) -> None:
    create = Mock(return_value=_response(tool_name="other-tool"))
    monkeypatch.setattr(gemini_binary, "openrouter_client", lambda **_kwargs: _client(create))

    with pytest.raises(ValueError, match="wrong tool"):
        evaluate_gemini_binary(_request(), execution_ref="trial-1")

    assert create.call_count == 1


def test_gemini_binary_preserves_response_accounting_when_output_is_malformed(
    monkeypatch,
) -> None:
    response = _response(content="not-json", model="google/gemini-2.5-flash-001")
    create = Mock(return_value=response)
    attempts: list[BinaryAttemptEvidence] = []
    monkeypatch.setattr(gemini_binary, "openrouter_client", lambda **_kwargs: _client(create))

    with pytest.raises(ValueError):
        evaluate_gemini_binary(
            _request(),
            execution_ref="trial-1",
            clock=iter((3.0, 3.25)).__next__,
            on_attempt=attempts.append,
        )

    assert create.call_count == 1
    assert len(attempts) == 1
    assert attempts[0].status == "terminal_error"
    assert attempts[0].provider_request_id == "openrouter-request-1"
    assert attempts[0].actual_model == "google/gemini-2.5-flash-001"
    assert attempts[0].input_tokens == 101
    assert attempts[0].output_tokens == 7
    assert attempts[0].cost_usd == Decimal("0.00042")
    assert attempts[0].latency_ms == 250
    assert attempts[0].probability is None
    assert attempts[0].error is not None
    assert attempts[0].error.retryable is False


@pytest.mark.parametrize("clocks", ((2.0, 1.0), (1.0, float("inf"))))
def test_gemini_binary_rejects_invalid_latency(monkeypatch, clocks) -> None:
    create = Mock(return_value=_response())
    monkeypatch.setattr(gemini_binary, "openrouter_client", lambda **_kwargs: _client(create))

    with pytest.raises(ValueError, match="latency"):
        evaluate_gemini_binary(_request(), execution_ref="trial-1", clock=iter(clocks).__next__)


def test_gemini_binary_does_not_import_production_relevance_prompts() -> None:
    source = ast.parse(inspect.getsource(gemini_binary))
    direct_imports = {
        alias.name
        for node in ast.walk(source)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imports = set()
    for node in ast.walk(source):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is not None:
            from_imports.add(node.module)
            from_imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        else:
            from_imports.update(alias.name for alias in node.names)
    imports = direct_imports | from_imports

    forbidden = {
        "relevance",
        "relevance_v3",
        "romanian_news.analysis.relevance",
        "romanian_news.analysis.relevance_v3",
    }
    assert imports.isdisjoint(forbidden)


def _request() -> BinaryRequest:
    state = "Canonical state"
    return BinaryRequest(
        question=BinaryQuestion(
            question_id="benchmark-question",
            instructions="Canonical instructions",
            true_criteria="Canonical true criteria",
            false_criteria="Canonical false criteria",
            threshold=Decimal("0.8"),
        ),
        state=state,
        state_digest=binary_state_digest(state),
    )


def _client(create: Mock) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _response(
    *,
    id: object = "openrouter-request-1",
    model: object = "google/gemini-2.5-flash",
    probability: object = 0.73,
    prompt_tokens: object = 101,
    completion_tokens: object = 7,
    cost: object = 0.00042,
    content: object = None,
    tool_name: object = "report_probability",
) -> SimpleNamespace:
    if content is None:
        content = json.dumps({"probability": probability})
    tool_call = SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=content))
    payload = {
        "id": id,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        },
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": content,
                            },
                        }
                    ],
                }
            }
        ],
    }
    return SimpleNamespace(
        id=id,
        model=model,
        choices=(SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call])),),
        model_dump=lambda *, mode: payload,
    )
