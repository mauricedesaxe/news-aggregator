from datetime import date

import httpx
import pytest
from openai import OpenAIError

from romanian_news.analysis import client as client_module
from romanian_news.analysis.tracing import archive_model_day


def test_openrouter_client_retries_one_transient_failure(monkeypatch) -> None:
    attempts = 0
    openai = client_module.OpenAI

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"object": "list", "data": []}, request=request)

    def openai_with_test_transport(**kwargs):
        return openai(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    monkeypatch.setattr(client_module, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client_module, "OpenAI", openai_with_test_transport)

    client = client_module.openrouter_client()
    client.models.list()

    assert attempts == 2


def test_archive_provider_call_has_no_hidden_retry(monkeypatch) -> None:
    attempts = 0
    openai = client_module.OpenAI

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, request=request)

    monkeypatch.setattr(client_module, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        client_module,
        "OpenAI",
        lambda **kwargs: openai(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))
        ),
    )

    with archive_model_day(date(2025, 9, 27)):
        with pytest.raises(OpenAIError):
            client_module.openrouter_client().models.list()
    assert attempts == 1
