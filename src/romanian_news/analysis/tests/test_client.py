import httpx

from romanian_news.analysis import client as client_module


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
