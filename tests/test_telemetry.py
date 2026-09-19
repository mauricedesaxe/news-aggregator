import asyncio
from types import SimpleNamespace

from starlette.types import Message, Receive, Scope, Send

from romanian_news.telemetry import HttpTracingMiddleware


def test_http_tracing_skips_a_non_callable_route_matcher() -> None:
    messages: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/items/1",
        "headers": [],
        "scheme": "http",
        "server": ("test", 80),
    }
    middleware = HttpTracingMiddleware(app, routes=(SimpleNamespace(matches=1),))

    asyncio.run(middleware(scope, receive, send))

    assert [message["type"] for message in messages] == [
        "http.response.start",
        "http.response.body",
    ]
