from openai import OpenAI

from romanian_news.config import OPENROUTER_API_KEY


def openrouter_client(*, max_retries: int = 1, timeout_seconds: float = 30) -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_retries=max_retries,
        timeout=timeout_seconds,
    )
