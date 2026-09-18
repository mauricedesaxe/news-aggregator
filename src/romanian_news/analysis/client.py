from openai import OpenAI

from romanian_news.config import OPENROUTER_API_KEY


def openrouter_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_retries=1,
        timeout=30,
    )
