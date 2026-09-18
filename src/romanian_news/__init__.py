from typing import Annotated as _Annotated
from zoneinfo import ZoneInfo as _ZoneInfo

import pydantic as _pydantic

BUCHAREST = _ZoneInfo("Europe/Bucharest")
GENERATION_MODEL = "google/gemini-2.5-flash-lite"
GROUP_ANALYSIS_MODEL = "openai/gpt-4.1-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

Sha256 = _Annotated[str, _pydantic.StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FeedId = _Annotated[str, _pydantic.StringConstraints(pattern=r"^[a-z0-9-]+$")]
OutletId = _Annotated[str, _pydantic.StringConstraints(pattern=r"^[a-z0-9-]+$")]


class NewsModel(_pydantic.BaseModel):
    model_config = _pydantic.ConfigDict(extra="forbid", frozen=True, strict=True)
