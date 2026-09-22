import hashlib
import json

from romanian_news import Sha256


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(content: bytes) -> Sha256:
    return hashlib.sha256(content).hexdigest()
