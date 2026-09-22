import io
import threading

import pytest
from botocore.exceptions import ClientError

from romanian_news import storage as research_storage


def test_immutable_r2_objects_publish_concurrently(monkeypatch) -> None:
    barrier = threading.Barrier(2)
    client = _R2Client(barrier)
    monkeypatch.setattr(research_storage, "_r2_client", lambda: client)

    publication = research_storage.publish_immutable_r2_objects(
        (("first", b"first"), ("second", b"second"))
    )

    assert publication.uploaded_objects == 2
    assert publication.reused_objects == 0


def test_immutable_r2_objects_reuse_identical_existing_content(monkeypatch) -> None:
    import hashlib

    content = b"already stored"
    uploads: list[str] = []

    class ReusingClient:
        def __init__(self) -> None:
            self.store: dict[str, bytes] = {}

        def head_object(self, **kwargs: object) -> dict[str, object]:
            if kwargs["Key"] == "stored":
                return {"Metadata": {"sha256": hashlib.sha256(content).hexdigest()}}
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def put_object(self, **kwargs: object) -> dict[str, object]:
            uploads.append(kwargs["Key"])
            self.store[kwargs["Key"]] = kwargs["Body"]
            return {}

        def get_object(self, **kwargs: object) -> dict[str, object]:
            return {"Body": io.BytesIO(self.store[kwargs["Key"]])}

    client = ReusingClient()
    monkeypatch.setattr(research_storage, "_r2_client", lambda: client)

    publication = research_storage.publish_immutable_r2_objects(
        (("stored", content), ("fresh", b"new bytes"))
    )

    assert publication.uploaded_objects == 1
    assert publication.reused_objects == 1
    assert uploads == ["fresh"]


def test_immutable_r2_objects_preserve_integrity_failures(monkeypatch) -> None:
    class ConflictingClient:
        def head_object(self, **_kwargs) -> dict[str, object]:
            return {"Metadata": {"sha256": "different"}}

        def get_object(self, **_kwargs) -> dict[str, io.BytesIO]:
            return {"Body": io.BytesIO(b"different")}

    monkeypatch.setattr(research_storage, "_r2_client", ConflictingClient)

    with pytest.raises(research_storage.ResearchObjectIntegrityError, match="different content"):
        research_storage.publish_immutable_r2_objects((("key", b"expected"),))


def test_r2_access_check_reads_at_most_one_key(monkeypatch) -> None:
    calls = []

    class Client:
        def list_objects_v2(self, **kwargs) -> dict[str, object]:
            calls.append(kwargs)
            return {}

    monkeypatch.setattr(research_storage, "_r2_client", Client)

    research_storage.check_r2_access()

    assert calls == [{"Bucket": research_storage.NEWS_R2_BUCKET, "MaxKeys": 1}]


def test_r2_access_check_wraps_provider_failures(monkeypatch) -> None:
    class Client:
        def list_objects_v2(self, **_kwargs) -> dict[str, object]:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")

    monkeypatch.setattr(research_storage, "_r2_client", Client)

    with pytest.raises(research_storage.ResearchObjectUnavailable, match="availability check"):
        research_storage.check_r2_access()


class _R2Client:
    def __init__(self, barrier: threading.Barrier) -> None:
        self.barrier = barrier
        self.values: dict[str, tuple[bytes, dict[str, str]]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.barrier.wait(timeout=1)
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str],
    ) -> None:
        self.values[Key] = Body, Metadata

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        content, _metadata = self.values[Key]
        return {"Body": io.BytesIO(content)}
