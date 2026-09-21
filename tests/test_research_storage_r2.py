import hashlib
import io
import threading
from typing import cast

import pytest
from botocore.client import BaseClient
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


def test_immutable_r2_objects_preserve_integrity_failures(monkeypatch) -> None:
    class ConflictingClient:
        def head_object(self, **_kwargs) -> dict[str, object]:
            return {"Metadata": {"sha256": "different"}}

        def get_object(self, **_kwargs) -> dict[str, io.BytesIO]:
            return {"Body": io.BytesIO(b"different")}

    monkeypatch.setattr(research_storage, "_r2_client", ConflictingClient)

    with pytest.raises(research_storage.ResearchObjectIntegrityError, match="different content"):
        research_storage.publish_immutable_r2_objects((("key", b"expected"),))


def test_immutable_r2_objects_hash_existing_body_despite_matching_metadata() -> None:
    class LyingClient:
        def head_object(self, **_kwargs) -> dict[str, object]:
            return {
                "Metadata": {
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                }
            }

        def get_object(self, **_kwargs) -> dict[str, io.BytesIO]:
            return {"Body": io.BytesIO(b"different")}

    with pytest.raises(research_storage.ResearchObjectIntegrityError, match="different content"):
        research_storage.publish_immutable_r2_objects(
            (("key", b"expected"),),
            client=cast(BaseClient, cast(object, LyingClient())),
            bucket="private-media",
        )


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


def test_verified_read_wraps_client_errors_and_uses_explicit_bucket() -> None:
    class Client:
        def get_object(self, **kwargs) -> dict[str, object]:
            assert kwargs["Bucket"] == "private-media"
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    with pytest.raises(research_storage.ResearchObjectUnavailable, match="unavailable"):
        research_storage.read_verified_r2_object(
            "private/video.mp4",
            "0" * 64,
            client=cast(BaseClient, cast(object, Client())),
            bucket="private-media",
        )


def test_immutable_publication_wraps_client_errors_with_explicit_bucket() -> None:
    class Client:
        def head_object(self, **kwargs) -> dict[str, object]:
            assert kwargs["Bucket"] == "private-media"
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")

    with pytest.raises(research_storage.ResearchObjectUnavailable, match="publication failed"):
        research_storage.publish_immutable_r2_objects(
            (("evidence.json", b"evidence"),),
            client=cast(BaseClient, cast(object, Client())),
            bucket="private-media",
        )


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
