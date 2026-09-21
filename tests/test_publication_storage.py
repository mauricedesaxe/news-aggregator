import hashlib
import io
from collections.abc import Callable
from typing import Literal, cast

import pytest
import requests
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from romanian_news import storage
from romanian_news.storage import (
    PublicObjectConflict,
    PublicObjectUnavailable,
    PublicR2Object,
    publish_public_r2_object,
    verify_public_object_url,
)

CONTENT = b"video-bytes"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
SOURCE = "a" * 64


def _value() -> PublicR2Object:
    return PublicR2Object(
        key=f"video-digests/{'b' * 64}/{DIGEST}.mp4",
        content=CONTENT,
        content_digest=DIGEST,
        byte_size=len(CONTENT),
        content_type="video/mp4",
        cache_control="public,max-age=31536000,immutable",
        visibility="public",
        retention="permanent",
        source_lineage=SOURCE,
    )


def _head(value: PublicR2Object) -> dict[str, object]:
    return {
        "ContentLength": value.byte_size,
        "ContentType": value.content_type,
        "CacheControl": value.cache_control,
        "Metadata": {
            "sha256": value.content_digest,
            "visibility": value.visibility,
            "retention": value.retention,
            "source-lineage": value.source_lineage,
        },
        "ETag": '"validator"',
    }


class _Client:
    def __init__(self, value: PublicR2Object, *, exists: bool, race: bool = False) -> None:
        self.value = value
        self.exists = exists
        self.race = race
        self.content = value.content
        self.head = _head(value)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head", kwargs))
        if not self.exists:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return self.head

    def get_object(self, **kwargs: object) -> dict[str, io.BytesIO]:
        self.calls.append(("get", kwargs))
        return {"Body": io.BytesIO(self.content)}

    def put_object(self, **kwargs: object) -> None:
        self.calls.append(("put", kwargs))
        self.exists = True
        body = kwargs["Body"]
        metadata = kwargs["Metadata"]
        assert isinstance(body, bytes)
        assert isinstance(metadata, dict)
        self.content = body
        self.head = {
            "ContentLength": len(body),
            "ContentType": kwargs["ContentType"],
            "CacheControl": kwargs["CacheControl"],
            "Metadata": metadata,
            "ETag": '"validator"',
        }
        if self.race:
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 412},
                },
                "PutObject",
            )


def test_missing_public_object_writes_exact_policy_with_precondition() -> None:
    value = _value()
    client = _Client(value, exists=False)

    publish_public_r2_object(cast(BaseClient, cast(object, client)), "public-media", value)

    put = next(kwargs for operation, kwargs in client.calls if operation == "put")
    assert put == {
        "Bucket": "public-media",
        "Key": value.key,
        "Body": CONTENT,
        "ContentType": "video/mp4",
        "CacheControl": "public,max-age=31536000,immutable",
        "Metadata": {
            "sha256": DIGEST,
            "visibility": "public",
            "retention": "permanent",
            "source-lineage": SOURCE,
        },
        "IfNoneMatch": "*",
    }
    assert [operation for operation, _kwargs in client.calls] == ["head", "put", "head", "get"]


def test_matching_public_object_is_adopted_only_after_get_and_hash() -> None:
    value = _value()
    client = _Client(value, exists=True)

    publish_public_r2_object(cast(BaseClient, cast(object, client)), "public-media", value)

    assert [operation for operation, _kwargs in client.calls] == ["head", "get"]


def test_precondition_race_reinspects_and_adopts_matching_object() -> None:
    value = _value()
    client = _Client(value, exists=False, race=True)

    publish_public_r2_object(cast(BaseClient, cast(object, client)), "public-media", value)

    assert [operation for operation, _kwargs in client.calls] == ["head", "put", "head", "get"]


def test_non_412_precondition_error_is_unavailable() -> None:
    value = _value()

    class WrongStatusClient(_Client):
        def put_object(self, **kwargs: object) -> None:
            self.calls.append(("put", kwargs))
            self.exists = True
            raise ClientError(
                {
                    "Error": {"Code": "PreconditionFailed"},
                    "ResponseMetadata": {"HTTPStatusCode": 409},
                },
                "PutObject",
            )

    client = WrongStatusClient(value, exists=False)

    with pytest.raises(PublicObjectUnavailable):
        publish_public_r2_object(cast(BaseClient, cast(object, client)), "public-media", value)


@pytest.mark.parametrize("retention", ("candidate-7d", "permanent"))
def test_private_video_classification_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    retention: Literal["candidate-7d", "permanent"],
) -> None:
    value = _value()
    client = _Client(value, exists=False)
    monkeypatch.setattr(storage, "_r2_client", lambda: client)
    monkeypatch.setattr(storage, "NEWS_R2_BUCKET", "private-media")

    storage.publish_private_video_object(
        "news/video-digest/candidates/7d/request/video.mp4",
        CONTENT,
        retention=retention,
        source_lineage="request-id",
    )

    put = next(kwargs for operation, kwargs in client.calls if operation == "put")
    assert put["Bucket"] == "private-media"
    assert put["CacheControl"] == "private,no-store"
    assert put["Metadata"] == {
        "sha256": DIGEST,
        "visibility": "private",
        "retention": retention,
        "source-lineage": "request-id",
    }


def test_storage_exposes_no_deletion_api() -> None:
    assert not any(name.startswith("delete") for name in vars(storage))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda client: setattr(client, "content", b"wrong"),
        lambda client: client.head.__setitem__("ContentLength", len(CONTENT) + 1),
        lambda client: client.head.pop("ContentType"),
        lambda client: client.head.__setitem__("ContentType", "video/webm"),
        lambda client: client.head.__setitem__("CacheControl", "public,max-age=60"),
        lambda client: client.head["Metadata"].__setitem__("visibility", "private"),
        lambda client: client.head["Metadata"].__setitem__("retention", "candidate-7d"),
        lambda client: client.head["Metadata"].__setitem__("source-lineage", "c" * 64),
        lambda client: client.head["Metadata"].__setitem__("sha256", "d" * 64),
        lambda client: client.head["Metadata"].__setitem__("unexpected", "value"),
    ),
    ids=(
        "checksum",
        "size",
        "missing-header",
        "type",
        "cache",
        "visibility",
        "retention",
        "lineage",
        "digest",
        "unexpected-metadata",
    ),
)
def test_public_object_mismatch_never_overwrites(
    mutate: Callable[[_Client], None],
) -> None:
    value = _value()
    client = _Client(value, exists=True)
    mutate(client)

    with pytest.raises(PublicObjectConflict):
        publish_public_r2_object(cast(BaseClient, cast(object, client)), "public-media", value)

    assert "put" not in [operation for operation, _kwargs in client.calls]


class _Response:
    def __init__(
        self,
        status_code: int,
        content: bytes,
        headers: dict[str, str],
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers
        self.closed = False

    def iter_content(self, *, chunk_size: int) -> tuple[bytes, ...]:
        assert chunk_size == 64 * 1024
        return (self.content,)

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _full_response(value: PublicR2Object) -> _Response:
    return _Response(
        200,
        value.content,
        {
            "Content-Length": str(value.byte_size),
            "Content-Type": value.content_type,
            "Cache-Control": value.cache_control,
            "ETag": '"validator"',
        },
    )


def _range_response(value: PublicR2Object, offset: int) -> _Response:
    return _Response(
        206,
        value.content[offset : offset + 1],
        {
            "Content-Length": "1",
            "Content-Type": value.content_type,
            "Cache-Control": value.cache_control,
            "Content-Range": f"bytes {offset}-{offset}/{value.byte_size}",
            "ETag": '"validator"',
        },
    )


def test_public_verification_hashes_full_body_and_probes_first_and_last_bytes() -> None:
    value = _value()
    session = _Session(
        [
            _full_response(value),
            _range_response(value, 0),
            _range_response(value, value.byte_size - 1),
        ]
    )

    result = verify_public_object_url(
        "https://media.example.com/video.mp4",
        value,
        session=cast(requests.Session, cast(object, session)),
    )

    assert result.content_digest == value.content_digest
    assert [call[1].get("headers") for call in session.calls] == [
        None,
        {"Range": "bytes=0-0"},
        {"Range": f"bytes={value.byte_size - 1}-{value.byte_size - 1}"},
    ]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    assert session.responses == []


@pytest.mark.parametrize(
    "mutate",
    (
        lambda response: setattr(response, "status_code", 302),
        lambda response: setattr(response, "status_code", 206),
        lambda response: response.headers.__setitem__("Content-Length", "1"),
        lambda response: response.headers.__setitem__("Content-Type", "video/webm"),
        lambda response: response.headers.__setitem__("Cache-Control", "public,max-age=60"),
        lambda response: response.headers.pop("ETag"),
        lambda response: setattr(response, "content", b"wrong-body"),
    ),
    ids=("redirect", "wrong-status", "length", "type", "cache", "validator", "body"),
)
def test_public_full_verification_rejects_wrong_response(
    mutate: Callable[[_Response], object],
) -> None:
    value = _value()
    response = _full_response(value)
    mutate(response)

    with pytest.raises(PublicObjectConflict):
        verify_public_object_url(
            "https://media.example.com/video.mp4",
            value,
            session=cast(requests.Session, cast(object, _Session([response]))),
        )


@pytest.mark.parametrize("status_code", (404, 429, 503))
def test_public_verification_retries_temporary_http_statuses(status_code: int) -> None:
    value = _value()
    response = _full_response(value)
    response.status_code = status_code

    with pytest.raises(PublicObjectUnavailable):
        verify_public_object_url(
            "https://media.example.com/video.mp4",
            value,
            session=cast(requests.Session, cast(object, _Session([response]))),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda response: setattr(response, "status_code", 200),
        lambda response: response.headers.__setitem__("Content-Range", "bytes 1-1/11"),
        lambda response: response.headers.__setitem__("ETag", '"changed"'),
        lambda response: setattr(response, "content", b"x"),
    ),
    ids=("status", "content-range", "validator", "body"),
)
def test_public_range_verification_rejects_wrong_response(
    mutate: Callable[[_Response], object],
) -> None:
    value = _value()
    first = _range_response(value, 0)
    mutate(first)

    with pytest.raises(PublicObjectConflict):
        verify_public_object_url(
            "https://media.example.com/video.mp4",
            value,
            session=cast(
                requests.Session,
                cast(object, _Session([_full_response(value), first])),
            ),
        )


def test_public_verification_rejects_wrong_last_byte() -> None:
    value = _value()
    last = _range_response(value, value.byte_size - 1)
    last.content = b"x"

    with pytest.raises(PublicObjectConflict):
        verify_public_object_url(
            "https://media.example.com/video.mp4",
            value,
            session=cast(
                requests.Session,
                cast(
                    object,
                    _Session([_full_response(value), _range_response(value, 0), last]),
                ),
            ),
        )
