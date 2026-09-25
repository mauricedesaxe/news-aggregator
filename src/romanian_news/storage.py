from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import requests
from botocore.exceptions import BotoCoreError as _BotoCoreError
from botocore.exceptions import ClientError as _ClientError
from pydantic import Field, StringConstraints

if TYPE_CHECKING:
    from botocore.client import BaseClient

from romanian_news import NewsModel, Sha256
from romanian_news.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    NEWS_R2_BUCKET,
)

_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True)
class ImmutableR2Publication:
    bucket: str
    uploaded_objects: int
    reused_objects: int


@dataclass(frozen=True)
class R2S3Config:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str = field(repr=False)


class ResearchObjectUnavailable(RuntimeError):
    pass


class ResearchObjectIntegrityError(RuntimeError):
    pass


class PublicObjectConflict(RuntimeError):
    pass


class PublicObjectUnavailable(RuntimeError):
    pass


class PublicR2Object(NewsModel):
    key: _NonEmpty
    content: bytes
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    content_type: Literal["video/mp4", "text/vtt"]
    cache_control: Literal["public,max-age=31536000,immutable"]
    visibility: Literal["public"]
    retention: Literal["permanent"]
    source_lineage: Sha256


class _R2Head(NewsModel):
    ContentLength: Annotated[int, Field(gt=0)]
    ContentType: _NonEmpty
    CacheControl: _NonEmpty
    Metadata: dict[str, str]
    ETag: _NonEmpty


class PublicObjectVerification(NewsModel):
    key: _NonEmpty
    content_digest: Sha256
    byte_size: Annotated[int, Field(gt=0)]
    content_type: Literal["video/mp4", "text/vtt"]
    cache_control: Literal["public,max-age=31536000,immutable"]
    visibility: Literal["public"]
    retention: Literal["permanent"]
    source_lineage: Sha256
    validator: _NonEmpty


def check_r2_access() -> None:
    """Verify that the configured R2 bucket accepts authenticated reads."""
    try:
        _r2_client().list_objects_v2(Bucket=NEWS_R2_BUCKET, MaxKeys=1)
    except (_BotoCoreError, _ClientError, requests.RequestException, RuntimeError) as error:
        raise ResearchObjectUnavailable("R2 availability check failed") from error


def read_verified_r2_object(
    key: str,
    digest: str,
    *,
    client: BaseClient | None = None,
    bucket: str | None = None,
) -> bytes:
    """Read one R2 object and verify its catalog digest."""
    try:
        content = (
            (client or _r2_client())
            .get_object(
                Bucket=bucket or NEWS_R2_BUCKET,
                Key=key,
            )["Body"]
            .read()
        )
    except (_BotoCoreError, _ClientError, requests.RequestException, RuntimeError) as error:
        raise ResearchObjectUnavailable(f"R2 object unavailable: {key}") from error
    if _sha256(content) != digest:
        raise ResearchObjectIntegrityError(f"R2 verification failed for {key}")
    return content


def download_verified_r2_object(key: str, digest: str, path: Path) -> None:
    """Download one verified R2 object with an atomic local replacement."""
    content = read_verified_r2_object(key, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def presigned_r2_url(key: str, *, expires_in: int = 3600) -> str:
    if expires_in <= 0:
        raise ValueError("expires_in must be positive")
    try:
        return _r2_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": NEWS_R2_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except (_BotoCoreError, _ClientError, RuntimeError) as error:
        raise ResearchObjectUnavailable(f"Could not sign R2 object: {key}") from error


def publish_immutable_r2_objects(
    objects: Iterable[tuple[str, bytes]],
    *,
    client: BaseClient | None = None,
    bucket: str | None = None,
) -> ImmutableR2Publication:
    """Publish immutable objects and verify every stored byte sequence."""
    values = dict(objects)
    if not values:
        return ImmutableR2Publication(
            bucket=bucket or NEWS_R2_BUCKET,
            uploaded_objects=0,
            reused_objects=0,
        )
    try:
        target_client = client or _r2_client()
        target_bucket = bucket or NEWS_R2_BUCKET
        with ThreadPoolExecutor(max_workers=min(16, len(values))) as executor:
            states = tuple(
                executor.map(
                    lambda value: _publish_immutable_r2_object(
                        target_client,
                        target_bucket,
                        *value,
                    ),
                    values.items(),
                )
            )
    except ResearchObjectIntegrityError:
        raise
    except (_BotoCoreError, _ClientError, requests.RequestException, RuntimeError) as error:
        raise ResearchObjectUnavailable("R2 publication failed") from error
    return ImmutableR2Publication(
        bucket=bucket or NEWS_R2_BUCKET,
        uploaded_objects=states.count("uploaded"),
        reused_objects=states.count("reused"),
    )


def publish_public_r2_object(
    client: BaseClient,
    bucket: str,
    value: PublicR2Object,
) -> None:
    if not bucket.strip():
        raise ValueError("Public R2 bucket is required")
    _validate_local_public_object(value)
    try:
        head = _head_r2_object(client, bucket, value.key)
        if head is not None:
            _require_matching_public_object(client, bucket, value, head)
            return
        try:
            client.put_object(
                Bucket=bucket,
                Key=value.key,
                Body=value.content,
                ContentType=value.content_type,
                CacheControl=value.cache_control,
                Metadata=_public_metadata(value),
                IfNoneMatch="*",
            )
        except _ClientError as error:
            if error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            ) != 412 or error.response.get("Error", {}).get("Code") not in {
                "412",
                "PreconditionFailed",
            }:
                raise
        head = _head_r2_object(client, bucket, value.key)
        if head is None:
            raise PublicObjectUnavailable(f"Public R2 object disappeared after write: {value.key}")
        _require_matching_public_object(client, bucket, value, head)
    except PublicObjectConflict:
        raise
    except ValueError as error:
        raise PublicObjectConflict(f"Public R2 metadata is invalid: {value.key}") from error
    except (_BotoCoreError, _ClientError, RuntimeError) as error:
        raise PublicObjectUnavailable(f"Public R2 operation failed: {value.key}") from error


def publish_private_video_object(
    key: str,
    content: bytes,
    *,
    retention: Literal["candidate-7d", "permanent"],
    source_lineage: str,
) -> None:
    publish_private_reference_media_object(
        key,
        content,
        content_type="video/mp4",
        retention=retention,
        source_lineage=source_lineage,
    )


def publish_private_reference_media_object(
    key: str,
    content: bytes,
    *,
    content_type: Literal["video/mp4", "audio/wav", "audio/mpeg"],
    retention: Literal["candidate-7d", "permanent"],
    source_lineage: str,
) -> None:
    if not source_lineage.strip():
        raise ValueError("Private media source lineage is required")
    digest = _sha256(content)
    metadata = {
        "sha256": digest,
        "visibility": "private",
        "retention": retention,
        "source-lineage": source_lineage,
    }
    try:
        client = _r2_client()
        head = _head_r2_object(client, NEWS_R2_BUCKET, key)
        if head is not None:
            _require_matching_private_video_object(
                client, key, content, metadata, head, content_type
            )
            return
        try:
            client.put_object(
                Bucket=NEWS_R2_BUCKET,
                Key=key,
                Body=content,
                ContentType=content_type,
                CacheControl="private,no-store",
                Metadata=metadata,
                IfNoneMatch="*",
            )
        except _ClientError as error:
            if error.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            ) != 412 or error.response.get("Error", {}).get("Code") not in {
                "412",
                "PreconditionFailed",
            }:
                raise
        head = _head_r2_object(client, NEWS_R2_BUCKET, key)
        if head is None:
            raise ResearchObjectUnavailable(f"Private R2 object disappeared after write: {key}")
        _require_matching_private_video_object(client, key, content, metadata, head, content_type)
    except ResearchObjectIntegrityError:
        raise
    except ValueError as error:
        raise ResearchObjectIntegrityError(
            f"R2 object already exists with invalid classification: {key}"
        ) from error
    except (_BotoCoreError, _ClientError, RuntimeError) as error:
        raise ResearchObjectUnavailable(f"R2 classified publication failed: {key}") from error


def verify_public_object_url(
    url: str,
    expected: PublicR2Object,
    *,
    session: requests.Session | None = None,
) -> PublicObjectVerification:
    requester = session or requests.Session()
    try:
        response = requester.get(url, allow_redirects=False, stream=True, timeout=30)
        try:
            _require_http_status(response, 200, url)
            validator = _required_header(response, "ETag")
            _require_public_http_headers(response, expected, content_length=expected.byte_size)
            digest = hashlib.sha256()
            byte_size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not isinstance(chunk, bytes):
                    raise PublicObjectUnavailable("Public media response returned a non-byte body")
                digest.update(chunk)
                byte_size += len(chunk)
            if byte_size != expected.byte_size or digest.hexdigest() != expected.content_digest:
                raise PublicObjectConflict(f"Public media body differs from intent: {url}")
        finally:
            response.close()
        _verify_range(requester, url, expected, offset=0, validator=validator)
        _verify_range(
            requester,
            url,
            expected,
            offset=expected.byte_size - 1,
            validator=validator,
        )
        return PublicObjectVerification(
            key=expected.key,
            content_digest=expected.content_digest,
            byte_size=expected.byte_size,
            content_type=expected.content_type,
            cache_control=expected.cache_control,
            visibility=expected.visibility,
            retention=expected.retention,
            source_lineage=expected.source_lineage,
            validator=validator,
        )
    except PublicObjectConflict:
        raise
    except (requests.RequestException, RuntimeError, ValueError) as error:
        raise PublicObjectUnavailable(f"Public media verification failed: {url}") from error


def _validate_local_public_object(value: PublicR2Object) -> None:
    if len(value.content) != value.byte_size or _sha256(value.content) != value.content_digest:
        raise PublicObjectConflict(f"Local public object differs from intent: {value.key}")


def _head_r2_object(client: BaseClient, bucket: str, key: str) -> _R2Head | None:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except _ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return _R2Head.model_validate(
        {
            "ContentLength": response.get("ContentLength"),
            "ContentType": response.get("ContentType"),
            "CacheControl": response.get("CacheControl"),
            "Metadata": response.get("Metadata"),
            "ETag": response.get("ETag"),
        },
        strict=True,
    )


def _read_r2_body(client: BaseClient, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise PublicObjectUnavailable(f"R2 object body is unavailable: {key}")
    content = body.read()
    if not isinstance(content, bytes):
        raise PublicObjectUnavailable(f"R2 object body is not bytes: {key}")
    return content


def _public_metadata(value: PublicR2Object) -> dict[str, str]:
    return {
        "sha256": value.content_digest,
        "visibility": value.visibility,
        "retention": value.retention,
        "source-lineage": value.source_lineage,
    }


def _require_matching_private_video_object(
    client: BaseClient,
    key: str,
    content: bytes,
    metadata: dict[str, str],
    head: _R2Head,
    content_type: str,
) -> None:
    remote = _read_r2_body(client, NEWS_R2_BUCKET, key)
    actual = (
        head.ContentLength,
        head.ContentType,
        head.CacheControl,
        head.Metadata,
        _sha256(remote),
    )
    expected = (
        len(content),
        content_type,
        "private,no-store",
        metadata,
        _sha256(content),
    )
    if actual != expected:
        raise ResearchObjectIntegrityError(
            f"R2 object already exists with different classification: {key}"
        )


def _require_matching_public_object(
    client: BaseClient,
    bucket: str,
    value: PublicR2Object,
    head: _R2Head,
) -> None:
    remote = _read_r2_body(client, bucket, value.key)
    actual = (
        _sha256(remote),
        head.ContentLength,
        head.ContentType,
        head.CacheControl,
        head.Metadata,
    )
    expected = (
        value.content_digest,
        value.byte_size,
        value.content_type,
        value.cache_control,
        _public_metadata(value),
    )
    if actual != expected:
        raise PublicObjectConflict(f"Public R2 object conflicts with intent: {value.key}")


def _required_header(response: requests.Response, name: str) -> str:
    value = response.headers.get(name)
    if value is None or not value.strip():
        raise PublicObjectConflict(f"Public media response is missing {name}")
    return value


def _require_http_status(response: requests.Response, status: int, url: str) -> None:
    if response.status_code != status:
        if response.status_code in {404, 408, 425, 429} or response.status_code >= 500:
            raise PublicObjectUnavailable(
                f"Public media is temporarily unavailable with HTTP {response.status_code}: {url}"
            )
        raise PublicObjectConflict(
            f"Public media returned HTTP {response.status_code}, expected {status}: {url}"
        )


def _require_public_http_headers(
    response: requests.Response,
    expected: PublicR2Object,
    *,
    content_length: int,
) -> None:
    try:
        actual_length = int(_required_header(response, "Content-Length"))
    except ValueError as error:
        raise PublicObjectConflict("Public media Content-Length is invalid") from error
    cache_control = tuple(
        directive.strip() for directive in _required_header(response, "Cache-Control").split(",")
    )
    if (
        actual_length,
        _required_header(response, "Content-Type"),
        cache_control,
    ) != (content_length, expected.content_type, tuple(expected.cache_control.split(","))):
        raise PublicObjectConflict("Public media response headers differ from intent")


def _verify_range(
    requester: requests.Session,
    url: str,
    expected: PublicR2Object,
    *,
    offset: int,
    validator: str,
) -> None:
    response = requester.get(
        url,
        headers={"Range": f"bytes={offset}-{offset}"},
        allow_redirects=False,
        stream=True,
        timeout=30,
    )
    try:
        _require_http_status(response, 206, url)
        _require_public_http_headers(response, expected, content_length=1)
        if (
            _required_header(response, "Content-Range")
            != f"bytes {offset}-{offset}/{expected.byte_size}"
        ):
            raise PublicObjectConflict("Public media Content-Range differs from request")
        if _required_header(response, "ETag") != validator:
            raise PublicObjectConflict("Public media validator changed between probes")
        if response.content != expected.content[offset : offset + 1]:
            raise PublicObjectConflict("Public media range body differs from intent")
    finally:
        response.close()


def _publish_immutable_r2_object(
    client: BaseClient,
    bucket: str,
    key: str,
    content: bytes,
) -> str:
    from botocore.exceptions import ClientError

    digest = _sha256(content)
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
    else:
        remote = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if _sha256(remote) != digest:
            raise ResearchObjectIntegrityError(
                f"R2 object already exists with different content: {key}"
            )
        return "reused"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content,
        Metadata={"sha256": digest},
    )
    remote = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if _sha256(remote) != digest:
        raise ResearchObjectIntegrityError(f"R2 verification failed for {key}")
    return "uploaded"


def r2_s3_config() -> R2S3Config:
    """Derive S3 credentials from the configured Cloudflare API token."""
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN or not NEWS_R2_BUCKET:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, and NEWS_R2_BUCKET are required"
        )
    return R2S3Config(
        endpoint_url=f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com",
        access_key_id=_cloudflare_token_id(),
        secret_access_key=_sha256(CLOUDFLARE_API_TOKEN.encode()),
    )


def r2_client() -> BaseClient:
    return _r2_client()


@cache
def _r2_client() -> BaseClient:
    import boto3

    config = r2_s3_config()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )


@cache
def _cloudflare_token_id() -> str:
    if not CLOUDFLARE_API_TOKEN:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is required")
    response = requests.get(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers=_cloudflare_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["result"]["id"]


def _cloudflare_headers() -> dict[str, str]:
    if not CLOUDFLARE_API_TOKEN:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is required")
    return {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
