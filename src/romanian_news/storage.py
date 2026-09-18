from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from botocore.exceptions import BotoCoreError as _BotoCoreError
from botocore.exceptions import ClientError as _ClientError

if TYPE_CHECKING:
    from botocore.client import BaseClient

from romanian_news.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    NEWS_R2_BUCKET,
)


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


def check_r2_access() -> None:
    """Verify that the configured R2 bucket accepts authenticated reads."""
    try:
        _r2_client().list_objects_v2(Bucket=NEWS_R2_BUCKET, MaxKeys=1)
    except (_BotoCoreError, _ClientError, requests.RequestException, RuntimeError) as error:
        raise ResearchObjectUnavailable("R2 availability check failed") from error


def read_verified_r2_object(key: str, digest: str) -> bytes:
    """Read one R2 object and verify its catalog digest."""
    try:
        content = _r2_client().get_object(Bucket=NEWS_R2_BUCKET, Key=key)["Body"].read()
    except (_BotoCoreError, requests.RequestException, RuntimeError) as error:
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


def publish_immutable_r2_objects(
    objects: Iterable[tuple[str, bytes]],
) -> ImmutableR2Publication:
    """Publish immutable objects and verify every stored byte sequence."""
    values = dict(objects)
    if not values:
        return ImmutableR2Publication(
            bucket=NEWS_R2_BUCKET,
            uploaded_objects=0,
            reused_objects=0,
        )
    try:
        client = _r2_client()
        with ThreadPoolExecutor(max_workers=min(16, len(values))) as executor:
            states = tuple(
                executor.map(
                    lambda value: _publish_immutable_r2_object(client, *value), values.items()
                )
            )
    except ResearchObjectIntegrityError:
        raise
    except (_BotoCoreError, requests.RequestException, RuntimeError) as error:
        raise ResearchObjectUnavailable("R2 publication failed") from error
    return ImmutableR2Publication(
        bucket=NEWS_R2_BUCKET,
        uploaded_objects=states.count("uploaded"),
        reused_objects=states.count("reused"),
    )


def _publish_immutable_r2_object(client: BaseClient, key: str, content: bytes) -> str:
    from botocore.exceptions import ClientError

    digest = _sha256(content)
    try:
        metadata = client.head_object(Bucket=NEWS_R2_BUCKET, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
    else:
        if metadata.get("Metadata", {}).get("sha256") == digest:
            return "reused"
        remote = client.get_object(Bucket=NEWS_R2_BUCKET, Key=key)["Body"].read()
        if _sha256(remote) != digest:
            raise ResearchObjectIntegrityError(
                f"R2 object already exists with different content: {key}"
            )
        return "reused"
    client.put_object(
        Bucket=NEWS_R2_BUCKET,
        Key=key,
        Body=content,
        Metadata={"sha256": digest},
    )
    remote = client.get_object(Bucket=NEWS_R2_BUCKET, Key=key)["Body"].read()
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
