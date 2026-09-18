import ipaddress
import socket
from collections.abc import Mapping
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5


class UnsafeNewsRedirect(requests.RequestException):
    """A news endpoint redirected to a non-public network destination."""


def create_news_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=1,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    return session


def get_with_validated_redirects(
    session: requests.Session,
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: int | tuple[int, int],
    stream: bool = False,
) -> requests.Response:
    """GET a trusted source URL after validating every redirect destination."""
    current_url = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        response = session.get(
            current_url,
            headers=headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )
        if getattr(response, "status_code", 200) not in _REDIRECT_STATUSES:
            return response
        location = response.headers.get("Location")
        if not location:
            response.close()
            raise UnsafeNewsRedirect("News redirect response has no Location header")
        if redirect_count == _MAX_REDIRECTS:
            response.close()
            raise requests.TooManyRedirects(f"Exceeded {_MAX_REDIRECTS} redirects")
        next_url = urljoin(current_url, location)
        try:
            _validate_public_destination(next_url)
        except UnsafeNewsRedirect:
            response.close()
            raise
        response.close()
        current_url = next_url
    raise AssertionError("redirect loop terminated without returning")


def _validate_public_destination(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise UnsafeNewsRedirect("News redirect must use HTTP or HTTPS")
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(
                parts.hostname,
                parts.port or (443 if parts.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as error:
        raise UnsafeNewsRedirect("News redirect hostname could not be resolved") from error
    if not addresses:
        raise UnsafeNewsRedirect("News redirect hostname returned no addresses")
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise UnsafeNewsRedirect("News redirect resolved to a non-public address")
