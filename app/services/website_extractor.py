"""Fetch a website safely and turn its HTML into clean text for the LLM.

Fetching rules (why each exists):
- Redirects are followed by hand, and the SSRF check runs before EVERY hop -
  otherwise a public site could 302 us to http://169.254.169.254/.
- Downloads stop at `max_html_bytes` so a huge page can't exhaust memory.
- Each network operation has a timeout, and the whole fetch has a total
  deadline, so a server that drips bytes forever can't hold a worker.
"""

import asyncio
import codecs
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.config import Settings
from app.errors import (
    InvalidURLError,
    SiteTimeoutError,
    SiteUnreachableError,
    UnsupportedContentError,
)
from app.utils.validators import ensure_public_url, normalize_url

REDIRECT_STATUSES = {301, 302, 303, 307, 308}
BOT_BLOCK_STATUSES = {401, 403, 429}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


@dataclass
class FetchedPage:
    url: str  # final URL after redirects
    status: int
    html: str


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=False,  # we follow them ourselves, with an SSRF check per hop
        timeout=settings.fetch_timeout_s,
        trust_env=False,  # ignore proxy env vars and ~/.netrc credentials
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


async def fetch_page(url: str, client: httpx.AsyncClient, settings: Settings) -> FetchedPage:
    """Download one HTML page, following redirects safely."""
    deadline = settings.fetch_timeout_s * 2
    try:
        async with asyncio.timeout(deadline):
            return await _fetch_following_redirects(url, client, settings)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise SiteTimeoutError("The website took too long to respond.") from exc
    except httpx.HTTPError as exc:
        raise SiteUnreachableError("Could not connect to the website.") from exc


async def _fetch_following_redirects(url: str, client: httpx.AsyncClient, settings: Settings) -> FetchedPage:
    current = url
    for _ in range(settings.max_redirects + 1):
        await ensure_public_url(current)
        async with client.stream("GET", current) as response:
            if response.status_code in REDIRECT_STATUSES:
                current = _next_hop(current, response.headers.get("location"))
                continue
            return await _read_final_response(current, response, settings)
    raise SiteUnreachableError("Too many redirects.")


def _next_hop(current: str, location: str | None) -> str:
    if not location:
        raise SiteUnreachableError("The website sent a redirect without a destination.")
    try:
        return normalize_url(urljoin(current, location))
    except InvalidURLError as exc:
        # The user's URL was fine; the *site* sent us somewhere we won't go.
        raise SiteUnreachableError("The website redirected to an unsupported address.") from exc


async def _read_final_response(url: str, response: httpx.Response, settings: Settings) -> FetchedPage:
    status = response.status_code
    if status >= 400:
        hint = " - the site blocks automated access" if status in BOT_BLOCK_STATUSES else ""
        raise SiteUnreachableError(f"Website returned HTTP {status}{hint}.")

    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type:
        raise UnsupportedContentError(
            f"The URL returned '{content_type.split(';')[0]}', not a web page."
        )

    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) >= settings.max_html_bytes:
            del body[settings.max_html_bytes:]
            break

    return FetchedPage(url=url, status=status, html=_decode(bytes(body), response.charset_encoding))


def _decode(body: bytes, charset: str | None) -> str:
    encoding = "utf-8"
    if charset:
        try:
            encoding = codecs.lookup(charset).name
        except LookupError:
            pass
    # "ignore" (not "replace") so a multi-byte char cut by the size cap can't grow the text.
    return body.decode(encoding, errors="ignore")
