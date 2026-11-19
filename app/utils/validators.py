"""URL validation and SSRF protection.

SSRF (Server-Side Request Forgery): our server fetches whatever URL a user sends.
Without a guard, someone could send `http://169.254.169.254/` (cloud metadata with
credentials), `http://localhost:6379/` (internal Redis) or `http://10.0.0.5/admin`
and make *our* server read internal resources for them.

Defence:
1. `normalize_url`   - only http/https, no credentials in the URL, only ports 80/443.
2. `ensure_public_url` - resolve the host and refuse if ANY resolved IP is not public.
   The fetcher calls this again for every redirect hop (Task 3).
"""

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedAddressError, InvalidURLError, SiteUnreachableError

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9_-]{1,63}\.)*[a-z0-9_-]{1,63}\.?$")


def normalize_url(raw: str) -> str:
    """Return a canonical http(s) URL or raise `InvalidURLError`."""
    url = (raw or "").strip()
    if not url:
        raise InvalidURLError("URL is required.")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError(f"URL is longer than {MAX_URL_LENGTH} characters.")
    if "://" not in url:
        url = "https://" + url  # users often type "example.com"

    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError for non-numeric / out-of-range ports
    except ValueError as exc:
        raise InvalidURLError("URL is not valid.") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError("Only http and https URLs are supported.")
    if parts.username is not None or parts.password is not None:
        raise InvalidURLError("URLs containing credentials are not allowed.")
    if port not in ALLOWED_PORTS:
        raise InvalidURLError("Only the default web ports (80 and 443) are allowed.")

    host = _canonical_host(parts.hostname)
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    # "https://a.com" and "https://a.com/" are the same page: one canonical form (also the cache key).
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _canonical_host(hostname: str | None) -> str:
    if not hostname:
        raise InvalidURLError("URL must contain a domain name.")
    host = hostname.lower()
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    try:
        host = host.encode("idna").decode("ascii")  # unicode domains -> punycode
    except UnicodeError as exc:
        raise InvalidURLError("URL contains an invalid domain name.") from exc
    if not _HOST_RE.match(host):
        raise InvalidURLError("URL contains an invalid domain name.")
    return host


def is_public_ip(ip: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        # IPv6 forms that embed an IPv4 address: judge the embedded one.
        # Python >= 3.12.4 already treats these as non-global; older versions got
        # IPv4-mapped addresses wrong, so this is defence in depth.
        embedded = addr.ipv4_mapped or addr.sixtofour or (addr.teredo[1] if addr.teredo else None)
        if embedded is not None and not is_public_ip(str(embedded)):
            return False
    return addr.is_global and not addr.is_multicast


async def resolve_host(host: str) -> list[str]:
    """Resolve a hostname to its unique IP addresses (IPv4 and IPv6)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise SiteUnreachableError("Domain could not be resolved.") from exc
    return list(dict.fromkeys(info[4][0] for info in infos))


async def ensure_public_url(url: str) -> None:
    """Raise `BlockedAddressError` unless every IP behind the URL's host is public."""
    host = urlsplit(url).hostname
    if not host:
        raise InvalidURLError("URL must contain a domain name.")
    try:
        ips = [str(ipaddress.ip_address(host))]  # IP literal: no DNS needed
    except ValueError:
        ips = await resolve_host(host)
    if not ips:
        raise SiteUnreachableError("Domain could not be resolved.")
    if not all(is_public_ip(ip) for ip in ips):
        raise BlockedAddressError("URL points to a private or internal network address.")
