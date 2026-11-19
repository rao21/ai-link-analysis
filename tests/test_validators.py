import pytest

from app.errors import BlockedAddressError, InvalidURLError, SiteUnreachableError
from app.utils.validators import ensure_public_url, is_public_ip, normalize_url, resolve_host


# ---------- normalize_url ----------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("example.com", "https://example.com"),
        ("  HTTPS://Example.COM/about#team ", "https://example.com/about"),
        ("http://example.com:80/x", "http://example.com:80/x"),
        ("https://example.com/search?q=web+design", "https://example.com/search?q=web+design"),
    ],
)
def test_normalize_ok(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not a url",
        "ftp://example.com",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "http://user:pass@example.com",
        "http://example.com:22",
        "http://example.com:abc",
        "https://",
        "http://" + "a" * 2050 + ".com",
    ],
)
def test_normalize_rejects(raw):
    with pytest.raises(InvalidURLError):
        normalize_url(raw)


# ---------- is_public_ip ----------

@pytest.mark.parametrize(
    "ip,public",
    [
        ("8.8.8.8", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),   # cloud metadata endpoint
        ("0.0.0.0", False),
        ("100.64.0.1", False),        # carrier-grade NAT
        ("224.0.0.1", False),         # multicast
        ("::1", False),
        ("fd00::1", False),           # IPv6 private
        ("fe80::1", False),           # IPv6 link-local
        ("::ffff:127.0.0.1", False),  # IPv4-mapped IPv6 loopback
        ("2002:7f00:1::", False),     # 6to4 wrapping 127.0.0.1
        ("2002:a00:1::", False),      # 6to4 wrapping 10.0.0.1
        ("fe80::1%en0", False),       # link-local with zone id
        ("not-an-ip", False),
    ],
)
def test_is_public_ip(ip, public):
    assert is_public_ip(ip) is public


# ---------- ensure_public_url (fake DNS) ----------

async def test_ensure_public_blocks_localhost_name(fake_resolver):
    with pytest.raises(BlockedAddressError):
        await ensure_public_url("http://localhost/")


async def test_ensure_public_blocks_if_any_ip_private(fake_resolver):
    with pytest.raises(BlockedAddressError):
        await ensure_public_url("https://mixed.test/")


async def test_ensure_public_blocks_ip_literal():
    with pytest.raises(BlockedAddressError):
        await ensure_public_url("http://169.254.169.254/latest/meta-data")


async def test_ensure_public_blocks_ipv6_literal():
    with pytest.raises(BlockedAddressError):
        await ensure_public_url("http://[::1]/")


async def test_ensure_public_allows_public(fake_resolver):
    await ensure_public_url("https://example.com/")


# ---------- real resolver: numeric tricks are parsed locally, no network needed ----------

@pytest.mark.parametrize("url", ["http://2130706433/", "http://0x7f.0.0.1/", "http://127.1/"])
async def test_obfuscated_loopback_blocked(url):
    with pytest.raises(BlockedAddressError):
        await ensure_public_url(url)


async def test_resolve_unknown_domain_raises():
    with pytest.raises(SiteUnreachableError):
        await resolve_host("this-domain-does-not-exist.invalid")
