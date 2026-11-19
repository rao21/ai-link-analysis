import asyncio

import httpx
import pytest

from app.config import Settings
from app.errors import (
    BlockedAddressError,
    SiteTimeoutError,
    SiteUnreachableError,
    UnsupportedContentError,
)
from app.services.website_extractor import build_http_client, fetch_page

HTML = {"content-type": "text/html; charset=utf-8"}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, max_html_bytes=2_000_000, max_redirects=5, fetch_timeout_s=5)


@pytest.fixture
async def client(settings):
    async with build_http_client(settings) as c:
        yield c


@pytest.fixture(autouse=True)
def _dns(fake_resolver):
    """Every test in this file uses fake DNS (*.test -> public IP)."""


async def test_fetch_ok(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(200, headers=HTML, text="<html>hi</html>")
    page = await fetch_page("https://site.test/", client, settings)
    assert page.status == 200
    assert "hi" in page.html
    assert page.url == "https://site.test/"


async def test_sends_browser_like_headers(respx_mock, client, settings):
    route = respx_mock.get("https://site.test/").respond(200, headers=HTML, text="ok")
    await fetch_page("https://site.test/", client, settings)
    sent = route.calls.last.request.headers
    assert "Mozilla" in sent["user-agent"]
    assert "text/html" in sent["accept"]


async def test_follows_redirect_and_reports_final_url(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(301, headers={"location": "https://www.site.test/"})
    respx_mock.get("https://www.site.test/").respond(200, headers=HTML, text="final")
    page = await fetch_page("https://site.test/", client, settings)
    assert page.url == "https://www.site.test/"
    assert "final" in page.html


async def test_relative_redirect(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(302, headers={"location": "/en/home"})
    respx_mock.get("https://site.test/en/home").respond(200, headers=HTML, text="en")
    page = await fetch_page("https://site.test/", client, settings)
    assert page.url == "https://site.test/en/home"


async def test_redirect_to_private_ip_blocked(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
    metadata_route = respx_mock.get("http://169.254.169.254/latest/meta-data").respond(200, text="SECRET")
    with pytest.raises(BlockedAddressError):
        await fetch_page("https://site.test/", client, settings)
    assert not metadata_route.called


async def test_redirect_to_localhost_name_blocked(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(302, headers={"location": "http://localhost/admin"})
    admin_route = respx_mock.get("http://localhost/admin").respond(200, text="ADMIN")
    with pytest.raises(BlockedAddressError):
        await fetch_page("https://site.test/", client, settings)
    assert not admin_route.called


async def test_redirect_to_unsupported_scheme(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(302, headers={"location": "ftp://files.site.test/"})
    with pytest.raises(SiteUnreachableError, match="redirected to an unsupported"):
        await fetch_page("https://site.test/", client, settings)


async def test_redirect_without_location(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(302)
    with pytest.raises(SiteUnreachableError):
        await fetch_page("https://site.test/", client, settings)


async def test_too_many_redirects(respx_mock, client, settings):
    respx_mock.get("https://a.test/").respond(302, headers={"location": "https://a.test/"})
    with pytest.raises(SiteUnreachableError, match="Too many redirects"):
        await fetch_page("https://a.test/", client, settings)


@pytest.mark.parametrize("status", [401, 403, 429])
async def test_blocked_by_site(respx_mock, client, settings, status):
    respx_mock.get("https://site.test/").respond(status, headers=HTML, text="denied")
    with pytest.raises(SiteUnreachableError, match="blocks automated access"):
        await fetch_page("https://site.test/", client, settings)


async def test_404(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(404, headers=HTML, text="nope")
    with pytest.raises(SiteUnreachableError, match="HTTP 404"):
        await fetch_page("https://site.test/", client, settings)


async def test_pdf_rejected(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
    with pytest.raises(UnsupportedContentError):
        await fetch_page("https://site.test/", client, settings)


async def test_missing_content_type_is_accepted(respx_mock, client, settings):
    respx_mock.get("https://site.test/").respond(200, content=b"<html>no header</html>")
    page = await fetch_page("https://site.test/", client, settings)
    assert "no header" in page.html


async def test_timeout(respx_mock, client, settings):
    respx_mock.get("https://site.test/").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(SiteTimeoutError):
        await fetch_page("https://site.test/", client, settings)


async def test_connection_error(respx_mock, client, settings):
    respx_mock.get("https://site.test/").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(SiteUnreachableError, match="Could not connect"):
        await fetch_page("https://site.test/", client, settings)


async def test_large_body_truncated(respx_mock, client, settings):
    respx_mock.get("https://big.test/").respond(200, headers=HTML, content=b"a" * 5_000_000)
    page = await fetch_page("https://big.test/", client, settings)
    assert len(page.html.encode()) <= settings.max_html_bytes


class _DripStream(httpx.AsyncByteStream):
    """Sends a few bytes, then waits — a server that never finishes."""

    async def __aiter__(self):
        for _ in range(100):
            yield b"<p>x</p>"
            await asyncio.sleep(0.1)


async def test_slow_drip_hits_total_deadline(respx_mock):
    settings = Settings(_env_file=None, fetch_timeout_s=0.2)
    respx_mock.get("https://drip.test/").respond(200, headers=HTML, stream=_DripStream())
    async with build_http_client(settings) as client:
        with pytest.raises(SiteTimeoutError):
            await fetch_page("https://drip.test/", client, settings)


async def test_charset_from_header(respx_mock, client, settings):
    body = "Café Ümlaut".encode("latin-1")
    respx_mock.get("https://site.test/").respond(200, headers={"content-type": "text/html; charset=latin-1"}, content=body)
    page = await fetch_page("https://site.test/", client, settings)
    assert "Café Ümlaut" in page.html
