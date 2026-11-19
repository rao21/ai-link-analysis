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
import json
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.errors import (
    InvalidURLError,
    SiteTimeoutError,
    SiteUnreachableError,
    UnsupportedContentError,
)
from app.schemas import SEOAudit
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


# ======================================================================
# Parsing: HTML -> clean, structured content (no LLM involved)
# ======================================================================

REMOVE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "template", "canvas"]
BLOCK_TAGS = [
    "p", "div", "section", "article", "main", "header", "aside", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table", "blockquote",
    "br", "dd", "dt", "figcaption", "address", "button", "label",
]
JSON_LD_TYPES = {"Organization", "LocalBusiness", "Corporation", "Service", "Product", "WebSite",
                 "ProfessionalService", "Store", "OnlineStore"}
SUBPAGE_KEYWORDS = ["service", "solution", "product", "about", "what-we-do", "pricing", "industr"]
MAX_SUBPAGE_DEPTH = 2
FILE_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".mp4", ".doc", ".docx")
_HIDDEN_STYLE = re.compile(r"display:none|visibility:hidden|font-size:0(?:px|em|rem|pt|%)?(?:;|$|!)")
_BREAK = " "  # paragraph separator used as a line marker while extracting text


@dataclass
class ExtractedContent:
    url: str
    title: str | None
    meta_description: str | None
    og: dict[str, str]
    headings: list[str]
    nav_links: list[tuple[str, str]]  # (link text, absolute URL), document order
    json_ld: list[dict]
    body_text: str
    footer_text: str
    seo_audit: SEOAudit


def extract_content(html: str, url: str) -> ExtractedContent:
    soup = BeautifulSoup(html, "lxml")

    # 1. Facts that need the untouched document.
    audit = _seo_audit(soup)
    json_ld = _json_ld(soup)
    og = {
        m["property"].lower(): _clean(m.get("content", ""))
        for m in soup.find_all("meta", property=re.compile(r"^og:", re.I))
        if m.get("content")
    }

    # 2. Drop what a visitor never sees (also blocks hidden prompt-injection text).
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
    for tag in soup.find_all(_is_hidden):
        if not tag.decomposed:
            tag.decompose()

    links = _links(soup, url)
    headings = _dedupe(_clean(h.get_text(" ")) for h in soup.find_all(["h1", "h2", "h3"]))

    footer_text = " ".join(_clean(f.get_text(" ")) for f in soup.find_all("footer"))[:500]
    for tag in soup.find_all(["nav", "footer"]):
        tag.decompose()

    body = soup.body or soup
    return ExtractedContent(
        url=url,
        title=audit.title,
        meta_description=audit.meta_description,
        og=og,
        headings=[h for h in headings if h],
        nav_links=links,
        json_ld=json_ld,
        body_text=_visible_text(body),
        footer_text=footer_text,
        seo_audit=audit,
    )


def _clean(text: str) -> str:
    return " ".join(text.split())


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _is_hidden(tag) -> bool:
    if tag.has_attr("hidden") or tag.get("aria-hidden", "").lower() == "true":
        return True
    style = "".join(tag.get("style", "").lower().split())
    return bool(style and _HIDDEN_STYLE.search(style))


def _visible_text(root) -> str:
    """Text with one line per block element, so inline tags (<a>, <strong>) don't split sentences."""
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before(_BREAK)
        tag.insert_after(_BREAK)
    text = root.get_text("")
    lines = (_clean(line) for line in text.split(_BREAK))
    # One-word lines are UI noise ("Menu", "Login", "×"); 2+ words can be real services ("Local SEO").
    return "\n".join(_dedupe(line for line in lines if len(line.split()) >= 2))


def _seo_audit(soup: BeautifulSoup) -> SEOAudit:
    title_tag = soup.find("title")
    title = _clean(title_tag.get_text()) if title_tag else None
    desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    desc = _clean(desc_tag.get("content", "")) if desc_tag else None
    desc = desc or None
    images = soup.find_all("img")
    missing_alt = sum(1 for img in images if not img.has_attr("alt"))
    h1_count = len(soup.find_all("h1"))
    has_canonical = soup.find("link", rel=lambda r: bool(r) and r.lower() == "canonical") is not None
    has_viewport = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)}) is not None
    html_tag = soup.find("html")
    lang = html_tag.get("lang") if html_tag else None
    has_og = soup.find("meta", property=re.compile(r"^og:", re.I)) is not None

    issues = []
    if not title:
        issues.append("Missing <title> tag.")
    elif len(title) > 60:
        issues.append(f"Title is {len(title)} characters; recommended maximum is 60.")
    if not desc:
        issues.append("Missing meta description.")
    elif len(desc) > 160:
        issues.append(f"Meta description is {len(desc)} characters; recommended maximum is 160.")
    if h1_count != 1:
        issues.append(f"Page has {h1_count} H1 headings; recommended exactly 1.")
    if missing_alt:
        issues.append(f"{missing_alt} of {len(images)} images are missing alt text.")
    if not has_canonical:
        issues.append("No canonical URL tag.")
    if not has_viewport:
        issues.append("No viewport meta tag (page may not be mobile-friendly).")

    return SEOAudit(
        title=title,
        title_length=len(title or ""),
        meta_description=desc,
        meta_description_length=len(desc or ""),
        h1_count=h1_count,
        images_total=len(images),
        images_missing_alt=missing_alt,
        has_canonical=has_canonical,
        has_viewport=has_viewport,
        lang=lang,
        has_open_graph=has_og,
        issues=issues,
    )


def _json_ld(soup: BeautifulSoup) -> list[dict]:
    found: list[dict] = []
    for script in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue  # broken JSON-LD on real sites is common; skip it
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop(0)
            if not isinstance(item, dict):
                continue
            if "@graph" in item and isinstance(item["@graph"], list):
                stack.extend(item["@graph"])
                continue
            types = item.get("@type")
            types = types if isinstance(types, list) else [types]
            if any(t in JSON_LD_TYPES for t in types):
                found.append(item)
    return found


def _links(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"].strip())
        if absolute not in seen:
            seen.add(absolute)
            out.append((_clean(a.get_text(" ")), absolute))
    return out


def _bare_host(host: str | None) -> str:
    host = (host or "").lower()
    return host[4:] if host.startswith("www.") else host


def select_subpages(nav_links: list[tuple[str, str]], base_url: str, limit: int) -> list[str]:
    """Pick the internal pages most likely to describe the business (services, about...)."""
    base_host = _bare_host(urlsplit(base_url).hostname)
    ranked: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    for position, (text, href) in enumerate(nav_links):
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https") or _bare_host(parts.hostname) != base_host:
            continue
        path = parts.path or "/"
        if path in ("", "/") or path.lower().endswith(FILE_EXTENSIONS):
            continue
        key = path.rstrip("/").lower()
        if key in seen:
            continue
        depth = len([seg for seg in path.split("/") if seg])
        if depth > MAX_SUBPAGE_DEPTH:
            continue  # deep links are usually help articles / blog posts, not business pages
        haystack = f"{text} {path}".lower()
        rank = next((i for i, kw in enumerate(SUBPAGE_KEYWORDS) if kw in haystack), None)
        if rank is None:
            continue
        try:
            clean_url = normalize_url(urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, "")))
        except InvalidURLError:
            continue
        seen.add(key)
        ranked.append((depth, rank, position, clean_url))
    ranked.sort()  # shallow pages first, then keyword priority, then page order
    return [u for *_, u in ranked[:limit]]


def content_chars(home: ExtractedContent, subpages: list[ExtractedContent]) -> int:
    return sum(len(p.body_text) + len(" ".join(p.headings)) for p in [home, *subpages])


def build_llm_context(home: ExtractedContent, subpages: list[ExtractedContent], max_chars: int) -> str:
    """Compact, labelled text for the LLM.

    Budget: metadata first (small, highest value), then the remaining budget is
    split between the homepage and each sub-page so a huge homepage can't push
    the /services page out of the prompt.
    """
    meta_lines = [f"URL: {home.url}"]
    if home.title:
        meta_lines.append(f"Title: {home.title}")
    if home.meta_description:
        meta_lines.append(f"Meta description: {home.meta_description}")
    if home.og.get("og:site_name"):
        meta_lines.append(f"og:site_name: {home.og['og:site_name']}")
    og_desc = home.og.get("og:description")
    if og_desc and og_desc != home.meta_description:  # don't pay for the same text twice
        meta_lines.append(f"og:description: {og_desc}")
    head = ["### PAGE METADATA\n" + "\n".join(meta_lines)]
    if home.json_ld:
        head.append("### STRUCTURED DATA\n" + json.dumps(home.json_ld, separators=(",", ":"), ensure_ascii=False)[:2000])
    if home.headings:
        head.append("### HEADINGS\n" + "\n".join(home.headings))
    head_text = "\n\n".join(head)

    footer = f"### FOOTER\n{home.footer_text}" if home.footer_text else ""
    remaining = max(0, max_chars - len(head_text) - len(footer) - 8)
    share = remaining // (1 + len(subpages)) if remaining else 0

    sections = [head_text, "### HOMEPAGE TEXT\n" + home.body_text[:share]]
    for page in subpages:
        body = "\n".join([*page.headings, page.body_text])
        sections.append(f"### PAGE: {page.url}\n" + body[:share])
    if footer:
        sections.append(footer)
    return "\n\n".join(s for s in sections if s)[:max_chars]
