"""The /link-analysis pipeline.

    URL -> validate -> cache? -> fetch home -> extract -> fetch sub-pages (parallel)
        -> enough text? -> budget? -> ONE LLM call -> validate + business rules
        -> (one corrective retry) -> competitors + grounding -> response -> cache

Every check that can fail cheaply runs before the LLM call, so a bad URL, a blocked
address, an unreachable site or a JavaScript-only page costs nothing.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
from cachetools import TTLCache
from pydantic import ValidationError

from app.config import Settings
from app.errors import AppError, BudgetExceededError, InsufficientContentError, LLMOutputError
from app.pricing import estimate_cost
from app.prompts import SYSTEM_PROMPT, build_retry_prompt, build_user_prompt
from app.schemas import ALL_PLATFORMS, Analysis, LinkAnalysisResponse, LLMAnalysis, Meta, SEOImprovements
from app.services.competitor_service import finalize_competitors
from app.services.grounding import ground_services
from app.services.llm.base import LLMClient
from app.services.website_extractor import (
    ExtractedContent,
    build_llm_context,
    content_chars,
    extract_content,
    fetch_page,
    select_subpages,
)
from app.utils.validators import normalize_url

logger = logging.getLogger("link_analysis")

MAX_LLM_ATTEMPTS = 2  # first call + one corrective retry


@dataclass
class SiteContext:
    url: str                       # final URL after redirects
    home: ExtractedContent
    subpages: list[ExtractedContent]
    context: str                   # exactly what the LLM receives
    raw_html_chars: int


async def prepare_site(url: str, http: httpx.AsyncClient, settings: Settings) -> SiteContext:
    """Fetch + extract + build the LLM context. Shared by the API and the benchmark,
    so benchmark numbers describe exactly what production sends."""
    home_page = await fetch_page(url, http, settings)
    # Parsing is CPU work: run it in a thread so other requests aren't blocked meanwhile.
    home = await asyncio.to_thread(extract_content, home_page.html, home_page.url)
    subpages, sub_html_chars = await _fetch_subpages(home, http, settings)
    if content_chars(home, subpages) < settings.min_content_chars:
        raise InsufficientContentError(
            "The website returned too little readable text (it may render content with JavaScript), "
            "so a reliable analysis isn't possible."
        )
    return SiteContext(
        url=home_page.url,
        home=home,
        subpages=subpages,
        context=build_llm_context(home, subpages, settings.max_context_chars),
        raw_html_chars=len(home_page.html) + sub_html_chars,
    )


async def _fetch_subpages(home: ExtractedContent, http: httpx.AsyncClient,
                          settings: Settings) -> tuple[list[ExtractedContent], int]:
    urls = select_subpages(home.nav_links, home.url, settings.max_subpages)

    async def one(u: str):
        page = await fetch_page(u, http, settings)
        return await asyncio.to_thread(extract_content, page.html, page.url), len(page.html)

    results = await asyncio.gather(*(one(u) for u in urls), return_exceptions=True)
    pages, html_chars = [], 0
    for u, result in zip(urls, results):
        if isinstance(result, (AppError, httpx.HTTPError)):
            logger.info("sub-page skipped %s: %s", u, result)
            continue
        if isinstance(result, BaseException):
            raise result
        pages.append(result[0])
        html_chars += result[1]
    return pages, html_chars


class BudgetGuard:
    """Caps estimated LLM spend per UTC day, so a public demo can't drain the API key."""

    def __init__(self, daily_budget_usd: float, today: Callable[[], date] | None = None) -> None:
        self._budget = daily_budget_usd
        self._today = today or (lambda: datetime.now(UTC).date())
        self._day = self._today()
        self._spent = 0.0

    def _roll(self) -> None:
        if self._today() != self._day:
            self._day, self._spent = self._today(), 0.0

    def check(self) -> None:
        self._roll()
        if self._spent >= self._budget:
            raise BudgetExceededError("Daily AI budget for this demo has been reached; try again tomorrow.")

    def add(self, usd: float) -> None:
        self._roll()
        self._spent += usd


def check_business_rules(a: LLMAnalysis) -> list[str]:
    """Rules the JSON schema can't express. Each string is fed back to the model on retry."""
    problems: list[str] = []
    if len(a.target_search_queries) != 5:
        problems.append(f"target_search_queries must contain exactly 5 queries (got {len(a.target_search_queries)}).")
    used = {p for q in a.target_search_queries for p in q.platforms}
    missing = [p for p in ALL_PLATFORMS if p not in used]
    if missing:
        problems.append(f"Each platform must appear in at least one query; missing: {', '.join(missing)}.")
    if len(a.top_services) != 5:
        problems.append(f"top_services must contain exactly 5 services (got {len(a.top_services)}).")
    if len(a.competitor_analysis.competitors) < 3:
        problems.append("competitors must contain at least 3 companies.")
    brand = " ".join(a.brand.name.lower().split())
    if any(" ".join(c.name.lower().split()) == brand for c in a.competitor_analysis.competitors):
        problems.append("competitors must not include the brand itself.")
    if len(a.seo_improvements.recommended_title) > 70:
        problems.append("recommended_title must be at most 60 characters.")
    if len(a.seo_improvements.meta_description) > 170:
        problems.append("meta_description must be at most 160 characters.")
    return problems


def _validation_problems(exc: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
        for err in exc.errors()[:10]
    ]


class AnalysisService:
    def __init__(self, settings: Settings, llm: LLMClient | Callable[[], LLMClient], http: httpx.AsyncClient,
                 budget: BudgetGuard | None = None) -> None:
        """`llm` may be a client or a factory; a factory is called on the first LLM call,
        so every cheap check (URL, SSRF, fetch, content) works even without an API key."""
        self._settings = settings
        self._llm_source = llm
        self._llm_client: LLMClient | None = llm if hasattr(llm, "generate_json") else None
        self._http = http
        self._budget = budget or BudgetGuard(settings.daily_budget_usd)
        self._cache: TTLCache[str, LinkAnalysisResponse] = TTLCache(maxsize=256, ttl=settings.cache_ttl_s)
        self._inflight: dict[str, asyncio.Future[LinkAnalysisResponse]] = {}

    async def analyze(self, raw_url: str) -> LinkAnalysisResponse:
        started = time.perf_counter()
        url = normalize_url(raw_url)

        if (hit := self._cache.get(url)) is not None:
            return self._as_cached(hit, started)

        # Same URL already being analyzed (double click, two users)? Wait for that result
        # instead of paying for a second identical LLM call.
        if (pending := self._inflight.get(url)) is not None:
            return self._as_cached(await asyncio.shield(pending), started)

        future: asyncio.Future[LinkAnalysisResponse] = asyncio.get_running_loop().create_future()
        self._inflight[url] = future
        try:
            response = await self._analyze_uncached(url, started)
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved: no "never retrieved" warning when nobody else waits
            raise
        else:
            future.set_result(response)
            return response
        finally:
            del self._inflight[url]

    @property
    def _llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = self._llm_source()
        return self._llm_client

    @staticmethod
    def _as_cached(response: LinkAnalysisResponse, started: float) -> LinkAnalysisResponse:
        copy = response.model_copy(deep=True)
        copy.meta.cached = True
        copy.meta.total_latency_ms = int((time.perf_counter() - started) * 1000)
        return copy

    async def _analyze_uncached(self, url: str, started: float) -> LinkAnalysisResponse:
        site = await prepare_site(url, self._http, self._settings)
        home, subpages, context = site.home, site.subpages, site.context

        analysis, usage = await self._call_llm(build_user_prompt(site.url, context))

        services, relabeled = ground_services(analysis.top_services, context)
        cost = estimate_cost(self._llm.model, usage["input"], usage["cached"], usage["output"])
        response = LinkAnalysisResponse(
            website=site.url,
            brand=analysis.brand,
            analysis=Analysis(
                competitor_analysis=finalize_competitors(analysis.competitor_analysis, analysis.brand.name),
                target_search_queries=analysis.target_search_queries,
                seo_improvements=SEOImprovements(**analysis.seo_improvements.model_dump(), current_audit=home.seo_audit),
                top_services=services,
                target_audience=analysis.target_audience,
            ),
            meta=Meta(
                provider=self._llm.provider,
                model=self._llm.model,
                input_tokens=usage["input"],
                cached_input_tokens=usage["cached"],
                output_tokens=usage["output"],
                estimated_cost_usd=cost,
                raw_html_chars=site.raw_html_chars,
                context_chars=len(context),
                services_relabeled=relabeled,
                llm_latency_ms=usage["latency"],
                total_latency_ms=int((time.perf_counter() - started) * 1000),
                pages_analyzed=[home.url, *(p.url for p in subpages)],
                llm_attempts=usage["attempts"],
                cached=False,
                analyzed_at=datetime.now(UTC),
            ),
        )
        self._cache[url] = response
        logger.info(json.dumps({
            "event": "analysis", "url": site.url, "model": self._llm.model,
            "input_tokens": usage["input"], "cached_input_tokens": usage["cached"],
            "output_tokens": usage["output"], "cost_usd": cost, "attempts": usage["attempts"],
            "latency_ms": response.meta.total_latency_ms, "services_relabeled": relabeled,
        }))
        return response

    async def _call_llm(self, user_prompt: str) -> tuple[LLMAnalysis, dict]:
        usage = {"input": 0, "cached": 0, "output": 0, "latency": 0, "attempts": 0}
        prompt = user_prompt
        problems: list[str] = []
        for _ in range(MAX_LLM_ATTEMPTS):
            self._budget.check()
            result = await self._llm.generate_json(SYSTEM_PROMPT, prompt, LLMAnalysis)
            usage["attempts"] += 1
            usage["input"] += result.input_tokens
            usage["cached"] += result.cached_input_tokens
            usage["output"] += result.output_tokens
            usage["latency"] += result.latency_ms
            cost = estimate_cost(self._llm.model, result.input_tokens, result.cached_input_tokens, result.output_tokens)
            if cost is None:
                logger.warning("no price for model %s; budget not charged", self._llm.model)
            self._budget.add(cost or 0.0)

            if result.truncated or result.data is None:
                problems = ["Your output was cut off at the token limit; shorten every field and respect the length limits."]
            else:
                try:
                    analysis = LLMAnalysis.model_validate(result.data)
                except ValidationError as exc:
                    problems = _validation_problems(exc)
                else:
                    problems = check_business_rules(analysis)
                    if not problems:
                        return analysis, usage
            logger.info("LLM output rejected (attempt %d): %s", usage["attempts"], problems)
            prompt = build_retry_prompt(user_prompt, problems)

        raise LLMOutputError("The AI could not produce a valid analysis for this website. Please try again.")
