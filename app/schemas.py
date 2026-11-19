"""Request/response models.

Two families:
- `LLMAnalysis` and its parts: exactly what the LLM must return. They forbid extra
  fields, and their JSON schema is sent to the provider as a structured-output contract.
- `Analysis` / `LinkAnalysisResponse`: what the API returns. They extend the LLM
  models with fields our code fills in (SEO audit, disclaimer, meta).

There is deliberately no field for search volume, ranking or traffic: the spec says
not to invent them, and a schema without the field makes that impossible.
"""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
ALL_PLATFORMS: tuple[str, ...] = ("Google", "Facebook", "Instagram", "LinkedIn", "X")

_WEBSITE_RE = re.compile(r"^https?://[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}(/\S*)?$")


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- SEO audit

class SEOAudit(BaseModel):
    """Facts measured from the page's HTML (deterministic, no LLM)."""

    title: str | None
    title_length: int
    meta_description: str | None
    meta_description_length: int
    h1_count: int
    images_total: int
    images_missing_alt: int
    has_canonical: bool
    has_viewport: bool
    lang: str | None
    has_open_graph: bool
    issues: list[str]


# ---------------------------------------------------------------- LLM output

class Brand(_LLMModel):
    name: str
    category: str
    summary: str


class Competitor(_LLMModel):
    name: str
    website: str | None
    description: str
    target_market: str
    strengths: list[str]
    weaknesses: list[str]
    basis: Literal["ai_suggestion"] = "ai_suggestion"

    @field_validator("website", mode="before")
    @classmethod
    def _sanitize_website(cls, value: object) -> str | None:
        """Keep only plausible http(s) URLs; anything else becomes null."""
        if not isinstance(value, str) or not value.strip():
            return None
        url = value.strip().lower()
        if "://" not in url:
            url = "https://" + url
        return url if _WEBSITE_RE.match(url) else None


class ComparisonRow(_LLMModel):
    aspect: str
    your_brand: str
    competitor_name: str
    competitor_value: str


class CompetitorAnalysisLLM(_LLMModel):
    brand_overview: str
    main_category: str
    main_products_services: list[str]
    competitors: list[Competitor]
    comparison_table: list[ComparisonRow]
    competitive_advantages: list[str]
    market_opportunities: list[str]


class SearchQuery(_LLMModel):
    query: str
    platforms: list[Platform]
    intent: str
    business_relevance: str
    target_audience: str


class ContentIdea(_LLMModel):
    type: Literal["service_page", "industry_page", "faq", "blog_post", "location_page", "article"]
    title: str
    purpose: str


class SEORecommendations(_LLMModel):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    business_keywords: list[str]
    recommended_title: str
    title_rationale: str
    meta_description: str
    supporting_content: list[ContentIdea]


class Service(_LLMModel):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: Literal["website_content", "inferred"]


class AudienceSegment(_LLMModel):
    segment: str
    business_type: str
    requirement: str


class TargetAudience(_LLMModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    business_types: list[str]
    customer_profiles: list[str]
    geographic_focus: str | None
    customer_needs: list[str]
    customer_challenges: list[str]
    interests: list[str]
    recommended_segments: list[AudienceSegment]


class LLMAnalysis(_LLMModel):
    """The complete structured output requested from the LLM in one call."""

    brand: Brand
    competitor_analysis: CompetitorAnalysisLLM
    target_search_queries: list[SearchQuery]
    seo_improvements: SEORecommendations
    top_services: list[Service]
    target_audience: TargetAudience


# ---------------------------------------------------------------- API response

class CompetitorAnalysis(CompetitorAnalysisLLM):
    disclaimer: str


class SEOImprovements(SEORecommendations):
    current_audit: SEOAudit


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: TargetAudience


class Meta(BaseModel):
    provider: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    raw_html_chars: int
    context_chars: int
    services_relabeled: int
    llm_latency_ms: int
    total_latency_ms: int
    pages_analyzed: list[str]
    llm_attempts: int
    cached: bool
    analyzed_at: datetime


class LinkAnalysisRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048, examples=["https://example.com"])


class LinkAnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: Meta


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
