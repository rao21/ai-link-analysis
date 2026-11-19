# AI Link Analysis API

A FastAPI service that reads a business website and uses an LLM to return a structured
**competitor, search-query, SEO, services and target-audience** analysis as JSON.

| | |
|---|---|
| **Live demo** | _added after deployment_ |
| **API docs** | `/docs` on the live URL (Swagger) |
| **Endpoint** | `POST /link-analysis` with `{"url": "https://example.com"}` |
| **Tests** | 159 automated tests (`pytest`) — no API key or network needed |
| **Model choice** | See [LLM research](docs/llm-research.md): official prices, calculated cost per analysis, and the benchmark that measures cost, latency and accuracy |

**What makes it production-minded**

- **Grounded, not generic** — analysis comes from the site's own text (home page + up to 3
  relevant sub-pages such as `/services`). Pages with too little text (JavaScript-only sites)
  return an error instead of a made-up analysis.
- **Honest output** — competitors are always labelled `ai_suggestion` with a disclaimer; each
  service says whether it came from the website or was inferred, and a runtime check relabels
  any "from the website" claim whose words aren't actually on the site. The schema has no
  field for search volume, so it can't be invented.
- **Secure fetching** — SSRF protection on the URL and on **every redirect hop** (private,
  loopback, link-local, cloud-metadata, IPv6 and obfuscated forms like `2130706433`), size and
  time limits, hidden-text stripping against prompt injection.
- **Cost-aware** — one LLM call for all five sections, cleaned and capped context
  (on mailchimp.com's homepage: ~108k tokens of raw HTML → ~4.9k tokens sent, estimated), bounded output, prompt caching, response cache,
  in-flight de-duplication, and a daily budget cap. Every response reports its own tokens,
  cost and latency.
- **Provider-agnostic** — the same code runs Gemini, Claude or OpenAI (one env var), which is
  how the models were benchmarked on identical inputs.

---

## Quick start

```bash
git clone <this repo> && cd ai-link-analysis
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add ONE API key and set LLM_PROVIDER
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- Demo page: http://localhost:4000/
- Swagger: http://localhost:4000/docs
- Tests: `pytest` (live provider tests: `pytest -m live`, needs keys)

Docker:

```bash
docker build -t ai-link-analysis .
docker run --rm -p 4000:4000 --env-file .env ai-link-analysis
```

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `anthropic` or `openai` |
| `LLM_MODEL` | provider default | `gemini-3.1-flash-lite`, `claude-haiku-5-5`, `gpt-6-luna` |
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | key for the chosen provider |
| `LLM_EFFORT` | `low` | reasoning effort (reasoning tokens are billed as output) |
| `MAX_CONTEXT_CHARS` | `24000` | cap on website text sent to the LLM |
| `CACHE_TTL_S` | `21600` | response cache lifetime (6 h) |
| `RATE_LIMIT` | `10/minute` | per client IP |
| `TRUST_PROXY_HEADERS` | `false` | `true` only behind a proxy that appends the client IP to `X-Forwarded-For` |
| `DAILY_BUDGET_USD` | `2.0` | hard daily cap on estimated LLM spend |

---

## API

### Request

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.example-business.com"}'
```

### Response (abridged)

```json
{
  "success": true,
  "website": "https://www.example-business.com/",
  "brand": { "name": "...", "category": "...", "summary": "..." },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "...", "main_category": "...", "main_products_services": ["..."],
      "competitors": [
        { "name": "...", "website": "https://...", "description": "...", "target_market": "...",
          "strengths": ["..."], "weaknesses": ["..."], "basis": "ai_suggestion" }
      ],
      "comparison_table": [ { "aspect": "Target Market", "your_brand": "...", "competitor_name": "...", "competitor_value": "..." } ],
      "competitive_advantages": ["..."], "market_opportunities": ["..."],
      "disclaimer": "Competitors are AI-generated suggestions ... verify before use."
    },
    "target_search_queries": [
      { "query": "...", "platforms": ["Google", "LinkedIn"], "intent": "...",
        "business_relevance": "...", "target_audience": "..." }
    ],
    "seo_improvements": {
      "current_audit": { "title_length": 75, "h1_count": 2, "images_missing_alt": 3,
                         "issues": ["Title is 75 characters; recommended maximum is 60.", "..."] },
      "primary_keyword": "...", "secondary_keywords": ["..."], "long_tail_keywords": ["..."],
      "business_keywords": ["..."], "recommended_title": "...", "title_rationale": "...",
      "meta_description": "...",
      "supporting_content": [ { "type": "service_page", "title": "...", "purpose": "..." } ]
    },
    "top_services": [
      { "name": "...", "description": "...", "target_industry": "...", "target_customer": "...",
        "main_benefit": "...", "source": "website_content" }
    ],
    "target_audience": {
      "primary_audience": ["..."], "secondary_audience": ["..."], "industries": ["..."],
      "business_types": ["..."], "customer_profiles": ["..."], "geographic_focus": "...",
      "customer_needs": ["..."], "customer_challenges": ["..."], "interests": ["..."],
      "recommended_segments": [ { "segment": "...", "business_type": "...", "requirement": "..." } ]
    }
  },
  "meta": {
    "provider": "anthropic", "model": "claude-haiku-5-5",
    "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
    "raw_html_chars": 0, "context_chars": 0, "services_relabeled": 0,
    "llm_latency_ms": 0, "total_latency_ms": 0, "pages_analyzed": ["..."],
    "llm_attempts": 1, "cached": false, "analyzed_at": "..."
  }
}
```

A full real response is in [docs/sample-response.json](docs/sample-response.json) _(added after the measured run)_.

### Errors

Every error has the same shape: `{"success": false, "error": {"code": "...", "message": "..."}}`

| Status | Code | When |
|---|---|---|
| 400 | `BLOCKED_ADDRESS` | URL resolves to a private/internal address (SSRF protection) |
| 422 | `INVALID_REQUEST` / `INVALID_URL` | Bad body, or not an http(s) URL on port 80/443 |
| 422 | `UNSUPPORTED_CONTENT` | URL is a PDF, image, etc. |
| 422 | `INSUFFICIENT_CONTENT` | Too little readable text (e.g. JavaScript-only site) |
| 422 | `LLM_REFUSED` | The AI provider declined the content |
| 429 | `RATE_LIMITED` | Too many requests from this client |
| 502 | `SITE_UNREACHABLE` | DNS failure, connection error, HTTP 4xx/5xx, or the site blocks bots |
| 502 | `LLM_UNAVAILABLE` / `LLM_INVALID_OUTPUT` | Provider error, or no valid output after one corrective retry |
| 503 | `BUDGET_EXCEEDED` | Daily AI budget for the demo is used up |
| 504 | `SITE_TIMEOUT` | Website too slow (per-read timeout and a total deadline) |

---

## How it works

```
POST /link-analysis {url}
  │
  ├─ 1. normalize_url ........ http(s) only, ports 80/443, no credentials        → 422
  ├─ 2. cache / in-flight .... same URL within 6 h, or already running          → reused, $0
  ├─ 3. fetch (per hop) ...... DNS → every IP must be public → GET → redirect?   → 400 / 502 / 504
  ├─ 4. extract .............. title, meta, JSON-LD, headings, clean text,
  │                            deterministic SEO audit; hidden text removed
  ├─ 5. sub-pages ............ up to 3 (services/about/products), fetched in parallel
  ├─ 6. enough text? ......... otherwise INSUFFICIENT_CONTENT                    → 422, $0
  ├─ 7. budget check ......... daily cap                                         → 503
  ├─ 8. ONE LLM call ......... static system prompt (cacheable) + site text as data,
  │                            native JSON-schema structured output
  ├─ 9. validate ............. Pydantic + business rules (5 queries, all 5 platforms,
  │                            5 services, ≥3 competitors, title/meta length)
  │                            → one retry that lists the problems, else 502
  ├─10. post-process ......... competitor labels + disclaimer, service grounding
  └─11. respond .............. analysis + meta (tokens, cost, latency) → cache
```

```text
app/
├── main.py                   # FastAPI app, route, error handlers, rate limit, demo page
├── config.py                 # settings from environment
├── errors.py                 # error types → HTTP status + code
├── schemas.py                # request/response and LLM output models
├── prompts.py                # system prompt and user prompt
├── pricing.py                # official token prices + cost estimate
├── static/index.html         # demo page
├── services/
│   ├── website_extractor.py  # safe fetch, parsing, SEO audit, sub-pages, context
│   ├── analysis_service.py   # pipeline, validation, retry, budget, cache
│   ├── competitor_service.py # competitor clean-up + disclaimer
│   ├── grounding.py          # service hallucination guard
│   └── llm/                  # base interface + gemini / anthropic / openai adapters
└── utils/validators.py       # URL normalization + SSRF guard
scripts/                      # benchmark_llms.py, evaluate.py, eval_metrics.py
eval/                         # gold set, prompt-injection fixture, method
docs/llm-research.md          # model comparison and recommendation
```

---

## LLM selection

Full write-up: **[docs/llm-research.md](docs/llm-research.md)**.

| | Claude Haiku 5.5 | Gemini 3.1 Flash-Lite | GPT-6 Luna |
|---|---|---|---|
| Input / output $ per 1M tokens | $0.10 / $0.50 | $0.25 / $1.50 | $0.10 / $0.50 |
| Cached input $ per 1M | $0.01 | $0.025 | $0.01 |
| Structured JSON output | Yes | Yes | Yes |

_Prices from official pricing pages, retrieved 2026-11-19. Estimated cost per analysis:
~$0.0014–0.0022 for Claude Haiku 5.5 / GPT-6 Luna, ~$0.004–0.006 for Gemini 3.1 Flash-Lite
(calculated from estimated tokens; see the research doc, which also holds the measured results)._

How accuracy is measured: a hand-written gold set (brand, services, location per site) that is
written before any model runs, automatic metrics (services recall/precision, brand, location, JSON
validity, stability), a prompt-injection test page, and blind human scoring with model names
hidden — see [eval/README.md](eval/README.md).

---

## Security notes

- **SSRF:** hosts are resolved and **all** returned IPs must be public; checked again on every
  redirect hop; only ports 80/443; redirects to other schemes are refused.
  *Known limit:* DNS rebinding — httpx resolves the host again after our check, so a DNS server
  that answers differently the second time is not stopped by the application. In production this is closed at the network layer
  (egress firewall blocking private ranges) or by connecting to the pre-resolved IP.
- **Rate-limit key:** the client IP comes from the socket, or — only when
  `TRUST_PROXY_HEADERS=true` behind a proxy such as Render — from the **rightmost**
  `X-Forwarded-For` entry, which the proxy wrote. The leftmost entry is client-controlled.
- **Prompt injection:** invisible elements are removed before the LLM sees the text; site text
  is passed inside `<website_content>` tags as data; output is schema-locked, with no tools.
  *Known limit:* text hidden only through external CSS classes can't be detected without
  rendering.
- **Secrets:** keys are read from the environment as `SecretStr`, never logged, never in the
  repo (`.env` is git-ignored); unexpected errors return a generic message.
- **Abuse/cost:** per-IP rate limit and a daily budget cap; the container runs as a non-root user.

## Limitations and next steps

- JavaScript-rendered sites are rejected rather than analyzed; a headless-browser fallback
  (Playwright) would cover them at the cost of memory and latency.
- Competitors come from the model's general knowledge (clearly labelled). A web-search API
  would let them be verified.
- Cache, rate limit and budget are in-memory (one instance). With several instances they move
  to Redis.

## Tests

```bash
pytest            # 159 tests: validators/SSRF, fetcher, extraction, schemas, LLM adapters,
                  # pipeline (retry, grounding, cache, budget), endpoint, eval metrics
pytest -m live    # real provider calls (needs keys)
```
