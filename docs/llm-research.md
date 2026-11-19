# LLM Platform & Model Research

How the model for this API was chosen, what each option costs, and how cost was reduced.

**Rule followed throughout:** prices come from the providers' official pricing pages
(retrieved 2026-11-19). Latency, token counts, cost per analysis and accuracy are
**measured** by `scripts/benchmark_llms.py` and `scripts/evaluate.py` on real websites;
anything not measured is labelled *estimate* or *calculated*.

> **Status:** sections A–D are complete from official documentation and calculated estimates
> (each labelled). Columns marked *pending measurement* are filled by running the benchmark
> (see [How to reproduce](#how-to-reproduce)).

---

## Why a small, fast model

The task is extraction + summarization + light marketing judgment over ~5k tokens of
cleaned website text, returning ~1.5–3k tokens of JSON. It needs:

1. Reliable schema-constrained JSON (native structured outputs).
2. Good reading comprehension of short marketing text.
3. Broad general knowledge (for competitor suggestions).
4. Low latency (a user is waiting) and low cost per call.

It does **not** need deep multi-step reasoning or a large context window: the context is
capped at 24,000 characters (~6k tokens). That points to each provider's small tier,
run at **low reasoning effort** (reasoning tokens are billed as output, the expensive side).

## A. Comparison (official specs)

| Criteria | Anthropic | Google | OpenAI |
|---|---|---|---|
| Model | `claude-haiku-5-5` | `gemini-3.1-flash-lite` | `gpt-6-luna` |
| Input $/1M tokens | $0.10 | $0.25 | $0.10 |
| Cached input $/1M | $0.01 | $0.025 (+ storage for explicit caches) | $0.01 |
| Output $/1M tokens | $0.50 | $1.50 | $0.50 |
| Context window | 1M | 1,048,576 | 1,050,000 |
| Max output | 128k | 65,536 | 128,000 |
| Structured JSON | Yes — `output_config.format` JSON schema | Yes — `response_json_schema` | Yes — `response_format` strict JSON schema |
| Reasoning control | `effort` low…max | `thinking_level` | `reasoning_effort` none…max |
| Prompt caching | Explicit `cache_control` (min 512-token prefix) | Implicit + explicit | Automatic (+ `prompt_cache_key`) |
| Free tier | No | Yes | No |
| Response speed | *pending measurement* | *pending measurement* | *pending measurement* |
| Accuracy | *pending measurement* | *pending measurement* | *pending measurement* |

Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
[Gemini pricing](https://aistudio.google.com/docs/pricing),
[Gemini 3.1 Flash-Lite](https://aistudio.google.com/docs/models/gemini-3.1-flash-lite),
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).
Claude Haiku 5.5 prices apply to prompts up to 100k tokens (ours are ~6k).

**Reading the table:** on list price, Claude Haiku 5.5 and GPT-6 Luna are identical and
2.5–3× cheaper than Gemini 3.1 Flash-Lite. Price alone doesn't decide: a cheaper model that
fails the schema or the business rules more often pays for a retry (a second full call).
That is why the benchmark measures *first-attempt* validity and the cost per **successful**
analysis.

## B. Selection criteria and how each is measured

| Spec criterion | How it is measured |
|---|---|
| 1. Accuracy in understanding website content | Gold set: brand correct, services recall, services precision, location correct (`scripts/evaluate.py`) |
| 2. Fast response time | p50 LLM latency per call, measured |
| 3. Low API cost | Measured tokens × official prices, per analysis and per 1,000 |
| 4. Reliable structured JSON | % schema-valid and % rule-clean on the first attempt |
| 5. Relevant competitor and SEO insights | Blind human scores (1–5) + competitor domain check + SEO format check |
| 6. Large content handled efficiently | Context capped at 24k chars; tokens measured per site |

Evaluation method: [eval/README.md](../eval/README.md) — gold set written before any model
ran, token-based matching, blind human scoring with model names hidden, plus a prompt-injection
page.

## C. Cost optimization

Implemented levers, in order of impact. *Measured effect* is filled from
`results/cost_ablation.json`.

| # | Lever | Where | Measured effect |
|---|---|---|---|
| 1 | Cheapest model that passes quality | `LLM_PROVIDER` / `LLM_MODEL` | see D |
| 2 | Bounded output — length limits in the prompt (lists ≤ 5, fields ≤ 30 words), `max_output_tokens` cap, low reasoning effort | `app/prompts.py`, `LLM_EFFORT` | *pending*: output tokens with vs. without limits |
| 3 | Cleaned, deduplicated, capped input (scripts/styles/nav/hidden text removed; 24k-char cap split fairly across pages) | `website_extractor.py` | Dry run, homepage only: mailchimp.com ~107.8k raw-HTML tokens → ~4.9k context tokens (≈95% less); basecamp.com ~23.8k → ~4.2k. *Estimate (chars/4); provider token counts pending.* |
| 4 | One LLM call for all five sections | `analysis_service.py` | *calculated*: 5 separate calls would resend the context 5× |
| 5 | Provider prompt caching — static, byte-identical system prompt | `prompts.py`, adapters | *pending*: `cached_input_tokens` on repeat calls. Note: system prompt ≈585 tokens (estimate) is just above Haiku's 512-token minimum |
| 6 | Response cache (6 h TTL) + in-flight dedupe of concurrent identical requests | `analysis_service.py` | repeat URL: 0 tokens, $0 (tested) |
| 7 | No LLM call for invalid/blocked URLs, unreachable sites, JS-only pages | `analysis_service.py` | $0 (tested) |
| 8 | Deterministic extraction + SEO audit in code, not an LLM | `website_extractor.py` | $0 for those parts |
| 9 | At most one corrective retry, only on failure | `analysis_service.py` | retry rate from benchmark |
| 10 | Daily budget cap on the public demo | `BudgetGuard` | hard stop at `DAILY_BUDGET_USD` |

Not implemented, considered: **Batch APIs** (≈50% cheaper) suit bulk/offline analysis but not
a request/response endpoint; **a separate small model for extraction** is unnecessary because
extraction is done by code at zero LLM cost.

## D. Final recommendation

### Provisional recommendation (from official specs + calculated estimates)

| | Value | Basis |
|---|---|---|
| Recommended provider / model | **Anthropic `claude-haiku-5-5`** | lowest list price (tied with GPT-6 Luna), cheapest cache reads, native JSON-schema output, explicit prompt-cache control |
| Fallback model | **OpenAI `gpt-6-luna`** | same price tier, different provider, so an outage at one vendor doesn't take the API down; switch with `LLM_PROVIDER`/`LLM_MODEL` |
| Estimated input tokens | ~6,100–6,800 per analysis | *estimate*: system prompt ~585 + output schema ~1,250 + site context ~4,200–4,900 (context sizes from the dry run on basecamp.com / mailchimp.com; chars ÷ 4) |
| Estimated output tokens | ~1,500–3,000 | *estimate*: a complete sample answer is ~1,400 tokens; upper bound allows for low-effort reasoning tokens |
| Estimated cost per analysis | **~$0.0014–0.0022** (≈ $1.4–2.2 per 1,000) | *calculated*: estimated tokens × official prices. Same for GPT-6 Luna; Gemini 3.1 Flash-Lite ≈ $0.0038–0.0062 |
| Expected response time | *not claimed until measured* | latency depends on provider load; the benchmark reports p50 |
| Balancing approach | small-tier model at low reasoning effort, one call, bounded output, cached prompt prefix, validated output with a single corrective retry | see section C |

**Why provisional:** list price is only half of cost. A model that fails the schema or the
business rules on its first try pays for a second full call, and accuracy decides whether the
output is usable at all. The measured run below can overturn this choice.

### Measured result

*Pending measurement* — filled from `results/benchmark.md`, `results/eval.json` and
`results/human_scores.json` after running the benchmark below: measured tokens, cost per
analysis and per 1,000, p50 latency, first-try JSON validity, accuracy metrics and blind
human scores per model, and the final pick.

## How to reproduce

```bash
# 1. put API keys in .env (see .env.example)
# 2. list sites in eval/urls.txt and write eval/gold/<domain>.json for each (before running)
python -m scripts.benchmark_llms \
  --models gemini:gemini-3.1-flash-lite anthropic:claude-haiku-5-5 openai:gpt-6-luna \
  --urls eval/urls.txt --runs 2 --ablation
python -m scripts.evaluate
python -m scripts.evaluate --blind     # then score eval/blind/scores.csv
python -m scripts.evaluate --unblind
```
