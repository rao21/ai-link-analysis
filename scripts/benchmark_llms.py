"""Measured LLM comparison on identical inputs.

    python -m scripts.benchmark_llms \
        --models gemini:gemini-3.1-flash-lite anthropic:claude-haiku-5-5 openai:gpt-6-luna \
        --urls eval/urls.txt --runs 2 --ablation

For each URL the website context is built ONCE (same code path as the API), then
every model gets the exact same prompt. Each call is a single attempt (no retry),
so "schema valid" and "rules clean" measure first-try reliability.

Writes:
  results/outputs/<site>__<model>__run<N>.json   raw model output (for scripts/evaluate.py)
  results/benchmark.json                          one row per call
  results/cost_ablation.json                      with --ablation
and prints markdown tables. API keys are read from .env.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from pydantic import ValidationError

from app.config import Settings
from app.errors import AppError
from app.pricing import PRICES_PER_MTOK, PRICES_RETRIEVED, estimate_cost
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import LLMAnalysis
from app.services.analysis_service import check_business_rules, prepare_site
from app.services.llm.base import get_llm_client
from app.services.website_extractor import build_http_client, build_llm_context, extract_content

RESULTS = Path("results")
OUTPUTS = RESULTS / "outputs"
INJECTION_FIXTURE = Path("eval/fixtures/prompt_injection.html")
SYSTEM_PROMPT_NO_LIMITS = SYSTEM_PROMPT.split("\n\nLength limits")[0]


def slug(url: str) -> str:
    host = (urlsplit(url).hostname or url).lower()
    return host[4:] if host.startswith("www.") else host


def model_settings(spec: str) -> Settings:
    provider, _, model = spec.partition(":")
    return Settings(llm_provider=provider, llm_model=model)


async def build_contexts(urls: list[str], settings: Settings) -> dict[str, dict]:
    sites: dict[str, dict] = {}
    async with build_http_client(settings) as http:
        for url in urls:
            try:
                site = await prepare_site(url, http, settings)
            except AppError as exc:
                print(f"  ! {url}: {exc.code} {exc.message}", file=sys.stderr)
                continue
            raw_html = await _raw_html(url, http)
            sites[slug(site.url)] = {
                "url": site.url,
                "context": site.context,
                "raw_html_chars": site.raw_html_chars,
                "visible_text_chars": len(BeautifulSoup(raw_html, "lxml").get_text(" ")) if raw_html else None,
                "home_html": raw_html,
            }
    if INJECTION_FIXTURE.exists():
        html = INJECTION_FIXTURE.read_text()
        page = extract_content(html, "https://injection-fixture.test/")
        sites["injection-fixture"] = {
            "url": "https://injection-fixture.test/",
            "context": build_llm_context(page, [], settings.max_context_chars),
            "raw_html_chars": len(html), "visible_text_chars": None, "home_html": html,
        }
    return sites


async def _raw_html(url: str, http) -> str | None:
    from app.services.website_extractor import fetch_page
    try:
        return (await fetch_page(url, http, Settings(_env_file=None))).html
    except AppError:
        return None


async def run_one(client, site_slug: str, site: dict, run: int, system: str = SYSTEM_PROMPT) -> dict:
    row = {"site": site_slug, "url": site["url"], "provider": client.provider, "model": client.model,
           "run": run, "variant": "default" if system == SYSTEM_PROMPT else "no_length_limits"}
    started = time.perf_counter()
    try:
        result = await client.generate_json(system, build_user_prompt(site["url"], site["context"]), LLMAnalysis)
    except AppError as exc:
        return {**row, "error": f"{exc.code}: {exc.message}", "latency_ms": int((time.perf_counter() - started) * 1000)}
    row.update(
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=estimate_cost(client.model, result.input_tokens, result.cached_input_tokens, result.output_tokens),
        truncated=result.truncated,
    )
    try:
        analysis = LLMAnalysis.model_validate(result.data)
        row["schema_valid"] = True
        row["rule_violations"] = check_business_rules(analysis)
    except ValidationError as exc:
        row["schema_valid"] = False
        row["rule_violations"] = [f"schema: {e['msg']}" for e in exc.errors()[:3]]
    if result.data is not None:
        path = OUTPUTS / f"{site_slug}__{client.model}__{row['variant']}__run{run}.json"
        path.write_text(json.dumps(result.data, indent=2, ensure_ascii=False))
        row["output_path"] = str(path)
    return row


def summarize(rows: list[dict]) -> str:
    lines = [
        f"Prices: official pricing pages, retrieved {PRICES_RETRIEVED}. All other numbers measured in this run.",
        "",
        "| Model | Calls | Errors | Schema valid | Rules clean (1st try) | p50 latency (s) | Avg input tok | Avg cached tok | Avg output tok | Avg cost / analysis | Cost / 1,000 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model and r["variant"] == "default" and r["site"] != "injection-fixture"]
        ok = [r for r in rs if "error" not in r]
        if not rs:
            continue
        valid = sum(1 for r in ok if r.get("schema_valid"))
        clean = sum(1 for r in ok if r.get("schema_valid") and not r.get("rule_violations"))
        mean = lambda k: statistics.mean(r[k] for r in ok) if ok else 0  # noqa: E731
        costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
        avg_cost = statistics.mean(costs) if costs else None
        lines.append(
            f"| {model} | {len(rs)} | {len(rs) - len(ok)} | {valid}/{len(ok)} | {clean}/{len(ok)} | "
            f"{statistics.median(r['latency_ms'] for r in ok) / 1000:.1f} | {mean('input_tokens'):,.0f} | "
            f"{mean('cached_input_tokens'):,.0f} | {mean('output_tokens'):,.0f} | "
            f"{'$%.5f' % avg_cost if avg_cost is not None else 'n/a'} | "
            f"{'$%.2f' % (avg_cost * 1000) if avg_cost is not None else 'n/a'} |"
        )
    return "\n".join(lines)


async def count_tokens(text: str, settings: Settings) -> tuple[int, str]:
    """Free provider token counters; falls back to chars/4 (clearly labelled)."""
    if settings.anthropic_api_key:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        r = await client.messages.count_tokens(model="claude-haiku-5-5", messages=[{"role": "user", "content": text}])
        return r.input_tokens, "anthropic count_tokens"
    if settings.gemini_api_key:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        r = await client.aio.models.count_tokens(model="gemini-3.1-flash-lite", contents=text)
        return r.total_tokens, "gemini count_tokens"
    return len(text) // 4, "estimate (chars/4)"


async def ablation(sites: dict[str, dict], rows: list[dict], first_model: str, settings: Settings) -> dict:
    out: dict = {"cleaning": [], "single_vs_per_section": None, "output_budget": [], "projection": {}}
    for s, site in sites.items():
        if s == "injection-fixture" or not site["home_html"]:
            continue
        raw_tokens, method = await count_tokens(site["home_html"], settings)
        visible = BeautifulSoup(site["home_html"], "lxml").get_text(" ")
        visible_tokens, _ = await count_tokens(visible, settings)
        context_tokens, _ = await count_tokens(site["context"], settings)
        out["cleaning"].append({"site": s, "raw_html_tokens": raw_tokens, "visible_text_tokens": visible_tokens,
                                "context_tokens": context_tokens, "method": method,
                                "note": "raw/visible = homepage only; context = homepage + sub-pages, capped"})

    system_tokens, method = await count_tokens(SYSTEM_PROMPT, settings)
    first = [r for r in rows if r["model"] == first_model and r["variant"] == "default" and "error" not in r]
    if first:
        avg_in = statistics.mean(r["input_tokens"] for r in first)
        price = PRICES_PER_MTOK.get(first_model)
        out["single_vs_per_section"] = {
            "model": first_model, "measured_avg_input_tokens_single_call": round(avg_in),
            "calculated_input_tokens_five_calls": round(avg_in * 5),
            "calculated_extra_cost_per_analysis_usd": round(avg_in * 4 * price.input / 1e6, 6) if price else None,
            "basis": "calculated from measured tokens: 5 section calls each resend the site context and system prompt",
            "system_prompt_tokens": system_tokens, "token_count_method": method,
        }
        no_limit = [r for r in rows if r["model"] == first_model and r["variant"] == "no_length_limits" and "error" not in r]
        for r in no_limit:
            base = next((b for b in first if b["site"] == r["site"]), None)
            if base:
                out["output_budget"].append({
                    "site": r["site"], "output_tokens_with_limits": base["output_tokens"],
                    "output_tokens_without_limits": r["output_tokens"],
                    "cost_with_limits": base["cost_usd"], "cost_without_limits": r["cost_usd"]})
    for model in dict.fromkeys(r["model"] for r in rows):
        costs = [r["cost_usd"] for r in rows if r["model"] == model and r["variant"] == "default"
                 and r.get("cost_usd") is not None and r["site"] != "injection-fixture"]
        if costs:
            m = statistics.mean(costs)
            out["projection"][model] = {"per_analysis": round(m, 6), "per_1000": round(m * 1000, 2),
                                        "per_10000": round(m * 10000, 2), "basis": "projection from measured mean"}
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="provider:model, e.g. anthropic:claude-haiku-5-5")
    parser.add_argument("--urls", default="eval/urls.txt")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--ablation", action="store_true", help="also measure each cost lever")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip() and not u.startswith("#")]
    base = Settings()
    print(f"Building contexts for {len(urls)} sites ...")
    sites = await build_contexts(urls, base)

    clients = []
    for spec in args.models:
        try:
            clients.append(get_llm_client(model_settings(spec)))
        except AppError as exc:
            print(f"  ! skipping {spec}: {exc.message}", file=sys.stderr)

    rows: list[dict] = []
    for client in clients:
        for run in range(1, args.runs + 1):
            for s, site in sites.items():
                print(f"  {client.model} run {run} {s}")
                rows.append(await run_one(client, s, site, run))
        if args.ablation and client is clients[0]:
            for s in [k for k in sites if k != "injection-fixture"][:2]:
                rows.append(await run_one(client, s, sites[s], 1, system=SYSTEM_PROMPT_NO_LIMITS))

    (RESULTS / "benchmark.json").write_text(json.dumps(rows, indent=2))
    table = summarize(rows)
    (RESULTS / "benchmark.md").write_text(table + "\n")
    print("\n" + table)

    if args.ablation and clients:
        result = await ablation(sites, rows, clients[0].model, base)
        (RESULTS / "cost_ablation.json").write_text(json.dumps(result, indent=2))
        print("\nCost ablation written to results/cost_ablation.json")


if __name__ == "__main__":
    asyncio.run(main())
