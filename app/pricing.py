"""LLM token prices (USD per 1M tokens) from the providers' official pricing pages.

Sources (retrieved on PRICES_RETRIEVED):
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- Google:    https://aistudio.google.com/docs/pricing   (redirect from ai.google.dev/gemini-api/docs/pricing)
- OpenAI:    https://developers.openai.com/api/docs/pricing

Notes:
- Claude Haiku 5.5 prices are for prompts up to 100k tokens (ours are ~10k).
- gemini-3.8-flash doubles in price on 2027-01-01.
- Cache writes (Anthropic: 1.25x input) are billed here at the base input rate;
  the system prompt is small, so this under-counts by well under 1%.
"""

from dataclasses import dataclass

PRICES_RETRIEVED = "2026-11-19"


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float


PRICES_PER_MTOK: dict[str, Price] = {
    # Anthropic
    "claude-haiku-5-5": Price(input=0.10, cached_input=0.01, output=0.50),
    "claude-sonnet-5-5": Price(input=2.00, cached_input=0.10, output=10.00),
    # Google
    "gemini-3.1-flash-lite": Price(input=0.25, cached_input=0.025, output=1.50),
    "gemini-2.5-flash-lite": Price(input=0.10, cached_input=0.01, output=0.40),
    "gemini-3.8-flash": Price(input=0.75, cached_input=0.075, output=3.75),
    # OpenAI
    "gpt-6-luna": Price(input=0.10, cached_input=0.01, output=0.50),
    "gpt-5-nano": Price(input=0.05, cached_input=0.005, output=0.40),
    "gpt-5.4-mini": Price(input=0.75, cached_input=0.075, output=4.50),
}


def estimate_cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one call in USD; None when the model's price is unknown."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    uncached = max(0, input_tokens - cached_input_tokens)
    usd = (uncached * price.input + cached_input_tokens * price.cached_input + output_tokens * price.output) / 1_000_000
    return round(usd, 6)
