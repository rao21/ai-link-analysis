import logging

import anthropic
from pydantic import BaseModel

from app.config import Settings
from app.errors import LLMRefusedError, LLMUnavailableError
from app.services.llm.base import LLMResult, Timer, parse_json, strict_json_schema

logger = logging.getLogger("link_analysis.llm")


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, settings: Settings, api_key: str) -> None:
        self.model = settings.resolved_model
        self._settings = settings
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=settings.llm_timeout_s, max_retries=1)

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult:
        try:
            with Timer() as t:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=self._settings.max_output_tokens,
                    # Static system prompt marked cacheable: repeat requests read it at 0.1x price.
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                    output_config={
                        "effort": self._settings.llm_effort,
                        "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
                    },
                )
        except anthropic.APITimeoutError as exc:
            raise LLMUnavailableError("The AI provider (anthropic) timed out.") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMUnavailableError("The AI provider (anthropic) rejected the API key.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("The AI provider (anthropic) rate limit was reached; try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            logger.warning("anthropic API error %s: %s", exc.status_code, exc.message)
            raise LLMUnavailableError(f"The AI provider (anthropic) returned an error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the AI provider (anthropic).") from exc
        return to_result(response, t.ms)


def to_result(response, latency_ms: int) -> LLMResult:
    if response.stop_reason == "refusal":
        raise LLMRefusedError("The AI provider declined to analyze this website's content.")
    usage = response.usage
    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    truncated = response.stop_reason == "max_tokens"
    text = next((b.text for b in response.content if b.type == "text"), None)
    return LLMResult(
        data=None if truncated else parse_json(text),
        input_tokens=usage.input_tokens + cached + written,  # input_tokens excludes cache reads/writes
        cached_input_tokens=cached,
        output_tokens=usage.output_tokens,
        latency_ms=latency_ms,
        truncated=truncated,
    )
