import logging

import openai
from pydantic import BaseModel

from app.config import Settings
from app.errors import LLMRefusedError, LLMUnavailableError
from app.services.llm.base import LLMResult, Timer, parse_json, strict_json_schema

logger = logging.getLogger("link_analysis.llm")


class OpenAIClient:
    provider = "openai"

    def __init__(self, settings: Settings, api_key: str) -> None:
        self.model = settings.resolved_model
        self._settings = settings
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=settings.llm_timeout_s, max_retries=1)

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult:
        try:
            with Timer() as t:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True},
                    },
                    max_completion_tokens=self._settings.max_output_tokens,
                    reasoning_effort=self._settings.llm_effort,
                    # Same key for every request -> routed to the same cache for the shared prefix.
                    prompt_cache_key="link-analysis-v1",
                )
        except openai.APITimeoutError as exc:
            raise LLMUnavailableError("The AI provider (openai) timed out.") from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMUnavailableError("The AI provider (openai) rejected the API key.") from exc
        except openai.RateLimitError as exc:
            raise LLMUnavailableError("The AI provider (openai) rate limit was reached; try again shortly.") from exc
        except openai.APIStatusError as exc:
            logger.warning("openai API error %s: %s", exc.status_code, exc.message)
            raise LLMUnavailableError(f"The AI provider (openai) returned an error ({exc.status_code}).") from exc
        except openai.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the AI provider (openai).") from exc
        return to_result(response, t.ms)


def to_result(response, latency_ms: int) -> LLMResult:
    choice = response.choices[0]
    if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
        raise LLMRefusedError("The AI provider declined to analyze this website's content.")
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    truncated = choice.finish_reason == "length"
    return LLMResult(
        data=None if truncated else parse_json(choice.message.content),
        input_tokens=usage.prompt_tokens,  # includes cached tokens
        cached_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        output_tokens=usage.completion_tokens,  # includes reasoning tokens
        latency_ms=latency_ms,
        truncated=truncated,
    )
