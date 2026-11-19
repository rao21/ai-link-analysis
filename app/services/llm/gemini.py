import logging

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.config import Settings
from app.errors import LLMRefusedError, LLMUnavailableError
from app.services.llm.base import LLMResult, Timer, parse_json, strict_json_schema

logger = logging.getLogger("link_analysis.llm")

_BLOCKED = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "MODEL_ARMOR"}
_EFFORT = {"low": types.ThinkingLevel.LOW, "medium": types.ThinkingLevel.MEDIUM, "high": types.ThinkingLevel.HIGH}


class GeminiClient:
    provider = "gemini"

    def __init__(self, settings: Settings, api_key: str) -> None:
        self.model = settings.resolved_model
        self._settings = settings
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000)),  # milliseconds
        )

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult:
        config = types.GenerateContentConfig(
            # Gemini caches repeated prompt prefixes implicitly; the system prompt is the stable prefix.
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=strict_json_schema(schema),
            max_output_tokens=self._settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=_EFFORT[self._settings.llm_effort]),
        )
        try:
            with Timer() as t:
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=user, config=config
                )
        except errors.ClientError as exc:
            logger.warning("gemini client error %s: %s", exc.code, exc.message)
            if exc.code in (401, 403):
                raise LLMUnavailableError("The AI provider (gemini) rejected the API key.") from exc
            if exc.code == 429:
                raise LLMUnavailableError("The AI provider (gemini) rate limit was reached; try again shortly.") from exc
            raise LLMUnavailableError(f"The AI provider (gemini) returned an error ({exc.code}).") from exc
        except errors.ServerError as exc:
            raise LLMUnavailableError(f"The AI provider (gemini) returned an error ({exc.code}).") from exc
        except Exception as exc:  # network errors / timeouts surface as httpx or asyncio exceptions
            logger.warning("gemini call failed: %r", exc)
            raise LLMUnavailableError("Could not reach the AI provider (gemini).") from exc
        return to_result(response, t.ms)


def to_result(response, latency_ms: int) -> LLMResult:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMRefusedError("The AI provider declined to analyze this website's content.")
    finish = response.candidates[0].finish_reason.name if response.candidates and response.candidates[0].finish_reason else "STOP"
    if finish in _BLOCKED:
        raise LLMRefusedError("The AI provider declined to analyze this website's content.")
    usage = response.usage_metadata
    truncated = finish == "MAX_TOKENS"
    return LLMResult(
        data=None if truncated else parse_json(response.text),
        input_tokens=usage.prompt_token_count or 0,          # includes cached tokens
        cached_input_tokens=usage.cached_content_token_count or 0,
        output_tokens=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        latency_ms=latency_ms,
        truncated=truncated,
    )
