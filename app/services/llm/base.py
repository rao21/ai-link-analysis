"""Provider-agnostic LLM interface.

Every adapter exposes the same `generate_json(system, user, schema)` call, so the
analysis code never imports a provider SDK and the benchmark can swap providers
with one setting. Each adapter uses the provider's native structured-output mode
(JSON-schema constrained decoding), so the reply is valid JSON of the right shape.
"""

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from app.config import Settings
from app.errors import LLMOutputError, LLMUnavailableError


@dataclass
class LLMResult:
    data: dict | None              # parsed JSON, or None if the output was cut off
    input_tokens: int              # all input tokens, cached ones included
    cached_input_tokens: int       # input tokens served from the provider's prompt cache
    output_tokens: int             # visible output + reasoning tokens (both billed as output)
    latency_ms: int
    truncated: bool = False        # provider stopped at the max-output-tokens limit


class LLMClient(Protocol):
    provider: str
    model: str

    async def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult: ...


def get_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    key = {
        "gemini": settings.gemini_api_key,
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
    }[provider]
    if key is None or not key.get_secret_value():
        raise LLMUnavailableError(f"LLM API key not configured for provider '{provider}'.")

    if provider == "gemini":
        from app.services.llm.gemini import GeminiClient
        return GeminiClient(settings, key.get_secret_value())
    if provider == "anthropic":
        from app.services.llm.anthropic import AnthropicClient
        return AnthropicClient(settings, key.get_secret_value())
    from app.services.llm.openai import OpenAIClient
    return OpenAIClient(settings, key.get_secret_value())


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema -> the strict subset all three providers accept.

    - `$ref`/`$defs` inlined (some providers don't resolve references)
    - every object: `additionalProperties: false` and every property `required`
      (OpenAI strict mode demands both; nullable fields stay nullable via anyOf)
    - `const` -> single-value `enum`; `default` and `title` dropped (titles are
      just extra input tokens on every request)
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        out = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if "const" in out:
            out["enum"] = [out.pop("const")]
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(raw)


def parse_json(text: str | None) -> dict:
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError as exc:
        raise LLMOutputError("The AI returned malformed JSON.") from exc
    if not isinstance(data, dict):
        raise LLMOutputError("The AI returned JSON that is not an object.")
    return data


class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *_: object) -> None:
        self.ms = int((time.perf_counter() - self._start) * 1000)
