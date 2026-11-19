"""Runtime hallucination guard for services.

If the LLM labels a service "website_content", the words of that service name
should actually appear in the text we sent it. If they don't, we relabel the
service "inferred" instead of passing an unsupported claim to the user.
"""

import re

from app.schemas import Service

_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "by", "at", "from", "or", "your", "our",
    "we", "you", "is", "are", "be", "as", "it", "its", "this", "that", "&",
    # generic business words that would match any site
    "service", "solution", "company", "professional", "best", "business", "online", "custom", "expert",
}


def _stem(word: str) -> str:
    # Light plural stemming so "audits" matches "audit"; enough for short service names.
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def significant_tokens(text: str) -> set[str]:
    words = (_stem(w) for w in re.findall(r"[^\W_]+", text.lower()))
    return {w for w in words if w not in _STOPWORDS}


def ground_services(services: list[Service], context: str) -> tuple[list[Service], int]:
    vocabulary = significant_tokens(context)
    out: list[Service] = []
    relabeled = 0
    for service in services:
        if service.source == "website_content":
            tokens = significant_tokens(service.name)
            found = sum(1 for t in tokens if t in vocabulary)
            if not tokens or found / len(tokens) < 0.5:
                service = service.model_copy(update={"source": "inferred"})
                relabeled += 1
        out.append(service)
    return out, relabeled
