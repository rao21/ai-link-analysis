from pydantic import BaseModel


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
