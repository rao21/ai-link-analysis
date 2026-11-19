from app.schemas import CompetitorAnalysis, CompetitorAnalysisLLM

COMPETITOR_DISCLAIMER = (
    "Competitors are AI-generated suggestions based on the model's general knowledge "
    "and the website's category; verify before use."
)


def _key(name: str) -> str:
    return " ".join(name.lower().split())


def finalize_competitors(analysis: CompetitorAnalysisLLM, brand_name: str) -> CompetitorAnalysis:
    """Drop the brand itself and duplicates, force the AI-suggestion label, add the disclaimer."""
    brand_key = _key(brand_name)
    seen: set[str] = set()
    competitors = []
    for competitor in analysis.competitors:
        key = _key(competitor.name)
        if not key or key == brand_key or key in seen:
            continue
        seen.add(key)
        competitors.append(competitor.model_copy(update={"basis": "ai_suggestion"}))
    data = analysis.model_dump()
    data["competitors"] = [c.model_dump() for c in competitors]
    return CompetitorAnalysis(**data, disclaimer=COMPETITOR_DISCLAIMER)
