from pathlib import Path

import pytest

from app.services.website_extractor import (
    build_llm_context,
    content_chars,
    extract_content,
    select_subpages,
)

FIXTURES = Path(__file__).parent / "fixtures"
REPEATED = "We have helped more than 120 local businesses grow online."


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def sample():
    return extract_content(read("sample_site.html"), "https://acme.test/")


@pytest.fixture
def services():
    return extract_content(read("services_page.html"), "https://acme.test/services")


# ---------- metadata ----------

def test_metadata(sample):
    assert sample.title.startswith("Acme Digital")
    assert sample.meta_description is None
    assert sample.og["og:site_name"] == "Acme Digital"


def test_json_ld_keeps_business_types_only(sample):
    assert len(sample.json_ld) == 1
    assert sample.json_ld[0]["name"] == "Acme Digital"
    assert sample.json_ld[0]["address"]["addressLocality"] == "Lahore"


def test_json_ld_graph_and_bad_json():
    html = """<html><head>
    <script type="application/ld+json">{"@graph": [{"@type": "LocalBusiness", "name": "Graph Co"}, {"@type": "WebPage"}]}</script>
    <script type="application/ld+json">{not valid json</script>
    </head><body><p>Some body text here.</p></body></html>"""
    page = extract_content(html, "https://g.test/")
    assert [item["name"] for item in page.json_ld] == ["Graph Co"]


# ---------- cleaning ----------

def test_noise_removed(sample):
    assert "TRACKING_PIXEL" not in sample.body_text
    assert "Subscribe to newsletter" not in sample.body_text  # forms removed


def test_hidden_text_removed(sample):
    for marker in ["IGNORE PREVIOUS", "SECRET_HIDDEN", "ARIA_HIDDEN", "ZERO_FONT"]:
        assert marker not in sample.body_text


def test_duplicates_removed(sample):
    assert sample.body_text.count(REPEATED) == 1


def test_inline_tags_do_not_split_sentences(sample):
    assert "runs SEO campaigns for small businesses and startups in Lahore." in sample.body_text


def test_short_service_items_kept_single_words_dropped(sample):
    lines = sample.body_text.splitlines()
    assert "Local SEO" in lines
    assert "Google Ads Management" in lines
    assert "Menu" not in lines


def test_header_hero_kept_nav_removed(sample):
    assert "Acme Digital builds fast websites" in sample.body_text
    assert "About Us" not in sample.body_text


def test_footer_separate(sample):
    assert "Gulberg" not in sample.body_text
    assert "Gulberg" in sample.footer_text
    assert len(sample.footer_text) <= 500


def test_headings_in_order_deduped(sample):
    assert sample.headings[:3] == ["Websites that bring you customers", "Our Services", "Web Design"]
    assert len(sample.headings) == len(set(sample.headings))


# ---------- SEO audit ----------

def test_seo_audit(sample):
    a = sample.seo_audit
    assert a.title_length == 75
    assert a.h1_count == 2
    assert a.images_total == 4
    assert a.images_missing_alt == 3
    assert a.lang == "en"
    assert a.has_open_graph is True
    assert a.has_canonical is False
    issues = " ".join(a.issues).lower()
    assert "meta description" in issues
    assert "title" in issues and "60" in issues
    assert "h1" in issues
    assert "alt" in issues
    assert "canonical" in issues
    assert "viewport" in issues


def test_seo_audit_clean_page():
    html = """<html lang="en"><head><title>Short Title</title>
    <meta name="description" content="A good description of the business.">
    <meta name="viewport" content="width=device-width"><link rel="canonical" href="https://c.test/">
    </head><body><h1>One</h1><img src="x.png" alt="x"></body></html>"""
    assert extract_content(html, "https://c.test/").seo_audit.issues == []


# ---------- sub-pages ----------

def test_select_subpages(sample):
    assert select_subpages(sample.nav_links, "https://acme.test/", 3) == [
        "https://acme.test/services",
        "https://acme.test/about",
    ]


def test_select_subpages_rules():
    links = [
        ("Our solutions", "https://www.acme.test/solutions#top"),
        ("Solutions", "https://acme.test/solutions"),            # duplicate after fragment strip
        ("Brochure", "https://acme.test/services.pdf"),          # file
        ("Partner services", "https://other.test/services"),     # other host
        ("Home", "https://acme.test/#services"),                 # homepage fragment
        ("Pricing", "https://acme.test/pricing"),
        ("Products", "https://acme.test/products"),
    ]
    assert select_subpages(links, "https://acme.test/", 2) == [
        "https://www.acme.test/solutions",
        "https://acme.test/products",
    ]


# ---------- LLM context ----------

def test_context_order_and_limit(sample, services):
    ctx = build_llm_context(sample, [services], max_chars=24_000)
    order = ["### PAGE METADATA", "### STRUCTURED DATA", "### HEADINGS", "### HOMEPAGE TEXT",
             "### PAGE: https://acme.test/services", "### FOOTER"]
    positions = [ctx.index(label) for label in order]
    assert positions == sorted(positions)
    assert len(build_llm_context(sample, [services], max_chars=500)) <= 500


def test_spa_shell_has_little_content():
    spa = extract_content(read("spa_shell.html"), "https://spa.test/")
    assert content_chars(spa, []) < 300


def _big_page(words: int) -> str:
    sentence = "Our consultants deliver bespoke enterprise transformation programmes worldwide. "
    paragraphs = "".join(f"<p>{i} {sentence * 5}</p>" for i in range(words // 50))
    return f"<html><body><h1>Big</h1>{paragraphs}</body></html>"


def test_huge_text_respects_budget():
    page = extract_content(_big_page(50_000), "https://big.test/")
    assert len(build_llm_context(page, [], 24_000)) <= 24_000


def test_subpage_survives_huge_homepage(services):
    home = extract_content(_big_page(50_000), "https://acme.test/")
    ctx = build_llm_context(home, [services], 24_000)
    assert "E-commerce Websites" in ctx
    assert "Shopify and WooCommerce" in ctx


def test_select_subpages_prefers_shallow_pages():
    links = [
        ("Charges", "https://acme.test/help/billing/about-additional-charges/"),  # too deep
        ("Launch", "https://acme.test/projects/product-launch"),
        ("About", "https://acme.test/about"),
    ]
    assert select_subpages(links, "https://acme.test/", 3) == [
        "https://acme.test/about",
        "https://acme.test/projects/product-launch",
    ]


def test_og_description_not_repeated_when_same_as_meta():
    html = """<html><head><title>T</title><meta name="description" content="Same text here.">
    <meta property="og:description" content="Same text here."></head><body><p>Body text here.</p></body></html>"""
    ctx = build_llm_context(extract_content(html, "https://d.test/"), [], 5000)
    assert ctx.count("Same text here.") == 1
