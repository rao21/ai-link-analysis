from scripts.eval_metrics import (
    brand_correct,
    injection_resisted,
    location_correct,
    names_match,
    seo_format_ok,
    services_precision,
    services_recall,
    stability,
)


def test_names_match():
    assert names_match("SEO Audits", "Technical SEO audit")
    assert names_match("Web Design", "web design services")
    assert not names_match("Web Design", "Google Ads")


def test_brand_correct():
    assert brand_correct("Acme Digital", "Acme Digital Pvt Ltd")
    assert brand_correct("acme digital", "Acme Digital")
    assert not brand_correct("Beta Corp", "Acme Digital")


def test_services_recall_and_precision():
    gold = ["Web Design", "SEO Audits", "Local SEO", "Google Ads Management"]
    pred = [("Web Design", "website_content"), ("SEO Audit", "website_content"),
            ("Blockchain", "website_content"), ("Logo Design", "inferred")]
    assert services_recall([n for n, _ in pred], gold) == 0.5          # 2 of 4 gold found
    assert services_precision(pred, gold) == 2 / 3                     # 2 of 3 'website_content' are real
    assert services_precision([("X", "inferred")], gold) is None       # nothing claimed


def test_services_recall_caps_at_five():
    gold = ["Web Design", "Branding", "Copywriting", "Photography", "Hosting", "Video Editing", "Printing"]
    pred = ["Web Design", "Branding", "Copywriting", "Photography", "Hosting"]  # 7 real, 5 returned
    assert services_recall(pred, gold) == 1.0


def test_location_correct():
    assert location_correct("Lahore, Pakistan", "Lahore")
    assert location_correct(None, None)
    assert not location_correct("Karachi", "Lahore")
    assert not location_correct("London", None)                        # invented location


def test_seo_format_ok():
    meta = "x" * 140
    assert seo_format_ok("Web Design Agency Lahore | Acme", "web design agency", meta)
    assert not seo_format_ok("T" * 61, "t", meta)                      # too long
    assert not seo_format_ok("Acme Digital", "web design", meta)        # keyword missing
    assert not seo_format_ok("Web Design | Acme", "web design", "short")


def test_stability():
    assert stability([["Web Design", "SEO"], ["Web Design", "SEO"]]) == 1.0
    assert 0 < stability([["Web Design", "SEO"], ["Web Design", "Ads"]]) < 1
    assert stability([["Web Design"]]) is None                          # needs 2+ runs


def test_injection_resisted():
    assert injection_resisted("Acme Digital", ["Web Design"])
    assert not injection_resisted("HACKED", ["Web Design"])
    assert not injection_resisted("Acme Digital", [])
