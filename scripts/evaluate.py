"""Accuracy evaluation against a hand-written gold set.

    python -m scripts.evaluate                 # automatic metrics -> results/eval.json + table
    python -m scripts.evaluate --blind         # write anonymised outputs to eval/blind/ for human scoring
    python -m scripts.evaluate --unblind       # join eval/blind/scores.csv back to models

Gold files (eval/gold/<site>.json) are written by a human who read the site,
BEFORE looking at any model output. See eval/README.md.
"""

import argparse
import csv
import json
import random
import socket
import statistics
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from scripts.eval_metrics import (
    brand_correct,
    injection_resisted,
    location_correct,
    seo_format_ok,
    services_precision,
    services_recall,
    stability,
)

RESULTS = Path("results")
GOLD = Path("eval/gold")
BLIND = Path("eval/blind")


def _domain_resolves(url: str | None) -> bool | None:
    if not url:
        return None
    host = urlsplit(url).hostname
    try:
        socket.getaddrinfo(host, 443)
        return True
    except (socket.gaierror, UnicodeError):
        return False


def _rows() -> list[dict]:
    rows = json.loads((RESULTS / "benchmark.json").read_text())
    return [r for r in rows if r.get("output_path") and r.get("variant") == "default"]


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def evaluate() -> None:
    per_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    services_by_run: dict[tuple[str, str], list[list[str]]] = defaultdict(list)

    for row in _rows():
        out = json.loads(Path(row["output_path"]).read_text())
        m = per_model[row["model"]]
        services = [(s["name"], s["source"]) for s in out.get("top_services", [])]
        brand = out.get("brand", {}).get("name", "")
        services_by_run[(row["model"], row["site"])].append([n for n, _ in services])

        if row["site"] == "injection-fixture":
            m["injection_resisted"].append(injection_resisted(brand, [n for n, _ in services]))
            continue
        gold_path = GOLD / f"{row['site']}.json"
        if not gold_path.exists():
            continue
        gold = json.loads(gold_path.read_text())
        m["schema_valid"].append(row.get("schema_valid", False))
        m["rules_clean"].append(bool(row.get("schema_valid")) and not row.get("rule_violations"))
        m["brand_correct"].append(brand_correct(brand, gold["brand_name"]))
        m["services_recall"].append(services_recall([n for n, _ in services], gold["services"]))
        m["services_precision"].append(services_precision(services, gold["services"]))
        m["location_correct"].append(location_correct(out.get("target_audience", {}).get("geographic_focus"), gold.get("location")))
        seo = out.get("seo_improvements", {})
        m["seo_format"].append(seo_format_ok(seo.get("recommended_title", ""), seo.get("primary_keyword", ""), seo.get("meta_description", "")))
        for c in out.get("competitor_analysis", {}).get("competitors", []):
            m["competitor_domain_resolves"].append(_domain_resolves(c.get("website")))

    for (model, _site), runs in services_by_run.items():
        per_model[model]["stability"].append(stability(runs))

    report = {model: {k: _mean(v) for k, v in metrics.items()} | {"n_outputs": len(metrics["schema_valid"])}
              for model, metrics in per_model.items()}
    (RESULTS / "eval.json").write_text(json.dumps(report, indent=2))

    cols = ["n_outputs", "schema_valid", "rules_clean", "brand_correct", "services_recall", "services_precision",
            "location_correct", "seo_format", "competitor_domain_resolves", "stability", "injection_resisted"]
    print("| Model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for model, values in report.items():
        print(f"| {model} | " + " | ".join("n/a" if values.get(c) is None else str(values.get(c)) for c in cols) + " |")


def _render(out: dict) -> str:
    ca, seo, aud = out["competitor_analysis"], out["seo_improvements"], out["target_audience"]
    lines = [f"# {out['brand']['name']}", out["brand"]["summary"], "", "## Competitors"]
    lines += [f"- **{c['name']}** ({c.get('website') or 'no site'}): {c['description']}" for c in ca["competitors"]]
    lines += ["", "## Search queries"] + [f"- {q['query']} ({', '.join(q['platforms'])}) - {q['intent']}" for q in out["target_search_queries"]]
    lines += ["", "## SEO", f"- Title: {seo['recommended_title']}", f"- Meta: {seo['meta_description']}",
              f"- Primary keyword: {seo['primary_keyword']}", f"- Long-tail: {', '.join(seo['long_tail_keywords'])}"]
    lines += [f"- Content idea: {c['title']}" for c in seo["supporting_content"]]
    lines += ["", "## Audience", f"- Primary: {', '.join(aud['primary_audience'])}", f"- Needs: {', '.join(aud['customer_needs'])}"]
    return "\n".join(lines)


def blind() -> None:
    BLIND.mkdir(parents=True, exist_ok=True)
    key, seen = {}, set()
    rows = [r for r in _rows() if r["site"] != "injection-fixture"]
    random.shuffle(rows)
    for row in rows:
        if (row["site"], row["model"]) in seen:
            continue  # score the first run per (site, model)
        seen.add((row["site"], row["model"]))
        item_id = f"{random.randrange(16**6):06x}"
        key[item_id] = {"site": row["site"], "model": row["model"]}
        out = json.loads(Path(row["output_path"]).read_text())
        (BLIND / f"{item_id}.md").write_text(f"Site: {row['url']}\n\n{_render(out)}\n")
    (BLIND / "_key.json").write_text(json.dumps(key, indent=2))
    with (BLIND / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "site", "competitors_1to5", "seo_specific_1to5", "audience_fit_1to5", "queries_realistic_1to5"])
        for item_id, v in sorted(key.items(), key=lambda kv: kv[1]["site"]):
            w.writerow([item_id, v["site"], "", "", "", ""])
    print(f"Wrote {len(key)} anonymised outputs to {BLIND}/. Score them in scores.csv WITHOUT opening _key.json.")


def unblind() -> None:
    key = json.loads((BLIND / "_key.json").read_text())
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (BLIND / "scores.csv").open() as f:
        for row in csv.DictReader(f):
            model = key[row["id"]]["model"]
            for col in ["competitors_1to5", "seo_specific_1to5", "audience_fit_1to5", "queries_realistic_1to5"]:
                if row[col].strip():
                    scores[model][col].append(float(row[col]))
    report = {m: {c: _mean(v) for c, v in cols.items()} for m, cols in scores.items()}
    (RESULTS / "human_scores.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--unblind", action="store_true")
    args = parser.parse_args()
    blind() if args.blind else unblind() if args.unblind else evaluate()
