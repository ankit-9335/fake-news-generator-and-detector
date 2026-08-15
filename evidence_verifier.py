"""NewsMorph evidence retrieval and claim verification.

This is an evidence-assisted demo, not a replacement for professional fact-checking.
It searches Google News RSS for the claim, then uses an NLI model to estimate whether
retrieved snippets support or contradict the claim.
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET

import requests
from transformers import pipeline

RSS_URL = "https://news.google.com/rss/search"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", " ", text).strip()


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown source"


def search_news(claim: str, max_results: int = 8) -> list[dict]:
    params = {
        "q": claim,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    }
    response = requests.get(RSS_URL, params=params, timeout=12,
                            headers={"User-Agent": "NewsMorph/1.0"})
    response.raise_for_status()

    root = ET.fromstring(response.content)
    items = []
    for item in root.findall("./channel/item")[:max_results]:
        title = clean_html(item.findtext("title", ""))
        link = item.findtext("link", "")
        description = clean_html(item.findtext("description", ""))
        pub_date = item.findtext("pubDate", "")
        source_el = item.find("source")
        source = (source_el.text or "") if source_el is not None else domain(link)

        if title and link:
            items.append({
                "title": title,
                "description": description,
                "url": link,
                "source": source,
                "domain": domain(link),
                "published": pub_date,
            })
    return items


def load_nli():
    return pipeline(
        "text-classification",
        model=NLI_MODEL,
        tokenizer=NLI_MODEL,
        top_k=None,
    )


def _nli_scores(nli_output):
    if nli_output and isinstance(nli_output[0], list):
        nli_output = nli_output[0]
    scores = {x["label"].lower(): float(x["score"]) for x in nli_output}

    # The MiniLM NLI checkpoint exposes labels such as entailment,
    # contradiction and neutral. Handle common capitalization variants.
    return {
        "entailment": scores.get("entailment", 0.0),
        "contradiction": scores.get("contradiction", 0.0),
        "neutral": scores.get("neutral", 0.0),
    }


def verify_claim(claim: str, max_results: int = 8) -> dict:
    articles = search_news(claim, max_results=max_results)
    if not articles:
        return {
            "verdict": "NO EVIDENCE FOUND",
            "confidence": 0.0,
            "articles": [],
            "support": 0.0,
            "contradiction": 0.0,
        }

    nli = load_nli()
    scored = []
    for article in articles:
        evidence = f"{article['title']}. {article['description']}".strip()

        # Claim -> evidence: entailment means the evidence is consistent with the claim;
        # contradiction means the evidence conflicts with it.
        result = nli(f"{claim}", evidence, truncation=True, max_length=256)
        scores = _nli_scores(result)
        article = {**article, **scores}
        scored.append(article)

    # Reward multiple independent sources instead of one article.
    support = sum(max(0.0, a["entailment"] - a["contradiction"]) for a in scored)
    contradiction = sum(max(0.0, a["contradiction"] - a["entailment"]) for a in scored)

    support_domains = len({a["domain"] for a in scored if a["entailment"] > a["contradiction"]})
    contradiction_domains = len({a["domain"] for a in scored if a["contradiction"] > a["entailment"]})

    support_score = support * (1 + min(support_domains, 4) * 0.15)
    contradiction_score = contradiction * (1 + min(contradiction_domains, 4) * 0.15)
    total = support_score + contradiction_score

    if total < 0.9:
        verdict = "UNCERTAIN"
    elif contradiction_score > support_score * 1.20 and contradiction_domains >= 1:
        verdict = "LIKELY FALSE"
    elif support_score > contradiction_score * 1.20 and support_domains >= 1:
        verdict = "LIKELY TRUE"
    else:
        verdict = "UNCERTAIN"

    confidence = 0.0 if total == 0 else abs(support_score - contradiction_score) / total

    return {
        "verdict": verdict,
        "confidence": confidence,
        "articles": sorted(
            scored,
            key=lambda a: max(a["entailment"], a["contradiction"]),
            reverse=True,
        ),
        "support": support_score,
        "contradiction": contradiction_score,
    }
