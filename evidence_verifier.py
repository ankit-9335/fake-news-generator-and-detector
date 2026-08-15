"""NewsMorph evidence retrieval and claim verification.

This is an evidence-assisted demo, not a replacement for professional fact-checking.
It searches Google News RSS for the claim, then uses an NLI model to estimate whether
retrieved snippets support or contradict the claim.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    response = requests.get(
        RSS_URL,
        params=params,
        timeout=12,
        headers={"User-Agent": "NewsMorph/1.0"},
    )
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


@lru_cache(maxsize=1)
def load_nli():
    """Load the NLI checkpoint once and reuse it for later claims."""
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def nli_scores(premise: str, hypothesis: str) -> dict:
    """Return contradiction/entailment/neutral probabilities for a sentence pair."""
    tokenizer, model, device = load_nli()

    features = tokenizer(
        premise,
        hypothesis,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    features = {key: value.to(device) for key, value in features.items()}

    with torch.no_grad():
        logits = model(**features).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    # This checkpoint defines: 0=contradiction, 1=entailment, 2=neutral.
    return {
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
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

    scored = []
    for article in articles:
        evidence = f"{article['title']}. {article['description']}".strip()
        scores = nli_scores(claim, evidence)
        scored.append({**article, **scores})

    # Reward multiple independent sources instead of one article.
    support = sum(max(0.0, a["entailment"] - a["contradiction"]) for a in scored)
    contradiction = sum(max(0.0, a["contradiction"] - a["entailment"]) for a in scored)

    support_domains = len({
        a["domain"] for a in scored if a["entailment"] > a["contradiction"]
    })
    contradiction_domains = len({
        a["domain"] for a in scored if a["contradiction"] > a["entailment"]
    })

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
