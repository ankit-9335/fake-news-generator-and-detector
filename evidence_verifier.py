"""NewsMorph evidence retrieval and claim verification.

Evidence-assisted demo, not a replacement for professional fact-checking.
It combines recent Google News results with a general-knowledge fallback from
Wikipedia, then uses an NLI model to compare retrieved evidence with the claim.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from urllib.parse import quote, urlparse

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RSS_URL = "https://news.google.com/rss/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

DEATH_TERMS = (
    "shot dead", "killed", "dead", "died", "assassinated", "murdered",
    "has died", "was killed", "was dead"
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "and", "or",
    "as", "that", "this", "these", "those", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "has", "have", "had", "will", "would",
    "can", "could", "may", "might", "do", "does", "did", "not", "all", "any",
    "about", "after", "before", "into", "than", "then", "there", "here",
}


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", " ", text).strip()


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown source"


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def relevance_score(claim: str, evidence: str) -> float:
    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return len(overlap) / len(claim_tokens)


def _rss_search(query: str, max_results: int = 8) -> list[dict]:
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    response = requests.get(
        RSS_URL, params=params, timeout=12,
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
                "title": title, "description": description, "url": link,
                "source": source, "domain": domain(link),
                "published": pub_date, "evidence_type": "news",
            })
    return items


def _wiki_search(claim: str, max_results: int = 5) -> list[dict]:
    """Fallback for evergreen/general facts when current news is unsuitable."""
    params = {
        "action": "query", "list": "search", "srsearch": claim,
        "format": "json", "utf8": 1, "srlimit": max_results,
        "srprop": "snippet",
    }
    response = requests.get(
        WIKI_API, params=params, timeout=10,
        headers={"User-Agent": "NewsMorph/1.0 educational project"},
    )
    response.raise_for_status()
    data = response.json()

    items = []
    for result in data.get("query", {}).get("search", []):
        title = clean_html(result.get("title", ""))
        snippet = clean_html(result.get("snippet", ""))
        if not title or not snippet:
            continue
        items.append({
            "title": title,
            "description": snippet,
            "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
            "source": "Wikipedia",
            "domain": "wikipedia.org",
            "published": "General knowledge source",
            "evidence_type": "knowledge",
        })
    return items


def _dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        key = item["url"] or item["title"].lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _extract_subject(claim: str) -> str | None:
    title_match = re.search(
        r"(?:Prime Minister|President|Chief Minister|Minister|PM)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", claim,
    )
    if title_match:
        return title_match.group(1).strip()
    proper_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim)
    return max(proper_phrases, key=len).strip() if proper_phrases else None


def search_news(claim: str, max_results: int = 8) -> list[dict]:
    queries = [f'"{claim}" when:7d', f"{claim} when:7d"]
    lower_claim = claim.lower()
    if any(term in lower_claim for term in DEATH_TERMS):
        subject = _extract_subject(claim)
        if subject:
            queries.extend([
                f'"{subject}" speech when:7d',
                f'"{subject}" latest when:7d',
                f'"{subject}" killed when:7d',
            ])

    articles: list[dict] = []
    for query in queries:
        try:
            articles.extend(_rss_search(query, max_results=max_results))
        except Exception:
            continue

    articles = _dedupe(articles)
    relevant_news = [
        a for a in articles if relevance_score(
            claim, f"{a['title']} {a['description']}"
        ) >= 0.20
    ]

    # Current news is the wrong source for evergreen facts. If there are not
    # enough relevant news results, supplement with general-knowledge evidence.
    if len(relevant_news) < 2:
        try:
            wiki_items = _wiki_search(claim, max_results=5)
            articles.extend(
                a for a in wiki_items
                if relevance_score(claim, f"{a['title']} {a['description']}") >= 0.20
            )
        except Exception:
            pass

    return _dedupe(articles)[: max_results * 2]


@lru_cache(maxsize=1)
def load_nli():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def nli_scores(premise: str, hypothesis: str) -> dict:
    tokenizer, model, device = load_nli()
    features = tokenizer(
        premise, hypothesis, padding=True, truncation=True,
        max_length=256, return_tensors="pt",
    )
    features = {key: value.to(device) for key, value in features.items()}
    with torch.no_grad():
        logits = model(**features).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return {
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
    }


def _source_weight(article: dict) -> float:
    if article.get("evidence_type") == "knowledge":
        return 0.90
    domain_name = article["domain"].lower()
    trusted = (
        "reuters.com", "apnews.com", "bbc.com", "thehindu.com",
        "indianexpress.com", "ndtv.com", "hindustantimes.com",
        "pib.gov.in", "pmo.gov.in", "gov.in"
    )
    return 1.25 if any(domain_name.endswith(d) for d in trusted) else 1.0


def verify_claim(claim: str, max_results: int = 8) -> dict:
    articles = search_news(claim, max_results=max_results)
    if not articles:
        return {
            "verdict": "NO EVIDENCE FOUND", "confidence": 0.0,
            "articles": [], "support": 0.0, "contradiction": 0.0,
        }

    scored = []
    for article in articles:
        evidence = f"{article['title']}. {article['description']}".strip()
        relevance = relevance_score(claim, evidence)
        if relevance < 0.20:
            continue
        scores = nli_scores(evidence, claim)
        scored.append({
            **article, **scores, "relevance": relevance,
            "source_weight": _source_weight(article),
        })

    if not scored:
        return {
            "verdict": "UNCERTAIN", "confidence": 0.0,
            "articles": articles[:max_results], "support": 0.0,
            "contradiction": 0.0,
        }

    support = 0.0
    contradiction = 0.0
    for article in scored:
        if article["entailment"] >= 0.55 and article["entailment"] > article["contradiction"]:
            support += (
                (article["entailment"] - article["contradiction"])
                * article["relevance"] * article["source_weight"]
            )
        elif article["contradiction"] >= 0.55 and article["contradiction"] > article["entailment"]:
            contradiction += (
                (article["contradiction"] - article["entailment"])
                * article["relevance"] * article["source_weight"]
            )

    support_domains = len({
        a["domain"] for a in scored
        if a["entailment"] >= 0.55 and a["entailment"] > a["contradiction"]
    })
    contradiction_domains = len({
        a["domain"] for a in scored
        if a["contradiction"] >= 0.55 and a["contradiction"] > a["entailment"]
    })

    support_score = support * (1 + min(support_domains, 4) * 0.15)
    contradiction_score = contradiction * (1 + min(contradiction_domains, 4) * 0.15)

    if any(term in claim.lower() for term in DEATH_TERMS):
        subject = _extract_subject(claim)
        if subject:
            status_articles = []
            for query in (f'"{subject}" speech when:7d', f'"{subject}" latest when:7d'):
                try:
                    status_articles.extend(_rss_search(query, max_results=4))
                except Exception:
                    pass
            for article in _dedupe(status_articles):
                evidence = f"{article['title']}. {article['description']}".strip()
                relevance = relevance_score(claim, evidence)
                if relevance < 0.20:
                    continue
                scores = nli_scores(evidence, claim)
                if scores["contradiction"] >= 0.55 and scores["contradiction"] > scores["entailment"]:
                    contradiction_score += (
                        scores["contradiction"] - scores["entailment"]
                    ) * relevance * _source_weight(article) * 1.5

    total = support_score + contradiction_score
    if total < 0.35:
        verdict = "UNCERTAIN"
    elif contradiction_score > support_score * 1.20 and contradiction_domains >= 1:
        verdict = "LIKELY FALSE"
    elif support_score > contradiction_score * 1.20 and support_domains >= 1:
        verdict = "LIKELY TRUE"
    else:
        verdict = "UNCERTAIN"

    confidence = 0.0 if total == 0 else abs(support_score - contradiction_score) / total
    return {
        "verdict": verdict, "confidence": confidence,
        "articles": sorted(
            scored,
            key=lambda a: a["relevance"] * max(a["entailment"], a["contradiction"]),
            reverse=True,
        )[:max_results],
        "support": support_score, "contradiction": contradiction_score,
    }
