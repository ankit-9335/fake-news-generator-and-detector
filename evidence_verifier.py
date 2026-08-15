"""NewsMorph evidence retrieval and claim verification.

Evidence-assisted demo, not a replacement for professional fact-checking.
It searches recent Google News RSS results and uses an NLI model to compare
retrieved evidence with the user's claim.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RSS_URL = "https://news.google.com/rss/search"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

# Claims containing these words are especially dangerous to verify from
# semantically related search results. For them we also perform a current-status
# search so that an unrelated "unknown gunmen" story cannot count as support.
DEATH_TERMS = (
    "shot dead", "killed", "dead", "died", "assassinated", "murdered",
    "has died", "was killed", "was dead"
)


def clean_html(text: str) -> str:
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", " ", text).strip()


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown source"


def _rss_search(query: str, max_results: int = 8) -> list[dict]:
    params = {
        "q": query,
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
    """Extract a likely named person/entity for a status search."""
    title_match = re.search(
        r"(?:Prime Minister|President|Chief Minister|Minister|PM)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        claim,
    )
    if title_match:
        return title_match.group(1).strip()

    proper_phrases = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", claim
    )
    if proper_phrases:
        return max(proper_phrases, key=len).strip()
    return None


def search_news(claim: str, max_results: int = 8) -> list[dict]:
    """Search recent news using several queries instead of one loose query."""
    queries = [
        f'"{claim}" when:7d',
        f"{claim} when:7d",
    ]

    # For death/attack claims, search the subject's current activity separately.
    # This is critical for claims such as "X was shot dead" when Google News
    # otherwise returns unrelated stories containing words like "unknown gunmen".
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
            # One failed search should not kill the entire verification attempt.
            continue

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
    """Return contradiction/entailment/neutral probabilities for a pair."""
    tokenizer, model, device = load_nli()

    # IMPORTANT: for NLI, the retrieved article is the premise and the user's
    # claim is the hypothesis. Reversing these can change the result materially.
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

    # cross-encoder/nli-MiniLM2-L6-H768 defines:
    # 0 = contradiction, 1 = entailment, 2 = neutral.
    return {
        "contradiction": float(probabilities[0]),
        "entailment": float(probabilities[1]),
        "neutral": float(probabilities[2]),
    }


def _source_weight(article: dict) -> float:
    """Give modest extra weight to well-established news/official sources."""
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
            "verdict": "NO EVIDENCE FOUND",
            "confidence": 0.0,
            "articles": [],
            "support": 0.0,
            "contradiction": 0.0,
        }

    scored = []
    for article in articles:
        evidence = f"{article['title']}. {article['description']}".strip()
        scores = nli_scores(evidence, claim)
        weight = _source_weight(article)
        scored.append({**article, **scores, "source_weight": weight})

    # Ignore articles where the NLI model sees mostly neutral/unrelated text.
    relevant = [
        a for a in scored
        if max(a["entailment"], a["contradiction"]) >= 0.45
    ]

    if not relevant:
        relevant = scored

    support = sum(
        max(0.0, a["entailment"] - a["contradiction"]) * a["source_weight"]
        for a in relevant
    )
    contradiction = sum(
        max(0.0, a["contradiction"] - a["entailment"]) * a["source_weight"]
        for a in relevant
    )

    support_domains = len({
        a["domain"] for a in relevant if a["entailment"] > a["contradiction"]
    })
    contradiction_domains = len({
        a["domain"] for a in relevant if a["contradiction"] > a["entailment"]
    })

    support_score = support * (1 + min(support_domains, 4) * 0.15)
    contradiction_score = contradiction * (1 + min(contradiction_domains, 4) * 0.15)

    # Death claims get an additional safety check: current reporting that the
    # named person is actively speaking/appearing is strong counter-evidence.
    if any(term in claim.lower() for term in DEATH_TERMS):
        subject = _extract_subject(claim)
        if subject:
            status_articles = []
            for query in (
                f'"{subject}" speech when:7d',
                f'"{subject}" latest when:7d',
            ):
                try:
                    status_articles.extend(_rss_search(query, max_results=4))
                except Exception:
                    pass

            status_articles = _dedupe(status_articles)
            for article in status_articles:
                evidence = f"{article['title']}. {article['description']}".strip()
                scores = nli_scores(evidence, claim)
                # A current article directly contradicting a death claim is
                # stronger than an unrelated article that merely shares words.
                if scores["contradiction"] > scores["entailment"]:
                    contradiction_score += (
                        scores["contradiction"] - scores["entailment"]
                    ) * _source_weight(article) * 1.5

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
        )[:max_results],
        "support": support_score,
        "contradiction": contradiction_score,
    }
