# Recuperación de evidencia para una afirmación desde RSS, Wikipedia y DuckDuckGo.

import os
import re
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List

from config import MAX_EVIDENCE_CHARS, MAX_SEARCH_RESULTS, RSS_FEEDS
from scrapers.rss_scraper import fetch_article_text as rss_fetch, search_rss_feed
from scrapers.web_scraper import fetch_article_text as web_fetch, search_web
from scrapers.wikipedia_scraper import search_wikipedia
from utils.multilingual import detect_language
from utils.text_preprocessing import build_search_query

_VERIFICADOR_DOMAINS = {
    "maldita.es", "newtral.es", "efeverifica.es", "factual.afp.com",
    "verificat.cat", "colombiacheck.com", "chequeado.com",
    "snopes.com", "factcheck.org", "politifact.com",
}
_ENCICLOPEDIA_DOMAINS = {"wikipedia.org"}

# Jaccard threshold para considerar dos documentos duplicados
_DEDUP_THRESHOLD = 0.50


def _classify_source_type(url: str, current_type: str) -> str:
    """Eleva el tipo de fuente si la URL pertenece a un dominio reconocido."""
    try:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return current_type
    if any(vd in domain for vd in _VERIFICADOR_DOMAINS):
        return "verificador"
    if any(ed in domain for ed in _ENCICLOPEDIA_DOMAINS):
        return "enciclopedia"
    return current_type


def _word_set(text: str) -> set:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _is_spanish(doc: Dict) -> bool:
    """Descarta documentos no en español/catalán/gallego (el NLI solo funciona en estas lenguas)."""
    text = (doc.get("full_text") or doc.get("summary", ""))[:400]
    if not text:
        return True
    return detect_language(text) in ("es", "ca", "gl")


def _deduplicate_content(docs: List[Dict], threshold: float = _DEDUP_THRESHOLD) -> List[Dict]:
    """Elimina artículos de agencia repetidos usando similitud Jaccard sobre palabras."""
    kept: List[Dict] = []
    kept_word_sets: List[set] = []
    for doc in docs:
        text = doc.get("full_text") or doc.get("summary", "")
        ws = _word_set(text)
        if not ws:
            kept.append(doc)
            kept_word_sets.append(ws)
            continue
        is_dup = any(
            existing_ws and len(ws & existing_ws) / len(ws | existing_ws) >= threshold
            for existing_ws in kept_word_sets
        )
        if not is_dup:
            kept.append(doc)
            kept_word_sets.append(ws)
    return kept


class EvidenceRetriever:
    def __init__(
        self,
        use_rss: bool = False,
        use_wikipedia: bool = True,
        use_web: bool = True,
        fetch_full_text: bool = True,
    ):
        self.use_rss = use_rss
        self.use_wikipedia = use_wikipedia
        self.use_web = use_web
        self.fetch_full_text = fetch_full_text

    def retrieve(self, claim: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
        """Recupera y filtra documentos de evidencia para la afirmación dada."""
        query = build_search_query(claim)
        evidence: List[Dict] = []

        if self.use_rss:
            for source_name, feed_url in RSS_FEEDS.items():
                for h in search_rss_feed(feed_url, query, max_results=2):
                    h["source_type"] = "verificador"
                    evidence.append(h)

        if self.use_wikipedia:
            for h in search_wikipedia(query, lang="es", max_results=3):
                h["source_type"] = "enciclopedia"
                evidence.append(h)

        if self.use_web:
            for h in search_web(query, max_results=max_results):
                h["source_type"] = _classify_source_type(h.get("url", ""), "web")
                evidence.append(h)

        # Deduplicación por URL
        seen_urls: set = set()
        unique_evidence = []
        for doc in evidence:
            url = doc.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_evidence.append(doc)

        if self.fetch_full_text:
            for doc in unique_evidence:
                if not doc.get("full_text") and doc.get("url"):
                    doc["full_text"] = web_fetch(doc["url"], max_chars=MAX_EVIDENCE_CHARS)

        spanish_evidence = [doc for doc in unique_evidence if _is_spanish(doc)]
        deduplicated = _deduplicate_content(spanish_evidence)
        return deduplicated[:max_results]
