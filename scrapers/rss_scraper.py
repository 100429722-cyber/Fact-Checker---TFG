# Búsqueda y extracción de texto de feeds RSS de medios verificadores.

import re
from typing import Dict, List

import feedparser
import requests
from bs4 import BeautifulSoup


def _clean_html(html_text: str) -> str:
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ", strip=True)


def fetch_article_text(url: str, max_chars: int = 3000, timeout: int = 10) -> str:
    """Descarga y extrae el texto principal de una URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FactCheckerBot/1.0; academic research)",
            "Accept-Language": "es-ES,es;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|article|post|body", re.I))
        )
        text = article.get_text(separator=" ", strip=True) if article else soup.get_text(separator=" ", strip=True)
        return re.sub(r'\s+', ' ', text).strip()[:max_chars]
    except Exception:
        return ""


def _relevance(title: str, summary: str, query_terms: List[str]) -> float:
    """Puntuación de relevancia: fracción de términos de la query presentes en título+resumen."""
    content = (title + " " + summary).lower()
    # Stopwords básicas que no aportan señal de relevancia
    stops = {"el", "la", "los", "las", "un", "una", "de", "del", "en", "que", "y", "a", "es", "se"}
    meaningful = [t for t in query_terms if t not in stops and len(t) > 2]
    if not meaningful:
        return 0.0
    return sum(1 for t in meaningful if t in content) / len(meaningful)


def search_rss_feed(
    feed_url: str,
    query: str,
    max_results: int = 2,
    min_relevance: float = 0.25,
) -> List[Dict]:
    """
    Busca en un feed RSS entradas relacionadas con la query.
    Solo devuelve resultados con relevancia >= min_relevance para evitar
    artículos que comparten una sola palabra genérica con la afirmación.
    """
    try:
        feed = feedparser.parse(feed_url)
        query_terms = re.sub(r'[^\w\s]', '', query.lower()).split()
        results = []

        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = _clean_html(entry.get("summary", ""))
            link    = entry.get("link", "")

            score = _relevance(title, summary, query_terms)
            if score >= min_relevance:
                results.append({
                    "title":      title,
                    "summary":    summary,
                    "url":        link,
                    "source":     feed.feed.get("title", feed_url),
                    "_relevance": score,
                })

        results.sort(key=lambda x: x["_relevance"], reverse=True)
        for r in results:
            r.pop("_relevance", None)
        return results[:max_results]
    except Exception:
        return []
