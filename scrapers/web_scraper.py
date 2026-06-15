# Búsqueda web (DuckDuckGo) y scraping de artículos con BeautifulSoup.

import re
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def search_web(query: str, max_results: int = 5, region: str = "es-es") -> List[Dict]:
    """Busca en la web usando DuckDuckGo (sin API key)."""
    results = []
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, region=region, safesearch="moderate", max_results=max_results))
        for hit in hits:
            host = hit.get("href", "").split("/")[2] if hit.get("href") else "web"
            results.append({
                "title": hit.get("title", ""),
                "summary": hit.get("body", ""),
                "url": hit.get("href", ""),
                "source": host,
            })
    except Exception:
        pass
    return results


def fetch_article_text(url: str, max_chars: int = 3000, timeout: int = 10) -> str:
    """Descarga y extrae el texto principal de una página web."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
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
