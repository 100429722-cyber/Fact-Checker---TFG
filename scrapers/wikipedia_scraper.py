# Búsqueda y extracción de resúmenes de artículos de Wikipedia (API REST pública).

import re
from typing import Dict, List, Optional

import requests

_HEADERS = {"User-Agent": "FactCheckerBot/1.0 (academic research; contact via UC3M)"}


def search_wikipedia(query: str, lang: str = "es", max_results: int = 4) -> List[Dict]:
    """Busca en Wikipedia y devuelve artículos con texto completo."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
        "utf8": 1,
    }
    try:
        resp = requests.get(api_url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", [])
    except Exception:
        return []

    results = []
    for item in items:
        article = _get_article_text(item["title"], lang)
        if article:
            results.append(article)
    return results


def _get_article_text(title: str, lang: str = "es", max_chars: int = 4000) -> Optional[Dict]:
    """Obtiene el texto completo del artículo (no solo el intro)."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
        "utf8": 1,
    }
    try:
        resp = requests.get(api_url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        if not text:
            return None
        # Limpiar saltos de línea múltiples
        text = re.sub(r'\n{2,}', ' ', text).strip()
        page_url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return {
            "title": page.get("title", title),
            "summary": text[:500],
            "full_text": text[:max_chars],
            "url": page_url,
            "source": f"Wikipedia ({lang})",
        }
    except Exception:
        return None
