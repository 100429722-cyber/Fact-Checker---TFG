# Extractor heurístico de afirmaciones verificables.
# Puntúa cada frase con señales lingüísticas de factualidad; sin modelo ML.

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List

from utils.text_preprocessing import clean_text, split_into_sentences

# ── Señales positivas ─────────────────────────────────────────────────────────

_NUMBERS = re.compile(
    r"\b\d+([.,]\d+)?\s*(%|€|\$|£|km|kg|m²|hab\.?|millones?|miles?|mil)?\b"
)
_YEARS = re.compile(r"\b(19|20)\d{2}\b")
_FACTUAL_VERBS = re.compile(
    r"\b(es|son|fue|fueron|será|serán|ha sido|han sido|había|hubo|"
    r"existe|existen|tiene|tienen|tuvo|tuvieron|hay|"
    r"ocurrió|sucedió|pasó|aconteció|"
    r"afirma|asegura|indica|señala|muestra|revela|confirma|demuestra|"
    r"aumentó|disminuyó|creció|bajó|subió|alcanzó|superó|llegó a|"
    r"se ha|se han|se encontró|se detectó|se registró)\b",
    re.IGNORECASE,
)
_INNER_CAPS = re.compile(r"(?<!\.\s)\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}")

# ── Señales negativas ─────────────────────────────────────────────────────────

_OPINION = re.compile(
    r"\b(creo|pienso|opino|me parece|quizás|quizá|tal vez|podría|"
    r"posiblemente|probablemente|a mi juicio|en mi opinión|"
    r"supongo|imagino|diría)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(r"\b(yo |mi |mí |me |nos |nuestro|nuestra)\b", re.IGNORECASE)

# Saludos y muletillas: descarte inmediato
_SOCIAL_RE = re.compile(
    r"^\s*(gracias|muchas gracias|muchísimas gracias|hola|adiós|hasta luego|"
    r"buenos días|buenas tardes|buenas noches|buenas|por favor|claro que sí|"
    r"claro|exacto|muy bien|desde luego|por supuesto|efectivamente|de nada|"
    r"ok\b|okay|perfecto|entendido|correcto|así es|bienvenidos?|"
    r"soy [A-Z]|hola soy|les presento|a continuación|vamos a ver|"
    r"veamos|mira\b|escucha\b|oye\b|bueno\b)\b",
    re.IGNORECASE,
)


def _score(sentence: str) -> float:
    """Puntuación 0-1 de factualidad de una frase."""
    s = sentence.strip()
    if s.endswith("?") or _SOCIAL_RE.match(s):
        return 0.0

    score = 0.0
    if _NUMBERS.search(s):       score += 0.30
    if _YEARS.search(s):         score += 0.15
    if _FACTUAL_VERBS.search(s): score += 0.20
    score += min(len(_INNER_CAPS.findall(s[2:])) * 0.08, 0.20)

    words = len(s.split())
    if words >= 12: score += 0.08
    if words >= 20: score += 0.07

    if _OPINION.search(s):  score -= 0.25
    if _PERSONAL.search(s): score -= 0.10

    return min(max(score, 0.0), 1.0)


class ClaimExtractor:
    def __init__(self, min_words: int = 5):
        self.min_words = min_words

    def extract_claims(self, text: str, min_confidence: float = 0.30) -> List[Dict]:
        """Extrae afirmaciones verificables de un texto, ordenadas por puntuación."""
        sentences = split_into_sentences(clean_text(text))
        claims = []
        for idx, sentence in enumerate(sentences):
            if len(sentence.split()) < self.min_words:
                continue
            confidence = _score(sentence)
            if confidence >= min_confidence:
                claims.append({"id": idx, "text": sentence, "confidence": round(confidence, 4)})
        claims.sort(key=lambda x: x["confidence"], reverse=True)
        return claims
