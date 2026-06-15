# Limpieza de texto, segmentación en frases y construcción de queries de búsqueda.

import re
from typing import List

import spacy

try:
    nlp = spacy.load("es_core_news_lg")
except OSError:
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        raise OSError(
            "Modelo spaCy no encontrado. "
            "Ejecuta: python -m spacy download es_core_news_lg"
        )


def clean_text(text: str) -> str:
    """Elimina HTML, URLs y espacios múltiples."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def split_into_sentences(text: str) -> List[str]:
    """Segmenta el texto en frases usando spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]


def build_search_query(claim: str, max_keywords: int = 8) -> str:
    """Construye una query de búsqueda extrayendo entidades y términos clave."""
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]
    numbers  = re.findall(r'\b\d[\d,.%]*\b', claim)
    keywords = [
        token.lemma_ for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")
        and not token.is_stop and len(token.text) > 2
    ]
    seen: set = set()
    all_terms: list = []
    for t in entities + numbers + keywords:
        if t.lower() not in seen:
            seen.add(t.lower())
            all_terms.append(t)
    query = " ".join(all_terms[:max_keywords])
    return query if query else claim
