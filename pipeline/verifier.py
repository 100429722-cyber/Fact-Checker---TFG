# Verificación NLI de afirmaciones contra evidencia recuperada.
# Usa CrossEncoder para reranking y mDeBERTa-v3 para inferencia de entailment/refutación.

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    NLI_MAX_LEN, NLI_PRETRAINED_ES,
    RERANKER_GATE, RERANKER_MIN_SCORE, RERANKER_MODEL, RERANKER_TOP_K,
)
from pipeline.quality_judge import QualityJudge

_SOURCE_WEIGHT = {"verificador": 2.0, "enciclopedia": 1.5, "web": 1.0}
_RETRIEVAL_WEIGHTS = [0.60, 0.30, 0.10]  # media ponderada de los top-3 reranker scores

# Palabras clave para detectar automáticamente qué índice es cada clase
_ENTAILMENT_KEYS    = {"entailment", "apoya", "verdadero", "true", "supports"}
_NEUTRAL_KEYS       = {"neutral", "no_verifica", "no_verificable", "nei", "not_enough_info"}
_CONTRADICTION_KEYS = {"contradiction", "refuta", "falso", "false", "refutes"}

_NUM_RE = re.compile(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?\b')

# Palabras de resultado (campeón, ganó...) que implican entailment semántico no siempre
# inferido por el NLI en oraciones descriptivas ("Argentina, campeona, desfiló...").
_RESULT_WORDS = re.compile(
    r'\b(campe[oó]n|campeona|campeones|campeonas|gan[oó]|ganaron|venci[oó]|vencieron|'
    r'victoria|victorias|subcampe[oó]n|finalista|proclam[oó]\s+campe[oó]n|'
    r'se\s+coron[oó]|corona[ds]o|titul[oó]|se\s+adjudic[oó]|alzó\s+(con\s+)?el\s+trofeo|'
    r'conquistó|conquistaron|primer\s+puesto|medalla\s+de\s+oro)\b',
    re.IGNORECASE,
)

# Detecta cuando la claim dice "la más X" (#1) pero el snippet indica un puesto inferior
_SUPERLATIVE_CLAIM = re.compile(
    r'\bla\s+(ciudad|nación|país|economía|empresa|región|comunidad|provincia|'
    r'universidad|aeropuerto|puerto|banco|hospital)\s+más\b',
    re.IGNORECASE,
)
_ORDINAL_SNIPPET = re.compile(
    r'\b(segunda?|tercera?|cuarta?|quinta?|sexta?|s[eé]ptima?|octava?|'
    r'novena?|d[eé]cima?|2\.?\s*[ªa°]|3\.?\s*[ªa°]|4\.?\s*[ªa°]|5\.?\s*[ªa°])\b',
    re.IGNORECASE,
)

_STOPWORDS_ES = {'de', 'la', 'el', 'los', 'las', 'en', 'a', 'y', 'que', 'del',
                 'al', 'es', 'un', 'una', 'por', 'con', 'se', 'su', 'más', 'no'}


def _extract_numbers(text: str) -> List[float]:
    nums = []
    for m in _NUM_RE.finditer(text):
        raw = m.group().replace(".", "").replace(",", ".")
        try:
            nums.append(float(raw))
        except ValueError:
            pass
    return nums


def _ordinal_contradiction(snippet: str, claim: str) -> bool:
    """Detecta si el snippet contradice el superlativo de posición de la claim."""
    if not _SUPERLATIVE_CLAIM.search(claim):
        return False
    if not _ORDINAL_SNIPPET.search(snippet):
        return False
    claim_kw = set(re.sub(r'[^\w\s]', '', claim.lower()).split()) - _STOPWORDS_ES
    snippet_l = snippet.lower()
    return sum(1 for w in claim_kw if w in snippet_l) >= 3


def _semantic_entailment_hint(snippet: str, claim: str) -> bool:
    """Detecta entailment implícito en oraciones descriptivas que el NLI no infiere."""
    if not _RESULT_WORDS.search(snippet):
        return False
    # Al menos dos palabras de la claim deben aparecer en el snippet
    claim_words = set(re.sub(r'[^\w\s]', '', claim.lower()).split()) - {
        'de', 'la', 'el', 'los', 'las', 'en', 'a', 'y', 'que', 'del', 'al'
    }
    snippet_lower = snippet.lower()
    matches = sum(1 for w in claim_words if w in snippet_lower)
    return matches >= 2


def _numeric_contradiction(snippet: str, claim: str) -> bool:
    """Devuelve True si el número del snippet difiere >15% del número de la claim."""
    claim_nums = _extract_numbers(claim)
    if len(claim_nums) != 1:
        return False
    snippet_nums = _extract_numbers(snippet)
    if not snippet_nums:
        return False
    claim_val = claim_nums[0]
    if claim_val == 0:
        return False
    # Buscar el número del snippet más cercano al de la claim en contexto
    closest = min(snippet_nums, key=lambda x: abs(x - claim_val))
    ratio = abs(closest - claim_val) / abs(claim_val)
    return ratio > 0.15


def _try_simplify_claim(claim: str) -> Optional[str]:
    """Simplifica afirmaciones largas con cláusulas de relativo para facilitar la verificación."""
    if len(claim.split()) <= 15:
        return None
    m = re.match(
        r'^(.+?)\s*,\s+'
        r'(a\s+la\s+que|al\s+que|a\s+los\s+que|a\s+las\s+que|'
        r'en\s+la\s+que|en\s+el\s+que|en\s+los\s+que|en\s+las\s+que)\s+'
        r'(.+)$',
        claim, re.IGNORECASE
    )
    if not m:
        return None

    main_part   = m.group(1).strip()
    prep_phrase = m.group(2).strip()               # "a la que", "en el que", …
    rel_body    = m.group(3).rstrip('.').strip()   # "se incorporó en 1986 junto con Portugal"

    # Sujeto: primer grupo de tokens en mayúscula al inicio
    subj_m  = re.match(r'^([A-ZÁÉÍÓÚÑ][a-záéíóúñü]*(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)*)', main_part)
    subject = subj_m.group(1) if subj_m else main_part.split()[0]

    # Antecedente: últimas 2 palabras del main_part (el SN más cercano al pronombre relativo)
    antecedent = ' '.join(main_part.split()[-2:])

    # Preposición sin "que": "a la que" → "a la", "al que" → "al", "en la que" → "en la"
    prep = ' '.join(prep_phrase.split()[:-1])

    return f"{subject} {rel_body} {prep} {antecedent}"


def _detect_label_indices(id2label: Dict) -> Tuple[int, int, int]:
    """Detecta índices APOYA/NEUTRAL/REFUTA de cualquier modelo NLI estándar de 3 clases."""
    idx_apoya, idx_neutral, idx_refuta = 0, 1, 2
    for idx, label in id2label.items():
        key = str(label).lower()
        if key in _ENTAILMENT_KEYS:
            idx_apoya = int(idx)
        elif key in _NEUTRAL_KEYS:
            idx_neutral = int(idx)
        elif key in _CONTRADICTION_KEYS:
            idx_refuta = int(idx)
    return idx_apoya, idx_neutral, idx_refuta


class Verifier:
    def __init__(self, model_path: str = NLI_PRETRAINED_ES):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prioridad: (1) ruta local si existe, (2) ID de HuggingFace
        if os.path.isdir(model_path):
            source = model_path
            print(f"[Verifier] Cargando modelo local: {model_path}")
        else:
            source = model_path
            print(f"[Verifier] Descargando modelo: {source}")

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

        self._idx_apoya, self._idx_neutral, self._idx_refuta = _detect_label_indices(
            self.model.config.id2label
        )
        print(
            f"[Verifier] Etiquetas detectadas — "
            f"APOYA={self._idx_apoya}, NEUTRAL={self._idx_neutral}, REFUTA={self._idx_refuta}"
        )

        print(f"[Verifier] Cargando reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("[Verifier] Reranker listo.")

        self._judge = QualityJudge()

    def _nli(self, premise: str, hypothesis: str) -> Dict:
        """Ejecuta inferencia NLI sobre el par (premisa, hipótesis)."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=NLI_MAX_LEN,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]

        label_id = probs.argmax().item()
        if label_id == self._idx_apoya:
            label = "VERDADERO"
        elif label_id == self._idx_refuta:
            label = "FALSO"
        else:
            label = "NO_VERIFICABLE"

        n = probs.shape[0]
        p_apoya   = round(probs[self._idx_apoya].item(), 4)
        p_neutral = round(probs[self._idx_neutral].item(), 4) if n > 2 else 0.0
        p_refuta  = round(probs[self._idx_refuta].item(), 4) if n > 2 else 0.0
        confidence = round(probs[label_id].item(), 4)

        return {
            "label":      label,
            "p_apoya":    p_apoya,
            "p_neutral":  p_neutral,
            "p_refuta":   p_refuta,
            "confidence": confidence,
        }

    @staticmethod
    def _relevant_snippet(text: str, claim: str, top_k: int = 4, max_chars: int = 1000) -> str:
        """Devuelve las top_k oraciones más relevantes, puntuadas por cobertura y densidad."""
        if not text:
            return ""
        claim_words = set(re.sub(r'[^\w\s]', '', claim.lower()).split())
        sentences = re.split(r'(?<=[.!?])\s+', text)

        scored = []
        for sent in sentences:
            words = set(re.sub(r'[^\w\s]', '', sent.lower()).split())
            n_match = len(claim_words & words)
            if n_match == 0:
                continue
            coverage = n_match / max(len(claim_words), 1)   # qué parte del claim cubre
            density  = n_match / max(len(words), 1)          # qué parte de la frase es claim
            # Bonus para frases que contienen palabras de resultado (ganó, campeón...).
            # Reducido a 0.20 para no desplazar en exceso frases con alta cobertura léxica.
            result_bonus = 0.20 if _RESULT_WORDS.search(sent) else 0.0
            score = coverage + 0.4 * density + result_bonus
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = " ".join(s for _, s in scored[:top_k])
        return best[:max_chars] if best else text[:max_chars]

    @staticmethod
    def _overlap_score(text: str, claim: str) -> float:
        """Fracción de palabras de la claim presentes en el texto."""
        claim_words = set(re.sub(r'[^\w\s]', '', claim.lower()).split())
        text_words  = set(re.sub(r'[^\w\s]', '', text.lower()).split())
        return len(claim_words & text_words) / max(len(claim_words), 1)

    @staticmethod
    def _generate_explanation(claim: str, verdict: str, source_results: List[Dict]) -> str:
        """
        Genera una explicación legible del veredicto a partir de las fuentes analizadas.
        source_results ya está ordenado por relevancia (mayor reranker_score primero).
        """
        if verdict == "NO_VERIFICABLE" or not source_results:
            return "No se encontró evidencia suficiente para verificar esta afirmación."

        top = source_results[0]
        snippet = top.get("relevant_snippet", "")
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
        src_name = top.get("source", "fuente desconocida")

        n = len(source_results)
        agreeing = sum(1 for s in source_results if s["verdict"] == verdict)

        _VERDICT_ES = {"VERDADERO": "verdadera", "FALSO": "falsa"}
        verdict_es = _VERDICT_ES.get(verdict, verdict.lower())

        parts = [f"La afirmación parece {verdict_es}."]
        if n > 1:
            parts.append(f"{agreeing} de {n} fuentes analizadas concuerdan con este veredicto.")
        if snippet:
            parts.append(f"Evidencia principal ({src_name}): «{snippet}».")

        return " ".join(parts)

    def _rerank(
        self, claim: str, candidates: List[Tuple[Dict, str]]
    ) -> List[Tuple[Dict, str, float]]:
        """Puntúa y ordena candidatos por relevancia semántica con el CrossEncoder."""
        if not candidates:
            return []
        pairs = [[claim, snippet] for _, snippet in candidates]
        logits = self.reranker.predict(pairs, show_progress_bar=False)
        scores = [1.0 / (1.0 + np.exp(-float(l))) for l in logits]

        ranked = [
            (doc, snippet, score)
            for (doc, snippet), score in zip(candidates, scores)
            if score >= RERANKER_MIN_SCORE
        ]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:RERANKER_TOP_K]

    @staticmethod
    def _retrieval_confidence(ranked: List[Tuple]) -> float:
        """Media ponderada (60/30/10%) de los top-3 reranker scores; mide calidad de la evidencia."""
        weights = _RETRIEVAL_WEIGHTS[:len(ranked)]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(ranked[i][2] * weights[i] for i in range(len(weights))) / total_w

    def verify_claim(self, claim: str, evidence_list: List[Dict]) -> Dict:
        if not evidence_list:
            return {
                "claim": claim,
                "verdict": "NO_VERIFICABLE",
                "confidence": 0.0,
                "explanation": "No se encontraron fuentes de evidencia.",
                "sources": [],
            }

        # Pre-filtro: descartar docs sin ninguna palabra de la afirmación
        candidates = []
        for doc in evidence_list:
            text = doc.get("full_text") or doc.get("summary", "")
            if not text or len(text.split()) < 10:
                continue
            if self._overlap_score(text, claim) == 0.0:
                continue
            snippet = self._relevant_snippet(text, claim) or text[:600]
            candidates.append((doc, snippet))

        if not candidates:
            return {
                "claim": claim,
                "verdict": "NO_VERIFICABLE",
                "confidence": 0.0,
                "explanation": "No se pudo analizar la evidencia encontrada.",
                "sources": [],
            }

        ranked = self._rerank(claim, candidates)

        if not ranked:
            return {
                "claim": claim,
                "verdict": "NO_VERIFICABLE",
                "confidence": 0.0,
                "retrieval_confidence": 0.0,
                "explanation": "No se encontró evidencia suficientemente relevante.",
                "sources": [],
            }

        retrieval_conf = self._retrieval_confidence(ranked)

        # Si la evidencia no supera el umbral de calidad, no ejecutar NLI
        if retrieval_conf < RERANKER_GATE:
            return {
                "claim": claim,
                "verdict": "NO_VERIFICABLE",
                "confidence": 0.0,
                "retrieval_confidence": round(retrieval_conf, 4),
                "explanation": "La evidencia recuperada no es suficientemente relevante para verificar la afirmación.",
                "sources": [],
            }

        hypothesis = _try_simplify_claim(claim) or claim
        source_results = []

        for doc, snippet, reranker_score in ranked:
            nli = self._nli(premise=snippet, hypothesis=hypothesis)

            # Override numérico (solo con evidencia relevante)
            if (nli["label"] == "NO_VERIFICABLE"
                    and reranker_score >= 0.4
                    and _numeric_contradiction(snippet, claim)):
                nli = {
                    "label":      "FALSO",
                    "p_apoya":    nli["p_apoya"],
                    "p_neutral":  0.15,
                    "p_refuta":   0.70,
                    "confidence": 0.70,
                }

            # Override semántico: corrige NLI neutral en oraciones descriptivas con resultado implícito
            elif (nli["label"] == "NO_VERIFICABLE"
                    and reranker_score >= 0.4
                    and _semantic_entailment_hint(snippet, claim)):
                nli = {
                    "label":      "VERDADERO",
                    "p_apoya":    0.80,
                    "p_neutral":  0.15,
                    "p_refuta":   nli["p_refuta"],
                    "confidence": 0.80,
                }

            # Override ordinal: NLI confunde "la segunda más X" con "la más X"
            elif (nli["label"] == "VERDADERO"
                    and reranker_score >= 0.30
                    and _ordinal_contradiction(snippet, claim)):
                nli = {
                    "label":      "FALSO",
                    "p_apoya":    0.10,
                    "p_neutral":  0.15,
                    "p_refuta":   0.75,
                    "confidence": 0.75,
                }

            source_type = doc.get("source_type", "web")
            type_weight = _SOURCE_WEIGHT.get(source_type, 1.0)
            weight = type_weight * reranker_score  # tipo × relevancia semántica

            source_results.append({
                "title":            doc.get("title", ""),
                "source":           doc.get("source", "Desconocida"),
                "source_type":      source_type,
                "url":              doc.get("url", ""),
                "verdict":          nli["label"],
                "confidence":       nli["confidence"],
                "p_apoya":          nli["p_apoya"],
                "p_neutral":        nli["p_neutral"],
                "p_refuta":         nli["p_refuta"],
                "relevant_snippet": snippet,
                "reranker_score":   round(reranker_score, 4),
                "weight":           round(weight, 3),
                "overlap":          round(self._overlap_score(snippet, claim), 3),
            })

        # Agregación ponderada: peso × solapamiento (penaliza snippets de ruido)
        eff_w   = [s["weight"] * max(s["overlap"], 0.15) for s in source_results]
        total_w = sum(eff_w)
        avg_apoya   = sum(s["p_apoya"]   * w for s, w in zip(source_results, eff_w)) / total_w
        avg_neutral = sum(s["p_neutral"] * w for s, w in zip(source_results, eff_w)) / total_w
        avg_refuta  = sum(s["p_refuta"]  * w for s, w in zip(source_results, eff_w)) / total_w

        STRONG = 0.70

        # Boost de señal fuerte de REFUTA
        best_refuta = max((s["p_refuta"] for s in source_results), default=0.0)
        if best_refuta >= STRONG:
            avg_refuta = max(avg_refuta, best_refuta * 0.85)
        for s in source_results:
            if s["source_type"] == "verificador" and s["p_refuta"] >= STRONG:
                avg_refuta = max(avg_refuta, s["p_refuta"] * 0.9)

        # Boost de señal fuerte de APOYA (solo cuando el snippet era realmente relevante)
        best_apoya_src = max(
            (s for s in source_results if s["reranker_score"] >= 0.50),
            key=lambda s: s["p_apoya"],
            default=None,
        )
        if best_apoya_src and best_apoya_src["p_apoya"] >= STRONG:
            avg_apoya = max(avg_apoya, best_apoya_src["p_apoya"] * 0.90)

        # Acuerdo ponderado: fuente con snippet relevante pesa más que varias con ruido
        apoya_agreement = (
            sum(w for s, w in zip(source_results, eff_w) if s["verdict"] == "VERDADERO") / total_w
            if total_w > 0 else 0.0
        )

        VERDADERO_MARGIN    = 0.05   # margen mínimo sobre REFUTA para emitir VERDADERO
        MIN_APOYA_AGREEMENT = 0.40   # fracción mínima del peso efectivo que debe votar VERDADERO
        if (avg_apoya >= avg_refuta + VERDADERO_MARGIN
                and avg_apoya >= avg_neutral
                and apoya_agreement >= MIN_APOYA_AGREEMENT):
            verdict, nli_conf = "VERDADERO", avg_apoya
        elif avg_refuta > avg_apoya and avg_refuta >= avg_neutral:
            verdict, nli_conf = "FALSO", avg_refuta
        else:
            verdict, nli_conf = "NO_VERIFICABLE", avg_neutral

        # Fallback: si el NLI queda indeciso pero la evidencia es buena, usar la señal secundaria
        # para reducir la sobrepredicción de NO_VERIFICABLE debida al bias neutral del NLI.
        if (verdict == "NO_VERIFICABLE"
                and retrieval_conf >= 0.50
                and avg_neutral < 0.80):
            if avg_apoya >= 0.40 and avg_apoya >= avg_refuta + 0.10:
                verdict, nli_conf = "VERDADERO", avg_apoya
            elif avg_refuta >= 0.40 and avg_refuta >= avg_apoya + 0.10:
                verdict, nli_conf = "FALSO", avg_refuta

        # Confianza final: señal NLI × calidad de evidencia (evidencia marginal → confianza baja)
        calibrated_conf = nli_conf * retrieval_conf

        result = {
            "claim":                claim,
            "verdict":              verdict,
            "confidence":           round(calibrated_conf, 4),
            "nli_confidence":       round(nli_conf, 4),
            "retrieval_confidence": round(retrieval_conf, 4),
            "avg_apoya":            round(avg_apoya, 4),
            "avg_refuta":           round(avg_refuta, 4),
            "avg_neutral":          round(avg_neutral, 4),
            "explanation":          self._generate_explanation(claim, verdict, source_results),
            "sources":              source_results,
        }
        return self._judge.evaluate(result)
