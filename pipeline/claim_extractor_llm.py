# Extractor de afirmaciones en dos pasos mediante LLM (Claude Haiku).
# Paso 1: el modelo extrae afirmaciones verificables del texto (JSON).
# Paso 2: cada afirmación se reescribe para ser autocontenida y verificable.
# Requiere: pip install anthropic  y  ANTHROPIC_API_KEY en entorno o constructor.

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional

# ── Prompts ───────────────────────────────────────────────────────────────────

_STEP1_SYSTEM = (
    "Eres un asistente especializado en fact-checking. "
    "Tu tarea es identificar afirmaciones factuales verificables en textos en español. "
    "Respondes siempre con JSON válido, sin texto adicional."
)

_STEP1_USER = """\
Analiza el siguiente texto (puede ser transcripción de vídeo, noticia o artículo) \
y extrae TODAS las afirmaciones factuales verificables que aparezcan.

Una afirmación verificable ES:
- Un hecho concreto comprobable: dato, cifra, estadística, fecha, evento, resultado
- Una característica atribuida a una entidad concreta (país, persona, organización)
- Una declaración sobre el mundo real con valor de verdad

Una afirmación verificable NO ES:
- Opinión, valoración subjetiva o predicción especulativa
- Metáfora, ironía o expresión retórica
- Saludo, presentación, muletilla o frase de cierre
- Publicidad o promoción de productos/servicios
- Pregunta

Devuelve un JSON con esta estructura exacta (solo el JSON, sin markdown):
{{"claims": [{{"text": "texto de la afirmación tal como aparece en el texto original", "score": 0.0}}]}}

El campo "score" indica cuán verificable es la afirmación (0.0–1.0).
Incluye solo afirmaciones con score >= 0.30.
Si no hay afirmaciones verificables, devuelve {{"claims": []}}.

TEXTO A ANALIZAR:
{text}"""

_STEP2_SYSTEM = (
    "Eres un asistente especializado en fact-checking. "
    "Tu tarea es reescribir afirmaciones para que sean autocontenidas y fáciles de verificar. "
    "Respondes solo con la afirmación limpia o con la palabra null, sin explicaciones ni markdown."
)

_STEP2_USER = """\
Afirmación extraída de una transcripción de vídeo:
"{claim}"

CONTEXTO GLOBAL (inicio del texto — establece el tema, fecha, personajes y lugar):
"{global_context}"

CONTEXTO LOCAL (fragmento cercano a la afirmación):
"{local_context}"

Usando AMBOS contextos, reescribe la afirmación para que sea:
1. AUTOCONTENIDA — sin pronombres o referencias ambiguas; añade el sujeto, lugar, año o \
   cualquier dato necesario si se puede deducir del contexto global o local
2. CONCISA — una sola afirmación factual por frase; elimina subordinadas de relleno
3. VERIFICABLE — elimina valoraciones subjetivas; solo el hecho comprobable

Ejemplos de transformación:
  Afirmación: "La tasa de paro crece hasta el 10,8%"
  Contexto global: "Análisis de la economía española en el primer trimestre de 2025..."
  → "La tasa de paro en España creció hasta el 10,8% en el primer trimestre de 2025"

  Afirmación: "Ganó la medalla de oro"
  Contexto global: "Los Juegos Olímpicos de París 2024. Hoy hablamos de Carolina Marín..."
  → "Carolina Marín ganó la medalla de oro en bádminton en los Juegos Olímpicos de París 2024"

  Afirmación: "Es el peor dato desde 2013"
  Contexto global: "Informe sobre desempleo juvenil en España, primer trimestre 2025..."
  → "El desempleo juvenil en España en el primer trimestre de 2025 es el peor dato desde 2013"

  Afirmación: "Se suele decir que la ciencia no se diferencia de la magia"
  → null  (expresión popular, no un hecho verificable)

  Afirmación: "Ganar el Abierto de Australia demuestra que era el más grande"
  → null  (opinión, no un hecho)

Devuelve ÚNICAMENTE la afirmación limpia como texto plano, \
o la palabra null si no es verificable. Sin explicaciones."""


# ── Clase principal ───────────────────────────────────────────────────────────

class ClaimExtractorLLM:
    """
    Extractor de afirmaciones en dos pasos mediante llamadas al API de Claude.

    Args:
        model    : ID del modelo Claude a usar (por defecto haiku, rápido y económico)
        api_key  : Clave de API de Anthropic. Si es None, usa la variable de entorno
                   ANTHROPIC_API_KEY.
        min_words: Longitud mínima de una afirmación para ser procesada en el paso 2.
        context_window: Número de caracteres a cada lado de la afirmación que se
                        envían como contexto al paso 2.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str = None,
        min_words: int = 4,
        context_window: int = 300,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "El paquete 'anthropic' no está instalado. "
                "Ejecuta: pip install anthropic"
            )

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "No se encontró ANTHROPIC_API_KEY. "
                "Establece la variable de entorno o pasa api_key al constructor."
            )

        self._client = Anthropic(api_key=key)
        self.model = model
        self.min_words = min_words
        self.context_window = context_window

    # ── Paso 1 ────────────────────────────────────────────────────────────────

    def _step1_extract_raw(self, text: str) -> List[Dict]:
        """
        Llama al LLM para que extraiga las afirmaciones verificables del texto.
        Devuelve lista de dicts con 'text' y 'score'.
        """
        prompt = _STEP1_USER.format(text=text[:8000])  # limitar tokens de entrada

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_STEP1_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Extraer JSON aunque el modelo añada texto extra
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            print(f"  [ClaimExtractorLLM] Paso 1: respuesta inesperada del modelo.")
            return []

        try:
            data = json.loads(match.group())
            claims = data.get("claims", [])
            return [
                c for c in claims
                if isinstance(c, dict)
                and "text" in c
                and len(str(c["text"]).split()) >= self.min_words
            ]
        except json.JSONDecodeError as e:
            print(f"  [ClaimExtractorLLM] Paso 1: error JSON — {e}")
            return []

    # ── Paso 2 ────────────────────────────────────────────────────────────────

    def _get_context(self, claim: str, full_text: str) -> tuple[str, str]:
        """
        Devuelve (global_context, local_context).
        - global_context: inicio del texto, que suele contener tema, año y personajes.
        - local_context: fragmento alrededor de la afirmación.
        """
        global_context = full_text[:500]

        idx = full_text.find(claim[:40])
        if idx == -1:
            local_context = full_text[:self.context_window * 2]
        else:
            start = max(0, idx - self.context_window)
            end   = min(len(full_text), idx + len(claim) + self.context_window)
            local_context = full_text[start:end]

        return global_context, local_context

    def _step2_clean(self, claim: str, global_context: str, local_context: str) -> Optional[str]:
        """
        Llama al LLM para limpiar y contextualizar una afirmación.
        Devuelve la afirmación limpia, o None si no es verificable.
        """
        prompt = _STEP2_USER.format(
            claim=claim,
            global_context=global_context[:500],
            local_context=local_context[:400],
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_STEP2_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.content[0].text.strip()

        if result.lower() in ("null", "none", "ninguna", "no verificable", ""):
            return None

        # Limpiar comillas si el modelo las devuelve
        result = result.strip('"\'')
        return result if len(result.split()) >= self.min_words else None

    # ── Pipeline completo ─────────────────────────────────────────────────────

    def extract_claims(self, text: str, min_confidence: float = 0.30) -> List[Dict]:
        """
        Extrae y limpia afirmaciones verificables de un texto en dos pasos.

        Interfaz compatible con ClaimExtractor.extract_claims():
        Devuelve lista de dicts ordenada por confianza descendente:
          [{"id": int, "text": str, "confidence": float, "original": str}, ...]

        Args:
            text           : Texto de entrada
            min_confidence : Umbral mínimo de score (0–1) para incluir una afirmación
        """
        print(f"  [ClaimExtractorLLM] Paso 1 — extrayendo afirmaciones del texto...")
        raw_claims = self._step1_extract_raw(text)

        # Filtrar por umbral
        raw_claims = [c for c in raw_claims if float(c.get("score", 0)) >= min_confidence]

        if not raw_claims:
            print(f"  [ClaimExtractorLLM] Paso 1 — sin afirmaciones con score >= {min_confidence}.")
            return []

        print(f"  [ClaimExtractorLLM] Paso 1 — {len(raw_claims)} afirmaciones candidatas.")
        print(f"  [ClaimExtractorLLM] Paso 2 — limpiando y contextualizando...")

        results = []
        for idx, item in enumerate(raw_claims):
            original = str(item["text"])
            score    = float(item.get("score", 0.5))
            global_ctx, local_ctx = self._get_context(original, text)

            cleaned = self._step2_clean(original, global_ctx, local_ctx)

            if cleaned is None:
                print(f"    [{idx+1}] Descartada (no verificable): {original[:60]}...")
                continue

            changed = cleaned.lower().strip() != original.lower().strip()
            if changed:
                print(f"    [{idx+1}] [{score*100:.0f}%] {cleaned}")
                print(f"          (original: {original[:70]}{'...' if len(original)>70 else ''})")
            else:
                print(f"    [{idx+1}] [{score*100:.0f}%] {cleaned}")

            results.append({
                "id":         idx,
                "text":       cleaned,
                "confidence": round(score, 4),
                "original":   original,
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        print(f"  [ClaimExtractorLLM] Paso 2 — {len(results)} afirmaciones listas para verificar.")
        return results
