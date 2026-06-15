# Pipeline completo de fact-checking.
# Uso: python main.py [archivo.txt] [--save] [--llm]

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from typing import Dict, List

from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_retriever import EvidenceRetriever
from pipeline.verifier import Verifier
from utils.multilingual import detect_language, lang_name, translate_to_spanish

USE_LLM_EXTRACTOR = "--llm" in sys.argv


# ── Presentación ─────────────────────────────────────────────────────────────

def _sep(char="─", width=65):
    print(char * width)


def _print_result(result: Dict):
    _sep()
    quality_flag = result.get("quality_flag", "N/A")

    print(f"\n  Afirmación : {result['claim']}")
    print(f"  Veredicto  : {result['verdict']}")
    print(f"  Confianza  : {result['confidence'] * 100:.1f}%"
          f"  (NLI: {result.get('nli_confidence', 0)*100:.1f}%"
          f"  |  Retrieval: {result.get('retrieval_confidence', 0)*100:.1f}%)")
    print(
        f"  Probabilidades — Verdadero: {result.get('avg_apoya', 0)*100:.1f}%  |  "
        f"Falso: {result.get('avg_refuta', 0)*100:.1f}%  |  "
        f"No verificable: {result.get('avg_neutral', 0)*100:.1f}%"
    )
    print(
        f"  Calidad    : {quality_flag}"
        f"  (acuerdo fuentes: {result.get('source_agreement', 0)*100:.0f}%)"
    )

    explanation = result.get("explanation", "")
    if explanation:
        print(f"\n  Explicación: {explanation}")

    sources = result.get("sources", [])
    if sources:
        print(f"\n  Fuentes analizadas ({len(sources)}):")
        for i, src in enumerate(sources, 1):
            print(f"\n    [{i}] {src.get('title', 'Sin título')}")
            print(f"        {src.get('source', '')}  —  {src.get('url', '')}")
            print(f"        {src['verdict']} ({src['confidence']*100:.1f}%)")
            snippet = src.get("relevant_snippet", "")
            if snippet:
                print(f"        Fragmento: «{snippet[:220]}»")


def _save_results(results: List[Dict], path: str = "results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {path}")


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_fact_checker(
    text: str,
    min_claim_confidence: float = 0.30,
    max_claims: int = 5,
    max_evidence: int = 5,
    use_llm: bool = False,
) -> List[Dict]:
    """Ejecuta el pipeline completo: extracción → evidencia → verificación → calidad."""
    print("\n" + "═" * 65)
    print("  FACT CHECKER — Verificador automático de afirmaciones")
    print("═" * 65)

    detected_lang = detect_language(text)
    if detected_lang != "es":
        print(f"\n[Paso 0] Idioma detectado: {lang_name(detected_lang)} ('{detected_lang}'). Traduciendo al español...")
        text = translate_to_spanish(text, source_lang=detected_lang)
        print(f"  → Traducción completada.")
    else:
        print(f"\n[Paso 0] Idioma detectado: español. No se requiere traducción.")

    if use_llm:
        print("\n[Paso 1] Extrayendo afirmaciones con LLM (dos pasos)...")
        from pipeline.claim_extractor_llm import ClaimExtractorLLM
        extractor = ClaimExtractorLLM()
    else:
        print("\n[Paso 1] Extrayendo afirmaciones del texto...")
        extractor = ClaimExtractor(min_words=5)
    claims = extractor.extract_claims(text, min_confidence=min_claim_confidence)

    if not claims:
        print("  No se encontraron afirmaciones verificables.")
        return []

    claims = claims[:max_claims]
    print(f"  → {len(claims)} afirmación(es) a verificar:")
    for i, c in enumerate(claims, 1):
        print(f"     {i}. [{c['confidence']*100:.0f}%] {c['text']}")

    retriever = EvidenceRetriever()
    verifier = Verifier()
    results = []

    for i, claim in enumerate(claims, 1):
        print(f"\n[Paso 2] Buscando evidencia para la afirmación {i}/{len(claims)}...")
        evidence = retriever.retrieve(claim["text"], max_results=max_evidence)
        print(f"  → {len(evidence)} fuente(s) encontrada(s)")

        print(f"[Paso 3] Verificando afirmación {i}/{len(claims)}...")
        result = verifier.verify_claim(claim["text"], evidence)
        results.append(result)
        _print_result(result)

    print("\n" + "═" * 65)
    print(f"  Análisis completado: {len(results)} afirmación(es) verificada(s)")
    print("═" * 65 + "\n")

    return results


# ── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    save = "--save" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        with open(args[0], "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        print("\nPega o escribe el texto a verificar.")
        print("Cuando termines, pulsa ENTER dos veces (línea en blanco) para continuar.\n")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        input_text = "\n".join(lines).strip()

        if not input_text:
            print("No se introdujo texto. Usando ejemplo por defecto.\n")
            input_text = (
                "El gobierno español aumentó el presupuesto de educación un 20% este año.\n"
                "España tiene la tasa de desempleo juvenil más alta de toda la Unión Europea.\n"
                "La inflación en la eurozona superó el 10% en 2022.\n"
                "Madrid es la ciudad más poblada de la Unión Europea.\n"
                "La vacuna contra el COVID-19 redujo las hospitalizaciones en un 90%."
            )

    results = run_fact_checker(input_text, use_llm=USE_LLM_EXTRACTOR)

    if save:
        _save_results(results)
