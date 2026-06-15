# Control de calidad del veredicto: acuerdo entre fuentes, detección de conflictos
# y penalización de confianza según el flag asignado.

from typing import Dict

# Fracción mínima de fuentes que deben coincidir con el veredicto
_AGREEMENT_MIN = 0.50
# Umbral de probabilidad opuesta para considerar que una fuente "contradice fuertemente"
_CONFLICT_THRESHOLD = 0.60
# Factores de penalización sobre la confianza calibrada
_PENALTY = {
    "OK":            1.00,
    "SINGLE_SOURCE": 0.90,
    "LOW_AGREEMENT": 0.85,
    "CONFLICT":      0.65,
}


class QualityJudge:
    """Evalúa la coherencia interna del veredicto y ajusta la confianza."""

    def evaluate(self, result: Dict) -> Dict:
        """
        Añade métricas de calidad al resultado y aplica penalización de confianza.
        Devuelve el dict enriquecido con source_agreement, has_conflict y quality_flag.
        """
        sources = result.get("sources", [])
        verdict = result.get("verdict", "NO_VERIFICABLE")

        if not sources or verdict == "NO_VERIFICABLE":
            return {**result, "source_agreement": 0.0, "has_conflict": False, "quality_flag": "N/A"}

        # Acuerdo entre fuentes
        matching  = sum(1 for s in sources if s["verdict"] == verdict)
        agreement = matching / len(sources)

        # Detección de conflicto fuerte
        has_conflict = False
        if verdict == "VERDADERO":
            has_conflict = any(s["p_refuta"] > _CONFLICT_THRESHOLD for s in sources)
        elif verdict == "FALSO":
            has_conflict = any(s["p_apoya"] > _CONFLICT_THRESHOLD for s in sources)

        # Flag de calidad
        if has_conflict:
            flag = "CONFLICT"
        elif len(sources) == 1:
            flag = "SINGLE_SOURCE"
        elif agreement < _AGREEMENT_MIN:
            flag = "LOW_AGREEMENT"
        else:
            flag = "OK"

        adjusted_conf = round(result["confidence"] * _PENALTY[flag], 4)

        return {
            **result,
            "confidence":       adjusted_conf,
            "source_agreement": round(agreement, 3),
            "has_conflict":     has_conflict,
            "quality_flag":     flag,
        }
