"""
Utilidades para cargar y preparar los datasets de entrenamiento.

Datasets utilizados:
  - Detección de afirmaciones: CLEF CheckThat! 2022 (español)
  - NLI / verificación:        XNLI (español, multilingüe)
"""

from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict


# ---------------------------------------------------------------------------
# NLI (verificación) — XNLI español
# Labels: 0=entailment (APOYA), 1=neutral (NO_VERIFICABLE), 2=contradiction (REFUTA)
# ---------------------------------------------------------------------------

def load_xnli_spanish() -> DatasetDict:
    """Carga XNLI filtrado para español. Siempre disponible en HuggingFace."""
    return load_dataset("xnli", "es")


# ---------------------------------------------------------------------------
# Detección de afirmaciones — CLEF CheckThat! 2022
# ---------------------------------------------------------------------------

def load_claim_detection_dataset() -> Optional[DatasetDict]:
    """
    Intenta cargar el dataset CLEF CheckThat! para detección de afirmaciones.
    Si no está disponible devuelve None (se usará dataset sintético de demo).
    """
    candidates = [
        ("clef2022_checkthat_v2", "spanish"),
        ("clef2021_checkthat_v2", "spanish"),
        ("clef2022_checkthat_v2", "es"),
    ]
    for name, config in candidates:
        try:
            return load_dataset(name, config)
        except Exception:
            continue
    return None


def prepare_checkthat_for_training(dataset: DatasetDict) -> DatasetDict:
    """Normaliza el dataset CLEF a columnas 'text' y 'label' (0/1)."""

    def _normalize(example):
        text = (
            example.get("tweet_text")
            or example.get("sentence")
            or example.get("text", "")
        )
        label = example.get("class_label") or example.get("label", 0)
        if isinstance(label, str):
            label = 1 if label.lower() in ("yes", "1", "true", "checkworthy") else 0
        return {"text": text, "label": int(label)}

    return dataset.map(_normalize)


def build_synthetic_claim_dataset() -> DatasetDict:
    """
    Dataset sintético mínimo para probar el pipeline sin datos reales.
    SUSTITUIR por CLEF CheckThat! para uso real.
    """
    examples = {
        "text": [
            "El gobierno español aumentó el presupuesto de educación un 20%.",
            "España tiene la tasa de desempleo más alta de la UE.",
            "La inflación en la eurozona superó el 10% en 2022.",
            "El presidente visitó Bruselas ayer por la tarde.",
            "Madrid es la ciudad más poblada de la Unión Europea.",
            "La vacuna COVID redujo las hospitalizaciones en un 90%.",
            "Hoy hace un día muy bonito en la capital.",
            "Me parece que la película estuvo bien.",
            "El tiempo es agradable esta semana.",
            "Esta canción me encanta mucho.",
            "El PIB de España creció un 5,5% en 2021.",
            "España tiene más de 47 millones de habitantes.",
            "El Banco Central Europeo subió los tipos de interés.",
            "Mañana hay partido de fútbol.",
            "El nuevo álbum del artista salió ayer.",
        ],
        "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    }

    # 80/20 split manual
    split = int(len(examples["text"]) * 0.8)
    train = {k: v[:split] for k, v in examples.items()}
    test = {k: v[split:] for k, v in examples.items()}

    return DatasetDict({
        "train": Dataset.from_dict(train),
        "test": Dataset.from_dict(test),
    })
