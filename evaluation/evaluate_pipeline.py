"""
Evaluación cuantitativa del pipeline de fact-checking.

Uso:
    python evaluation/evaluate_pipeline.py                    # usa sample_claims.csv
    python evaluation/evaluate_pipeline.py mis_claims.csv     # archivo propio
    python evaluation/evaluate_pipeline.py sample_claims.csv --save  # guarda resultados

Formato del CSV de entrada (columnas requeridas: claim, label):
    claim,label
    "España ingresó en la UE en 1986.",VERDADERO
    "Barcelona es la capital de España.",FALSO

Etiquetas válidas: VERDADERO / V, FALSO / F, NO_VERIFICABLE / N / NEI
"""

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evidence_retriever import EvidenceRetriever
from pipeline.verifier import Verifier

CLASSES = ["VERDADERO", "FALSO", "NO_VERIFICABLE"]

_LABEL_MAP = {
    "V": "VERDADERO", "VERDADERO": "VERDADERO", "TRUE": "VERDADERO",
    "F": "FALSO",     "FALSO": "FALSO",         "FALSE": "FALSO",
    "N": "NO_VERIFICABLE", "NO_VERIFICABLE": "NO_VERIFICABLE", "NEI": "NO_VERIFICABLE",
}


def _normalize(label: str) -> str:
    return _LABEL_MAP.get(str(label).strip().upper(), "NO_VERIFICABLE")


def _print_confusion_matrix(cm: List[List[int]], classes: List[str]) -> None:
    col_w = max(len(c) for c in classes) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    print("  " + header)
    print("  " + "-" * len(header))
    for i, row_cls in enumerate(classes):
        row = f"  {row_cls:<{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(classes)))
        print(row)


def evaluate(csv_path: str, max_evidence: int = 5) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(csv_path)
    if "claim" not in df.columns or "label" not in df.columns:
        sys.exit("Error: el CSV debe tener columnas 'claim' y 'label'.")

    df["label"] = df["label"].apply(_normalize)
    n = len(df)

    print(f"\n  Cargando pipeline...")
    retriever = EvidenceRetriever()
    verifier = Verifier()

    print(f"\n{'═'*65}")
    print(f"  Evaluando {n} afirmaciones")
    print(f"{'═'*65}")

    records = []
    for i, row in df.iterrows():
        claim = str(row["claim"]).strip()
        true_label = row["label"]

        print(f"\n[{i+1}/{n}] {claim[:72]}...")
        t0 = time.time()

        evidence = retriever.retrieve(claim, max_results=max_evidence)
        result = verifier.verify_claim(claim, evidence)
        elapsed = time.time() - t0

        pred_label = result["verdict"]
        correct = pred_label == true_label
        icon = "✅" if correct else "❌"
        print(
            f"  {icon}  Real: {true_label:<14}  Pred: {pred_label:<14}"
            f"  Conf: {result['confidence']*100:.1f}%  Fuentes: {len(result.get('sources',[]))}  [{elapsed:.1f}s]"
        )

        records.append({
            "claim":                claim,
            "true_label":           true_label,
            "pred_label":           pred_label,
            "correct":              correct,
            "confidence":           round(result["confidence"], 4),
            "nli_confidence":       round(result.get("nli_confidence", result["confidence"]), 4),
            "retrieval_confidence": round(result.get("retrieval_confidence", 0), 4),
            "source_agreement":     round(result.get("source_agreement", 0), 4),
            "quality_flag":         result.get("quality_flag", "N/A"),
            "has_conflict":         result.get("has_conflict", False),
            "explanation":          result.get("explanation", ""),
            "avg_apoya":            round(result.get("avg_apoya", 0), 4),
            "avg_refuta":           round(result.get("avg_refuta", 0), 4),
            "avg_neutral":          round(result.get("avg_neutral", 0), 4),
            "n_sources":            len(result.get("sources", [])),
            "elapsed_s":            round(elapsed, 2),
        })

    results_df = pd.DataFrame(records)
    y_true = results_df["true_label"].tolist()
    y_pred = results_df["pred_label"].tolist()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy":        round(acc, 4),
        "n_claims":        n,
        "n_correct":       int(results_df["correct"].sum()),
        "per_class":       {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in CLASSES if cls in report
        },
        "macro_avg":       {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall":    round(report["macro avg"]["recall"], 4),
            "f1":        round(report["macro avg"]["f1-score"], 4),
        },
        "avg_confidence":  round(float(results_df["confidence"].mean()), 4),
        "avg_n_sources":   round(float(results_df["n_sources"].mean()), 2),
        "avg_elapsed_s":   round(float(results_df["elapsed_s"].mean()), 2),
        "total_elapsed_s": round(float(results_df["elapsed_s"].sum()), 2),
    }

    # ── Resumen en consola ───────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  RESUMEN DE EVALUACIÓN")
    print(f"{'═'*65}")
    print(f"  Afirmaciones evaluadas : {n}")
    print(f"  Correctas              : {metrics['n_correct']} / {n}  ({acc*100:.1f}%)")
    print(f"  Confianza media        : {metrics['avg_confidence']*100:.1f}%")
    print(f"  Fuentes medias / claim : {metrics['avg_n_sources']}")
    print(f"  Tiempo medio / claim   : {metrics['avg_elapsed_s']}s")
    print(f"  Tiempo total           : {metrics['total_elapsed_s']}s")

    print(f"\n  {'Clase':<18} {'Precisión':>10} {'Recall':>8} {'F1':>8} {'N':>5}")
    print(f"  {'-'*51}")
    for cls in CLASSES:
        m = metrics["per_class"].get(cls, {})
        print(
            f"  {cls:<18} {m.get('precision',0)*100:>9.1f}%"
            f" {m.get('recall',0)*100:>7.1f}%"
            f" {m.get('f1',0)*100:>7.1f}%"
            f" {m.get('support',0):>5}"
        )
    m = metrics["macro_avg"]
    print(f"  {'MACRO AVG':<18} {m['precision']*100:>9.1f}% {m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}%")

    print(f"\n  Matriz de confusión  (filas = real,  columnas = predicho):")
    _print_confusion_matrix(cm, CLASSES)

    errors = results_df[~results_df["correct"]]
    if not errors.empty:
        print(f"\n  Errores ({len(errors)}):")
        for _, row in errors.iterrows():
            print(
                f"    ✗  Real={row['true_label']:<14}  Pred={row['pred_label']:<14}"
                f"  «{row['claim'][:55]}...»"
            )

    print(f"{'═'*65}\n")
    return results_df, metrics


def save_results(results_df: pd.DataFrame, metrics: Dict, out_dir: str = None) -> None:
    out_path = Path(out_dir) if out_dir else Path(__file__).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_out  = out_path / f"results_{ts}.csv"
    json_out = out_path / f"metrics_{ts}.json"

    results_df.to_csv(csv_out, index=False, encoding="utf-8")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"  Resultados  → {csv_out}")
    print(f"  Métricas    → {json_out}\n")


if __name__ == "__main__":
    cli_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_save  = "--save" in sys.argv

    csv_path = cli_args[0] if cli_args else str(Path(__file__).parent / "sample_claims.csv")
    print(f"\nDataset : {csv_path}")

    results_df, metrics = evaluate(csv_path)

    if do_save:
        save_results(results_df, metrics)
    else:
        print("  (Usa --save para guardar los resultados en CSV/JSON)")
