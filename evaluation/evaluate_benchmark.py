"""
Evalúa el pipeline de fact-checking sobre el benchmark estático de afirmaciones.

USO BÁSICO (muestra de 100 afirmaciones aleatorias):
    py evaluation/evaluate_benchmark.py --sample 100

CON EXTRACTOR LLM (limpieza y contextualización previa con Claude Haiku):
    py evaluation/evaluate_benchmark.py --sample 100 --llm

CON TODAS LAS AFIRMACIONES (tarda muchas horas):
    py evaluation/evaluate_benchmark.py

FILTRAR POR CATEGORÍA O ETIQUETA:
    py evaluation/evaluate_benchmark.py --category ciencia --sample 50
    py evaluation/evaluate_benchmark.py --label FALSO --sample 50

CALCULAR MÉTRICAS SOBRE UN CSV YA GENERADO (sin volver a ejecutar el pipeline):
    py evaluation/evaluate_benchmark.py --metrics evaluation/benchmark_results_TIMESTAMP.csv

OPCIONES:
  --benchmark FILE   CSV de benchmark (default: evaluation/benchmark_claims.csv)
  --sample N         Número de afirmaciones a evaluar (muestra aleatoria estratificada)
  --category CAT     Filtrar por categoría (ciencia/historia/politica/deportes/cultura/economia)
  --label LABEL      Filtrar por etiqueta real (VERDADERO/FALSO/NO_VERIFICABLE)
  --seed N           Semilla aleatoria para reproducibilidad (default: 42)
  --metrics FILE     Calcular métricas sobre resultados ya guardados
  --llm              Aplica el paso 2 del ClaimExtractorLLM (Claude Haiku) a cada claim
                     antes de verificarla. Requiere ANTHROPIC_API_KEY.
"""

import argparse
import getpass
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSES = ["VERDADERO", "FALSO", "NO_VERIFICABLE"]
DEFAULT_BENCHMARK = str(Path(__file__).parent / "benchmark_claims.csv")


def _sep(char="═", width=65):
    print(char * width)


# ── LLM: API key + limpieza de claims ────────────────────────────────────────

def _get_api_key() -> str:
    """Devuelve la API key de Anthropic: del entorno o solicitándola al usuario."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        _sep("─")
        print("  ANTHROPIC_API_KEY no encontrada en las variables de entorno.")
        key = getpass.getpass("  Introduce tu Anthropic API key (no se mostrará): ").strip()
        if not key:
            sys.exit("  Error: se requiere ANTHROPIC_API_KEY para usar --llm.")
        os.environ["ANTHROPIC_API_KEY"] = key   # disponible para el resto del proceso
        _sep("─")
    return key


def _init_llm_client():
    """Inicializa el cliente Anthropic y devuelve (client, model_id)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("  Error: instala el paquete anthropic:  pip install anthropic")

    from pipeline.claim_extractor_llm import ClaimExtractorLLM
    api_key = _get_api_key()
    return Anthropic(api_key=api_key), ClaimExtractorLLM.DEFAULT_MODEL


def _clean_claim_llm(client, model: str, claim: str) -> str:
    """
    Aplica el Paso 2 del ClaimExtractorLLM (reescritura para autocontención y
    verificabilidad) a una claim ya extraída del benchmark.
    Devuelve la claim limpia, o la original si el modelo la rechaza/falla.
    """
    from pipeline.claim_extractor_llm import _STEP2_SYSTEM, _STEP2_USER

    prompt = _STEP2_USER.format(
        claim=claim,
        global_context="(claim extraída de benchmark estático — sin texto fuente disponible)",
        local_context=claim,
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=_STEP2_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip().strip('"\'')
        if result.lower() in ("null", "none", "ninguna", "no verificable", ""):
            return claim
        return result if len(result.split()) >= 4 else claim
    except Exception as exc:
        print(f"    [LLM] Error al limpiar claim: {exc}")
        return claim


# ── Cargar y filtrar el benchmark ─────────────────────────────────────────────

def load_benchmark(
    benchmark_csv: str,
    sample: int | None = None,
    category: str | None = None,
    label: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(benchmark_csv)
    # Force plain Python str (numpy object dtype) to avoid ArrowStringArray /
    # StringDtype bugs in groupby-apply that silently NaN the key column.
    df["claim"]      = list(map(str, df["claim"]))
    df["category"]   = list(map(str, df["category"]))
    df["true_label"] = list(map(str, df["true_label"]))

    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if label:
        df = df[df["true_label"].str.upper() == label.upper()]

    if sample and sample < len(df):
        # Muestra estratificada por true_label — loop explícito para evitar
        # el bug de groupby.apply con StringDtype en pandas ≥ 2.0 + pyarrow.
        per_class = max(1, sample // df["true_label"].nunique())
        parts = [
            grp.sample(min(len(grp), per_class), random_state=seed)
            for _, grp in df.groupby("true_label", sort=False)
        ]
        sampled = pd.concat(parts, ignore_index=True)
        remaining = sample - len(sampled)
        if remaining > 0:
            leftover = df.loc[~df.index.isin(sampled.index)]
            extra    = leftover.sample(min(remaining, len(leftover)), random_state=seed)
            sampled  = pd.concat([sampled, extra], ignore_index=True)
        df = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


# ── Evaluación claim por claim ────────────────────────────────────────────────

def run_evaluation(
    benchmark_csv: str = DEFAULT_BENCHMARK,
    sample: int | None  = None,
    category: str | None = None,
    label: str | None   = None,
    seed: int = 42,
    use_llm: bool = False,
) -> str:
    from pipeline.evidence_retriever import EvidenceRetriever
    from pipeline.verifier import Verifier

    df = load_benchmark(benchmark_csv, sample=sample, category=category,
                        label=label, seed=seed)
    n_total = len(df)

    _sep()
    print(f"  EVALUACIÓN SOBRE BENCHMARK ESTÁTICO")
    print(f"  Afirmaciones a evaluar : {n_total}")
    print(f"  Extractor LLM          : {'activado (Claude Haiku — Paso 2)' if use_llm else 'desactivado (heurístico)'}")
    print(f"  Tiempo estimado        : ~{n_total * (45 if use_llm else 30) // 60} min")
    if category:
        print(f"  Categoría filtrada     : {category}")
    if label:
        print(f"  Etiqueta filtrada      : {label}")
    _sep()

    # Inicializar cliente LLM si se solicita (solicita la API key si no está en entorno)
    llm_client = None
    llm_model  = None
    if use_llm:
        print("\n  Inicializando extractor LLM...")
        llm_client, llm_model = _init_llm_client()
        print(f"  Cliente LLM listo. Modelo: {llm_model}\n")

    print("\n  Cargando modelos NLI + reranker (puede tardar ~30 s la primera vez)...")
    retriever = EvidenceRetriever(fetch_full_text=True)
    verifier  = Verifier()
    print("  Modelos listos.\n")

    # Fichero de salida
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(benchmark_csv).parent
    out_csv = out_dir / f"benchmark_results_{ts}.csv"

    records = []
    t_global = time.time()

    for i, row in df.iterrows():
        claim_text  = str(row["claim"])
        true_label  = str(row["true_label"]).strip()
        cat         = str(row["category"])

        print(f"  [{len(records)+1}/{n_total}] [{cat}] {claim_text[:70]}...")
        t0 = time.time()

        # Paso 2 LLM: limpiar y contextualizar la claim antes de verificarla
        claim_to_verify = claim_text
        if use_llm and llm_client:
            claim_to_verify = _clean_claim_llm(llm_client, llm_model, claim_text)
            if claim_to_verify != claim_text:
                print(f"         LLM → «{claim_to_verify}»")

        try:
            evidence = retriever.retrieve(claim_to_verify, max_results=5)
            result   = verifier.verify_claim(claim_to_verify, evidence)
            elapsed  = round(time.time() - t0, 2)

            pred     = result["verdict"]
            conf     = result["confidence"]
            flag     = result.get("quality_flag", "N/A")
            correct  = "OK" if pred == true_label else "FAIL"
            print(f"         → {pred} ({conf*100:.0f}%) [{flag}] {correct}  [{elapsed}s]")

            records.append({
                "claim":                claim_text,
                "claim_verified":       claim_to_verify,
                "llm_rewritten":        claim_to_verify != claim_text,
                "use_llm":              use_llm,
                "category":             cat,
                "true_label":           true_label,
                "pred_label":           pred,
                "confidence":           round(conf, 4),
                "nli_confidence":       round(result.get("nli_confidence", conf), 4),
                "retrieval_confidence": round(result.get("retrieval_confidence", 0), 4),
                "quality_flag":         flag,
                "source_agreement":     round(result.get("source_agreement", 0), 4),
                "has_conflict":         result.get("has_conflict", False),
                "n_sources":            len(result.get("sources", [])),
                "explanation":          result.get("explanation", ""),
                "elapsed_s":            elapsed,
                "status":               "OK",
                "error_msg":            "",
            })

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            print(f"         ERROR: {exc}")
            records.append({
                "claim": claim_text, "claim_verified": claim_to_verify,
                "llm_rewritten": claim_to_verify != claim_text,
                "use_llm": use_llm,
                "category": cat,
                "true_label": true_label, "pred_label": "",
                "confidence": 0, "nli_confidence": 0, "retrieval_confidence": 0,
                "quality_flag": "N/A", "source_agreement": 0, "has_conflict": False,
                "n_sources": 0, "explanation": "", "elapsed_s": elapsed,
                "status": "ERROR", "error_msg": str(exc),
            })

        # Guardar progreso tras cada claim
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")

    total_elapsed = time.time() - t_global
    ok = sum(1 for r in records if r["status"] == "OK")

    _sep()
    print(f"\n  Evaluación completada: {ok}/{n_total} OK  ({total_elapsed/60:.1f} min)")
    if use_llm:
        n_rewritten = sum(1 for r in records if r.get("llm_rewritten", False))
        print(f"  Claims reescritas por LLM: {n_rewritten}/{n_total} ({n_rewritten/n_total*100:.0f}%)")
    print(f"  Resultados → {out_csv}")
    print(f"\n  Para calcular métricas:")
    print(f"    py evaluation/evaluate_benchmark.py --metrics {out_csv.name}")
    _sep()

    return str(out_csv)


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(results_csv: str) -> None:
    df = pd.read_csv(results_csv)
    df["true_label"] = [str(v).strip() for v in df["true_label"]]
    df["pred_label"] = [str(v).strip() for v in df["pred_label"]]
    df["status"]     = [str(v).strip() for v in df["status"]]
    df_eval = df[(df["status"] == "OK") &
                 df["true_label"].isin(CLASSES) &
                 df["pred_label"].isin(CLASSES)].copy()

    if df_eval.empty:
        print("No hay filas evaluables en el CSV. Comprueba que tiene columnas "
              "'true_label', 'pred_label' y 'status'.")
        return

    y_true = df_eval["true_label"].tolist()
    y_pred = df_eval["pred_label"].tolist()
    n      = len(df_eval)

    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)

    metrics = {
        "n_claims":  n,
        "n_correct": int(sum(t == p for t, p in zip(y_true, y_pred))),
        "accuracy":  round(acc, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in CLASSES if cls in report
        },
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"], 4),
            "recall":    round(report["macro avg"]["recall"],    4),
            "f1":        round(report["macro avg"]["f1-score"],  4),
        },
        "by_category": {},
        "avg_confidence":  round(float(df_eval["confidence"].mean()),  4),
        "avg_n_sources":   round(float(df_eval["n_sources"].mean()),   2),
        "avg_elapsed_s":   round(float(df_eval["elapsed_s"].mean()),   2),
    }

    for cat, grp in df_eval.groupby("category"):
        cat_acc = accuracy_score(grp["true_label"], grp["pred_label"])
        cat_rep = classification_report(
            grp["true_label"], grp["pred_label"],
            labels=CLASSES, output_dict=True, zero_division=0,
        )
        metrics["by_category"][cat] = {
            "n_claims": len(grp),
            "accuracy": round(cat_acc, 4),
            "macro_f1": round(cat_rep["macro avg"]["f1-score"], 4),
        }

    # ── Consola ───────────────────────────────────────────────────────────────
    _sep()
    print("  MÉTRICAS — BENCHMARK ESTÁTICO")
    _sep()
    print(f"  Afirmaciones evaluadas : {n}")
    print(f"  Correctas              : {metrics['n_correct']} / {n}  ({acc*100:.1f}%)")
    print(f"  Confianza media        : {metrics['avg_confidence']*100:.1f}%")
    print(f"  Fuentes medias/claim   : {metrics['avg_n_sources']}")
    print(f"  Tiempo medio/claim     : {metrics['avg_elapsed_s']}s")

    print(f"\n  {'Clase':<18} {'Precisión':>10} {'Recall':>8} {'F1':>8} {'N':>6}")
    print(f"  {'-'*54}")
    for cls in CLASSES:
        m = metrics["per_class"].get(cls, {})
        print(f"  {cls:<18} {m.get('precision',0)*100:>9.1f}%"
              f" {m.get('recall',0)*100:>7.1f}%"
              f" {m.get('f1',0)*100:>7.1f}%"
              f" {m.get('support',0):>6}")
    m = metrics["macro_avg"]
    print(f"  {'MACRO AVG':<18} {m['precision']*100:>9.1f}%"
          f" {m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}%")

    print(f"\n  Matriz de confusión  (filas=real, columnas=predicho):")
    col_w = max(len(c) for c in CLASSES) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in CLASSES)
    print("  " + header)
    print("  " + "─" * len(header))
    for i, row_cls in enumerate(CLASSES):
        row = f"  {row_cls:<{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(3))
        print(row)

    print(f"\n  Accuracy por categoría:")
    for cat, m in metrics["by_category"].items():
        print(f"    {cat:<18}  acc={m['accuracy']*100:.1f}%  macro-F1={m['macro_f1']*100:.1f}%  ({m['n_claims']} claims)")

    # Análisis de errores
    errors = df_eval[df_eval["true_label"] != df_eval["pred_label"]]
    if not errors.empty:
        print(f"\n  Errores frecuentes ({len(errors)} total):")
        err_types = errors.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
        for (real, pred), cnt in err_types.items():
            print(f"    Real={real:<14} → Pred={pred:<14} : {cnt} veces")
    _sep()

    # Guardar JSON
    out_path = Path(results_csv).parent
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = out_path / f"benchmark_metrics_{ts}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Métricas guardadas → {json_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación sobre el benchmark estático")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                        help="CSV del benchmark (default: benchmark_claims.csv)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Número de afirmaciones a muestrear (muestra estratificada)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filtrar por categoría: ciencia/historia/politica/deportes/cultura/economia")
    parser.add_argument("--label", type=str, default=None,
                        help="Filtrar por etiqueta: VERDADERO/FALSO/NO_VERIFICABLE")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Calcular métricas sobre un CSV de resultados ya generado")
    parser.add_argument("--llm", action="store_true", default=False,
                        help=(
                            "Aplica el Paso 2 del ClaimExtractorLLM (Claude Haiku) a cada "
                            "afirmación antes de verificarla: la reescribe para que sea "
                            "autocontenida y verificable. Requiere ANTHROPIC_API_KEY."
                        ))
    args = parser.parse_args()

    if args.metrics:
        compute_metrics(args.metrics)
    else:
        out = run_evaluation(
            benchmark_csv=args.benchmark,
            sample=args.sample,
            category=args.category,
            label=args.label,
            seed=args.seed,
            use_llm=args.llm,
        )
        compute_metrics(out)
