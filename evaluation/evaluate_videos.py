"""
Evaluación sistemática del pipeline con vídeos reales de YouTube.

FLUJO DE USO:
  1. Rellena evaluation/videos_to_analyze.csv con las URLs y categorías.
  2. (Opcional pero recomendado) Genera la caché de transcripciones — solo hay que
     hacerlo una vez; después reutiliza los audios ya transcritos:
         python evaluation/evaluate_videos.py --transcribe-only
  3. Ejecuta el batch de extracción + verificación:
         python evaluation/evaluate_videos.py
         python evaluation/evaluate_videos.py --llm   # extractor LLM en dos pasos
     Genera evaluation/video_claims_TIMESTAMP.csv con las predicciones.
     La columna 'true_label' queda vacía para que la rellenes a mano.
  4. Abre el CSV, rellena 'true_label' (VERDADERO / FALSO / NO_VERIFICABLE)
     para cada afirmación detectada por el sistema.
  5. Calcula las métricas finales:
         python evaluation/evaluate_videos.py --metrics video_claims_TIMESTAMP.csv

Opciones:
  --llm              Usa el extractor LLM (ClaimExtractorLLM) en dos pasos.
                     Requiere la variable de entorno ANTHROPIC_API_KEY.
  --max-claims N     Máximo de afirmaciones por vídeo (por defecto 5)
  --threshold F      Umbral mínimo del ClaimExtractor (por defecto 0.30)
  --resume FILE      Reanuda un batch anterior: salta vídeos ya procesados en FILE
  --metrics FILE     Calcula métricas sobre un CSV ya etiquetado manualmente
  --transcribe-only  Solo descarga y transcribe vídeos; guarda la caché y termina.
                     No extrae afirmaciones ni verifica. Útil para preparar la caché
                     antes de iterar sobre el extractor.
  --transcript-cache Ruta del JSON de caché de transcripciones
                     (por defecto: evaluation/transcripts_cache.json)
"""

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSES = ["VERDADERO", "FALSO", "NO_VERIFICABLE"]

_LABEL_MAP = {
    "V": "VERDADERO", "VERDADERO": "VERDADERO", "TRUE": "VERDADERO",
    "F": "FALSO",     "FALSO": "FALSO",         "FALSE": "FALSO",
    "N": "NO_VERIFICABLE", "NO_VERIFICABLE": "NO_VERIFICABLE", "NEI": "NO_VERIFICABLE",
}

DEFAULT_CACHE = str(Path(__file__).parent / "transcripts_cache.json")


def _load_cache(path: str) -> dict:
    """Carga la caché de transcripciones desde disco. Devuelve {} si no existe."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, cache: dict) -> None:
    """Guarda la caché de transcripciones en disco."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _normalize_label(label) -> str:
    if pd.isna(label):
        return ""
    return _LABEL_MAP.get(str(label).strip().upper(), "")


def _sep(char="═", width=65):
    print(char * width)


# ── Procesamiento de un único vídeo ──────────────────────────────────────────

def process_video(
    url: str,
    category: str,
    notes: str,
    retriever,
    verifier,
    max_claims: int = 5,
    threshold: float = 0.30,
    use_llm: bool = False,
    transcript_cache: dict = None,
    cache_path: str = None,
) -> list[dict]:
    """
    Descarga, transcribe y verifica afirmaciones de un vídeo de YouTube.
    Si transcript_cache contiene la URL, omite descarga y transcripción.
    Devuelve lista de dicts (una fila por afirmación detectada).
    """
    from video.youtube_downloader import download_audio
    from video.transcriber import transcribe
    from utils.multilingual import detect_language, translate_to_spanish, lang_name

    records = []
    t_video = time.time()

    # ── Descarga + Transcripción (con caché) ──────────────────────────────────
    if transcript_cache is not None and url in transcript_cache:
        cached = transcript_cache[url]
        transcript = cached["transcript"]
        n_words    = cached["n_words"]
        translated = cached["translated"]
        print(f"  [1-2/4] Transcripción cargada desde caché ({n_words} palabras)")
    else:
        print(f"\n  [1/4] Descargando audio...")
        try:
            audio_path = download_audio(url)
        except Exception as exc:
            print(f"  ERROR descarga: {exc}")
            return [_error_row(url, category, notes, "ERROR_DESCARGA", str(exc))]

        print(f"  [2/4] Transcribiendo con Whisper...")
        try:
            transcript = transcribe(audio_path)
        except Exception as exc:
            print(f"  ERROR transcripción: {exc}")
            return [_error_row(url, category, notes, "ERROR_TRANSCRIPCION", str(exc))]
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        if not transcript.strip():
            print("  AVISO: transcripción vacía.")
            return [_error_row(url, category, notes, "TRANSCRIPCION_VACIA", "")]

        n_words = len(transcript.split())
        print(f"  Transcripción: {n_words} palabras")

        lang = detect_language(transcript)
        translated = False
        if lang != "es":
            print(f"  Idioma: {lang_name(lang)} → traduciendo al español...")
            transcript = translate_to_spanish(transcript, source_lang=lang)
            translated = True

        if transcript_cache is not None:
            transcript_cache[url] = {
                "transcript": transcript,
                "n_words":    n_words,
                "translated": translated,
            }
            if cache_path:
                _save_cache(cache_path, transcript_cache)
                print(f"  Transcripción guardada en caché.")

    # ── Extracción de afirmaciones ────────────────────────────────────────────
    if use_llm:
        print(f"  [3/4] Extrayendo afirmaciones con LLM (umbral={threshold})...")
        from pipeline.claim_extractor_llm import ClaimExtractorLLM
        extractor = ClaimExtractorLLM()
    else:
        print(f"  [3/4] Extrayendo afirmaciones (umbral={threshold})...")
        from pipeline.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor(min_words=5)
    claims = extractor.extract_claims(transcript, min_confidence=threshold)[:max_claims]

    if not claims:
        print("  Sin afirmaciones detectadas con el umbral actual.")
        return [_error_row(url, category, notes, "SIN_AFIRMACIONES", transcript[:300])]

    print(f"  {len(claims)} afirmación(es) a verificar.")

    # ── Verificación ──────────────────────────────────────────────────────────
    print(f"  [4/4] Verificando...")
    for idx, claim in enumerate(claims, 1):
        t_c = time.time()
        try:
            evidence = retriever.retrieve(claim["text"], max_results=5)
            result   = verifier.verify_claim(claim["text"], evidence)
        except Exception as exc:
            print(f"    [{idx}] ERROR verificación: {exc}")
            records.append(_error_row(url, category, notes, "ERROR_VERIFICACION", str(exc),
                                      claim=claim["text"]))
            continue

        elapsed = time.time() - t_c
        verdict = result["verdict"]
        conf    = result["confidence"]
        flag    = result.get("quality_flag", "N/A")
        print(f"    [{idx}/{len(claims)}] {verdict} ({conf*100:.0f}%) [{flag}]  {claim['text'][:60]}...")

        records.append({
            "video_url":            url,
            "category":             category,
            "notes":                notes,
            "transcript_words":     n_words,
            "translated":           translated,
            "claim_score":          round(claim["confidence"], 4),
            "claim":                claim["text"],
            "pred_label":           verdict,
            "true_label":           "",          # ← rellenar manualmente
            "confidence":           round(conf, 4),
            "nli_confidence":       round(result.get("nli_confidence", conf), 4),
            "retrieval_confidence": round(result.get("retrieval_confidence", 0), 4),
            "quality_flag":         flag,
            "source_agreement":     round(result.get("source_agreement", 0), 4),
            "has_conflict":         result.get("has_conflict", False),
            "n_sources":            len(result.get("sources", [])),
            "explanation":          result.get("explanation", ""),
            "elapsed_s":            round(elapsed, 2),
            "status":               "OK",
            "error_msg":            "",
        })

    elapsed_video = time.time() - t_video
    print(f"  Vídeo completado en {elapsed_video:.1f}s  ({len(records)} afirmaciones)")
    return records


def _error_row(url, category, notes, status, error_msg, claim="") -> dict:
    return {
        "video_url": url, "category": category, "notes": notes,
        "transcript_words": 0, "translated": False,
        "claim_score": 0, "claim": claim,
        "pred_label": "", "true_label": "",
        "confidence": 0, "nli_confidence": 0, "retrieval_confidence": 0,
        "quality_flag": "N/A", "source_agreement": 0, "has_conflict": False,
        "n_sources": 0, "explanation": "", "elapsed_s": 0,
        "status": status, "error_msg": error_msg,
    }


# ── Solo transcripción ───────────────────────────────────────────────────────

def transcribe_only(input_csv: str, cache_path: str = DEFAULT_CACHE) -> None:
    """Descarga y transcribe todos los vídeos del CSV; guarda la caché y termina."""
    from video.youtube_downloader import download_audio
    from video.transcriber import transcribe
    from utils.multilingual import detect_language, translate_to_spanish, lang_name

    df = pd.read_csv(input_csv)
    df = df[df["url"].notna() & (df["url"].str.strip() != "")]
    if df.empty:
        sys.exit("Error: no hay URLs en el CSV de entrada.")

    cache = _load_cache(cache_path)
    pending = df[~df["url"].isin(cache)]
    total   = len(pending)
    cached  = len(df) - total

    _sep()
    print(f"  MODO SOLO-TRANSCRIPCIÓN")
    print(f"  Vídeos en caché   : {cached} / {len(df)}")
    print(f"  Vídeos pendientes : {total}")
    print(f"  Caché → {cache_path}")
    _sep()

    for i, (_, row) in enumerate(pending.iterrows(), 1):
        url      = str(row["url"]).strip()
        category = str(row.get("category", "sin_categoria")).strip()

        _sep("─")
        print(f"  [{i}/{total}]  {category.upper()}  |  {url}")
        _sep("─")

        try:
            print("  [1/2] Descargando audio...")
            audio_path = download_audio(url)
        except Exception as exc:
            print(f"  ERROR descarga: {exc}")
            continue

        try:
            print("  [2/2] Transcribiendo con Whisper...")
            transcript = transcribe(audio_path)
        except Exception as exc:
            print(f"  ERROR transcripción: {exc}")
            continue
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        if not transcript.strip():
            print("  AVISO: transcripción vacía, no se guarda.")
            continue

        n_words = len(transcript.split())
        lang = detect_language(transcript)
        translated = False
        if lang != "es":
            print(f"  Idioma: {lang_name(lang)} → traduciendo al español...")
            transcript = translate_to_spanish(transcript, source_lang=lang)
            translated = True

        cache[url] = {"transcript": transcript, "n_words": n_words, "translated": translated}
        _save_cache(cache_path, cache)
        print(f"  Guardado en caché ({n_words} palabras).")

    _sep()
    print(f"  Caché completada: {len(cache)} vídeos → {cache_path}")
    _sep()


# ── Batch principal ───────────────────────────────────────────────────────────

def run_batch(
    input_csv: str,
    max_claims: int = 5,
    threshold: float = 0.30,
    resume_file: str = None,
    use_llm: bool = False,
    cache_path: str = DEFAULT_CACHE,
) -> str:
    """Procesa todos los vídeos del CSV de entrada. Devuelve ruta del CSV de salida."""
    from pipeline.evidence_retriever import EvidenceRetriever
    from pipeline.verifier import Verifier

    df = pd.read_csv(input_csv)
    df = df[df["url"].notna() & (df["url"].str.strip() != "")]
    if df.empty:
        sys.exit("Error: no hay URLs en el CSV de entrada.")

    # Caché de transcripciones
    transcript_cache = _load_cache(cache_path)
    n_cached = sum(1 for url in df["url"] if url in transcript_cache)

    # URLs ya procesadas en un batch anterior (--resume)
    done_urls: set = set()
    existing_records: list = []
    if resume_file and Path(resume_file).exists():
        prev = pd.read_csv(resume_file)
        done_urls = set(prev["video_url"].dropna().unique())
        existing_records = prev.to_dict("records")
        print(f"  Reanudando desde {resume_file} — {len(done_urls)} vídeos ya procesados.")

    pending = df[~df["url"].isin(done_urls)]
    total   = len(pending)

    extractor_name = "LLM (Claude, dos pasos)" if use_llm else "Heurístico"
    _sep()
    print(f"  EVALUACIÓN EN BATCH — {total} vídeos pendientes")
    print(f"  Extractor: {extractor_name}")
    print(f"  Umbral: {threshold}   Max afirmaciones/vídeo: {max_claims}")
    print(f"  Transcripciones en caché: {n_cached} / {len(df)}")
    _sep()

    print("\n  Cargando modelos (puede tardar ~30 s la primera vez)...")
    retriever = EvidenceRetriever()
    verifier  = Verifier()
    print("  Modelos listos.\n")

    # Ruta de salida (fija desde el inicio para poder hacer --resume sobre ella)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(input_csv).parent
    out_csv  = out_dir / f"video_claims_{ts}.csv"

    all_records = list(existing_records)
    t_global    = time.time()

    for i, (_, row) in enumerate(pending.iterrows(), 1):
        url      = str(row["url"]).strip()
        category = str(row.get("category", "sin_categoria")).strip()
        notes    = str(row.get("notes", "")).strip() if pd.notna(row.get("notes")) else ""

        _sep("─")
        print(f"  [{i}/{total}]  {category.upper()}  |  {url}")
        _sep("─")

        records = process_video(
            url, category, notes, retriever, verifier,
            max_claims=max_claims, threshold=threshold, use_llm=use_llm,
            transcript_cache=transcript_cache, cache_path=cache_path,
        )
        all_records.extend(records)

        # Guardar tras cada vídeo (tolerancia a fallos)
        pd.DataFrame(all_records).to_csv(out_csv, index=False, encoding="utf-8")

    total_elapsed = time.time() - t_global
    ok_records    = [r for r in all_records if r["status"] == "OK"]

    _sep()
    print(f"\n  BATCH COMPLETADO")
    print(f"  Vídeos procesados : {total}")
    print(f"  Afirmaciones OK   : {len(ok_records)}")
    print(f"  Tiempo total      : {total_elapsed/60:.1f} min")
    print(f"\n  CSV generado → {out_csv}")
    print(f"\n  PRÓXIMO PASO:")
    print(f"  Abre el CSV y rellena la columna 'true_label' para cada afirmación.")
    print(f"  Etiquetas válidas: VERDADERO  /  FALSO  /  NO_VERIFICABLE")
    print(f"  Cuando termines ejecuta:")
    print(f"    python evaluation/evaluate_videos.py --metrics {out_csv.name}")
    _sep()

    return str(out_csv)


# ── Métricas post-etiquetado ──────────────────────────────────────────────────

def compute_metrics(labeled_csv: str) -> None:
    """Calcula métricas sobre un CSV ya etiquetado manualmente."""
    df = pd.read_csv(labeled_csv, encoding="utf-8", engine="python")

    # Solo filas con etiqueta real y predicción válidas
    df["true_label"] = df["true_label"].apply(_normalize_label)
    df["pred_label"] = df["pred_label"].apply(_normalize_label)
    df_eval = df[(df["true_label"] != "") & (df["pred_label"] != "") &
                 (df["status"] == "OK")].copy()

    if df_eval.empty:
        print("No hay filas con 'true_label' relleno y status OK. Rellena el CSV primero.")
        return

    y_true = df_eval["true_label"].tolist()
    y_pred = df_eval["pred_label"].tolist()
    n      = len(df_eval)

    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)

    metrics = {
        "n_claims":   n,
        "n_correct":  int(sum(t == p for t, p in zip(y_true, y_pred))),
        "accuracy":   round(acc, 4),
        "per_class":  {
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

    # Métricas por categoría
    for cat, grp in df_eval.groupby("category"):
        cat_acc = accuracy_score(grp["true_label"], grp["pred_label"])
        metrics["by_category"][cat] = {
            "n_claims": len(grp),
            "accuracy": round(cat_acc, 4),
        }

    # ── Consola ───────────────────────────────────────────────────────────────
    _sep()
    print("  MÉTRICAS DE EVALUACIÓN CON VÍDEOS REALES")
    _sep()
    print(f"  Afirmaciones evaluadas : {n}")
    print(f"  Correctas              : {metrics['n_correct']} / {n}  ({acc*100:.1f}%)")
    print(f"  Confianza media        : {metrics['avg_confidence']*100:.1f}%")
    print(f"  Fuentes medias/claim   : {metrics['avg_n_sources']}")
    print(f"  Tiempo medio/claim     : {metrics['avg_elapsed_s']}s")

    print(f"\n  {'Clase':<18} {'Precisión':>10} {'Recall':>8} {'F1':>8} {'N':>5}")
    print(f"  {'-'*51}")
    for cls in CLASSES:
        m = metrics["per_class"].get(cls, {})
        print(f"  {cls:<18} {m.get('precision',0)*100:>9.1f}%"
              f" {m.get('recall',0)*100:>7.1f}%"
              f" {m.get('f1',0)*100:>7.1f}%"
              f" {m.get('support',0):>5}")
    m = metrics["macro_avg"]
    print(f"  {'MACRO AVG':<18} {m['precision']*100:>9.1f}%"
          f" {m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}%")

    print(f"\n  Matriz de confusión  (filas=real, columnas=predicho):")
    col_w = max(len(c) for c in CLASSES) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in CLASSES)
    print("  " + header)
    print("  " + "-" * len(header))
    for i, row_cls in enumerate(CLASSES):
        row = f"  {row_cls:<{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(3))
        print(row)

    print(f"\n  Accuracy por categoría:")
    for cat, m in metrics["by_category"].items():
        print(f"    {cat:<18} {m['accuracy']*100:.1f}%  ({m['n_claims']} claims)")

    errors = df_eval[df_eval["true_label"] != df_eval["pred_label"]]
    if not errors.empty:
        print(f"\n  Errores ({len(errors)}):")
        for _, r in errors.iterrows():
            print(f"    [{r['category']}] Real={r['true_label']:<14}"
                  f" Pred={r['pred_label']:<14} «{r['claim'][:55]}...»")
    _sep()

    # ── Guardar métricas JSON ─────────────────────────────────────────────────
    out_path = Path(labeled_csv).parent
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = out_path / f"video_metrics_{ts}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Métricas guardadas → {json_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluación en batch con vídeos de YouTube")
    parser.add_argument("input", nargs="?",
                        default=str(Path(__file__).parent / "videos_to_analyze.csv"),
                        help="CSV de entrada con URLs (por defecto: videos_to_analyze.csv)")
    parser.add_argument("--llm", action="store_true",
                        help="Usar extractor LLM en dos pasos (requiere ANTHROPIC_API_KEY)")
    parser.add_argument("--max-claims", type=int, default=5,
                        help="Máximo de afirmaciones por vídeo (por defecto: 5)")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Umbral mínimo ClaimExtractor (por defecto: 0.30)")
    parser.add_argument("--resume", type=str, default=None,
                        help="CSV de un batch anterior para reanudar desde donde se dejó")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Calcula métricas sobre un CSV ya etiquetado manualmente")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Solo descarga y transcribe; guarda la caché y termina")
    parser.add_argument("--transcript-cache", type=str, default=DEFAULT_CACHE,
                        help=f"Ruta del JSON de caché de transcripciones (por defecto: {DEFAULT_CACHE})")
    args = parser.parse_args()

    if args.metrics:
        compute_metrics(args.metrics)
    elif args.transcribe_only:
        transcribe_only(args.input, cache_path=args.transcript_cache)
    else:
        run_batch(
            input_csv=args.input,
            max_claims=args.max_claims,
            threshold=args.threshold,
            resume_file=args.resume,
            use_llm=args.llm,
            cache_path=args.transcript_cache,
        )
