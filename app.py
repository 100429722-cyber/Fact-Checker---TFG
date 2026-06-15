"""
Interfaz web del Fact Checker — v2.
Caché de transcripciones y resultados, visualización paso a paso.

Arrancar con:
    streamlit run app.py
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Fact Checker — Verificador de afirmaciones",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .verdict-VERDADERO      { background:#d4edda; border-left:5px solid #28a745;
                              padding:12px 16px; border-radius:4px; margin-bottom:8px; }
    .verdict-FALSO          { background:#f8d7da; border-left:5px solid #dc3545;
                              padding:12px 16px; border-radius:4px; margin-bottom:8px; }
    .verdict-NO_VERIFICABLE { background:#fff3cd; border-left:5px solid #ffc107;
                              padding:12px 16px; border-radius:4px; margin-bottom:8px; }
    .claim-header           { font-size:1.05rem; font-weight:600; margin-bottom:4px; }
    .source-row             { font-size:0.85rem; color:#555; margin:4px 0; }
    .quality-chip           { display:inline-block; padding:2px 10px; border-radius:12px;
                              font-size:0.78rem; font-weight:600; }
    .chip-OK            { background:#d4edda; color:#155724; }
    .chip-SINGLE_SOURCE { background:#cce5ff; color:#004085; }
    .chip-LOW_AGREEMENT { background:#fff3cd; color:#856404; }
    .chip-CONFLICT      { background:#f8d7da; color:#721c24; }
    .chip-NA            { background:#e2e3e5; color:#383d41; }
    .cache-badge        { background:#e3f2fd; border:1px solid #90caf9; padding:3px 10px;
                          border-radius:12px; font-size:0.80rem; color:#1565c0;
                          font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Rutas de caché ─────────────────────────────────────────────────────────────
_ROOT             = Path(__file__).parent
_CACHE_DIR        = _ROOT / "cache"
_TRANSCRIPTS_FILE = _CACHE_DIR / "transcripts.json"
_RESULTS_DIR      = _CACHE_DIR / "results"

_CACHE_DIR.mkdir(exist_ok=True)
_RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers: transcripciones ──────────────────────────────────────────────────

def _load_transcripts() -> dict:
    if _TRANSCRIPTS_FILE.exists():
        try:
            return json.loads(_TRANSCRIPTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_transcript(url: str, transcript: str) -> None:
    data = _load_transcripts()
    data[url] = {
        "transcript": transcript,
        "n_words":    len(transcript.split()),
        "cached_at":  datetime.now().isoformat(),
    }
    _TRANSCRIPTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Helpers: resultados ────────────────────────────────────────────────────────

def _make_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _load_result(h: str) -> dict | None:
    p = _RESULTS_DIR / f"{h}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_result(h: str, data: dict) -> None:
    p = _RESULTS_DIR / f"{h}.json"
    try:
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


def _list_history() -> list[dict]:
    items = []
    for p in sorted(_RESULTS_DIR.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "hash":        p.stem,
                "input_type":  d.get("input_type", "text"),
                "preview":     d.get("input_preview", ""),
                "n_claims":    len(d.get("claims", [])),
                "analyzed_at": d.get("analyzed_at", "")[:16].replace("T", " "),
                "use_llm":     d.get("use_llm", False),
            })
        except Exception:
            continue
    return items


# ── Modelos (cacheados en sesión) ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelos de IA… (solo ocurre la primera vez)")
def _load_pipeline():
    from pipeline.evidence_retriever import EvidenceRetriever
    from pipeline.verifier import Verifier
    return EvidenceRetriever(), Verifier()


# ── Renderizado de un resultado ────────────────────────────────────────────────

_VERDICT_LABEL = {
    "VERDADERO":      "VERDADERO",
    "FALSO":          "FALSO",
    "NO_VERIFICABLE": "NO VERIFICABLE",
}
_QUALITY_LABEL = {
    "OK":            ("OK",              "chip-OK"),
    "SINGLE_SOURCE": ("Una sola fuente", "chip-SINGLE_SOURCE"),
    "LOW_AGREEMENT": ("Bajo acuerdo",    "chip-LOW_AGREEMENT"),
    "CONFLICT":      ("Conflicto",       "chip-CONFLICT"),
    "N/A":           ("Sin datos",       "chip-NA"),
}


def _render_result(result: dict, idx: int) -> None:
    verdict = result.get("verdict", "NO_VERIFICABLE")
    conf    = float(result.get("confidence",          0.0))
    nli_c   = float(result.get("nli_confidence",      0.0))
    ret_c   = float(result.get("retrieval_confidence",0.0))
    flag    = result.get("quality_flag", "N/A")
    sources = result.get("sources", [])
    expl    = result.get("explanation", "")

    q_label, q_class = _QUALITY_LABEL.get(flag, ("?", "chip-NA"))

    st.markdown(f"""
    <div class="verdict-{verdict}">
      <div class="claim-header">Afirmación {idx}: {result.get("claim","")}</div>
      <strong>{_VERDICT_LABEL.get(verdict, verdict)}</strong>
      &nbsp;&nbsp;
      <span class="quality-chip {q_class}">{q_label}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Confianza total",   f"{conf*100:.1f}%")
    c2.metric("Confianza NLI",     f"{nli_c*100:.1f}%")
    c3.metric("Calidad evidencia", f"{ret_c*100:.1f}%")

    if expl:
        st.caption(expl)

    if sources:
        with st.expander(f"{len(sources)} fuente(s) analizadas"):
            for s in sources:
                st.markdown(
                    f"<div class='source-row'>"
                    f"<strong>{s.get('title','Sin título')}</strong> — "
                    f"<em>{s.get('source','')}</em> "
                    f"({float(s.get('confidence',0))*100:.0f}%)<br>"
                    f"<a href='{s.get('url','')}' target='_blank'>{s.get('url','')}</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                snippet = s.get("relevant_snippet", "")
                if snippet:
                    st.markdown(
                        f"> *«{snippet[:300]}{'…' if len(snippet)>300 else ''}»*"
                    )
                st.divider()

    st.markdown("---")


def _show_all_results(data: dict, from_cache: bool = False) -> None:
    """Muestra transcript + claims + veredictos de un análisis guardado."""
    if from_cache:
        st.markdown(
            '<span class="cache-badge">Resultado cargado desde caché</span>',
            unsafe_allow_html=True,
        )
        st.write("")

    if data.get("translated"):
        st.info(f"Texto traducido automáticamente desde **{data.get('lang','?')}** al español.")

    # Paso 1 — Texto / Transcripción
    st.markdown("### Paso 1 — Texto analizado")
    if data.get("input_type") == "video":
        st.caption(f"Vídeo: {data.get('video_url','')}")
        with st.expander("Ver transcripción completa", expanded=False):
            st.write(data.get("transcript", ""))
    else:
        with st.expander("Ver texto completo", expanded=False):
            st.write(data.get("text", ""))

    # Paso 2 — Afirmaciones
    claims = data.get("claims", [])
    st.markdown(f"### Paso 2 — {len(claims)} afirmación(es) detectadas")
    if not claims:
        st.warning("No se detectaron afirmaciones verificables.")
        return
    for i, c in enumerate(claims, 1):
        score = float(c.get("score", c.get("confidence", 0)))
        st.markdown(f"**{i}.** {c['text']}  *(puntuación: {score:.2f})*")

    # Paso 3 — Veredictos
    st.markdown("---")
    st.markdown("### Paso 3 — Resultados de verificación")
    for i, result in enumerate(data.get("results", []), 1):
        _render_result(result, i)


# ── Extracción de afirmaciones ─────────────────────────────────────────────────

def _extract_claims(text: str, min_conf: float, use_llm: bool, max_claims: int,
                    api_key: str = ""):
    from utils.multilingual import detect_language, translate_to_spanish
    lang       = detect_language(text)
    translated = False
    if lang != "es":
        text       = translate_to_spanish(text, source_lang=lang)
        translated = True

    if use_llm:
        from pipeline.claim_extractor_llm import ClaimExtractorLLM
        extractor = ClaimExtractorLLM(api_key=api_key or None)
    else:
        from pipeline.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor(min_words=5)

    claims = extractor.extract_claims(text, min_confidence=min_conf)[:max_claims]
    return text, lang, translated, claims


# ── Verificación paso a paso ──────────────────────────────────────────────────

def _verify_step_by_step(claims: list) -> list:
    """Verifica afirmaciones una a una y renderiza cada resultado en tiempo real."""
    retriever, verifier = _load_pipeline()
    results  = []
    n        = len(claims)
    progress = st.progress(0.0, text=f"Verificando 0 / {n}…")

    st.markdown("### Paso 3 — Resultados de verificación")
    box = st.container()

    for i, claim in enumerate(claims):
        progress.progress(
            i / n,
            text=f"Verificando {i+1}/{n}: «{claim['text'][:60]}…»",
        )
        evidence = retriever.retrieve(claim["text"], max_results=5)
        result   = verifier.verify_claim(claim["text"], evidence)
        results.append(result)
        with box:
            _render_result(result, i + 1)

    progress.progress(1.0, text="Verificación completada")
    return results


# ── Sidebar: configuración e historial ────────────────────────────────────────

with st.sidebar:
    st.title("Configuración")

    max_claims = st.slider("Máx. afirmaciones a verificar", 1, 10, 5)
    min_conf   = st.slider("Umbral detección (heurístico)", 0.10, 0.90, 0.30, step=0.05,
                           help="Solo se verifican frases cuya puntuación supere este valor.")
    use_llm    = st.toggle(
        "Extractor LLM (Claude Haiku)",
        value=False,
        help="Más preciso pero requiere ANTHROPIC_API_KEY y ~5 s extra por análisis.",
    )

    llm_api_key = ""
    if use_llm:
        llm_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Clave de API de Anthropic necesaria para usar Claude Haiku.",
        )
        if not llm_api_key.strip():
            st.warning("Introduce la API key para activar el extractor LLM.")
            use_llm = False

    st.markdown("---")
    st.markdown("### Historial de análisis")

    history = _list_history()
    if not history:
        st.caption("Aún no hay análisis guardados.")
    else:
        for item in history[:12]:
            type_label = "Video" if item["input_type"] == "video" else "Texto"
            llm_label  = " · LLM" if item["use_llm"] else ""
            n_cl       = item["n_claims"]
            date       = item["analyzed_at"]
            label      = item["preview"]
            label      = (label[:38] + "…") if len(label) > 40 else label
            btn_label  = f"[{type_label}] {date}{llm_label}  [{n_cl} claims]\n{label}"
            if st.button(btn_label, key=f"h_{item['hash']}", use_container_width=True):
                st.session_state["load_hash"] = item["hash"]
                st.rerun()

    st.markdown("---")
    st.markdown("""
**Sobre el sistema**

1. **Extracción** — heurístico o LLM en 2 pasos
2. **Evidencia** — Wikipedia ES + DuckDuckGo + RSS
3. **Verificación** — BERT NLI + CrossEncoder
4. **Calidad** — acuerdo entre fuentes

*TFG · Juan Barga García · UC3M*
""")


# ── Cabecera ──────────────────────────────────────────────────────────────────

st.title("Fact Checker")
st.subheader("Verificador automático de afirmaciones en español")
st.caption(
    "Introduce un texto o la URL de un vídeo de YouTube. "
    "El sistema detectará afirmaciones, buscará evidencias y emitirá un veredicto."
)

# ── Historial: mostrar resultado cargado ──────────────────────────────────────

if "load_hash" in st.session_state:
    loaded = _load_result(st.session_state["load_hash"])
    if loaded:
        st.markdown("---")
        col_title, col_close = st.columns([5, 1])
        col_title.markdown("#### Resultado del historial")
        if col_close.button("Cerrar", use_container_width=True):
            del st.session_state["load_hash"]
            st.rerun()
        _show_all_results(loaded, from_cache=True)
        st.stop()
    else:
        del st.session_state["load_hash"]

# ── Pestañas de entrada ───────────────────────────────────────────────────────

tab_text, tab_video = st.tabs(["Texto libre", "Vídeo de YouTube"])


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA: Texto libre
# ══════════════════════════════════════════════════════════════════════════════

with tab_text:
    st.markdown("#### Pega aquí el texto que deseas verificar")
    default_text = (
        "España ingresó en la Unión Europea en 1986.\n"
        "Madrid es la ciudad más poblada de la Unión Europea.\n"
        "La selección española ganó el Mundial de Fútbol de 2010.\n"
        "Barcelona es la capital de España."
    )
    user_text = st.text_area(
        "Texto de entrada",
        value=default_text,
        height=180,
        label_visibility="collapsed",
        placeholder="Pega aquí el artículo o las frases a verificar…",
    )

    col_btn, col_force = st.columns([4, 1])
    btn_text    = col_btn.button("Verificar texto", type="primary", use_container_width=True)
    force_text  = col_force.checkbox("Forzar re-análisis", key="force_text",
                                     help="Ignora la caché y re-ejecuta el pipeline.")

    if btn_text and user_text.strip():
        text_hash = _make_hash(f"text|{user_text}|{use_llm}|{max_claims}|{min_conf:.2f}")
        cached    = None if force_text else _load_result(text_hash)

        if cached:
            _show_all_results(cached, from_cache=True)
        else:
            # ── Paso 1: extraer afirmaciones ──────────────────────────────
            st.markdown("### Paso 1 — Texto analizado")
            with st.spinner("Detectando idioma y extrayendo afirmaciones…"):
                try:
                    text_es, lang, translated, claims = _extract_claims(
                        user_text, min_conf, use_llm, max_claims, llm_api_key
                    )
                except Exception as e:
                    st.error(f"Error en la extracción de afirmaciones: {e}")
                    st.stop()

            if translated:
                st.info(f"Texto traducido automáticamente desde **{lang}** al español.")

            with st.expander("Ver texto completo", expanded=False):
                st.write(text_es)

            # ── Paso 2: mostrar afirmaciones ──────────────────────────────
            st.markdown(f"### Paso 2 — {len(claims)} afirmación(es) detectadas")
            if not claims:
                st.warning(
                    "No se detectaron afirmaciones verificables con el umbral actual. "
                    "Prueba a bajarlo en la barra lateral."
                )
                st.stop()

            for i, c in enumerate(claims, 1):
                score = float(c.get("score", c.get("confidence", 0)))
                st.markdown(f"**{i}.** {c['text']}  *(puntuación: {score:.2f})*")

            st.markdown("---")

            # ── Paso 3: verificar (tiempo real) ──────────────────────────
            results = _verify_step_by_step(claims)

            # Guardar en caché
            _save_result(text_hash, {
                "input_type":    "text",
                "input_preview": user_text[:80],
                "text":          text_es,
                "lang":          lang,
                "translated":    translated,
                "use_llm":       use_llm,
                "max_claims":    max_claims,
                "min_conf":      min_conf,
                "claims":        claims,
                "results":       results,
                "analyzed_at":   datetime.now().isoformat(),
            })

    elif btn_text:
        st.warning("Escribe o pega algún texto antes de verificar.")


# ══════════════════════════════════════════════════════════════════════════════
# PESTAÑA: Vídeo de YouTube
# ══════════════════════════════════════════════════════════════════════════════

with tab_video:
    st.markdown("#### Introduce la URL del vídeo de YouTube")
    st.caption(
        "El sistema descargará el audio, lo transcribirá con Whisper y verificará "
        "las afirmaciones detectadas. Las transcripciones se guardan en caché "
        "para no repetir la descarga."
    )

    yt_url = st.text_input(
        "URL de YouTube",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    col_btn2, col_force2 = st.columns([4, 1])
    btn_video   = col_btn2.button("Analizar vídeo", type="primary", use_container_width=True)
    force_video = col_force2.checkbox("Forzar re-análisis", key="force_video",
                                      help="Ignora la caché y re-ejecuta descargar + transcribir.")

    if btn_video and yt_url.strip():
        url_clean  = yt_url.strip()
        vid_hash   = _make_hash(f"video|{url_clean}|{use_llm}|{max_claims}|{min_conf:.2f}")
        cached_res = None if force_video else _load_result(vid_hash)

        if cached_res:
            _show_all_results(cached_res, from_cache=True)

        else:
            # ── Paso 1: transcripción ────────────────────────────────────
            st.markdown("### Paso 1 — Transcripción del vídeo")

            transcripts = _load_transcripts()
            if not force_video and url_clean in transcripts:
                transcript = transcripts[url_clean]["transcript"]
                n_words    = transcripts[url_clean].get("n_words", len(transcript.split()))
                st.success(
                    f"Transcripción cargada desde caché "
                    f"({n_words} palabras) — sin re-descarga."
                )
            else:
                col_dl, col_tr, col_fc = st.columns(3)
                try:
                    with col_dl:
                        with st.spinner("Descargando audio…"):
                            from video.youtube_downloader import download_audio
                            audio_path = download_audio(url_clean)
                        st.success("Audio descargado")

                    with col_tr:
                        with st.spinner("Transcribiendo con Whisper…"):
                            from video.transcriber import transcribe
                            import os
                            transcript = transcribe(audio_path)
                            try:
                                os.remove(audio_path)
                            except OSError:
                                pass
                        st.success("Transcripción lista")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

                if not transcript.strip():
                    st.warning(
                        "La transcripción está vacía. "
                        "Comprueba que el vídeo tiene audio en español."
                    )
                    st.stop()

                _save_transcript(url_clean, transcript)
                n_words = len(transcript.split())
                col_fc.success(f"Guardado en caché ({n_words} palabras)")

            with st.expander("Ver transcripción completa", expanded=False):
                st.write(transcript)

            # ── Paso 2: extraer afirmaciones ─────────────────────────────
            with st.spinner("Extrayendo afirmaciones de la transcripción…"):
                try:
                    text_es, lang, translated, claims = _extract_claims(
                        transcript, min_conf, use_llm, max_claims, llm_api_key
                    )
                except Exception as e:
                    st.error(f"Error en la extracción: {e}")
                    st.stop()

            if translated:
                st.info(f"Transcripción traducida desde **{lang}** al español.")

            st.markdown(f"### Paso 2 — {len(claims)} afirmación(es) detectadas")
            if not claims:
                st.warning("No se detectaron afirmaciones verificables en el vídeo.")
                st.stop()

            for i, c in enumerate(claims, 1):
                score = float(c.get("score", c.get("confidence", 0)))
                st.markdown(f"**{i}.** {c['text']}  *(puntuación: {score:.2f})*")

            st.markdown("---")

            # ── Paso 3: verificar (tiempo real) ──────────────────────────
            results = _verify_step_by_step(claims)

            # Guardar resultado completo
            _save_result(vid_hash, {
                "input_type":    "video",
                "input_preview": url_clean,
                "video_url":     url_clean,
                "transcript":    transcript,
                "text":          text_es,
                "lang":          lang,
                "translated":    translated,
                "use_llm":       use_llm,
                "max_claims":    max_claims,
                "min_conf":      min_conf,
                "claims":        claims,
                "results":       results,
                "analyzed_at":   datetime.now().isoformat(),
            })

    elif btn_video:
        st.warning("Introduce una URL de YouTube antes de analizar.")
