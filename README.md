# Fact Checker — Verificador automático de afirmaciones en español

Sistema de verificación automática de afirmaciones (*fact-checking*) para contenido textual y audiovisual en español. El pipeline extrae afirmaciones verificables de un texto o vídeo de YouTube, recupera evidencia de fuentes abiertas (Wikipedia, DuckDuckGo, RSS) y emite un veredicto mediante un modelo NLI multilingüe.

---

## Requisitos

- **Python 3.11 o 3.12 (recomendado)** — descargable desde [python.org](https://www.python.org/downloads/). Durante la instalación en Windows, marcar la opción **"Add Python to PATH"**. Se desaconseja usar Python 3.13 o superior, ya que algunas dependencias como PyTorch o faster-whisper aún no tienen soporte estable para esas versiones.
- Conexión a internet (para recuperación de evidencia)
- CPU con al menos 8 GB de RAM recomendados (la inferencia NLI se ejecuta en CPU)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<tu-usuario>/<nombre-repo>.git
cd <nombre-repo>
```

### 2. Crear y activar un entorno virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Ejecutar el script de configuración

Instala todas las dependencias y descarga el modelo de spaCy en español:

```bash
python setup.py
```

Esto es equivalente a ejecutar manualmente:

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `transformers` | Modelo NLI (mDeBERTa-v3) |
| `sentence-transformers` | CrossEncoder para reranking |
| `torch` | Inferencia de modelos |
| `spacy` | Segmentación y extracción de entidades |
| `faster-whisper` | Transcripción de audio local |
| `yt-dlp` | Descarga de vídeos de YouTube |
| `duckduckgo-search` | Búsqueda web sin API key |
| `streamlit` | Interfaz web |
| `anthropic` | Extractor de afirmaciones basado en LLM (opcional) |
| `deep-translator` | Traducción automática para textos no españoles |
| `langdetect` | Detección de idioma |

---

## Uso

### Interfaz web (recomendado)

```bash
python -m streamlit run app.py
```

Abre el navegador en `http://localhost:8501`. Desde la interfaz se puede:
- Pegar un texto o introducir una URL de YouTube para analizarlo
- Seleccionar el modo de extracción: heurístico (por defecto) o basado en LLM
- Ver los veredictos, fuentes y nivel de confianza de cada afirmación

**Extractor LLM (opcional):** el modo LLM utiliza Claude Haiku para extraer afirmaciones de mayor calidad semántica. Requiere una API key de Anthropic definida como variable de entorno antes de lanzar la aplicación:

```bash
# Windows
set ANTHROPIC_API_KEY=tu_api_key
# Linux / macOS
export ANTHROPIC_API_KEY=tu_api_key
```

---

## Evaluación

### Ejecutar evaluación sobre el benchmark estático

```bash
python evaluation/evaluate_benchmark.py
```

### Ejecutar evaluación sobre vídeos de YouTube

```bash
python evaluation/evaluate_videos.py
```

### Generar un nuevo benchmark

```bash
python evaluation/generate_benchmark.py
```

---

## Uso por consola (avanzado)

### Texto libre

```bash
python main.py
```

Pega o escribe el texto cuando se solicite y pulsa Enter dos veces para iniciar el análisis.

### Desde archivo de texto

```bash
python main.py texto.txt
```

### Guardar resultados en JSON

```bash
python main.py texto.txt --save
```

Los resultados se guardan en `results.json`.

### Usar el extractor LLM desde consola

```bash
python main.py --llm
```

---

## Estructura del proyecto

```
├── main.py                  # Punto de entrada principal (consola)
├── app.py                   # Interfaz web con Streamlit
├── config.py                # Parámetros y configuración global
├── requirements.txt         # Dependencias Python
├── setup.py                 # Script de instalación inicial
├── pipeline/
│   ├── claim_extractor.py       # Extractor heurístico de afirmaciones
│   ├── claim_extractor_llm.py   # Extractor basado en LLM (Claude)
│   ├── evidence_retriever.py    # Recuperación de evidencia (Wikipedia, web, RSS)
│   ├── verifier.py              # Verificación NLI + reranking
│   └── quality_judge.py         # Control de calidad y calibración
├── scrapers/
│   ├── wikipedia_scraper.py     # Búsqueda en Wikipedia (API REST)
│   ├── web_scraper.py           # Búsqueda web (DuckDuckGo)
│   └── rss_scraper.py           # Búsqueda en feeds RSS
├── utils/
│   ├── text_preprocessing.py    # Segmentación y preprocesamiento con spaCy
│   └── multilingual.py          # Detección de idioma y traducción
├── video/
│   ├── youtube_downloader.py    # Descarga de vídeos con yt-dlp
│   ├── transcriber.py           # Transcripción con faster-whisper
│   └── video_pipeline.py        # Pipeline completo para vídeo
├── training/
│   ├── train_claim_detector.py  # Entrenamiento del detector de afirmaciones
│   ├── train_nli.py             # Entrenamiento del modelo NLI
│   └── dataset_utils.py         # Utilidades para datasets
├── evaluation/
│   ├── benchmark_claims.csv         # Dataset de evaluación (1.999 afirmaciones)
│   ├── evaluate_benchmark.py        # Evaluación sobre benchmark estático
│   ├── evaluate_videos.py           # Evaluación sobre vídeos reales
│   ├── evaluate_pipeline.py         # Evaluación de componentes individuales
│   ├── generate_benchmark.py        # Generación del benchmark
│   └── aplicar_etiquetas_videos.py  # Etiquetado manual de vídeos
└── notebooks/
    └── kaggle_train.py          # Script de entrenamiento en Kaggle (GPU)
```

---

## Modelos utilizados

Los modelos se descargan automáticamente de Hugging Face en el primer uso:

- **NLI:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Transcripción:** `faster-whisper` modelo `small` (descargado automáticamente)
- **spaCy:** `es_core_news_lg` (instalado con `setup.py`)
