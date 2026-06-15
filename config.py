import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelo base para entrenamiento de modelos propios
BASE_MODEL = "xlm-roberta-base"

# Rutas de los modelos entrenados (no usados en producción)
CLAIM_DETECTOR_PATH = os.path.join(BASE_DIR, "models", "claim_detector_notused")
NLI_VERIFIER_PATH   = os.path.join(BASE_DIR, "models", "nli_verifier_notused")

# Modelo NLI preentrenado usado en producción
NLI_PRETRAINED_ES = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Hiperparámetros de entrenamiento
CLAIM_DETECTOR_MAX_LEN = 256
NLI_MAX_LEN   = 512
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 3

# Recuperación de evidencia
MAX_SEARCH_RESULTS = 5
MAX_EVIDENCE_CHARS = 3000

# CrossEncoder para reranking semántico
RERANKER_MODEL     = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_TOP_K     = 5
RERANKER_MIN_SCORE = 0.05
RERANKER_GATE      = 0.15  # por debajo de este umbral → NO_VERIFICABLE sin correr NLI

# Transcripción con Whisper (local, sin API key)
WHISPER_MODEL    = "small"
WHISPER_LANGUAGE = "es"
WHISPER_DEVICE   = "cpu"
DOWNLOADS_DIR    = "downloads"

# Feeds RSS de medios verificadores
RSS_FEEDS = {
    "Maldita.es":   "https://maldita.es/feed/",
    "Newtral":      "https://newtral.es/feed/",
    "EFE Verifica": "https://efeverifica.es/feed/",
    "AFP Factual":  "https://factual.afp.com/feed/",
}
