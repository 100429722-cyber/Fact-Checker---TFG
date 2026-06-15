"""
Transcripción local de audio con faster-whisper (sin API key).
Limpia artefactos habituales de Whisper: [Música], [Aplausos], etc.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

# Artefactos que Whisper inserta y que no son texto real
_ARTIFACTS_RE = re.compile(
    r"\[(?:música|aplausos|risas|inaudible|silencio|music|applause|laughter|noise"
    r"|ruido|subtítulos|subtitulos|traducción|traduccion)\]",
    re.IGNORECASE,
)


def _clean_transcript(text: str) -> str:
    text = _ARTIFACTS_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def transcribe(audio_path: str) -> str:
    """
    Transcribe un archivo de audio WAV con faster-whisper.

    Args:
        audio_path: Ruta al archivo de audio (WAV/MP3/cualquier formato ffmpeg)

    Returns:
        Transcripción limpia como cadena de texto.

    Raises:
        ImportError : si faster-whisper no está instalado
        FileNotFoundError: si el archivo no existe
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper no está instalado. Ejecuta: pip install faster-whisper"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_path}")

    print(f"[Transcriber] Cargando modelo Whisper '{WHISPER_MODEL}' en {WHISPER_DEVICE}...")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")

    lang = WHISPER_LANGUAGE if WHISPER_LANGUAGE else None
    print(f"[Transcriber] Transcribiendo '{os.path.basename(audio_path)}'...")

    segments, info = model.transcribe(
        audio_path,
        language=lang,
        beam_size=5,
        vad_filter=True,          # filtra silencios con VAD
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected = info.language
    print(f"[Transcriber] Idioma detectado: {detected}  |  Duración: {info.duration:.1f}s")

    raw_text = " ".join(seg.text.strip() for seg in segments)
    return _clean_transcript(raw_text)
