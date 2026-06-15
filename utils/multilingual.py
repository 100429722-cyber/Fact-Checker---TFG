# Detección de idioma y traducción automática al español.
# Dependencias opcionales: langdetect, deep-translator.

from __future__ import annotations

_LANG_NAMES: dict[str, str] = {
    "es": "español", "en": "inglés",  "fr": "francés",   "pt": "portugués",
    "de": "alemán",  "it": "italiano","ca": "catalán",   "gl": "gallego",
    "eu": "euskera", "nl": "neerlandés","pl": "polaco",  "ru": "ruso",
    "ar": "árabe",   "zh": "chino",   "ja": "japonés",
}

_CHUNK_SIZE = 4500  # máximo de caracteres por petición a Google Translate


def lang_name(code: str) -> str:
    """Devuelve el nombre en español del código ISO 639-1."""
    return _LANG_NAMES.get(code, code)


def detect_language(text: str) -> str:
    """Detecta el idioma del texto. Devuelve 'es' si falla."""
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
        DetectorFactory.seed = 0
        return detect(text[:800])
    except ImportError:
        print("  [Multilingüe] 'langdetect' no instalado. Asumiendo español.")
        return "es"
    except Exception:
        return "es"


def translate_to_spanish(text: str, source_lang: str = "auto") -> str:
    """Traduce texto al español con Google Translate. Maneja textos largos por fragmentos."""
    if source_lang == "es":
        return text
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("  [Multilingüe] 'deep-translator' no instalado. Usando texto original.")
        return text
    try:
        translator = GoogleTranslator(source=source_lang, target="es")
        if len(text) <= _CHUNK_SIZE:
            return translator.translate(text) or text
        chunks = [text[i: i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
        return " ".join(translator.translate(chunk) or chunk for chunk in chunks)
    except Exception as e:
        print(f"  [Multilingüe] Traducción fallida: {e}. Usando texto original.")
        return text
