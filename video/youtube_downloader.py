"""
Descarga el audio de un vídeo de YouTube usando yt-dlp.
No requiere ffmpeg: descarga en el formato nativo (webm/m4a) que
faster-whisper puede transcribir directamente con su propio libav.
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DOWNLOADS_DIR

_BROWSERS = ["chrome", "firefox", "edge", "opera", "brave"]

_BASE_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    },
}


def _try_download(ydl_opts: dict, url: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=True)


def _find_audio_file(dest: str, video_id: str) -> str:
    """Busca el archivo descargado por video_id con cualquier extensión."""
    pattern = os.path.join(dest, f"{video_id}.*")
    matches = [f for f in glob.glob(pattern) if not f.endswith(".json")]
    if not matches:
        raise RuntimeError(
            f"Descarga completada pero no se encontró el archivo de audio "
            f"para id='{video_id}' en '{dest}'"
        )
    return os.path.abspath(matches[0])


def download_audio(url: str, output_dir: str = None) -> str:
    """
    Descarga el audio de una URL de YouTube en formato nativo (sin ffmpeg).

    Intenta primero sin cookies; si YouTube devuelve 403, prueba con las
    cookies de cada navegador instalado hasta que uno funcione.

    Args:
        url       : URL de YouTube (cualquier formato válido)
        output_dir: Carpeta de destino (por defecto DOWNLOADS_DIR de config.py)

    Returns:
        Ruta absoluta al archivo de audio descargado (webm, m4a, etc.)

    Raises:
        ImportError : si yt-dlp no está instalado
        ValueError  : si la URL no es válida
        RuntimeError: si la descarga falla con todos los métodos disponibles
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp no está instalado. Ejecuta: py -m pip install yt-dlp")

    if not url or not url.strip():
        raise ValueError("La URL no puede estar vacía.")

    dest = output_dir or DOWNLOADS_DIR
    os.makedirs(dest, exist_ok=True)

    base = dict(_BASE_OPTS)
    base["outtmpl"] = os.path.join(dest, "%(id)s.%(ext)s")

    last_error = None

    # ── Intento 1: sin cookies ───────────────────────────────────────────────
    try:
        info = _try_download(base, url)
        return _find_audio_file(dest, info["id"])
    except yt_dlp.utils.DownloadError as exc:
        last_error = exc
        if "403" not in str(exc) and "Forbidden" not in str(exc):
            raise RuntimeError(f"Error al descargar el vídeo: {exc}") from exc

    # ── Intentos 2-N: con cookies de navegadores ─────────────────────────────
    for browser in _BROWSERS:
        opts = dict(base)
        opts["cookiesfrombrowser"] = (browser,)
        try:
            print(f"[Downloader] Reintentando con cookies de {browser}...")
            info = _try_download(opts, url)
            return _find_audio_file(dest, info["id"])
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"No se pudo descargar el vídeo (403 Forbidden). "
        f"Abre YouTube en tu navegador e inicia sesión, luego vuelve a intentarlo.\n"
        f"Último error: {last_error}"
    ) from last_error
