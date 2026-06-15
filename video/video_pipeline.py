# Pipeline completo: URL de YouTube → descarga → transcripción → fact-checking.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.youtube_downloader import download_audio
from video.transcriber import transcribe


def analyze_video(url: str, keep_audio: bool = False) -> dict:
    """Descarga, transcribe y verifica afirmaciones de un vídeo de YouTube."""
    # Importación diferida para no cargar los modelos hasta que se necesiten
    from main import run_fact_checker

    print(f"\n{'═'*65}")
    print(f"  VIDEO FACT-CHECKER")
    print(f"{'═'*65}")
    print(f"\n[1/3] Descargando audio de:\n      {url}\n")
    audio_path = download_audio(url)
    print(f"      Guardado en: {audio_path}")

    print(f"\n[2/3] Transcribiendo audio...")
    transcript = transcribe(audio_path)

    if not keep_audio:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    if not transcript.strip():
        print("  La transcripcion esta vacia. Comprueba el video o el idioma configurado.")
        return {"transcript": "", "claims": [], "results": []}

    print(f"\n  Transcripcion ({len(transcript)} caracteres):")
    preview = transcript[:500]
    print(f"  «{preview}{'...' if len(transcript) > 500 else ''}»")

    print(f"\n[3/3] Analizando afirmaciones...\n")
    results = run_fact_checker(transcript)

    return {"transcript": transcript, "results": results}


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Fact-checker para videos de YouTube")
    parser.add_argument("url", help="URL de YouTube")
    parser.add_argument("--keep-audio", action="store_true", help="Conservar el archivo WAV tras la transcripcion")
    args = parser.parse_args()

    result = analyze_video(args.url, keep_audio=args.keep_audio)
    if not result["results"]:
        print("\n  No se encontraron afirmaciones verificables en el video.")


if __name__ == "__main__":
    _cli()
