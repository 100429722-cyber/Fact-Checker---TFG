# -*- coding: utf-8 -*-
"""
Script de configuracion inicial del entorno.

Ejecutar una sola vez antes de empezar:
    py setup.py
"""

import io
import os
import subprocess
import sys

# Forzar UTF-8 en la salida para evitar errores de codificacion en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run(cmd: str, description: str):
    print(f"\n>>> {description}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  AVISO: el comando terminó con código {result.returncode}")
    else:
        print(f"  OK")


def create_dirs():
    dirs = [
        "data/raw",
        "data/processed",
        "models/claim_detector",
        "models/nli_verifier",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directorios creados.")


def main():
    print("=" * 60)
    print("  Configuración del entorno — Fact Checker")
    print("=" * 60)

    create_dirs()

    run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalando dependencias de Python...",
    )

    run(
        f"{sys.executable} -m spacy download es_core_news_lg",
        "Descargando modelo spaCy en español (es_core_news_lg)...",
    )

    print("\n" + "=" * 60)
    print("  Configuración completada.")
    print("\n  Proximos pasos:")
    print("  1. Entrenar detector de afirmaciones:")
    print("       py training/train_claim_detector.py")
    print("  2. Entrenar modelo NLI:")
    print("       py training/train_nli.py")
    print("  3. Ejecutar el pipeline:")
    print("       py main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
