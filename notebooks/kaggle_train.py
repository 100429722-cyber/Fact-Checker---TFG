# ============================================================
#  FACT CHECKER — Entrenamiento en Kaggle / Google Colab
#  Entrena los dos modelos: detector de afirmaciones + NLI
#
#  Instrucciones:
#  1. Sube este archivo a un notebook de Kaggle (New Notebook >
#     File > Import > sube kaggle_train.py y ejecútalo como script,
#     o copia cada bloque en celdas separadas)
#  2. Activa la GPU: Settings > Accelerator > GPU T4 x2
#  3. Al terminar, descarga la carpeta /kaggle/working/models/
#     y colócala en tu proyecto local como TFG 3/models/
# ============================================================

# ── Celda 1: Instalar dependencias ──────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q",
    "transformers", "datasets", "evaluate", "accelerate", "scikit-learn"
])

# ── Celda 2: Imports ────────────────────────────────────────
import os
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Celda 3: Configuración ──────────────────────────────────
BASE_MODEL        = "xlm-roberta-base"
OUTPUT_BASE       = "/kaggle/working/models"   # En Colab: "/content/models"
CLAIM_OUTPUT      = f"{OUTPUT_BASE}/claim_detector"
NLI_OUTPUT        = f"{OUTPUT_BASE}/nli_verifier"

BATCH_SIZE        = 64    # T4 lo aguanta bien con seq_len=256
LEARNING_RATE     = 2e-5
NUM_EPOCHS        = 3
CLAIM_MAX_LEN     = 256
NLI_MAX_LEN       = 256   # 256 es suficiente para pares NLI cortos; 512 cuadruplica el tiempo

# Límite de ejemplos NLI para entrenamiento rápido (~30-45 min en T4).
# Con 50k ejemplos se obtiene >82% accuracy. Usar None para el dataset completo (392k, ~4h).
NLI_MAX_TRAIN_SAMPLES = 50_000
NLI_MAX_VAL_SAMPLES   = 2_490   # validación completa siempre

os.makedirs(CLAIM_OUTPUT, exist_ok=True)
os.makedirs(NLI_OUTPUT, exist_ok=True)
print("Directorios creados.")


# ════════════════════════════════════════════════════════════
#  PARTE 1 — Detector de afirmaciones
# ════════════════════════════════════════════════════════════

# ── Celda 4: Cargar dataset de detección ────────────────────
def load_claim_dataset():
    """Intenta CLEF CheckThat! 2022; si no está disponible usa dataset sintético."""
    for name, config in [
        ("clef2022_checkthat_v2", "spanish"),
        ("clef2021_checkthat_v2", "spanish"),
    ]:
        try:
            ds = load_dataset(name, config)
            print(f"Dataset cargado: {name} ({config})")
            return ds, True
        except Exception:
            continue

    print("CLEF no disponible. Usando dataset sintetico de demo.")
    examples = {
        "text": [
            "El gobierno espanol aumento el presupuesto de educacion un 20%.",
            "Espana tiene la tasa de desempleo juvenil mas alta de la UE.",
            "La inflacion en la eurozona supero el 10% en 2022.",
            "El presidente visito Bruselas ayer por la tarde.",
            "Madrid es la ciudad mas poblada de la Union Europea.",
            "La vacuna COVID redujo las hospitalizaciones en un 90%.",
            "Hoy hace un dia muy bonito en la capital.",
            "Me parece que la pelicula estuvo bien.",
            "El tiempo es agradable esta semana.",
            "Esta cancion me encanta mucho.",
            "El PIB de Espana crecio un 5,5% en 2021.",
            "Espana tiene mas de 47 millones de habitantes.",
            "El BCE subio los tipos de interes al 4,5%.",
            "Manana hay partido de futbol.",
            "El nuevo album del artista salio ayer.",
            "El paro bajo al 11% en el tercer trimestre.",
            "El Congreso aprobo los presupuestos generales.",
            "Me gusta mucho el cafe por las mananas.",
        ],
        "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    }
    split = int(len(examples["text"]) * 0.8)
    return DatasetDict({
        "train": Dataset.from_dict({k: v[:split] for k, v in examples.items()}),
        "test":  Dataset.from_dict({k: v[split:] for k, v in examples.items()}),
    }), False


def normalize_claim_dataset(ds):
    def _norm(ex):
        text = (ex.get("tweet_text") or ex.get("sentence") or ex.get("text", ""))
        label = ex.get("class_label") or ex.get("label", 0)
        if isinstance(label, str):
            label = 1 if label.lower() in ("yes", "1", "true", "checkworthy") else 0
        return {"text": text, "label": int(label)}
    return ds.map(_norm)


# ── Celda 5: Entrenar detector de afirmaciones ──────────────
def train_claim_detector():
    print("\n" + "="*55)
    print("  ENTRENANDO DETECTOR DE AFIRMACIONES")
    print("="*55)

    ds, is_clef = load_claim_dataset()
    if is_clef:
        ds = normalize_claim_dataset(ds)

    eval_split = "test" if "test" in ds else "validation"
    print(f"Train: {len(ds['train'])}  |  {eval_split}: {len(ds[eval_split])}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=CLAIM_MAX_LEN)

    tokenized = ds.map(tokenize, batched=True)
    keep = {"input_ids", "attention_mask", "label"}
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
        id2label={0: "NO_AFIRMACION", 1: "AFIRMACION"},
        label2id={"NO_AFIRMACION": 0, "AFIRMACION": 1},
    )

    acc_metric = evaluate.load("accuracy")
    f1_metric  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            **acc_metric.compute(predictions=preds, references=labels),
            **f1_metric.compute(predictions=preds, references=labels, average="binary"),
        }

    args = TrainingArguments(
        output_dir=CLAIM_OUTPUT,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model(CLAIM_OUTPUT)
    tokenizer.save_pretrained(CLAIM_OUTPUT)
    print(f"Modelo guardado en {CLAIM_OUTPUT}")

    # Liberar VRAM antes del siguiente entrenamiento
    import gc
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM liberada.")

    return eval_results


# ════════════════════════════════════════════════════════════
#  PARTE 2 — Verificador NLI
# ════════════════════════════════════════════════════════════

# ── Celda 6: Entrenar verificador NLI ───────────────────────
def train_nli_verifier():
    print("\n" + "="*55)
    print("  ENTRENANDO VERIFICADOR NLI (XNLI español)")
    print("="*55)

    ds = load_dataset("xnli", "es")

    # Subconjunto balanceado para entrenamiento rápido
    if NLI_MAX_TRAIN_SAMPLES and NLI_MAX_TRAIN_SAMPLES < len(ds["train"]):
        ds["train"] = ds["train"].shuffle(seed=42).select(range(NLI_MAX_TRAIN_SAMPLES))
        print(f"  Usando subconjunto de {NLI_MAX_TRAIN_SAMPLES} ejemplos de train.")
        print(f"  Para dataset completo pon NLI_MAX_TRAIN_SAMPLES = None (tarda ~4h en T4).")

    print(f"Train: {len(ds['train'])}  |  Val: {len(ds['validation'])}  |  Test: {len(ds['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_nli(examples):
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", truncation=True, max_length=NLI_MAX_LEN)

    tokenized = ds.map(tokenize_nli, batched=True,
                       remove_columns=["premise", "hypothesis"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3,
        id2label={0: "APOYA", 1: "NO_VERIFICA", 2: "REFUTA"},
        label2id={"APOYA": 0, "NO_VERIFICA": 1, "REFUTA": 2},
    )

    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc_metric.compute(predictions=preds, references=labels)

    # batch 16 + acumulación 4 = batch efectivo 64, con menor pico de VRAM
    args = TrainingArguments(
        output_dir=NLI_OUTPUT,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(NLI_OUTPUT)
    tokenizer.save_pretrained(NLI_OUTPUT)
    print(f"Modelo guardado en {NLI_OUTPUT}")

    results = trainer.evaluate(tokenized["test"])
    print(f"Test: {results}")
    return results


# ── Celda 7: Ejecutar todo ───────────────────────────────────
if __name__ == "__main__":
    claim_results = train_claim_detector()
    print(f"\nDetector afirmaciones: {claim_results}")

    nli_results = train_nli_verifier()
    print(f"\nVerificador NLI: {nli_results}")

    print("\n" + "="*55)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"  Modelos en: {OUTPUT_BASE}")
    print("  Descarga la carpeta 'models' y colócala en tu")
    print("  proyecto local: TFG 3/models/")
    print("="*55)

    # ── Celda 8: Comprimir modelos para descarga ─────────────────
    import shutil

    # Borrar checkpoints para liberar espacio (~1-2 GB)
    for model_dir in [CLAIM_OUTPUT, NLI_OUTPUT]:
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(model_dir, item))
                print(f"Eliminado: {item}")

    total, used, free = shutil.disk_usage("/kaggle/working")
    print(f"Espacio libre: {free / (1024**3):.1f} GB")

    shutil.make_archive("/kaggle/working/claim_detector", "zip",
                        OUTPUT_BASE, "claim_detector")
    print("claim_detector.zip:", os.path.getsize("/kaggle/working/claim_detector.zip") // (1024**2), "MB")

    shutil.make_archive("/kaggle/working/nli_verifier", "zip",
                        OUTPUT_BASE, "nli_verifier")
    print("nli_verifier.zip:", os.path.getsize("/kaggle/working/nli_verifier.zip") // (1024**2), "MB")
