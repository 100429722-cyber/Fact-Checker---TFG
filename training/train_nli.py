"""
Entrena un modelo de NLI (Natural Language Inference) sobre XLM-RoBERTa.

Dataset : XNLI (español) — disponible directamente en HuggingFace.
Labels  : 0 = entailment  → APOYA la afirmación
          1 = neutral      → NO VERIFICA
          2 = contradiction → REFUTA la afirmación

Uso (local, rápido ~20 min CPU):
    py training/train_nli.py

Uso (dataset completo, requiere GPU — recomendado en Google Colab):
    py training/train_nli.py --full
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import evaluate

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    LEARNING_RATE,
    NLI_MAX_LEN,
    NLI_VERIFIER_PATH,
    NUM_EPOCHS,
)
from training.dataset_utils import load_xnli_spanish

# Tamaño del subconjunto para entrenamiento local (sin GPU).
# Usar --full para el dataset completo (392k ejemplos, requiere GPU).
LOCAL_TRAIN_SAMPLES = 5_000
LOCAL_VAL_SAMPLES = 500


def _tokenize_nli(examples, tokenizer):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=NLI_MAX_LEN,
    )


def _compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


def train(output_dir: str = NLI_VERIFIER_PATH, full_dataset: bool = False):
    print("Cargando dataset XNLI (español)...")
    dataset = load_xnli_spanish()

    if not full_dataset:
        # Subconjunto balanceado por clase para entrenamiento local
        train_subset = dataset["train"].shuffle(seed=42).select(range(LOCAL_TRAIN_SAMPLES))
        val_subset = dataset["validation"].shuffle(seed=42).select(range(LOCAL_VAL_SAMPLES))
        dataset = dataset.copy()
        dataset["train"] = train_subset
        dataset["validation"] = val_subset
        print(f"  Modo LOCAL — subconjunto de {LOCAL_TRAIN_SAMPLES} ejemplos de train.")
        print(f"  Para entrenamiento completo usa: py training/train_nli.py --full")
    else:
        print("  Modo COMPLETO — dataset entero (recomendado con GPU).")

    print(
        f"  Train: {len(dataset['train'])}  |  "
        f"Val: {len(dataset['validation'])}  |  "
        f"Test: {len(dataset['test'])}"
    )

    print(f"Cargando tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized = dataset.map(
        lambda x: _tokenize_nli(x, tokenizer),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    print(f"Cargando modelo base: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label={0: "APOYA", 1: "NO_VERIFICA", 2: "REFUTA"},
        label2id={"APOYA": 0, "NO_VERIFICA": 1, "REFUTA": 2},
    )

    warmup = 200 if full_dataset else 50

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_steps=warmup,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Iniciando entrenamiento NLI...")
    trainer.train()

    print(f"Guardando modelo en {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = trainer.evaluate(tokenized["test"])
    print(f"Resultados en test: {results}")
    return trainer


if __name__ == "__main__":
    full = "--full" in sys.argv
    train(full_dataset=full)
