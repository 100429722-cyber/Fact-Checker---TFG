"""
Entrena un clasificador de afirmaciones verificables sobre XLM-RoBERTa.

Dataset preferido : CLEF CheckThat! 2022 (español)
Fallback          : dataset sintético de demo (ver dataset_utils.py)

Uso:
    python training/train_claim_detector.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import DatasetDict
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
    CLAIM_DETECTOR_MAX_LEN,
    CLAIM_DETECTOR_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from training.dataset_utils import (
    build_synthetic_claim_dataset,
    load_claim_detection_dataset,
    prepare_checkthat_for_training,
)


def _tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=CLAIM_DETECTOR_MAX_LEN,
    )


def _compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **acc_metric.compute(predictions=preds, references=labels),
        **f1_metric.compute(predictions=preds, references=labels, average="binary"),
    }


def train(output_dir: str = CLAIM_DETECTOR_PATH):
    print("Cargando dataset de detección de afirmaciones...")
    dataset = load_claim_detection_dataset()

    if dataset is None:
        print(
            "AVISO: Dataset CLEF CheckThat! no encontrado.\n"
            "Usando dataset sintético de demo. Para resultados reales descarga el dataset:\n"
            "  https://huggingface.co/datasets/clef2022_checkthat_v2\n"
        )
        dataset = build_synthetic_claim_dataset()
    else:
        print("Dataset CLEF CheckThat! cargado.")
        dataset = prepare_checkthat_for_training(dataset)

    eval_split = "test" if "test" in dataset else "validation"
    print(f"  Train: {len(dataset['train'])}  |  {eval_split}: {len(dataset[eval_split])}")

    print(f"Cargando tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized: DatasetDict = dataset.map(
        lambda x: _tokenize(x, tokenizer), batched=True
    )
    keep_cols = {"input_ids", "attention_mask", "label"}
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_cols]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    print(f"Cargando modelo base: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "NO_AFIRMACION", 1: "AFIRMACION"},
        label2id={"NO_AFIRMACION": 0, "AFIRMACION": 1},
    )

    args = TrainingArguments(
        output_dir=output_dir,
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
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Iniciando entrenamiento...")
    trainer.train()

    print(f"Guardando modelo en {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = trainer.evaluate()
    print(f"Resultados finales: {results}")
    return trainer


if __name__ == "__main__":
    train()
