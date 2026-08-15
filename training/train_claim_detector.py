"""
NewsMorph - Binary Short-Claim Detector

Fine-tunes RoBERTa on the LIAR short-statement dataset.

We intentionally REMOVE the ambiguous "half-true" examples instead of
forcing them into FAKE or REAL. The remaining labels are:
    FAKE = false, barely-true, pants-fire
    REAL = mostly-true, true

This model is for classifying claim-like text patterns. It is NOT a live
fact checker and cannot prove that a current event happened.
"""

import csv
import os
import random
import urllib.request

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "newsmorph-claim-detector"
MAX_LENGTH = 128
EPOCHS = 4
LEARNING_RATE = 1.5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01

LABEL_MAP = {
    "false": 0,
    "mostly-true": 1,
    "true": 1,
    "barely-true": 0,
    "pants-fire": 0,
    # half-true is deliberately excluded as ambiguous.
}

ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}

BASE_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/"
DATA_DIR = "liar_data"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_split(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.tsv")
    if not os.path.exists(path):
        url = BASE_URL + f"{name}.tsv"
        print(f"Downloading {name}.tsv...")
        urllib.request.urlretrieve(url, path)
    return path


def load_split(name):
    path = download_split(name)
    statements = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            label_name = row[1].strip().lower()
            statement = row[2].strip()

            if label_name not in LABEL_MAP or not statement:
                continue

            statements.append(statement)
            labels.append(LABEL_MAP[label_name])

    return Dataset.from_dict({"statement": statements, "labels": labels})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    set_seed()

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA GPU not detected. Use Colab T4.")

    print("Loading LIAR TSV files...")
    dataset = DatasetDict({
        "train": load_split("train"),
        "validation": load_split("valid"),
        "test": load_split("test"),
    })

    print(
        f"Dataset sizes after removing half-true: "
        f"train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}, "
        f"test={len(dataset['test'])}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["statement"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["statement"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting binary LIAR claim fine-tuning...")
    trainer.train()

    print("\nValidation metrics:")
    print(trainer.evaluate(tokenized["validation"]))

    print("\nTest metrics:")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    predictions = trainer.predict(tokenized["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    print("\nConfusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_true, y_pred))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
        for key, value in sorted(test_metrics.items()):
            f.write(f"{key}: {value}\n")

    print(f"\nModel saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
