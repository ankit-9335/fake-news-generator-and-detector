"""
NewsMorph - Fake News Detector Fine-Tuning

Fine-tunes DistilBERT on the WELFake dataset.
Labels: 0 = FAKE, 1 = REAL.
Designed for Google Colab/T4 GPU.
"""

import os
import random
import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

SEED = 42
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "newsmorph-fake-news-detector"
MAX_LENGTH = 512
TEST_SIZE = 0.10
VALID_SIZE = 0.10
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_text(example):
    title = clean_text(example.get("title"))
    body = clean_text(example.get("text"))
    if title and body:
        return f"{title} [SEP] {body}"
    return title or body


def prepare_dataset():
    print("Loading WELFake...")
    data = load_dataset("davanstrien/WELFake")["train"]

    data = data.filter(
        lambda x: x["label"] in [0, 1]
        and bool(clean_text(x["title"]) or clean_text(x["text"]))
    )

    data = data.map(lambda x: {"model_text": build_text(x)})

    # Remove exact duplicate articles before splitting to reduce leakage.
    before = len(data)
    seen = set()
    keep = []
    for i, text in enumerate(data["model_text"]):
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            keep.append(i)
    data = data.select(keep)
    print(f"Removed {before - len(data)} exact duplicates.")

    # Do stratification with scikit-learn instead of Dataset.train_test_split.
    # This avoids a torchvision dependency issue in recent Colab environments.
    indices = np.arange(len(data))
    labels = np.array(data["label"])

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    relative_valid = VALID_SIZE / (1.0 - TEST_SIZE)
    train_idx, valid_idx = train_test_split(
        train_val_idx,
        test_size=relative_valid,
        random_state=SEED,
        stratify=labels[train_val_idx],
    )

    return DatasetDict(
        train=data.select(train_idx.tolist()),
        validation=data.select(valid_idx.tolist()),
        test=data.select(test_idx.tolist()),
    )


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
        print("WARNING: CUDA GPU not detected. Use a Google Colab T4 for training.")

    dataset = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["model_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["title", "text", "model_text"],
    )

    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Starting fine-tuning...")
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
    print("Upload this directory to Hugging Face, then connect its model ID to Streamlit.")


if __name__ == "__main__":
    main()
