"""
train.py — Task 4: Fine-tune on Kaggle + track on W&B
Same script runs BOTH versions; change CONFIG between runs (one axis at a time).

On Kaggle: enable GPU T4, add WANDB_API_KEY + HF_TOKEN as Kaggle Secrets.
Owner: Sarthak Kapoor
"""
import json
import os

import numpy as np
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# ─────────────────────────────────────────────────────────────
# CONFIG — change for each version
# v1: model="microsoft/MiniLM-L12-H384-uncased", lr=3e-5, epochs=3
# v2: model="google/electra-small-discriminator", lr=5e-5, epochs=4
CONFIG = {
    "model_name":    "microsoft/MiniLM-L12-H384-uncased",
    "version":       "v1",
    "epochs":        3,
    "batch_size":    16,
    "learning_rate": 3e-5,
    "max_length":    128,
}
# ─────────────────────────────────────────────────────────────


def load_secrets():
    """Kaggle Secrets -> env. Falls back to existing env locally."""
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        s = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = s.get_secret("WANDB_API_KEY")
        login(token=s.get_secret("HF_TOKEN"))
    except Exception as e:
        print("Not on Kaggle / secrets unavailable:", e)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    load_secrets()

    with open("id2label.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    ds = load_dataset("dair-ai/emotion")
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    def tokenize(batch):
        return tok(batch["text"], truncation=True,
                   padding="max_length", max_length=CONFIG["max_length"])

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    ds.set_format("torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    wandb.init(
        project="mlops-assignment3",
        name=f"run-{CONFIG['version']}",
        config=CONFIG,
    )

    args = TrainingArguments(
        output_dir=f"./results-{CONFIG['version']}",
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="wandb",
        run_name=f"run-{CONFIG['version']}",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(ds["test"])
    print("TEST:", test_metrics)
    wandb.run.summary.update(test_metrics)

    # Save locally; push to HF in Task 5 (best run only)
    trainer.save_model(f"./best-{CONFIG['version']}")
    tok.save_pretrained(f"./best-{CONFIG['version']}")
    wandb.finish()


if __name__ == "__main__":
    main()
