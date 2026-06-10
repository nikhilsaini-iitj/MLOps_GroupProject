"""Training script (local use only).

All actual training must happen on Kaggle Notebooks with W&B tracking.
This file is for reproducibility and local testing.

Group Project — MLOps End-to-End Pipeline
"""
import os
import wandb
from transformers import TrainingArguments, Trainer

from utils import compute_metrics, load_label_map, TextDataset


def train_model(model, tokenizer, train_dataset, eval_dataset, config: dict):
    """Fine-tune model with W&B tracking.
    
    config keys:
        - epochs, batch_size, learning_rate, warmup_steps, weight_decay,
        - output_dir, run_name, project
    """
    wandb.init(
        project=config.get("project", "mlops-groupproject"),
        name=config.get("run_name", "run-v1"),
        config=config,
    )

    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./results"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", 32),
        warmup_steps=config.get("warmup_steps", 100),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=config.get("logging_steps", 50),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=config.get("run_name", "run-v1"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()
    return trainer


if __name__ == "__main__":
    # This script is a template — actual training runs on Kaggle.
    pass
