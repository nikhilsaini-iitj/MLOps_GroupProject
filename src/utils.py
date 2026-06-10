"""Shared helpers: label maps, dataset class, compute_metrics.

Group Project — MLOps End-to-End Pipeline
"""
import json
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score


def load_label_map(path: str = "id2label.json") -> Dict:
    """Load id2label / label2id mapping."""
    with open(path, "r") as f:
        return json.load(f)


class TextDataset:
    """Simple torch Dataset wrapper for Hugging Face Datasets.
    
    Adapt for your modality (text / image / tabular / audio).
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """Return accuracy and weighted F1 for Hugging Face Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}
