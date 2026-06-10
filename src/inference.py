"""
inference.py — used by Docker (Task 6) and the GitHub Actions inference workflow (Task 7).
Reads model from HF_MODEL_NAME and text from INPUT_TEXT, prints the predicted label.

Owner: Aryaveer Rathi
"""
import os

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME", "Nikhil-iitj/emotion-electra")  # default = winning v2 model
INPUT_TEXT = os.environ.get("INPUT_TEXT", "I am so happy today!")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    enc = tok(INPUT_TEXT, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = int(logits.argmax(-1))
    label = model.config.id2label[pred_id]
    score = torch.softmax(logits, -1).max().item()

    print(f"Input : {INPUT_TEXT}")
    print(f"Label : {label}  (confidence {score:.3f})")


if __name__ == "__main__":
    main()
