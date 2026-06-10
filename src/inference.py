"""Inference entrypoint for Docker and GitHub Actions.

Group Project — MLOps End-to-End Pipeline
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_from_hub(model_name: str, token: str = None):
    """Load tokenizer and model from Hugging Face Hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    return tokenizer, model


def predict(text: str, tokenizer, model, id2label: dict):
    """Run inference on a single input and return predicted label."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred_id = outputs.logits.argmax(-1).item()
    return id2label.get(str(pred_id), pred_id)


def main():
    """Read INPUT_TEXT and HF_MODEL_NAME from environment and predict."""
    input_text = os.environ.get("INPUT_TEXT", "")
    model_name = os.environ.get("HF_MODEL_NAME", os.environ.get("HF_MODEL", "your-username/your-model"))
    hf_token = os.environ.get("HF_TOKEN", None)

    if not input_text:
        print("ERROR: INPUT_TEXT environment variable not set.")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    tokenizer, model = load_model_from_hub(model_name, token=hf_token)

    # Load label map if available
    id2label = {}
    try:
        import json
        with open("id2label.json", "r") as f:
            mapping = json.load(f)
            id2label = mapping.get("id2label", {})
    except FileNotFoundError:
        print("Warning: id2label.json not found.")

    pred = predict(input_text, tokenizer, model, id2label)
    print(f"Input: {input_text}")
    print(f"Predicted: {pred}")


if __name__ == "__main__":
    main()
