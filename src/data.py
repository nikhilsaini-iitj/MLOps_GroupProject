"""
prepare_data.py — Task 2: Data Preparation & Normalisation
Loads dair-ai/emotion, inspects it, cleans text, builds id2label.json,
and saves processed splits locally (NOT committed).

Run:  python src/prepare_data.py
Owner: Y Sharathchandrika
"""
import json
import os
import re
import string
from collections import Counter

from datasets import load_dataset

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # 1. Load raw data
    ds = load_dataset("dair-ai/emotion")
    print("Splits:", {k: len(v) for k, v in ds.items()})

    # 2. Inspect class distribution (report this!)
    labels = ds["train"].features["label"].names
    dist = Counter(ds["train"]["label"])
    print("Classes:", labels)
    print("Train class distribution:",
          {labels[i]: dist[i] for i in range(len(labels))})

    # 3. Build id2label / label2id and save mapping (COMMIT THIS FILE)
    id2label = {str(i): name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}
    with open("id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    print("Wrote id2label.json:", id2label)

    # 4. Clean text + drop duplicates/empties per split
    def process(split):
        rows = ds[split]
        seen, texts, ys = set(), [], []
        for ex in rows:
            t = clean_text(ex["text"])
            if not t or t in seen:
                continue          # drop empty + duplicate
            seen.add(t)
            texts.append(t)
            ys.append(ex["label"])
        return {"text": texts, "label": ys}

    for split in ["train", "validation", "test"]:
        data = process(split)
        path = os.path.join(OUT_DIR, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"{split}: {len(data['text'])} rows -> {path}")


if __name__ == "__main__":
    main()
