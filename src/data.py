"""Data loading, cleaning, encoding, and train/test split.

Group Project — MLOps End-to-End Pipeline
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset from CSV or other format."""
    raise NotImplementedError("Implement based on your chosen dataset.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize data (handle nulls, duplicates, encoding).
    
    Document every decision in the report.
    """
    raise NotImplementedError("Implement based on your chosen dataset and modality.")


def encode_labels(df: pd.DataFrame, label_col: str, output_path: str = "id2label.json"):
    """Create id2label / label2id mappings and save to JSON."""
    unique_labels = sorted(df[label_col].unique().tolist())
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    
    mapping = {"id2label": id2label, "label2id": label2id}
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    
    return mapping


def split_data(df: pd.DataFrame, label_col: str, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )
    return train_df, test_df


if __name__ == "__main__":
    # Example workflow (replace with your dataset)
    # df = load_data("data/raw.csv")
    # df = clean_data(df)
    # mapping = encode_labels(df, label_col="label")
    # train_df, test_df = split_data(df, label_col="label")
    pass
