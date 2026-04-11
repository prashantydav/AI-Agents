import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from config import DATA_DIR


def load_documents(source_path: Path = None) -> List[dict]:
    path = source_path or DATA_DIR / "sample_docs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Document source not found: {path}")

    df = pd.read_csv(path)
    documents = []
    for _, row in df.iterrows():
        documents.append(
            {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
            }
        )

    return documents


def to_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value
