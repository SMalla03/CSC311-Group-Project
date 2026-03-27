from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as prep


def ensure_preprocessor_pickle_compatibility() -> None:
    """Expose preprocessing classes under __main__ so the existing pickle loads."""
    import __main__

    __main__.NumericColumnSpec = prep.NumericColumnSpec
    __main__.NumericPreprocessor = prep.NumericPreprocessor
    __main__.SimpleTfidfVectorizer = prep.SimpleTfidfVectorizer
    __main__.SimpleMultiLabelBinarizer = prep.SimpleMultiLabelBinarizer
    __main__.SimpleLabelEncoder = prep.SimpleLabelEncoder


def load_preprocessor(processed_dir: Path | str) -> dict:
    processed_dir = Path(processed_dir)
    ensure_preprocessor_pickle_compatibility()
    with (processed_dir / "preprocessor.pkl").open("rb") as handle:
        return pickle.load(handle)


def dense_load(path: Path | str) -> np.ndarray:
    with np.load(Path(path)) as loaded:
        return loaded["data"].astype(np.float32)


def load_feature_names(processed_dir: Path | str) -> list[str]:
    processed_dir = Path(processed_dir)
    with (processed_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_feature_indices(feature_names: list[str], feature_set: str) -> np.ndarray:
    if feature_set == "full":
        selected = list(range(len(feature_names)))
    elif feature_set == "text_season_feel":
        selected = [
            index
            for index, name in enumerate(feature_names)
            if name.startswith("text:") or name.startswith("season=") or name.startswith("feel_")
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if not selected:
        raise ValueError("No features matched the configured subset.")
    return np.array(selected, dtype=np.int64)


def apply_feature_subset(x: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    return x[:, feature_indices]


def preprocess_frame(frame: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    text_columns = preprocessor["text_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    vectorizer = preprocessor["vectorizer"]
    categorical_encoders = preprocessor["categorical_encoders"]
    numeric_processor = preprocessor["numeric_processor"]

    text_documents = prep.combine_text_columns(frame, text_columns)
    text_matrix = vectorizer.transform(text_documents)

    categorical_parts = []
    for feature_name, source_column in categorical_columns.items():
        values = frame[source_column].map(prep.split_multivalue_cell)
        categorical_parts.append(categorical_encoders[feature_name].transform(values))
    categorical_matrix = np.hstack(categorical_parts).astype(np.float32)

    numeric_matrix = numeric_processor.transform(frame)
    full_matrix = np.hstack([text_matrix, categorical_matrix, numeric_matrix]).astype(np.float32)
    return full_matrix


def read_rows(csv_path: Path | str, preprocessor: dict) -> pd.DataFrame:
    frame = pd.read_csv(Path(csv_path))
    missing = [column for column in preprocessor["resolved_columns"].values() if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")
    return frame


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_value, pred_value] += 1
    return matrix


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    per_class: list[dict[str, object]] = []

    for class_index, label in enumerate(labels):
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        support = int(matrix[class_index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    accuracy_value = float((y_true == y_pred).mean())
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    weighted_f1 = float(sum(row["f1"] * row["support"] for row in per_class) / max(1, len(y_true)))

    return {
        "accuracy": accuracy_value,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "support": int(len(y_true)),
        },
        "weighted_avg": {
            "f1": weighted_f1,
            "support": int(len(y_true)),
        },
        "confusion_matrix": matrix.tolist(),
    }
