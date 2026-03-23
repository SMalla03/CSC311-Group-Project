"""Pure Python/NumPy inference wrapper for the trained random forest classifier."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (  # noqa: E402
    apply_feature_subset,
    load_feature_names,
    load_preprocessor,
    preprocess_frame,
    read_rows,
    select_feature_indices,
)


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "processed" / "models" / "random_forest" / "rf_inference_params.pkl"


def _predict_tree(tree: dict[str, list], row: np.ndarray) -> np.ndarray:
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    features = tree["feature"]
    thresholds = tree["threshold"]
    values = tree["value"]

    node = 0
    while children_left[node] != children_right[node]:
        feature_index = features[node]
        threshold = thresholds[node]
        if row[feature_index] <= threshold:
            node = children_left[node]
        else:
            node = children_right[node]
    return np.array(values[node], dtype=np.float32)


def predict_all(filename: str) -> list[str]:
    with DEFAULT_PARAMS_PATH.open("rb") as handle:
        saved = pickle.load(handle)

    labels = saved["labels"]
    feature_set = saved["feature_set"]
    trees = saved["trees"]

    preprocessor = load_preprocessor(DEFAULT_PREPROCESSED_DIR)
    frame = read_rows(filename, preprocessor)
    feature_indices = select_feature_indices(load_feature_names(DEFAULT_PREPROCESSED_DIR), feature_set)
    x = apply_feature_subset(preprocess_frame(frame, preprocessor), feature_indices)

    predictions: list[str] = []
    for row in x:
        class_totals = None
        for tree in trees:
            leaf_values = _predict_tree(tree, row)
            class_totals = leaf_values if class_totals is None else class_totals + leaf_values
        predictions.append(labels[int(np.argmax(class_totals))])
    return predictions

__all__ = ["predict_all"]
