"""Pure NumPy inference wrapper for the trained MLP painting classifier."""

from __future__ import annotations

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
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "processed" / "models" / "mlp" / "mlp_inference_params.npz"


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def predict_all(filename: str) -> list[str]:
    saved = np.load(DEFAULT_PARAMS_PATH, allow_pickle=True)
    w1 = saved["w1"]
    b1 = saved["b1"]
    w2 = saved["w2"]
    b2 = saved["b2"]
    labels = saved["labels"].tolist()
    feature_set = saved["feature_set"].item()

    preprocessor = load_preprocessor(DEFAULT_PREPROCESSED_DIR)
    frame = read_rows(filename, preprocessor)
    feature_indices = select_feature_indices(load_feature_names(DEFAULT_PREPROCESSED_DIR), feature_set)
    x = apply_feature_subset(preprocess_frame(frame, preprocessor), feature_indices)

    hidden = relu(x @ w1 + b1)
    logits = hidden @ w2 + b2
    predictions = np.argmax(logits, axis=1)
    return [labels[index] for index in predictions]

__all__ = ["predict_all"]
