from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from common import load_preprocessor, preprocess_frame, read_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "processed" / "models"
NB_PARAMS_PATH = DEFAULT_MODELS_DIR / "naive_bayes" / "nb_inference_params.npz"
LOGISTIC_PARAMS_PATH = DEFAULT_MODELS_DIR / "logistic" / "logistic_inference_params.npz"
RF_PARAMS_PATH = DEFAULT_MODELS_DIR / "random_forest" / "rf_inference_params.pkl"


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def load_nb_params(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    params = np.load(path)
    return params["pi"], params["theta"], int(params["text_feature_count"])


def predict_nb(x_full: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    priors, theta, text_feature_count = load_nb_params(path)
    x_text = x_full[:, :text_feature_count]
    log_scores = x_text @ np.log(theta) + np.log(priors)
    predictions = np.argmax(log_scores, axis=1)
    probabilities = softmax(log_scores)
    return predictions, probabilities


def load_logistic_params(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.load(path, allow_pickle=True)
    return params["coef"], params["intercept"], params["classes"]


def predict_logistic(x_full: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    coef, intercept, _ = load_logistic_params(path)
    scores = x_full @ coef.T + intercept
    probabilities = softmax(scores)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def _tree_leaf_distribution(tree: dict[str, list], row: np.ndarray) -> np.ndarray:
    node_index = 0
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    features = tree["feature"]
    thresholds = tree["threshold"]
    values = tree["value"]

    while children_left[node_index] != children_right[node_index]:
        feature_index = features[node_index]
        threshold = thresholds[node_index]
        if row[feature_index] <= threshold:
            node_index = children_left[node_index]
        else:
            node_index = children_right[node_index]

    leaf_values = np.asarray(values[node_index], dtype=np.float64)
    total = leaf_values.sum()
    if total <= 0:
        return np.full_like(leaf_values, 1.0 / len(leaf_values), dtype=np.float64)
    return leaf_values / total


def predict_random_forest(x_full: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    trees = payload["trees"]
    num_rows = x_full.shape[0]
    num_classes = len(payload["labels"])
    probabilities = np.zeros((num_rows, num_classes), dtype=np.float64)

    for tree in trees:
        tree_probs = np.vstack([_tree_leaf_distribution(tree, row) for row in x_full])
        probabilities += tree_probs

    probabilities /= max(1, len(trees))
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def majority_vote_with_tiebreak(
    prediction_matrix: np.ndarray,
    probability_tensor: np.ndarray,
) -> np.ndarray:
    num_rows, _, num_classes = probability_tensor.shape
    final_predictions = np.empty(num_rows, dtype=np.int64)

    for row_index in range(num_rows):
        votes = np.bincount(prediction_matrix[row_index], minlength=num_classes)
        top_vote_count = int(votes.max())
        top_classes = np.flatnonzero(votes == top_vote_count)

        if len(top_classes) == 1:
            final_predictions[row_index] = int(top_classes[0])
            continue

        summed_probabilities = probability_tensor[row_index].sum(axis=0)
        final_predictions[row_index] = int(top_classes[np.argmax(summed_probabilities[top_classes])])

    return final_predictions


def predict_all(filename: str) -> list[str]:
    preprocessor = load_preprocessor(DEFAULT_PREPROCESSED_DIR)
    frame = read_rows(filename, preprocessor)
    x_full = preprocess_frame(frame, preprocessor)

    nb_pred, nb_prob = predict_nb(x_full, NB_PARAMS_PATH)
    logistic_pred, logistic_prob = predict_logistic(x_full, LOGISTIC_PARAMS_PATH)
    rf_pred, rf_prob = predict_random_forest(x_full, RF_PARAMS_PATH)

    prediction_matrix = np.column_stack([nb_pred, rf_pred, logistic_pred])
    probability_tensor = np.stack([nb_prob, rf_prob, logistic_prob], axis=1)
    final_predictions = majority_vote_with_tiebreak(prediction_matrix, probability_tensor)

    labels = np.asarray(preprocessor["label_encoder"].classes_, dtype=object)
    return labels[final_predictions].tolist()
