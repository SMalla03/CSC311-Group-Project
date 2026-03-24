from __future__ import annotations

"""Run a 3-model majority-vote ensemble on the saved train/validation/test splits.

Usage:
    python models/ensemble_majority_vote.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (  # noqa: E402
    apply_feature_subset,
    classification_report,
    dense_load,
    load_feature_names,
    load_preprocessor,
    select_feature_indices,
)
from models.naive_bayes.train_naivebayes import (  # noqa: E402
    make_prediction,
    naive_bayes_map,
)


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "ensemble"
COMBINED_REPORT_PATH = DEFAULT_OUTPUT_DIR / "majority_vote_report.md"
TRAIN_ROWS_PATH = DEFAULT_OUTPUT_DIR / "train_majority_vote_rows.csv"
VALIDATION_ROWS_PATH = DEFAULT_OUTPUT_DIR / "validation_majority_vote_rows.csv"
TEST_ROWS_PATH = DEFAULT_OUTPUT_DIR / "test_majority_vote_rows.csv"
RF_MODEL_PATH = PROJECT_ROOT / "processed" / "models" / "random_forest" / "rf_model.joblib"
LOGISTIC_MODEL_PATH = PROJECT_ROOT / "processed" / "models" / "logistic" / "logistic_model.joblib"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_split_arrays(processed_dir: Path) -> tuple[np.ndarray, ...]:
    x_train = dense_load(processed_dir / "train_X.npz")
    y_train = np.load(processed_dir / "train_y.npy")
    x_val = dense_load(processed_dir / "val_X.npz")
    y_val = np.load(processed_dir / "val_y.npy")
    x_test = dense_load(processed_dir / "test_X.npz")
    y_test = np.load(processed_dir / "test_y.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_split_rows(processed_dir: Path, split_name: str) -> pd.DataFrame:
    path = processed_dir / f"{split_name}_rows.csv"
    require_file(path)
    return pd.read_csv(path)


def softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def naive_bayes_predict_with_probs(
    x_train_text: np.ndarray,
    y_train: np.ndarray,
    x_eval_text: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    priors, theta = naive_bayes_map(x_train_text, y_train, alpha)
    log_scores = x_eval_text @ np.log(theta) + np.log(priors)
    predictions = make_prediction(x_eval_text, priors, theta)
    probabilities = softmax_from_scores(log_scores)
    return predictions, probabilities


def load_bundle_predictions(
    model_path: Path,
    full_features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    require_file(model_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_set = bundle.get("feature_set", "full")
    feature_indices = select_feature_indices(feature_names, feature_set)
    x_subset = apply_feature_subset(full_features, feature_indices)
    predictions = model.predict(x_subset)
    probabilities = model.predict_proba(x_subset)
    return predictions, probabilities


def majority_vote_with_tiebreak(
    prediction_matrix: np.ndarray,
    probability_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_rows, _, num_classes = probability_tensor.shape
    final_predictions = np.empty(num_rows, dtype=np.int64)
    tie_flags = np.zeros(num_rows, dtype=bool)

    for row_index in range(num_rows):
        votes = np.bincount(prediction_matrix[row_index], minlength=num_classes)
        top_vote_count = int(votes.max())
        top_classes = np.flatnonzero(votes == top_vote_count)

        if len(top_classes) == 1:
            final_predictions[row_index] = int(top_classes[0])
            continue

        tie_flags[row_index] = True
        summed_probabilities = probability_tensor[row_index].sum(axis=0)
        tied_scores = summed_probabilities[top_classes]
        final_predictions[row_index] = int(top_classes[np.argmax(tied_scores)])

    return final_predictions, tie_flags


def build_row_results(
    split_rows: pd.DataFrame,
    split_name: str,
    y_true: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    model_probabilities: dict[str, np.ndarray],
    labels: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    model_order = ["naive_bayes", "random_forest", "logistic"]
    prediction_matrix = np.column_stack([model_predictions[name] for name in model_order])
    probability_tensor = np.stack([model_probabilities[name] for name in model_order], axis=1)
    ensemble_predictions, tie_flags = majority_vote_with_tiebreak(prediction_matrix, probability_tensor)
    report = classification_report(y_true, ensemble_predictions, labels)

    per_model_accuracy = {
        name: float((model_predictions[name] == y_true).mean())
        for name in model_order
    }

    row_frame = split_rows.copy()
    row_frame.insert(0, "split", split_name)
    row_frame["true_label"] = [labels[index] for index in y_true]
    row_frame["naive_bayes_pred"] = [labels[index] for index in model_predictions["naive_bayes"]]
    row_frame["random_forest_pred"] = [labels[index] for index in model_predictions["random_forest"]]
    row_frame["logistic_pred"] = [labels[index] for index in model_predictions["logistic"]]
    row_frame["ensemble_pred"] = [labels[index] for index in ensemble_predictions]
    row_frame["ensemble_correct"] = ensemble_predictions == y_true
    row_frame["vote_tie_resolved"] = tie_flags

    return (
        {
            "split": split_name,
            "ensemble_accuracy": report["accuracy"],
            "tie_count": int(tie_flags.sum()),
            "per_model_accuracy": per_model_accuracy,
            "classification_report": report,
        },
        row_frame,
    )


def append_split_report(lines: list[str], split_summary: dict[str, object]) -> None:
    report = split_summary["classification_report"]
    per_model_accuracy = split_summary["per_model_accuracy"]
    lines.extend(
        [
            f"## {split_summary['split'].title()} Results",
            "",
            f"- Ensemble accuracy: `{split_summary['ensemble_accuracy']:.4f}`",
            f"- Ties resolved by summed probabilities: `{split_summary['tie_count']}`",
            "",
            "### Per-Model Accuracy",
            "",
            "| Model | Accuracy |",
            "| --- | ---: |",
        ]
    )
    for model_name, accuracy_value in per_model_accuracy.items():
        lines.append(f"| {model_name} | {accuracy_value:.4f} |")

    lines.extend(
        [
            "",
            "### Ensemble Per-Class Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["per_class"]:
        lines.append(
            f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    lines.extend(["", "Confusion matrix (`rows=true`, `cols=predicted`):", ""])
    header = " | ".join(["true \\ pred", *[row["label"] for row in report["per_class"]]])
    divider = " | ".join(["---"] * (len(report["per_class"]) + 1))
    lines.append(f"| {header} |")
    lines.append(f"| {divider} |")
    for label, row_values in zip([row["label"] for row in report["per_class"]], report["confusion_matrix"]):
        lines.append(f"| {label} | {' | '.join(str(value) for value in row_values)} |")
    lines.extend(["", ""])


def write_combined_markdown_report(
    path: Path,
    train_summary: dict[str, object],
    validation_summary: dict[str, object],
    test_summary: dict[str, object],
) -> None:
    lines = [
        "# Majority Vote Ensemble Report",
        "",
        "Models included: `naive_bayes`, `random_forest`, `logistic`.",
        "Tie-breaking rule: sum the tied classes' predicted probabilities across the included models and choose the larger total.",
        "",
    ]
    append_split_report(lines, train_summary)
    append_split_report(lines, validation_summary)
    append_split_report(lines, test_summary)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    processed_dir = DEFAULT_PREPROCESSED_DIR
    require_file(processed_dir / "feature_names.json")
    require_file(processed_dir / "metadata.json")
    require_file(processed_dir / "train_X.npz")
    require_file(processed_dir / "val_X.npz")
    require_file(processed_dir / "test_X.npz")

    preprocessor = load_preprocessor(processed_dir)
    labels = preprocessor["label_encoder"].classes_
    feature_names = load_feature_names(processed_dir)
    text_feature_count = sum(
        1 for feature_name in feature_names if isinstance(feature_name, str) and feature_name.startswith("text:")
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_split_arrays(processed_dir)
    train_rows = load_split_rows(processed_dir, "train")
    val_rows = load_split_rows(processed_dir, "val")
    test_rows = load_split_rows(processed_dir, "test")
    x_train_text = x_train[:, :text_feature_count]
    x_val_text = x_val[:, :text_feature_count]
    x_test_text = x_test[:, :text_feature_count]

    nb_alpha = 6.32
    nb_train_pred, nb_train_prob = naive_bayes_predict_with_probs(x_train_text, y_train, x_train_text, nb_alpha)
    nb_val_pred, nb_val_prob = naive_bayes_predict_with_probs(x_train_text, y_train, x_val_text, nb_alpha)
    nb_test_pred, nb_test_prob = naive_bayes_predict_with_probs(x_train_text, y_train, x_test_text, nb_alpha)

    rf_train_pred, rf_train_prob = load_bundle_predictions(RF_MODEL_PATH, x_train, feature_names)
    rf_val_pred, rf_val_prob = load_bundle_predictions(RF_MODEL_PATH, x_val, feature_names)
    rf_test_pred, rf_test_prob = load_bundle_predictions(RF_MODEL_PATH, x_test, feature_names)

    logistic_train_pred, logistic_train_prob = load_bundle_predictions(LOGISTIC_MODEL_PATH, x_train, feature_names)
    logistic_val_pred, logistic_val_prob = load_bundle_predictions(LOGISTIC_MODEL_PATH, x_val, feature_names)
    logistic_test_pred, logistic_test_prob = load_bundle_predictions(LOGISTIC_MODEL_PATH, x_test, feature_names)

    train_summary, train_rows_output = build_row_results(
        train_rows,
        "train",
        y_train,
        {
            "naive_bayes": nb_train_pred,
            "random_forest": rf_train_pred,
            "logistic": logistic_train_pred,
        },
        {
            "naive_bayes": nb_train_prob,
            "random_forest": rf_train_prob,
            "logistic": logistic_train_prob,
        },
        list(labels),
    )
    validation_summary, validation_rows = build_row_results(
        val_rows,
        "validation",
        y_val,
        {
            "naive_bayes": nb_val_pred,
            "random_forest": rf_val_pred,
            "logistic": logistic_val_pred,
        },
        {
            "naive_bayes": nb_val_prob,
            "random_forest": rf_val_prob,
            "logistic": logistic_val_prob,
        },
        list(labels),
    )
    test_summary, test_rows_output = build_row_results(
        test_rows,
        "test",
        y_test,
        {
            "naive_bayes": nb_test_pred,
            "random_forest": rf_test_pred,
            "logistic": logistic_test_pred,
        },
        {
            "naive_bayes": nb_test_prob,
            "random_forest": rf_test_prob,
            "logistic": logistic_test_prob,
        },
        list(labels),
    )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_combined_markdown_report(COMBINED_REPORT_PATH, train_summary, validation_summary, test_summary)
    train_rows_output.to_csv(TRAIN_ROWS_PATH, index=False)
    validation_rows.to_csv(VALIDATION_ROWS_PATH, index=False)
    test_rows_output.to_csv(TEST_ROWS_PATH, index=False)

    print(f"Train majority-vote accuracy: {train_summary['ensemble_accuracy']:.4f}")
    print(f"Train ties resolved: {train_summary['tie_count']}")
    print(f"Validation majority-vote accuracy: {validation_summary['ensemble_accuracy']:.4f}")
    print(f"Validation ties resolved: {validation_summary['tie_count']}")
    print(f"Test majority-vote accuracy: {test_summary['ensemble_accuracy']:.4f}")
    print(f"Test ties resolved: {test_summary['tie_count']}")
    print(f"Saved combined report to: {COMBINED_REPORT_PATH}")
    print(f"Saved train row results to: {TRAIN_ROWS_PATH}")
    print(f"Saved validation row results to: {VALIDATION_ROWS_PATH}")
    print(f"Saved test row results to: {TEST_ROWS_PATH}")


if __name__ == "__main__":
    main()
