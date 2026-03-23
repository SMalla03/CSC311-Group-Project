from __future__ import annotations

"""Train and run a scikit-learn MLP for painting prediction.

Usage:
    python models/mlp/mlp_model.py --train
    python models/mlp/mlp_model.py --predict data/training_data_202601.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (  # noqa: E402
    apply_feature_subset,
    classification_report,
    dense_load,
    load_feature_names,
    load_preprocessor,
    preprocess_frame,
    read_rows,
    select_feature_indices,
)


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "mlp"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "mlp_model.joblib"
DEFAULT_INFERENCE_PARAMS_PATH = DEFAULT_OUTPUT_DIR / "mlp_inference_params.npz"
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 25
DEFAULT_TOP_FEATURES = 12
DEFAULT_FEATURE_SET = "full"
RUN_LOG_PATH = DEFAULT_OUTPUT_DIR / "mlp_training_log.csv"


def load_processed_splits(processed_dir: Path) -> tuple[np.ndarray, ...]:
    x_train = dense_load(processed_dir / "train_X.npz")
    y_train = np.load(processed_dir / "train_y.npy")
    x_val = dense_load(processed_dir / "val_X.npz")
    y_val = np.load(processed_dir / "val_y.npy")
    x_test = dense_load(processed_dir / "test_X.npz")
    y_test = np.load(processed_dir / "test_y.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def top_feature_rows(weights: np.ndarray, feature_names: list[str], top_k: int) -> list[dict[str, float | str]]:
    indices = np.argsort(np.abs(weights))[::-1][:top_k]
    return [
        {"feature": feature_names[index], "weight": float(weights[index]), "abs_weight": float(abs(weights[index]))}
        for index in indices
    ]


def summarize_hidden_units(
    model: MLPClassifier,
    feature_names: list[str],
    labels: list[str],
    top_units: int = 6,
    top_features: int = 8,
) -> list[dict[str, object]]:
    input_to_hidden = model.coefs_[0]
    hidden_to_output = model.coefs_[-1]
    hidden_strength = np.max(np.abs(hidden_to_output), axis=1)
    unit_indices = np.argsort(hidden_strength)[::-1][:top_units]
    rows: list[dict[str, object]] = []
    for unit_index in unit_indices:
        class_index = int(np.argmax(np.abs(hidden_to_output[unit_index])))
        rows.append(
            {
                "hidden_unit": int(unit_index),
                "strongest_output_class": labels[class_index],
                "output_weight": float(hidden_to_output[unit_index, class_index]),
                "top_input_features": top_feature_rows(input_to_hidden[:, unit_index], feature_names, top_features),
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_training_log(path: Path, run_summary: dict[str, object], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature_set",
        "selected_feature_count",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "patience",
        "seed",
        "best_epoch",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
    ]
    row = {**run_summary, **metrics}
    frame = pd.DataFrame([{column: row[column] for column in columns}])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def write_training_report(path: Path, run_summary: dict[str, object], split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MLP Training Report",
        "",
        "## Run Summary",
        "",
        f"- Feature subset: `{run_summary['feature_set']}`",
        f"- Selected feature count: `{run_summary['selected_feature_count']}`",
        f"- Hidden size: `{run_summary['hidden_dim']}`",
        f"- Learning rate: `{run_summary['learning_rate']}`",
        f"- Weight decay: `{run_summary['weight_decay']}`",
        f"- Batch size: `{run_summary['batch_size']}`",
        f"- Max epochs: `{run_summary['epochs']}`",
        f"- Patience: `{run_summary['patience']}`",
        f"- Seed: `{run_summary['seed']}`",
        f"- Best epoch: `{run_summary['best_epoch']}`",
        f"- Solver: `{run_summary['solver']}`",
        f"- Activation: `{run_summary['activation']}`",
        "",
    ]
    for split_name, report in split_reports.items():
        lines.extend(
            [
                f"## {split_name.title()} Metrics",
                "",
                f"- Accuracy: `{report['accuracy']:.4f}`",
                f"- Macro F1: `{report['macro_avg']['f1']:.4f}`",
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
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_weight_summary(path: Path, model: MLPClassifier, feature_names: list[str], labels: list[str], top_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    input_to_hidden = model.coefs_[0]
    hidden_to_output = model.coefs_[-1]
    hidden_bias = model.intercepts_[0]
    output_bias = model.intercepts_[-1]
    lines = [
        "# MLP Weight Summary",
        "",
        "This file is a readable summary of the learned weights from scikit-learn's `MLPClassifier`.",
        f"The full trained model is stored in `{DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()}`, but this report highlights the strongest patterns.",
        "",
        "## Output Layer",
        "",
        "Each class score is computed from the hidden layer through the second weight matrix and output biases.",
        "",
    ]
    for class_index, label in enumerate(labels):
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Output bias: `{output_bias[class_index]:.6f}`",
                "",
                "| Hidden Unit | Weight To Class |",
                "| --- | ---: |",
            ]
        )
        top_hidden = np.argsort(np.abs(hidden_to_output[:, class_index]))[::-1][:top_features]
        for unit_index in top_hidden:
            lines.append(f"| {unit_index} | {hidden_to_output[unit_index, class_index]:.6f} |")
        lines.append("")
    lines.extend(["", "## Hidden Units And Their Strongest Input Features", "", "These are the hidden units with the strongest downstream effect on at least one class.", ""])
    for row in summarize_hidden_units(model, feature_names, labels, top_features=top_features):
        unit_index = row["hidden_unit"]
        lines.extend(
            [
                f"### Hidden Unit {unit_index}",
                "",
                f"- Hidden bias: `{hidden_bias[unit_index]:.6f}`",
                f"- Strongest output class: `{row['strongest_output_class']}`",
                f"- Output weight to that class: `{row['output_weight']:.6f}`",
                "",
                "| Input Feature | Weight | abs(weight) |",
                "| --- | ---: | ---: |",
            ]
        )
        for feature_row in row["top_input_features"]:
            lines.append(
                f"| {feature_row['feature']} | {feature_row['weight']:.6f} | {feature_row['abs_weight']:.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_performance_summary(path: Path, split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_report = split_reports["test"]
    best_test_class = max(test_report["per_class"], key=lambda row: row["f1"])
    weakest_test_class = min(test_report["per_class"], key=lambda row: row["f1"])
    lines = [
        "# MLP Performance Summary",
        "",
        (
            f"The sklearn MLP was trained on the full saved preprocessing pipeline and reached "
            f"`{split_reports['train']['accuracy']:.4f}` training accuracy, "
            f"`{split_reports['validation']['accuracy']:.4f}` validation accuracy, and "
            f"`{split_reports['test']['accuracy']:.4f}` test accuracy on the latest run."
        ),
        "",
        (
            f"On the test set, the strongest class by F1 was `{best_test_class['label']}` "
            f"with precision `{best_test_class['precision']:.4f}`, recall `{best_test_class['recall']:.4f}`, "
            f"and F1 `{best_test_class['f1']:.4f}`. The weakest class was `{weakest_test_class['label']}` "
            f"with precision `{weakest_test_class['precision']:.4f}`, recall `{weakest_test_class['recall']:.4f}`, "
            f"and F1 `{weakest_test_class['f1']:.4f}`."
        ),
        "",
        "Use the full training report for confusion matrices and the weight summary for the strongest hidden-unit patterns.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_inference_params(path: Path, model: MLPClassifier, labels: list[str], feature_set: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w1=model.coefs_[0].astype(np.float32),
        b1=model.intercepts_[0].astype(np.float32),
        w2=model.coefs_[1].astype(np.float32),
        b2=model.intercepts_[1].astype(np.float32),
        labels=np.array(labels, dtype=object),
        feature_set=np.array(feature_set, dtype=object),
    )


def train_model(
    preprocessed_dir: Path,
    output_dir: Path,
    model_path: Path,
    feature_set: str,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_splits(preprocessed_dir)
    preprocessor = load_preprocessor(preprocessed_dir)
    label_names = preprocessor["label_encoder"].classes_
    full_feature_names = load_feature_names(preprocessed_dir)
    feature_indices = select_feature_indices(full_feature_names, feature_set)
    feature_names = [full_feature_names[index] for index in feature_indices.tolist()]
    x_train = apply_feature_subset(x_train, feature_indices)
    x_val = apply_feature_subset(x_val, feature_indices)
    x_test = apply_feature_subset(x_test, feature_indices)

    model = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation="relu",
        solver="adam",
        alpha=weight_decay,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=patience,
        random_state=seed,
        shuffle=True,
        verbose=False,
    )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    split_reports = {
        "train": classification_report(y_train, train_pred, label_names),
        "validation": classification_report(y_val, val_pred, label_names),
        "test": classification_report(y_test, test_pred, label_names),
    }
    metrics = {
        "train_accuracy": split_reports["train"]["accuracy"],
        "val_accuracy": split_reports["validation"]["accuracy"],
        "test_accuracy": split_reports["test"]["accuracy"],
    }
    run_summary = {
        "feature_set": feature_set,
        "selected_feature_count": len(feature_names),
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "best_epoch": int(getattr(model, "n_iter_", epochs)),
        "solver": "adam",
        "activation": "relu",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": label_names, "feature_set": feature_set}, model_path)
    export_inference_params(output_dir / DEFAULT_INFERENCE_PARAMS_PATH.name, model, label_names, feature_set)
    write_json(output_dir / "mlp_metrics.json", split_reports)
    write_training_report(output_dir / "mlp_training_report.md", run_summary, split_reports)
    write_weight_summary(output_dir / "mlp_weight_summary.md", model, feature_names, label_names, DEFAULT_TOP_FEATURES)
    write_performance_summary(output_dir / "mlp_performance_summary.md", split_reports)
    append_training_log(output_dir / RUN_LOG_PATH.name, run_summary, metrics)
    return metrics


def predict_all(
    filename: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> list[str]:
    preprocessor = load_preprocessor(preprocessed_dir)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    labels = bundle["labels"]
    effective_feature_set = bundle.get("feature_set", feature_set)
    frame = read_rows(filename, preprocessor)
    feature_indices = select_feature_indices(load_feature_names(preprocessed_dir), effective_feature_set)
    x = apply_feature_subset(preprocess_frame(frame, preprocessor), feature_indices)
    predictions = model.predict(x)
    return [labels[index] for index in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a scikit-learn MLP for painting prediction.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train", action="store_true", help="Train the model on processed train/val/test splits.")
    parser.add_argument("--predict", type=Path, help="Predict painting labels for a CSV file.")
    parser.add_argument("--feature-set", choices=["full", "text_season_feel"], default=DEFAULT_FEATURE_SET)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()

    if args.train:
        metrics = train_model(
            preprocessed_dir=args.preprocessed_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            feature_set=args.feature_set,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))

    if args.predict is not None:
        predictions = predict_all(
            filename=str(args.predict),
            model_path=args.model_path,
            preprocessed_dir=args.preprocessed_dir,
            feature_set=args.feature_set,
        )
        print(json.dumps(predictions, indent=2))

    if not args.train and args.predict is None:
        parser.error("Choose at least one of --train or --predict.")


if __name__ == "__main__":
    main()
